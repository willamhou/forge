use forge_core::{Backend, ModelConfig, Result};
use forge_loader::SafeTensorsLoader;

use crate::layers::{LlamaAttention, LlamaDecoderLayer, LlamaMLP, RMSNorm};
use crate::model::LlamaModel;
use crate::rope::RopeFreqs;

/// Load a Llama model from SafeTensors files.
///
/// HuggingFace stores linear weights as `[out_features, in_features]`.
/// Our `matmul(x, W)` needs `[in_features, out_features]`, so all linear
/// weights are transposed at load time.
pub fn load_llama_model<B: Backend + Clone>(
    loader: &SafeTensorsLoader,
    config: ModelConfig,
    backend: &B,
) -> Result<LlamaModel<B>> {
    let embed_tokens = loader.load_tensor("model.embed_tokens.weight", backend)?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let layer = load_decoder_layer(loader, &prefix, &config, backend)?;
        layers.push(layer);
    }

    let norm_weight = loader.load_tensor("model.norm.weight", backend)?;
    let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

    // lm_head may share weights with embed_tokens (tied embeddings).
    // lm_head is [vocab_size, hidden_size] -> transpose to [hidden_size, vocab_size]
    let lm_head_raw = match loader.load_tensor("lm_head.weight", backend) {
        Ok(t) => t,
        Err(_) => loader.load_tensor("model.embed_tokens.weight", backend)?,
    };
    let lm_head = backend.transpose(&lm_head_raw, 0, 1)?;

    // Cap RoPE precomputation to avoid huge allocations
    let rope_max_len = config.max_position_embeddings.min(8192);
    let rope_freqs = RopeFreqs::precompute(&config, rope_max_len, backend)?;

    Ok(LlamaModel::new(
        config,
        embed_tokens,
        layers,
        norm,
        lm_head,
        rope_freqs,
        backend.clone(),
    ))
}

/// Load and transpose a linear weight: [out, in] -> [in, out].
fn load_linear<B: Backend>(
    loader: &SafeTensorsLoader,
    name: &str,
    backend: &B,
) -> Result<B::Tensor> {
    let raw = loader.load_tensor(name, backend)?;
    backend.transpose(&raw, 0, 1)
}

fn load_decoder_layer<B: Backend>(
    loader: &SafeTensorsLoader,
    prefix: &str,
    config: &ModelConfig,
    backend: &B,
) -> Result<LlamaDecoderLayer<B>> {
    // Attention weights (transposed at load)
    let wq = load_linear(loader, &format!("{prefix}.self_attn.q_proj.weight"), backend)?;
    let wk = load_linear(loader, &format!("{prefix}.self_attn.k_proj.weight"), backend)?;
    let wv = load_linear(loader, &format!("{prefix}.self_attn.v_proj.weight"), backend)?;
    let wo = load_linear(loader, &format!("{prefix}.self_attn.o_proj.weight"), backend)?;
    let attn = LlamaAttention::new(wq, wk, wv, wo, config);

    // MLP weights (transposed at load)
    let gate_proj = load_linear(loader, &format!("{prefix}.mlp.gate_proj.weight"), backend)?;
    let up_proj = load_linear(loader, &format!("{prefix}.mlp.up_proj.weight"), backend)?;
    let down_proj = load_linear(loader, &format!("{prefix}.mlp.down_proj.weight"), backend)?;
    let mlp = LlamaMLP::new(gate_proj, up_proj, down_proj);

    // LayerNorm weights (1D, no transpose needed)
    let input_ln_weight =
        loader.load_tensor(&format!("{prefix}.input_layernorm.weight"), backend)?;
    let input_ln = RMSNorm::new(input_ln_weight, config.rms_norm_eps);

    let post_attn_ln_weight =
        loader.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"), backend)?;
    let post_attn_ln = RMSNorm::new(post_attn_ln_weight, config.rms_norm_eps);

    Ok(LlamaDecoderLayer::new(input_ln, attn, post_attn_ln, mlp))
}
