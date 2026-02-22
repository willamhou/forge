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

    let rope_freqs = RopeFreqs::precompute(&config, config.max_position_embeddings, backend)?;

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

    // Concatenate wq/wk/wv into a single wqkv tensor at load time.
    // load_linear produces [hidden, proj_size]. We need to cat along the proj dimension (dim=1),
    // but cat only supports dim=0, so: transpose → cat along dim=0 → transpose back.
    let wq_t = backend.transpose(&wq, 0, 1)?; // [q_proj, hidden]
    let wk_t = backend.transpose(&wk, 0, 1)?; // [kv_proj, hidden]
    let wv_t = backend.transpose(&wv, 0, 1)?; // [kv_proj, hidden]
    let cat_t = backend.cat(&[&wq_t, &wk_t, &wv_t], 0)?; // [q+2*kv, hidden]
    let wqkv = backend.transpose(&cat_t, 0, 1)?; // [hidden, q+2*kv]
    let attn = LlamaAttention::new(wqkv, wo, config);

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
