use forge_core::{Backend, ForgeError, KvCache, Model, ModelConfig, ModelInput, ModelOutput, Result};

use crate::layers::{LlamaDecoderLayer, RMSNorm};
use crate::rope::RopeFreqs;

pub struct LlamaModel<B: Backend> {
    config: ModelConfig,
    embed_tokens: B::Tensor,
    layers: Vec<LlamaDecoderLayer<B>>,
    norm: RMSNorm<B>,
    lm_head: B::Tensor,
    rope_freqs: RopeFreqs<B>,
    backend: B,
}

impl<B: Backend> LlamaModel<B> {
    pub fn new(
        config: ModelConfig,
        embed_tokens: B::Tensor,
        layers: Vec<LlamaDecoderLayer<B>>,
        norm: RMSNorm<B>,
        lm_head: B::Tensor,
        rope_freqs: RopeFreqs<B>,
        backend: B,
    ) -> Self {
        Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope_freqs,
            backend,
        }
    }
}

impl<B: Backend + Clone> LlamaModel<B> {
    /// Single-sequence forward pass (prefill or decode).
    fn forward_single(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
    ) -> Result<ModelOutput<B::Tensor>> {
        let seq_meta = &input.seq_metadata[0];
        let token_ids = &input.token_ids[0];

        let hidden = self.backend.embedding(&self.embed_tokens, token_ids)?;

        // Position offset for RoPE: absolute position of the first token.
        let pos_offset = if seq_meta.is_prefill {
            seq_meta.prompt_len.saturating_sub(token_ids.len())
        } else {
            seq_meta.prompt_len + seq_meta.generated_len - token_ids.len()
        };

        let mut hidden = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden,
                &self.rope_freqs,
                pos_offset,
                kv_cache,
                seq_meta.seq_id,
                i,
                &self.backend,
            )?;
        }

        hidden = self.norm.forward(&hidden, &self.backend)?;
        let logits = self.backend.matmul(&hidden, &self.lm_head)?;

        Ok(ModelOutput { logits })
    }

    /// Batched decode forward: N sequences, 1 token each.
    ///
    /// Concatenates tokens into `[N, hidden_size]`. Linear ops (QKV, MLP, norms,
    /// LM head) batch naturally. Attention loops per-sequence for KV cache.
    fn forward_batch_decode(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
    ) -> Result<ModelOutput<B::Tensor>> {
        let all_tokens: Vec<u32> = input.token_ids.iter().flatten().copied().collect();
        let all_positions: Vec<u32> = input.positions.iter().flatten().copied().collect();
        let seq_ids: Vec<u64> = input.seq_metadata.iter().map(|m| m.seq_id).collect();

        let mut hidden = self.backend.embedding(&self.embed_tokens, &all_tokens)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_batch(
                &hidden,
                &self.rope_freqs,
                &all_positions,
                &seq_ids,
                kv_cache,
                i,
                &self.backend,
            )?;
        }

        hidden = self.norm.forward(&hidden, &self.backend)?;
        let logits = self.backend.matmul(&hidden, &self.lm_head)?;

        Ok(ModelOutput { logits })
    }
}

impl<B: Backend + Clone> Model for LlamaModel<B> {
    type T = B::Tensor;

    fn forward(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
    ) -> Result<ModelOutput<B::Tensor>> {
        if input.seq_metadata.len() == 1 {
            return self.forward_single(input, kv_cache);
        }

        // Multi-sequence: all must be decode with exactly 1 token each
        for (i, meta) in input.seq_metadata.iter().enumerate() {
            if meta.is_prefill {
                return Err(ForgeError::InvalidArgument(
                    "batch forward does not support prefill sequences".into(),
                ));
            }
            if input.token_ids[i].len() != 1 {
                return Err(ForgeError::InvalidArgument(format!(
                    "batch decode expects 1 token per sequence, seq {} has {}",
                    meta.seq_id,
                    input.token_ids[i].len()
                )));
            }
        }

        self.forward_batch_decode(input, kv_cache)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
