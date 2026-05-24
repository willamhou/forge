use forge_core::{
    Backend, ForgeError, KvCache, Model, ModelConfig, ModelInput, ModelOutput, Result, Tensor,
};

use crate::layers::{LlamaDecoderLayer, RMSNorm};
use crate::persistent_buffers::LlamaPersistentBuffers;
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

    /// Persistent-buffer batched-decode forward.
    ///
    /// All intermediates and the final logits land in `buffers`. The caller
    /// constructs `buffers` once per batch shape (see [`LlamaPersistentBuffers::new`])
    /// and reuses it across calls. After this returns, `buffers.logits`
    /// holds `[batch, vocab_size]` and is the model's output.
    ///
    /// **Requirements**:
    /// - All sequences in `input` must be decode (1 token each), not prefill.
    /// - `input` must contain exactly `buffers.batch_size` sequences.
    /// - `kv_cache` must be a paged cache (PagedKvCache); naive caches are
    ///   rejected by `LlamaAttention::forward_batch_into` upfront.
    ///
    /// Returns `Ok(())` on success; callers read `buffers.logits` to consume.
    pub fn forward_into(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        buffers: &mut LlamaPersistentBuffers<B>,
    ) -> Result<()> {
        let n = input.seq_metadata.len();
        if n != buffers.batch_size {
            return Err(ForgeError::InvalidArgument(format!(
                "forward_into: input has {n} seqs but buffers.batch_size = {}",
                buffers.batch_size
            )));
        }
        // Decode-only validation (mirrors forward()).
        for (i, meta) in input.seq_metadata.iter().enumerate() {
            if meta.is_prefill {
                return Err(ForgeError::InvalidArgument(
                    "forward_into does not support prefill sequences".into(),
                ));
            }
            if input.token_ids[i].len() != 1 {
                return Err(ForgeError::InvalidArgument(format!(
                    "forward_into expects 1 token per sequence, seq {} has {}",
                    meta.seq_id,
                    input.token_ids[i].len()
                )));
            }
        }

        let all_tokens: Vec<u32> = input.token_ids.iter().flatten().copied().collect();
        let all_positions: Vec<u32> = input.positions.iter().flatten().copied().collect();
        let seq_ids: Vec<u64> = input.seq_metadata.iter().map(|m| m.seq_id).collect();

        // Embedding lookup → buffers.embeddings → seeds buffers.hidden.
        self.backend.embedding_into(
            &mut buffers.embeddings,
            &self.embed_tokens,
            &all_tokens,
        )?;
        // hidden := embeddings (memcpy via reshape_into with same shape).
        self.backend.reshape_into(
            &mut buffers.hidden,
            &buffers.embeddings,
            buffers.embeddings.shape().to_vec().as_slice(),
        )?;

        // Per-layer decoder pass — each layer reads + writes buffers.hidden.
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_batch_into(
                &self.rope_freqs,
                &all_positions,
                &seq_ids,
                kv_cache,
                i,
                buffers,
                &self.backend,
            )?;
        }

        // Final norm + lm_head: hidden → final_normed → logits.
        self.backend.rms_norm_into(
            &mut buffers.final_normed,
            &buffers.hidden,
            &self.norm.weight,
            self.norm.eps,
        )?;
        self.backend.matmul_into(
            &mut buffers.logits,
            &buffers.final_normed,
            &self.lm_head,
        )?;
        Ok(())
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
