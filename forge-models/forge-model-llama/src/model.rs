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
    ///
    /// ## ⚠️ CUDA-Graph capture not yet safe
    ///
    /// `forward_into` is structurally compatible with capture (every
    /// kernel arg's device pointer is stable), but two host-side gaps
    /// remain that would cause silent corruption on replay:
    ///
    /// 1. `PagedKvCache::append` builds a fresh `Vec<i32>` slot_mapping
    ///    per call and passes a borrowed slice to `paged_write_kv`. The
    ///    captured `memcpy_htod` node bakes the Vec's host pointer; on
    ///    replay that pointer is stale (Vec dropped).
    /// 2. `forward_into` itself builds fresh `all_tokens` / `all_positions`
    ///    / `seq_ids` Vecs (and `RopeFreqs::apply_with_positions_into`
    ///    builds fresh `cos_data`/`sin_data`). Same captured-memcpy_htod
    ///    issue.
    ///
    /// Until the engine wires persistent host buffers for all of these,
    /// **do NOT wrap `forward_into` in `CudaGraphCache::run_or_capture`
    /// directly** — replays produce wrong KV / wrong token / wrong RoPE
    /// state. Use the alloc-variant `forward()` for eager runs in the
    /// meantime; `forward_into` is currently a foundation for the
    /// upcoming engine integration, not a drop-in user-facing API.
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

        // Final norm + lm_head:
        //   hidden → rms_norm → final_normed (activation dtype)
        //   → cast → final_normed_cast (lm_head dtype)
        //   → matmul(lm_head) → logits (lm_head dtype)
        // The cast is a no-op memcpy when dtypes match.
        self.backend.rms_norm_into(
            &mut buffers.final_normed,
            &buffers.hidden,
            &self.norm.weight,
            self.norm.eps,
        )?;
        self.backend
            .cast_into(&mut buffers.final_normed_cast, &buffers.final_normed)?;
        self.backend.matmul_into(
            &mut buffers.logits,
            &buffers.final_normed_cast,
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
