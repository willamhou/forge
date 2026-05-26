use forge_core::{
    Backend, DecodeStageInputs, ForgeError, KvCache, Model, ModelConfig, ModelInput, ModelOutput,
    Result, Tensor,
};

use crate::layers::{LlamaDecoderLayer, RMSNorm};
use crate::persistent_buffers::{LlamaDecodeState, LlamaPersistentBuffers, StagedDecodeMeta};
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

    /// Host-side staging for one decode step (the body of
    /// [`Model::stage_decode`]). Advances the cache bookkeeping, gathers RoPE
    /// frequencies and fixed-width paged metadata, stores them in `state`, and
    /// uploads them into backend-persistent scratch so a replayed capture
    /// reads fresh data. No capturable kernel launches.
    fn stage_decode_impl(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        state: &mut LlamaDecodeState<B>,
    ) -> Result<()> {
        let n = input.seq_metadata.len();
        if n != state.buffers.batch_size {
            return Err(ForgeError::InvalidArgument(format!(
                "stage_decode: input has {n} seqs but state.batch_size = {}",
                state.buffers.batch_size
            )));
        }
        // Guard the parallel-Vec indexing below: a malformed ModelInput must
        // return an error, not panic.
        if input.token_ids.len() != n || input.positions.len() != n {
            return Err(ForgeError::InvalidArgument(format!(
                "stage_decode: ragged input — {n} seq_metadata but {} token_ids / {} positions",
                input.token_ids.len(),
                input.positions.len()
            )));
        }
        for (i, meta) in input.seq_metadata.iter().enumerate() {
            if meta.is_prefill {
                return Err(ForgeError::InvalidArgument(
                    "stage_decode does not support prefill sequences".into(),
                ));
            }
            if input.token_ids[i].len() != 1 {
                return Err(ForgeError::InvalidArgument(format!(
                    "stage_decode expects 1 token per sequence, seq {} has {}",
                    meta.seq_id,
                    input.token_ids[i].len()
                )));
            }
        }

        let token_ids: Vec<u32> = input.token_ids.iter().flatten().copied().collect();
        let positions: Vec<u32> = input.positions.iter().flatten().copied().collect();
        let seq_ids: Vec<u64> = input.seq_metadata.iter().map(|m| m.seq_id).collect();

        // Advance the cache by this step's one token per seq (allocate blocks,
        // bump lengths) and get the absolute write slot for each.
        let slot_mapping = kv_cache.advance_decode(&seq_ids)?;

        // Fixed block-table width (anchored to max context) so the staged
        // buffer's shape never changes between steps — a captured graph's
        // metadata memcpy length must stay constant across replays.
        let block_size = kv_cache.usage().block_size.max(1);
        let max_blocks_per_seq = self
            .config
            .max_position_embeddings
            .div_ceil(block_size)
            .max(1);

        let mut block_tables = vec![-1i32; n * max_blocks_per_seq];
        let mut kv_lens = vec![0i32; n];
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let bt = kv_cache.get_block_table(seq_id)?;
            kv_lens[i] = kv_cache.get_seq_len(seq_id)? as i32;
            if bt.len() > max_blocks_per_seq {
                return Err(ForgeError::InvalidArgument(format!(
                    "stage_decode: seq {seq_id} needs {} blocks but fixed width is {max_blocks_per_seq} \
                     (prompt + generated exceeds max_position_embeddings)",
                    bt.len()
                )));
            }
            for (j, &b) in bt.iter().enumerate() {
                // Block ids feed paged attention as i32; a wrap to negative
                // would silently read garbage KV. Reject oversized pools.
                block_tables[i * max_blocks_per_seq + j] =
                    i32::try_from(b).map_err(|_| {
                        ForgeError::InvalidArgument(format!(
                            "stage_decode: block id {b} exceeds i32::MAX (KV pool too large)"
                        ))
                    })?;
            }
        }

        let (rope_cos, rope_sin) = self.rope_freqs.gather_host_freqs(&positions)?;

        // Stage into backend-persistent scratch (capture-safe: outside any
        // captured region). No-op on backends without graph capture.
        self.backend.stage_slot_mapping(&slot_mapping)?;
        self.backend.stage_decode_inputs(&DecodeStageInputs {
            token_indices: &token_ids,
            rope_cos: &rope_cos,
            rope_sin: &rope_sin,
            block_tables: &block_tables,
            kv_lens: &kv_lens,
        })?;

        state.token_ids = token_ids;
        state.meta = StagedDecodeMeta {
            positions,
            seq_ids,
            slot_mapping,
            block_tables,
            kv_lens,
            max_blocks_per_seq,
        };
        Ok(())
    }

    /// Capture-safe decode forward (the body of [`Model::compute_decode`]).
    /// Pure kernel sequence reading the pre-staged inputs in `state`; writes
    /// `[batch, vocab_size]` logits into `state.buffers.logits`.
    fn compute_decode_impl(
        &self,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        state: &mut LlamaDecodeState<B>,
    ) -> Result<()> {
        let LlamaDecodeState {
            buffers,
            token_ids,
            meta,
        } = state;

        // Embedding lookup → buffers.embeddings → seeds buffers.hidden.
        self.backend
            .embedding_into(&mut buffers.embeddings, &self.embed_tokens, token_ids)?;
        self.backend.reshape_into(
            &mut buffers.hidden,
            &buffers.embeddings,
            buffers.embeddings.shape().to_vec().as_slice(),
        )?;

        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_batch_into(
                &self.rope_freqs,
                meta,
                kv_cache,
                i,
                buffers,
                &self.backend,
            )?;
        }

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
    type DecodeState = LlamaDecodeState<B>;

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

    /// QKV bias (Qwen2) isn't wired into the persistent-buffer (capture) path,
    /// so models that have it must use eager decode. The engine then skips the
    /// CUDA-Graph path for them (device-side sampling still applies).
    fn supports_capture_decode(&self) -> bool {
        !self.layers.iter().any(|l| l.has_qkv_bias())
    }

    fn make_decode_state(&self, batch_size: usize) -> Result<LlamaDecodeState<B>> {
        // Uniform-dtype buffers (F16 weights + F16 activations is the
        // production norm, and what the standard load path yields). Mixed
        // activation/weight dtypes would need per-weight dtype threading here.
        let buffers = LlamaPersistentBuffers::new_uniform(&self.backend, &self.config, batch_size)?;
        Ok(LlamaDecodeState::new(buffers))
    }

    fn stage_decode(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        state: &mut LlamaDecodeState<B>,
    ) -> Result<()> {
        self.stage_decode_impl(input, kv_cache, state)
    }

    fn compute_decode(
        &self,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        state: &mut LlamaDecodeState<B>,
    ) -> Result<()> {
        self.compute_decode_impl(kv_cache, state)
    }

    fn decode_logits<'a>(&self, state: &'a LlamaDecodeState<B>) -> &'a B::Tensor {
        &state.buffers.logits
    }
}
