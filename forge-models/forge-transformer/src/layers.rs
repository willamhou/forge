use forge_core::{Backend, ForgeError, KvCache, ModelConfig, Result, Tensor};

use crate::persistent_buffers::{LlamaPersistentBuffers, StagedDecodeMeta};
use crate::quantized_linear::QuantizedLinear;
use crate::rope::RopeFreqs;

/// RMS normalization layer.
pub struct RMSNorm<B: Backend> {
    pub(crate) weight: B::Tensor,
    pub(crate) eps: f32,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(weight: B::Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &B::Tensor, backend: &B) -> Result<B::Tensor> {
        backend.rms_norm(x, &self.weight, self.eps)
    }

    /// Persistent-buffer variant — writes into a caller-provided `out`.
    pub fn forward_into(&self, out: &mut B::Tensor, x: &B::Tensor, backend: &B) -> Result<()> {
        backend.rms_norm_into(out, x, &self.weight, self.eps)
    }
}

/// SiLU-gated MLP (gate_proj, up_proj, down_proj).
pub struct LlamaMLP<B: Backend> {
    gate_proj: QuantizedLinear<B>,
    up_proj: QuantizedLinear<B>,
    down_proj: QuantizedLinear<B>,
}

impl<B: Backend> LlamaMLP<B> {
    pub fn new(
        gate_proj: QuantizedLinear<B>,
        up_proj: QuantizedLinear<B>,
        down_proj: QuantizedLinear<B>,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &B::Tensor, backend: &B) -> Result<B::Tensor> {
        // gate = x @ gate_proj  — weights already transposed at load
        let gate = self.gate_proj.matmul_prefill(x, backend)?;

        // up = x @ up_proj
        let up = self.up_proj.matmul_prefill(x, backend)?;

        // output = (silu(gate) * up) @ down_proj — fused into one kernel
        let fused = backend.fused_silu_mul(&gate, &up)?;
        self.down_proj.matmul_prefill(&fused, backend)
    }

    /// Persistent-buffer variant.
    ///
    /// **Buffer convention**: reads input from `buffers.normed` (caller
    /// must have populated it), writes the final MLP output to
    /// `buffers.mlp_out`. Stages through `buffers.{gate, up, silu_mul}`.
    /// Reading via the buffers struct (vs an `x: &B::Tensor` parameter)
    /// lets us use disjoint &mut/& borrows on buffer fields and avoids
    /// the borrow-checker conflict that an explicit `x` would create.
    pub fn forward_into(&self, buffers: &mut LlamaPersistentBuffers<B>, backend: &B) -> Result<()> {
        self.gate_proj
            .matmul_decode_into(&mut buffers.gate, &buffers.normed, backend)?;
        self.up_proj
            .matmul_decode_into(&mut buffers.up, &buffers.normed, backend)?;
        backend.fused_silu_mul_into(&mut buffers.silu_mul, &buffers.gate, &buffers.up)?;
        self.down_proj
            .matmul_decode_into(&mut buffers.mlp_out, &buffers.silu_mul, backend)?;
        Ok(())
    }
}

/// Self-attention with GQA and RoPE.
///
/// Weights are stored as [in_features, out_features] (transposed at load time).
/// KV cache is used: during prefill, K/V are appended to cache; during decode,
/// cached K/V are retrieved so attention sees the full context.
pub struct LlamaAttention<B: Backend> {
    // wqkv: [hidden_size, q_proj_size + 2 * kv_proj_size] (FP16, prefill);
    // its Q8_0 decode copy is [q_proj_size + 2 * kv_proj_size, hidden_size].
    wqkv: QuantizedLinear<B>,
    /// Optional fused QKV bias `[q_proj_size + 2 * kv_proj_size]`. Present
    /// for Qwen2-family models (q/k/v_proj have bias); `None` for Llama.
    qkv_bias: Option<B::Tensor>,
    /// Optional per-head RMSNorm on Q / K before RoPE (Qwen3 `q_norm`/`k_norm`,
    /// each `[head_dim]`). `None` for Llama/Qwen2. Applied to every head's
    /// `head_dim` slice independently.
    q_norm: Option<RMSNorm<B>>,
    k_norm: Option<RMSNorm<B>>,
    wo: QuantizedLinear<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_proj_size: usize,
    kv_proj_size: usize,
}

impl<B: Backend> LlamaAttention<B> {
    /// Construct without QKV bias (Llama).
    pub fn new(wqkv: QuantizedLinear<B>, wo: QuantizedLinear<B>, config: &ModelConfig) -> Self {
        Self::new_with_bias(wqkv, None, wo, config)
    }

    /// Construct with an optional fused QKV bias (Qwen2 sets this).
    pub fn new_with_bias(
        wqkv: QuantizedLinear<B>,
        qkv_bias: Option<B::Tensor>,
        wo: QuantizedLinear<B>,
        config: &ModelConfig,
    ) -> Self {
        Self {
            wqkv,
            qkv_bias,
            q_norm: None,
            k_norm: None,
            wo,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            q_proj_size: config.num_attention_heads * config.head_dim,
            kv_proj_size: config.num_key_value_heads * config.head_dim,
        }
    }

    /// Attach Qwen3 per-head QK-norm (RMSNorm on Q / K over `head_dim`,
    /// applied before RoPE). No-op for models without it.
    pub fn with_qk_norm(mut self, q_norm: Option<RMSNorm<B>>, k_norm: Option<RMSNorm<B>>) -> Self {
        self.q_norm = q_norm;
        self.k_norm = k_norm;
        self
    }

    /// Apply Qwen3 Q-norm (per-head RMSNorm over `head_dim`) to a 4D tensor
    /// `[1, n, num_heads, head_dim]`. Reshapes to `[n*num_heads, head_dim]` so
    /// each head is normalised independently, then back. No-op without q_norm.
    /// (Allocating; eager paths only — the capture path uses `*_into`.)
    fn maybe_q_norm(&self, q: B::Tensor, n: usize, backend: &B) -> Result<B::Tensor> {
        match &self.q_norm {
            Some(qn) => {
                let flat = backend.reshape(&q, &[n * self.num_heads, self.head_dim])?;
                let normed = backend.rms_norm(&flat, &qn.weight, qn.eps)?;
                backend.reshape(&normed, &[1, n, self.num_heads, self.head_dim])
            }
            None => Ok(q),
        }
    }

    /// K-norm sibling of [`Self::maybe_q_norm`] (uses `num_kv_heads`).
    fn maybe_k_norm(&self, k: B::Tensor, n: usize, backend: &B) -> Result<B::Tensor> {
        match &self.k_norm {
            Some(kn) => {
                let flat = backend.reshape(&k, &[n * self.num_kv_heads, self.head_dim])?;
                let normed = backend.rms_norm(&flat, &kn.weight, kn.eps)?;
                backend.reshape(&normed, &[1, n, self.num_kv_heads, self.head_dim])
            }
            None => Ok(k),
        }
    }

    /// Whether this attention has a QKV bias (Qwen2). The capture-safe decode
    /// path doesn't support it yet, so the engine falls back to eager decode.
    pub fn has_qkv_bias(&self) -> bool {
        self.qkv_bias.is_some()
    }

    /// Apply the fused QKV projection (matmul + optional bias).
    fn project_qkv(&self, x: &B::Tensor, backend: &B) -> Result<B::Tensor> {
        let qkv = self.wqkv.matmul_prefill(x, backend)?;
        match &self.qkv_bias {
            Some(bias) => backend.add_bias(&qkv, bias),
            None => Ok(qkv),
        }
    }

    /// Compute self-attention for a sequence.
    ///
    /// Input `x`: [seq_len, hidden_size]
    /// Output: [seq_len, hidden_size]
    ///
    /// `pos_offset`: the starting position index for RoPE (0 for prefill,
    /// cached_len for decode).
    pub fn forward(
        &self,
        x: &B::Tensor,
        rope_freqs: &RopeFreqs<B>,
        pos_offset: usize,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        seq_id: u64,
        layer_idx: usize,
        backend: &B,
    ) -> Result<B::Tensor> {
        let shape = x.shape();
        let seq_len = shape[0];

        // Fused QKV projection (matmul + optional Qwen2 bias) — single GEMM + split
        let qkv = self.project_qkv(x, backend)?;
        let (q, k, v) = backend.split_qkv(&qkv, self.q_proj_size, self.kv_proj_size)?;

        // Reshape for RoPE: [1, seq_len, num_heads/kv_heads, head_dim]
        let q = backend.reshape(&q, &[1, seq_len, self.num_heads, self.head_dim])?;
        let k = backend.reshape(&k, &[1, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = backend.reshape(&v, &[1, seq_len, self.num_kv_heads, self.head_dim])?;

        // Qwen3 QK-norm (no-op otherwise), then RoPE with position offset.
        let q = self.maybe_q_norm(q, seq_len, backend)?;
        let k = self.maybe_k_norm(k, seq_len, backend)?;
        let q = rope_freqs.apply_with_offset(&q, pos_offset, backend)?;
        let k = rope_freqs.apply_with_offset(&k, pos_offset, backend)?;

        // Flatten K, V back to 2D for cache: [seq_len, num_kv_heads * head_dim]
        let k_flat = backend.reshape(&k, &[seq_len, self.num_kv_heads * self.head_dim])?;
        let v_flat = backend.reshape(&v, &[seq_len, self.num_kv_heads * self.head_dim])?;

        // Append new K, V to cache
        kv_cache.append(seq_id, layer_idx, &k_flat, &v_flat)?;

        // Retrieve full cached K, V (includes all prior + current tokens)
        let (k_full, v_full) = kv_cache.get_kv(seq_id, layer_idx)?;

        // Reshape to 4D for attention: [1, kv_len, num_kv_heads, head_dim]
        let kv_len = k_full.shape()[0];
        let k_4d = backend.reshape(&k_full, &[1, kv_len, self.num_kv_heads, self.head_dim])?;
        let v_4d = backend.reshape(&v_full, &[1, kv_len, self.num_kv_heads, self.head_dim])?;

        // Compute attention with Q over full K,V (including cached)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let is_causal = seq_len > 1;
        let attn_out = backend.multi_head_attention(
            &q,
            &k_4d,
            &v_4d,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            scale,
            is_causal,
        )?;

        // Cast attention output to match weight dtype (naive attention produces F32,
        // but weights may be F16). FlashAttention will natively produce matching dtype.
        let attn_out = backend.cast(&attn_out, self.wo.dtype())?;

        // Output projection: [seq_len, num_heads * head_dim] @ wo
        self.wo.matmul_prefill(&attn_out, backend)
    }

    /// Batched attention for decode: each sequence contributes exactly 1 token.
    ///
    /// `x`: `[N, hidden_size]` (N concatenated decode tokens)
    /// `positions`: per-token absolute positions, length N
    /// `seq_ids`: per-sequence IDs for KV cache, length N
    ///
    /// Returns: `[N, hidden_size]`
    pub fn forward_batch(
        &self,
        x: &B::Tensor,
        rope_freqs: &RopeFreqs<B>,
        positions: &[u32],
        seq_ids: &[u64],
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        layer_idx: usize,
        backend: &B,
    ) -> Result<B::Tensor> {
        let n = x.shape()[0];

        // Fused QKV projection (matmul + optional Qwen2 bias) — single GEMM + split
        let qkv = self.project_qkv(x, backend)?;
        let (q, k, v) = backend.split_qkv(&qkv, self.q_proj_size, self.kv_proj_size)?;

        // Reshape for RoPE: [1, N, num_heads/kv_heads, head_dim]
        let q = backend.reshape(&q, &[1, n, self.num_heads, self.head_dim])?;
        let k = backend.reshape(&k, &[1, n, self.num_kv_heads, self.head_dim])?;

        // Qwen3 QK-norm (no-op otherwise), then RoPE with per-token positions.
        let q = self.maybe_q_norm(q, n, backend)?;
        let k = self.maybe_k_norm(k, n, backend)?;
        let q = rope_freqs.apply_with_positions(&q, positions, backend)?;
        let k = rope_freqs.apply_with_positions(&k, positions, backend)?;

        // Flatten back to 2D
        let q = backend.reshape(&q, &[n, self.num_heads * self.head_dim])?;
        let k = backend.reshape(&k, &[n, self.num_kv_heads * self.head_dim])?;

        // Per-sequence cache append (still per-seq)
        for i in 0..n {
            let k_row = backend.slice_rows(&k, i, 1)?;
            let v_row = backend.slice_rows(&v, i, 1)?;
            kv_cache.append(seq_ids[i], layer_idx, &k_row, &v_row)?;
        }

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Paged attention fast path (PagedKvCache only). Single fused kernel
        // launch reading K/V from the device-resident pool through block tables
        // — no per-seq gather + multi_head_attention dance, capture-friendly.
        //
        // TODO(task-3 capture): this calls the allocating variant, which
        // produces a fresh output CudaTensor every step. For CUDA Graph
        // capture, the engine must thread a persistent output buffer through
        // here and call `CudaBackend::paged_attention_into` instead so the
        // captured kernel's output arg points at a stable device address.
        if let Some(inputs) = kv_cache.paged_attention_inputs(layer_idx, seq_ids)? {
            let attn_out = backend.paged_attention(
                &q,
                inputs.k_pool,
                inputs.v_pool,
                &inputs.block_tables,
                &inputs.kv_lens,
                inputs.max_blocks_per_seq,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                scale,
            )?;
            let attn_out = backend.cast(&attn_out, self.wo.dtype())?;
            return self.wo.matmul_prefill(&attn_out, backend);
        }

        // Fallback for non-paged caches (e.g. NaiveKvCache): retrieve full KV
        // per seq and dispatch to batched_decode_attention.
        let mut k_caches = Vec::with_capacity(n);
        let mut v_caches = Vec::with_capacity(n);
        for i in 0..n {
            let (k_full, v_full) = kv_cache.get_kv(seq_ids[i], layer_idx)?;
            k_caches.push(k_full);
            v_caches.push(v_full);
        }

        let attn_out = backend.batched_decode_attention(
            &q,
            &k_caches,
            &v_caches,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            scale,
        )?;

        // Cast + output projection
        let attn_out = backend.cast(&attn_out, self.wo.dtype())?;
        self.wo.matmul_prefill(&attn_out, backend)
    }

    /// Persistent-buffer batched-decode forward.
    ///
    /// Same algorithm as [`Self::forward_batch`] but every intermediate
    /// is written into a pre-allocated buffer in `buffers`. Requires a
    /// `PagedKvCache` — non-paged caches (NaiveKvCache) are rejected
    /// upfront because their fallback path (per-seq gather +
    /// `batched_decode_attention`) is not capture-safe.
    ///
    /// **Buffer convention**: reads input from `buffers.normed` (caller
    /// must have populated it via `input_layernorm.forward_into`), writes
    /// the layer's attention output to `buffers.attn_proj`.
    pub fn forward_batch_into(
        &self,
        rope_freqs: &RopeFreqs<B>,
        meta: &StagedDecodeMeta,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        layer_idx: usize,
        buffers: &mut LlamaPersistentBuffers<B>,
        backend: &B,
    ) -> Result<()> {
        let n = buffers.batch_size;
        let positions = &meta.positions;

        // QKV bias not yet supported in the persistent-buffer (capture) path —
        // would need an add_bias_into op + a buffer for the biased qkv. The
        // QKV projection + optional bias (Qwen2). Reads from buffers.normed
        // (caller convention). `add_bias_into` mutates the persistent qkv buffer
        // in place (the bias is a static weight), so it stays capture-safe.
        self.wqkv
            .matmul_decode_into(&mut buffers.qkv, &buffers.normed, backend)?;
        if let Some(bias) = &self.qkv_bias {
            backend.add_bias_into(&mut buffers.qkv, bias)?;
        }
        backend.split_qkv_into(
            &mut buffers.q_2d,
            &mut buffers.k_2d,
            &mut buffers.v_2d,
            &buffers.qkv,
            self.q_proj_size,
            self.kv_proj_size,
        )?;

        // Reshape Q/K to 4D for RoPE (capture-safe via reshape_into memcpy).
        backend.reshape_into(
            &mut buffers.q_4d,
            &buffers.q_2d,
            &[1, n, self.num_heads, self.head_dim],
        )?;
        backend.reshape_into(
            &mut buffers.k_4d,
            &buffers.k_2d,
            &[1, n, self.num_kv_heads, self.head_dim],
        )?;

        // Qwen3 QK-norm (per-head RMSNorm over head_dim) into the normed
        // buffers, then RoPE reads those; without QK-norm RoPE reads q_4d/k_4d
        // directly. The norm weights are static, so this stays capture-safe.
        if let Some(qn) = &self.q_norm {
            backend.rms_norm_into(&mut buffers.q_normed_4d, &buffers.q_4d, &qn.weight, qn.eps)?;
        }
        if let Some(kn) = &self.k_norm {
            backend.rms_norm_into(&mut buffers.k_normed_4d, &buffers.k_4d, &kn.weight, kn.eps)?;
        }
        let q_rope_src = if self.q_norm.is_some() {
            &buffers.q_normed_4d
        } else {
            &buffers.q_4d
        };
        let k_rope_src = if self.k_norm.is_some() {
            &buffers.k_normed_4d
        } else {
            &buffers.k_4d
        };

        // Apply RoPE with per-token positions.
        rope_freqs.apply_with_positions_into(
            &mut buffers.q_rotated_4d,
            q_rope_src,
            positions,
            backend,
        )?;
        rope_freqs.apply_with_positions_into(
            &mut buffers.k_rotated_4d,
            k_rope_src,
            positions,
            backend,
        )?;

        // Flatten back to 2D for attention + cache append.
        backend.reshape_into(
            &mut buffers.q_rotated_2d,
            &buffers.q_rotated_4d,
            &[n, self.num_heads * self.head_dim],
        )?;
        backend.reshape_into(
            &mut buffers.k_rotated_2d,
            &buffers.k_rotated_4d,
            &[n, self.num_kv_heads * self.head_dim],
        )?;

        // Capture-safe KV write: scatter all N new rows into this layer's pool
        // at the slots staged by `stage_decode` (rotated K, un-rotated V). On
        // CUDA this reads a device-resident slot scratch (no host pointer baked
        // into the captured graph); the host `slot_mapping` is used only by the
        // CPU fallback. Replaces the old per-token `append` loop.
        kv_cache.scatter_decode(
            layer_idx,
            &buffers.k_rotated_2d,
            &buffers.v_2d,
            &meta.slot_mapping,
            n,
        )?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Paged attention. We borrow the device-resident pool tensors from the
        // cache but feed the *staged* fixed-width block_tables / kv_lens (not
        // the cache's per-step dynamic ones) so the captured kernel's metadata
        // buffer size is stable across replays.
        let inputs = kv_cache
            .paged_attention_inputs(layer_idx, &meta.seq_ids)?
            .ok_or_else(|| {
                ForgeError::InvalidArgument(
                    "forward_batch_into requires a paged KV cache (PagedKvCache); \
                     naive caches have no capture-safe attention path"
                        .into(),
                )
            })?;
        backend.paged_attention_into(
            &mut buffers.attn_out,
            &buffers.q_rotated_2d,
            inputs.k_pool,
            inputs.v_pool,
            &meta.block_tables,
            &meta.kv_lens,
            meta.max_blocks_per_seq,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            scale,
        )?;

        // Cast attn_out to wo's weight dtype (no-op memcpy if dtypes match).
        backend.cast_into(&mut buffers.attn_out_cast, &buffers.attn_out)?;

        // Output projection.
        self.wo
            .matmul_decode_into(&mut buffers.attn_proj, &buffers.attn_out_cast, backend)?;
        Ok(())
    }
}

/// A single Llama decoder layer (attention + MLP with residual connections).
pub struct LlamaDecoderLayer<B: Backend> {
    input_layernorm: RMSNorm<B>,
    self_attn: LlamaAttention<B>,
    post_attention_layernorm: RMSNorm<B>,
    mlp: LlamaMLP<B>,
}

impl<B: Backend> LlamaDecoderLayer<B> {
    pub fn new(
        input_layernorm: RMSNorm<B>,
        self_attn: LlamaAttention<B>,
        post_attention_layernorm: RMSNorm<B>,
        mlp: LlamaMLP<B>,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    /// See [`LlamaAttention::has_qkv_bias`].
    pub fn has_qkv_bias(&self) -> bool {
        self.self_attn.has_qkv_bias()
    }

    pub fn forward(
        &self,
        x: &B::Tensor,
        rope_freqs: &RopeFreqs<B>,
        pos_offset: usize,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        seq_id: u64,
        layer_idx: usize,
        backend: &B,
    ) -> Result<B::Tensor> {
        // Pre-attention norm + attention + residual
        let normed = self.input_layernorm.forward(x, backend)?;
        let attn_out = self.self_attn.forward(
            &normed, rope_freqs, pos_offset, kv_cache, seq_id, layer_idx, backend,
        )?;

        // Fused residual add + post-attention norm
        let (normed, x) = backend.fused_residual_rms_norm(
            &attn_out,
            x,
            &self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )?;
        let mlp_out = self.mlp.forward(&normed, backend)?;
        backend.add(&x, &mlp_out)
    }

    /// Persistent-buffer batched forward (decode).
    ///
    /// Reads `buffers.hidden` (residual stream from prior layer), runs
    /// pre-attn norm + attention + post-attn fused-norm + MLP + residual
    /// add. Writes the updated residual stream back into `buffers.hidden`
    /// so the next layer's iteration reads it.
    pub fn forward_batch_into(
        &self,
        rope_freqs: &RopeFreqs<B>,
        meta: &StagedDecodeMeta,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        layer_idx: usize,
        buffers: &mut LlamaPersistentBuffers<B>,
        backend: &B,
    ) -> Result<()> {
        // Pre-attention norm: buffers.normed = rms_norm(buffers.hidden).
        // Split-borrow on disjoint fields (&mut .normed, & .hidden).
        backend.rms_norm_into(
            &mut buffers.normed,
            &buffers.hidden,
            &self.input_layernorm.weight,
            self.input_layernorm.eps,
        )?;

        // Attention reads buffers.normed (by convention), writes buffers.attn_proj.
        self.self_attn
            .forward_batch_into(rope_freqs, meta, kv_cache, layer_idx, buffers, backend)?;

        // Fused residual + post-attn norm:
        //   buffers.normed = rms_norm(buffers.attn_proj + buffers.hidden)
        //   buffers.residual_after_attn = buffers.attn_proj + buffers.hidden
        backend.fused_residual_rms_norm_into(
            &mut buffers.normed,
            &mut buffers.residual_after_attn,
            &buffers.attn_proj,
            &buffers.hidden,
            &self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )?;

        // MLP reads buffers.normed (post-attn norm), writes buffers.mlp_out.
        self.mlp.forward_into(buffers, backend)?;

        // Final residual add: buffers.hidden = buffers.residual_after_attn + buffers.mlp_out.
        // Overwriting buffers.hidden in place is safe — residual_after_attn
        // and mlp_out are distinct buffer fields, so no aliasing.
        backend.add_into(
            &mut buffers.hidden,
            &buffers.residual_after_attn,
            &buffers.mlp_out,
        )?;
        Ok(())
    }

    /// Batched forward for decode: N sequences, 1 token each.
    ///
    /// Norms and MLP operate row-wise and batch naturally on `[N, hidden_size]`.
    /// Attention loops per-sequence for KV cache.
    pub fn forward_batch(
        &self,
        x: &B::Tensor,
        rope_freqs: &RopeFreqs<B>,
        positions: &[u32],
        seq_ids: &[u64],
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        layer_idx: usize,
        backend: &B,
    ) -> Result<B::Tensor> {
        // Pre-attention norm + batched attention + residual
        let normed = self.input_layernorm.forward(x, backend)?;
        let attn_out = self.self_attn.forward_batch(
            &normed, rope_freqs, positions, seq_ids, kv_cache, layer_idx, backend,
        )?;

        // Fused residual add + post-attention norm
        let (normed, x) = backend.fused_residual_rms_norm(
            &attn_out,
            x,
            &self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )?;
        let mlp_out = self.mlp.forward(&normed, backend)?;
        backend.add(&x, &mlp_out)
    }
}
