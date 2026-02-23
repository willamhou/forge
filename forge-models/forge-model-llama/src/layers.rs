use forge_core::{Backend, KvCache, ModelConfig, Result, Tensor};

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
}

/// SiLU-gated MLP (gate_proj, up_proj, down_proj).
pub struct LlamaMLP<B: Backend> {
    gate_proj: B::Tensor,
    up_proj: B::Tensor,
    down_proj: B::Tensor,
}

impl<B: Backend> LlamaMLP<B> {
    pub fn new(gate_proj: B::Tensor, up_proj: B::Tensor, down_proj: B::Tensor) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &B::Tensor, backend: &B) -> Result<B::Tensor> {
        // gate = x @ gate_proj  — weights already transposed at load
        let gate = backend.matmul(x, &self.gate_proj)?;

        // up = x @ up_proj
        let up = backend.matmul(x, &self.up_proj)?;

        // output = (silu(gate) * up) @ down_proj — fused into one kernel
        let fused = backend.fused_silu_mul(&gate, &up)?;
        backend.matmul(&fused, &self.down_proj)
    }
}

/// Self-attention with GQA and RoPE.
///
/// Weights are stored as [in_features, out_features] (transposed at load time).
/// KV cache is used: during prefill, K/V are appended to cache; during decode,
/// cached K/V are retrieved so attention sees the full context.
pub struct LlamaAttention<B: Backend> {
    wqkv: B::Tensor,    // [hidden_size, q_proj_size + 2 * kv_proj_size]
    wo: B::Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_proj_size: usize,
    kv_proj_size: usize,
}

impl<B: Backend> LlamaAttention<B> {
    pub fn new(
        wqkv: B::Tensor,
        wo: B::Tensor,
        config: &ModelConfig,
    ) -> Self {
        Self {
            wqkv,
            wo,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            q_proj_size: config.num_attention_heads * config.head_dim,
            kv_proj_size: config.num_key_value_heads * config.head_dim,
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

        // Fused QKV projection — single GEMM + split
        let qkv = backend.matmul(x, &self.wqkv)?;
        let (q, k, v) = backend.split_qkv(&qkv, self.q_proj_size, self.kv_proj_size)?;

        // Reshape for RoPE: [1, seq_len, num_heads/kv_heads, head_dim]
        let q = backend.reshape(&q, &[1, seq_len, self.num_heads, self.head_dim])?;
        let k = backend.reshape(&k, &[1, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = backend.reshape(&v, &[1, seq_len, self.num_kv_heads, self.head_dim])?;

        // Apply RoPE with position offset
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
        let attn_out = self.compute_attention(&q, &k_4d, &v_4d, seq_len, kv_len, backend)?;

        // Cast attention output to match weight dtype (naive attention produces F32,
        // but weights may be F16). FlashAttention will natively produce matching dtype.
        let attn_out = backend.cast(&attn_out, self.wo.dtype())?;

        // Output projection: [seq_len, num_heads * head_dim] @ wo
        backend.matmul(&attn_out, &self.wo)
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

        // Fused QKV projection — single GEMM + split
        let qkv = backend.matmul(x, &self.wqkv)?;
        let (q, k, v) = backend.split_qkv(&qkv, self.q_proj_size, self.kv_proj_size)?;

        // Reshape for RoPE: [1, N, num_heads/kv_heads, head_dim]
        let q = backend.reshape(&q, &[1, n, self.num_heads, self.head_dim])?;
        let k = backend.reshape(&k, &[1, n, self.num_kv_heads, self.head_dim])?;

        // Apply RoPE with per-token positions
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

        // Retrieve full KV caches for all sequences
        let mut k_caches = Vec::with_capacity(n);
        let mut v_caches = Vec::with_capacity(n);
        for i in 0..n {
            let (k_full, v_full) = kv_cache.get_kv(seq_ids[i], layer_idx)?;
            k_caches.push(k_full);
            v_caches.push(v_full);
        }

        // Batched decode attention -- single kernel for all sequences
        let attn_out = backend.batched_decode_attention(
            &q,
            &k_caches,
            &v_caches,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;

        // Cast + output projection
        let attn_out = backend.cast(&attn_out, self.wo.dtype())?;
        backend.matmul(&attn_out, &self.wo)
    }

    /// Compute scaled dot-product attention per head (GPU-native, no CPU copies).
    ///
    /// Q: [1, seq_len, num_heads, head_dim]
    /// K: [1, kv_len, num_kv_heads, head_dim]
    /// V: [1, kv_len, num_kv_heads, head_dim]
    ///
    /// Returns: [seq_len, num_heads * head_dim]
    fn compute_attention(
        &self,
        q: &B::Tensor,
        k: &B::Tensor,
        v: &B::Tensor,
        seq_len: usize,
        kv_len: usize,
        backend: &B,
    ) -> Result<B::Tensor> {
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let heads_per_group = self.num_heads / self.num_kv_heads;

        // Reshape to 3D for extract_head: strip batch dim (always 1)
        let q = backend.reshape(q, &[seq_len, self.num_heads, self.head_dim])?;
        let k = backend.reshape(k, &[kv_len, self.num_kv_heads, self.head_dim])?;
        let v = backend.reshape(v, &[kv_len, self.num_kv_heads, self.head_dim])?;

        let mut head_outputs = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let kv_h = h / heads_per_group;

            // Extract per-head slices — GPU-native, no CPU copies
            let q_head = backend.extract_head(&q, seq_len, self.num_heads, self.head_dim, h)?;
            let k_head =
                backend.extract_head(&k, kv_len, self.num_kv_heads, self.head_dim, kv_h)?;
            let v_head =
                backend.extract_head(&v, kv_len, self.num_kv_heads, self.head_dim, kv_h)?;

            // scores = Q @ K^T * scale
            let k_t = backend.transpose(&k_head, 0, 1)?;
            let scores = backend.matmul(&q_head, &k_t)?;
            let scores = backend.mul_scalar(&scores, scale)?;

            // Apply causal mask for prefill (seq_len > 1)
            let scores = if seq_len > 1 {
                backend.apply_causal_mask(&scores, seq_len, kv_len)?
            } else {
                scores
            };

            let attn = backend.softmax(&scores, -1)?;
            head_outputs.push(backend.matmul(&attn, &v_head)?);
        }

        // Interleave per-head outputs → [seq_len, num_heads * head_dim]
        let refs: Vec<&B::Tensor> = head_outputs.iter().collect();
        backend.interleave_heads(&refs, seq_len, self.head_dim)
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
            &normed,
            rope_freqs,
            pos_offset,
            kv_cache,
            seq_id,
            layer_idx,
            backend,
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
            &normed,
            rope_freqs,
            positions,
            seq_ids,
            kv_cache,
            layer_idx,
            backend,
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
