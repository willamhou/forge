use forge_core::{Backend, KvCache, ModelConfig, Result, Tensor};

use crate::rope::RopeFreqs;

/// RMS normalization layer.
pub struct RMSNorm<B: Backend> {
    weight: B::Tensor,
    eps: f32,
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
        // gate = silu(x @ gate_proj^T)
        let gate = backend.matmul(x, &self.gate_proj)?;
        let gate = backend.silu(&gate)?;

        // up = x @ up_proj^T
        let up = backend.matmul(x, &self.up_proj)?;

        // output = (gate * up) @ down_proj^T
        let fused = backend.mul(&gate, &up)?;
        backend.matmul(&fused, &self.down_proj)
    }
}

/// Self-attention with GQA and RoPE.
///
/// For Phase 1, computes attention using basic backend ops (matmul + softmax).
/// KV caching is prepared but the naive path computes full attention each time.
/// FlashAttention integration in Task 16 will replace the inner loop.
pub struct LlamaAttention<B: Backend> {
    wq: B::Tensor,
    wk: B::Tensor,
    wv: B::Tensor,
    wo: B::Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> LlamaAttention<B> {
    pub fn new(
        wq: B::Tensor,
        wk: B::Tensor,
        wv: B::Tensor,
        wo: B::Tensor,
        config: &ModelConfig,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
        }
    }

    /// Compute self-attention for a sequence.
    ///
    /// Input `x`: [seq_len, hidden_size]
    /// Output: [seq_len, hidden_size]
    pub fn forward(
        &self,
        x: &B::Tensor,
        rope_freqs: &RopeFreqs<B>,
        _kv_cache: &mut dyn KvCache<T = B::Tensor>,
        _seq_id: u64,
        _layer_idx: usize,
        backend: &B,
    ) -> Result<B::Tensor> {
        let shape = x.shape();
        let seq_len = shape[0];

        // Q, K, V projections
        let q = backend.matmul(x, &self.wq)?; // [seq_len, num_heads * head_dim]
        let k = backend.matmul(x, &self.wk)?; // [seq_len, num_kv_heads * head_dim]
        let v = backend.matmul(x, &self.wv)?; // [seq_len, num_kv_heads * head_dim]

        // Reshape for RoPE and attention
        let q = backend.reshape(&q, &[1, seq_len, self.num_heads, self.head_dim])?;
        let k = backend.reshape(&k, &[1, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = backend.reshape(&v, &[1, seq_len, self.num_kv_heads, self.head_dim])?;

        // Apply RoPE to Q and K
        let q = rope_freqs.apply(&q, backend)?;
        let k = rope_freqs.apply(&k, backend)?;

        // TODO: In Task 16, KV cache append/retrieval will be wired here.
        // For now, compute attention over the full current K,V.

        // Per-head attention using backend ops.
        // This calls the standalone attention helper that uses only Backend trait methods.
        let attn_out = self.compute_attention(&q, &k, &v, seq_len, backend)?;

        // Output projection: [seq_len, num_heads * head_dim] @ wo
        backend.matmul(&attn_out, &self.wo)
    }

    /// Compute scaled dot-product attention per head using only Backend trait ops.
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
        backend: &B,
    ) -> Result<B::Tensor> {
        let kv_len = k.shape()[1];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let heads_per_group = self.num_heads / self.num_kv_heads;

        // Read all data to host for per-head slicing (naive path).
        // FlashAttention in Task 16 will eliminate this CPU roundtrip.
        let q_data = backend.copy_to_host_f32(q)?;
        let k_data = backend.copy_to_host_f32(k)?;
        let v_data = backend.copy_to_host_f32(v)?;

        let mut all_head_outputs: Vec<f32> =
            Vec::with_capacity(seq_len * self.num_heads * self.head_dim);

        // Process in token-first order for correct output layout
        for t in 0..seq_len {
            for h in 0..self.num_heads {
                let kv_h = h / heads_per_group;

                // Extract Q[t,h]: single row [1, head_dim]
                let q_offset = t * self.num_heads * self.head_dim + h * self.head_dim;
                let q_row = &q_data[q_offset..q_offset + self.head_dim];
                let q_tensor = backend.copy_from_host_f32(q_row, &[1, self.head_dim])?;

                // Extract K[:,kv_h]: [kv_len, head_dim]
                let mut k_head = Vec::with_capacity(kv_len * self.head_dim);
                for kv_t in 0..kv_len {
                    let k_offset =
                        kv_t * self.num_kv_heads * self.head_dim + kv_h * self.head_dim;
                    k_head.extend_from_slice(&k_data[k_offset..k_offset + self.head_dim]);
                }
                let k_tensor = backend.copy_from_host_f32(&k_head, &[kv_len, self.head_dim])?;
                let k_t = backend.transpose(&k_tensor, 0, 1)?;

                // scores = Q @ K^T * scale: [1, head_dim] @ [head_dim, kv_len] -> [1, kv_len]
                let scores = backend.matmul(&q_tensor, &k_t)?;
                let scores = backend.mul_scalar(&scores, scale)?;
                let attn_weights = backend.softmax(&scores, -1)?;

                // Extract V[:,kv_h]: [kv_len, head_dim]
                let mut v_head = Vec::with_capacity(kv_len * self.head_dim);
                for kv_t in 0..kv_len {
                    let v_offset =
                        kv_t * self.num_kv_heads * self.head_dim + kv_h * self.head_dim;
                    v_head.extend_from_slice(&v_data[v_offset..v_offset + self.head_dim]);
                }
                let v_tensor = backend.copy_from_host_f32(&v_head, &[kv_len, self.head_dim])?;

                // out = attn_weights @ V: [1, kv_len] @ [kv_len, head_dim] -> [1, head_dim]
                let head_out = backend.matmul(&attn_weights, &v_tensor)?;
                let head_out_data = backend.copy_to_host_f32(&head_out)?;
                all_head_outputs.extend_from_slice(&head_out_data);
            }
        }

        // Result is [seq_len, num_heads * head_dim]
        backend.copy_from_host_f32(
            &all_head_outputs,
            &[seq_len, self.num_heads * self.head_dim],
        )
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
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
        seq_id: u64,
        layer_idx: usize,
        backend: &B,
    ) -> Result<B::Tensor> {
        // Pre-attention norm + attention + residual
        let normed = self.input_layernorm.forward(x, backend)?;
        let attn_out =
            self.self_attn
                .forward(&normed, rope_freqs, kv_cache, seq_id, layer_idx, backend)?;
        let x = backend.add(x, &attn_out)?;

        // Post-attention norm + MLP + residual
        let normed = self.post_attention_layernorm.forward(&x, backend)?;
        let mlp_out = self.mlp.forward(&normed, backend)?;
        backend.add(&x, &mlp_out)
    }
}
