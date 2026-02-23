use crate::tensor::Tensor;
use crate::{DType, Result};

pub trait Backend: Send + Sync + 'static {
    type Tensor: Tensor;

    fn name(&self) -> &str;
    fn device_count(&self) -> usize;

    // Allocation
    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;
    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;

    // Data transfer
    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_to_host_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>>;

    // Synchronization
    fn synchronize(&self) -> Result<()>;

    // Core ops
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn mul_scalar(&self, a: &Self::Tensor, scalar: f32) -> Result<Self::Tensor>;
    fn silu(&self, a: &Self::Tensor) -> Result<Self::Tensor>;

    /// Fused SiLU activation and element-wise multiply: out = silu(gate) * up
    fn fused_silu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor) -> Result<Self::Tensor> {
        let activated = self.silu(gate)?;
        self.mul(&activated, up)
    }

    fn rms_norm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor>;

    /// Fused residual addition + RMSNorm.
    /// Returns (normalized, updated_residual) where:
    ///   updated_residual = x + residual_in
    ///   normalized = rms_norm(updated_residual, weight, eps)
    fn fused_residual_rms_norm(
        &self,
        x: &Self::Tensor,
        residual: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<(Self::Tensor, Self::Tensor)> {
        let sum = self.add(x, residual)?;
        let normed = self.rms_norm(&sum, weight, eps)?;
        Ok((normed, sum))
    }

    /// Split a concatenated QKV tensor [rows, q_size + kv_size + kv_size] into
    /// (Q [rows, q_size], K [rows, kv_size], V [rows, kv_size]).
    fn split_qkv(
        &self,
        qkv: &Self::Tensor,
        q_size: usize,
        kv_size: usize,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor)> {
        let rows = qkv.shape()[0];
        let total_cols = q_size + 2 * kv_size;
        let data = self.copy_to_host_f32(qkv)?;
        let mut q_data = Vec::with_capacity(rows * q_size);
        let mut k_data = Vec::with_capacity(rows * kv_size);
        let mut v_data = Vec::with_capacity(rows * kv_size);
        for r in 0..rows {
            let row = &data[r * total_cols..(r + 1) * total_cols];
            q_data.extend_from_slice(&row[..q_size]);
            k_data.extend_from_slice(&row[q_size..q_size + kv_size]);
            v_data.extend_from_slice(&row[q_size + kv_size..]);
        }
        Ok((
            self.copy_from_host_f32(&q_data, &[rows, q_size])?,
            self.copy_from_host_f32(&k_data, &[rows, kv_size])?,
            self.copy_from_host_f32(&v_data, &[rows, kv_size])?,
        ))
    }

    /// Batched decode attention: N sequences x 1 query token each.
    /// Processes Q@K^T -> softmax -> @V for all sequences.
    ///
    /// `q`: [num_seqs, num_heads * head_dim]
    /// `k_caches`: per-sequence K caches, each [kv_len_i, num_kv_heads * head_dim]
    /// `v_caches`: per-sequence V caches, each [kv_len_i, num_kv_heads * head_dim]
    ///
    /// Returns: [num_seqs, num_heads * head_dim]
    fn batched_decode_attention(
        &self,
        q: &Self::Tensor,
        k_caches: &[Self::Tensor],
        v_caches: &[Self::Tensor],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Self::Tensor> {
        let n = k_caches.len();
        let heads_per_group = num_heads / num_kv_heads;
        let mut seq_outputs = Vec::with_capacity(n);

        for i in 0..n {
            let q_row = self.slice_rows(q, i, 1)?;
            let kv_len = k_caches[i].shape()[0];

            // Reshape for per-head extraction: [seq_len, num_heads, head_dim]
            let q_3d = self.reshape(&q_row, &[1, num_heads, head_dim])?;
            let k_3d = self.reshape(&k_caches[i], &[kv_len, num_kv_heads, head_dim])?;
            let v_3d = self.reshape(&v_caches[i], &[kv_len, num_kv_heads, head_dim])?;

            let mut head_outputs = Vec::with_capacity(num_heads);
            for h in 0..num_heads {
                let kv_h = h / heads_per_group;
                let q_head = self.extract_head(&q_3d, 1, num_heads, head_dim, h)?;
                let k_head = self.extract_head(&k_3d, kv_len, num_kv_heads, head_dim, kv_h)?;
                let v_head = self.extract_head(&v_3d, kv_len, num_kv_heads, head_dim, kv_h)?;

                let k_t = self.transpose(&k_head, 0, 1)?;
                let scores = self.matmul(&q_head, &k_t)?;
                let scores = self.mul_scalar(&scores, scale)?;
                // No causal mask needed for decode (seq_len=1)
                let attn = self.softmax(&scores, -1)?;
                head_outputs.push(self.matmul(&attn, &v_head)?);
            }

            let refs: Vec<&Self::Tensor> = head_outputs.iter().collect();
            let seq_out = self.interleave_heads(&refs, 1, head_dim)?;
            seq_outputs.push(seq_out);
        }

        let refs: Vec<&Self::Tensor> = seq_outputs.iter().collect();
        self.cat(&refs, 0)
    }

    /// Multi-head scaled dot-product attention.
    ///
    /// Q: [1, seq_len, num_heads, head_dim]
    /// K: [1, kv_len, num_kv_heads, head_dim]
    /// V: [1, kv_len, num_kv_heads, head_dim]
    ///
    /// Returns: [seq_len, num_heads * head_dim]
    ///
    /// Default impl: per-head loop (extract_head -> matmul -> softmax -> interleave).
    /// CUDA override: FlashAttention v2 when available.
    fn multi_head_attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<Self::Tensor> {
        let q_shape = q.shape();
        let seq_len = q_shape[1];
        let kv_len = k.shape()[1];
        let heads_per_group = num_heads / num_kv_heads;

        // Reshape to 3D for extract_head: strip batch dim (always 1)
        let q = self.reshape(q, &[seq_len, num_heads, head_dim])?;
        let k = self.reshape(k, &[kv_len, num_kv_heads, head_dim])?;
        let v = self.reshape(v, &[kv_len, num_kv_heads, head_dim])?;

        let mut head_outputs = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let kv_h = h / heads_per_group;

            let q_head = self.extract_head(&q, seq_len, num_heads, head_dim, h)?;
            let k_head = self.extract_head(&k, kv_len, num_kv_heads, head_dim, kv_h)?;
            let v_head = self.extract_head(&v, kv_len, num_kv_heads, head_dim, kv_h)?;

            let k_t = self.transpose(&k_head, 0, 1)?;
            let scores = self.matmul(&q_head, &k_t)?;
            let scores = self.mul_scalar(&scores, scale)?;

            let scores = if is_causal && seq_len > 1 {
                self.apply_causal_mask(&scores, seq_len, kv_len)?
            } else {
                scores
            };

            let attn = self.softmax(&scores, -1)?;
            head_outputs.push(self.matmul(&attn, &v_head)?);
        }

        let refs: Vec<&Self::Tensor> = head_outputs.iter().collect();
        self.interleave_heads(&refs, seq_len, head_dim)
    }

    fn rope(
        &self,
        x: &Self::Tensor,
        freqs_cos: &Self::Tensor,
        freqs_sin: &Self::Tensor,
    ) -> Result<Self::Tensor>;
    fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor>;
    fn embedding(&self, weight: &Self::Tensor, indices: &[u32]) -> Result<Self::Tensor>;
    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor>;
    fn transpose(&self, x: &Self::Tensor, dim0: usize, dim1: usize) -> Result<Self::Tensor>;
    fn cat(&self, tensors: &[&Self::Tensor], dim: usize) -> Result<Self::Tensor>;

    /// Cast a tensor to a different dtype. Returns the input unchanged if already the target dtype.
    fn cast(&self, x: &Self::Tensor, dtype: DType) -> Result<Self::Tensor>;

    // ── Attention helpers ───────────────────────────────────────
    // Default impls go through host; CudaBackend overrides with GPU kernels.

    /// Extract head `head` from `[seq_len, num_heads, head_dim]` layout → `[seq_len, head_dim]`.
    fn extract_head(
        &self,
        tensor: &Self::Tensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        head: usize,
    ) -> Result<Self::Tensor> {
        let data = self.copy_to_host_f32(tensor)?;
        let stride = num_heads * head_dim;
        let mut out = Vec::with_capacity(seq_len * head_dim);
        for t in 0..seq_len {
            let start = t * stride + head * head_dim;
            out.extend_from_slice(&data[start..start + head_dim]);
        }
        self.copy_from_host_f32(&out, &[seq_len, head_dim])
    }

    /// Apply causal mask: `scores[q][k] = -inf` for `k > kv_len - seq_len + q`.
    ///
    /// Input/output: `[seq_len, kv_len]`.
    fn apply_causal_mask(
        &self,
        scores: &Self::Tensor,
        seq_len: usize,
        kv_len: usize,
    ) -> Result<Self::Tensor> {
        let mut data = self.copy_to_host_f32(scores)?;
        for q_pos in 0..seq_len {
            let abs_pos = kv_len - seq_len + q_pos;
            for k_pos in (abs_pos + 1)..kv_len {
                data[q_pos * kv_len + k_pos] = f32::NEG_INFINITY;
            }
        }
        self.copy_from_host_f32(&data, scores.shape())
    }

    /// Extract rows `[start_row..start_row+num_rows]` from a tensor.
    /// Input: `[total_rows, cols...]`, Output: `[num_rows, cols...]`.
    fn slice_rows(
        &self,
        tensor: &Self::Tensor,
        start_row: usize,
        num_rows: usize,
    ) -> Result<Self::Tensor> {
        let shape = tensor.shape();
        let cols: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            1
        };
        let data = self.copy_to_host_f32(tensor)?;
        let offset = start_row * cols;
        let len = num_rows * cols;
        let mut out_shape = shape.to_vec();
        out_shape[0] = num_rows;
        self.copy_from_host_f32(&data[offset..offset + len], &out_shape)
    }

    /// Interleave per-head `[seq_len, head_dim]` outputs → `[seq_len, num_heads * head_dim]`.
    fn interleave_heads(
        &self,
        heads: &[&Self::Tensor],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Self::Tensor> {
        let num_heads = heads.len();
        let mut head_data: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
        for h in heads {
            head_data.push(self.copy_to_host_f32(h)?);
        }
        let mut result = Vec::with_capacity(seq_len * num_heads * head_dim);
        for t in 0..seq_len {
            for data in &head_data {
                let offset = t * head_dim;
                result.extend_from_slice(&data[offset..offset + head_dim]);
            }
        }
        self.copy_from_host_f32(&result, &[seq_len, num_heads * head_dim])
    }
}
