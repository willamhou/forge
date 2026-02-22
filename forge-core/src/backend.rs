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
    fn rms_norm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor>;
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
