//! Naive scaled dot-product attention using cuBLAS matmul + softmax kernel.
//!
//! This is the Level 1 attention from the design doc — correct but not optimized.
//! Will be replaced by FlashAttention in Task 16.
//!
//! NOTE: Currently only supports F32 tensors. A cast layer will be added when
//! the model forward pass is integrated (F16 weights will be cast to F32 for compute).

use forge_core::{Backend, DType, ForgeError, Result, Tensor};

use crate::backend::CudaBackend;
use crate::tensor::CudaTensor;

/// Compute naive scaled dot-product attention.
///
/// Q: [batch, seq_len, num_heads, head_dim]
/// K: [batch, kv_len, num_kv_heads, head_dim]
/// V: [batch, kv_len, num_kv_heads, head_dim]
///
/// Returns: [batch, seq_len, num_heads, head_dim]
///
/// For simplicity, this implementation works on single-sequence inputs
/// (batch=1) and handles GQA by repeating KV heads.
pub fn naive_attention(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
) -> Result<CudaTensor> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    if q_shape.len() != 4 {
        return Err(ForgeError::InvalidArgument(
            "Q must be 4D [batch, seq_len, heads, head_dim]".into(),
        ));
    }
    if k_shape.len() != 4 {
        return Err(ForgeError::InvalidArgument(
            "K must be 4D [batch, kv_len, kv_heads, head_dim]".into(),
        ));
    }
    if v_shape.len() != 4 {
        return Err(ForgeError::InvalidArgument(
            "V must be 4D [batch, kv_len, kv_heads, head_dim]".into(),
        ));
    }
    if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
        return Err(ForgeError::InvalidArgument(
            "naive_attention currently only supports F32 tensors".into(),
        ));
    }

    let batch = q_shape[0];
    let seq_len = q_shape[1];
    let num_heads = q_shape[2];
    let head_dim = q_shape[3];
    let kv_len = k_shape[1];
    let num_kv_heads = k_shape[2];

    // For now, process one head at a time by reshaping into 2D matmuls.
    // This is correct but slow — FlashAttention will replace this.

    let heads_per_group = num_heads / num_kv_heads;

    // Flatten all tensors to 2D for slicing
    // Q: [seq_len * num_heads, head_dim]
    let q_2d = backend.reshape(q, &[seq_len * num_heads, head_dim])?;

    // K: [kv_len * num_kv_heads, head_dim]
    let k_2d = backend.reshape(k, &[kv_len * num_kv_heads, head_dim])?;
    let k_t = backend.transpose(&k_2d, 0, 1)?; // [head_dim, kv_len * num_kv_heads]

    // V: [kv_len * num_kv_heads, head_dim]
    let v_2d = backend.reshape(v, &[kv_len * num_kv_heads, head_dim])?;

    // Process per-token, per-head to produce correct interleaved layout.
    // Output layout: [seq_len, num_heads, head_dim] which reshapes to [batch, seq_len, num_heads, head_dim].
    //
    // We compute each head's attention and collect results in token-first order.
    // Each head output is [seq_len, head_dim], but we need to interleave by token.

    // Compute all head outputs first
    let mut head_outputs_data: Vec<Vec<f32>> = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;

        // Extract Q slice for this head: rows [seq_len*h .. seq_len*(h+1)]
        let q_head = extract_rows(backend, &q_2d, h * seq_len, seq_len, head_dim)?;

        // Extract K^T slice for this KV head: cols [kv_len*kv_h .. kv_len*(kv_h+1)]
        let k_t_head = extract_cols(backend, &k_t, kv_h * kv_len, kv_len, head_dim)?;

        // Extract V slice for this KV head
        let v_head = extract_rows(backend, &v_2d, kv_h * kv_len, kv_len, head_dim)?;

        // scores = Q @ K^T * scale: [seq_len, head_dim] @ [head_dim, kv_len] = [seq_len, kv_len]
        let scores = backend.matmul(&q_head, &k_t_head)?;
        let scores = backend.mul_scalar(&scores, scale)?;

        // attn_weights = softmax(scores, dim=-1)
        let attn_weights = backend.softmax(&scores, -1)?;

        // output = attn_weights @ V: [seq_len, kv_len] @ [kv_len, head_dim] = [seq_len, head_dim]
        let head_out = backend.matmul(&attn_weights, &v_head)?;
        let data = backend.copy_to_host_f32(&head_out)?;
        head_outputs_data.push(data);
    }

    // Interleave: build output in [seq_len, num_heads, head_dim] order
    let mut result = Vec::with_capacity(seq_len * num_heads * head_dim);
    for t in 0..seq_len {
        for h in 0..num_heads {
            let offset = t * head_dim;
            result.extend_from_slice(&head_outputs_data[h][offset..offset + head_dim]);
        }
    }

    let out = backend.copy_from_host_f32(&result, &[batch, seq_len, num_heads, head_dim])?;
    Ok(out)
}

/// Extract `num_rows` rows starting at `start_row` from a 2D tensor.
fn extract_rows(
    backend: &CudaBackend,
    tensor: &CudaTensor,
    start_row: usize,
    num_rows: usize,
    cols: usize,
) -> Result<CudaTensor> {
    // Read all data, slice on CPU, copy back. This is slow but correct.
    // TODO: Replace with a CUDA kernel for slicing.
    let all_data = backend.copy_to_host_f32(tensor)?;
    let start = start_row * cols;
    let end = start + num_rows * cols;
    backend.copy_from_host_f32(&all_data[start..end], &[num_rows, cols])
}

/// Extract `num_cols` columns starting at `start_col` from a 2D tensor [rows, total_cols].
fn extract_cols(
    backend: &CudaBackend,
    tensor: &CudaTensor,
    start_col: usize,
    num_cols: usize,
    rows: usize,
) -> Result<CudaTensor> {
    // Read all data, slice columns on CPU, copy back.
    let all_data = backend.copy_to_host_f32(tensor)?;
    let total_cols = tensor.shape()[1];
    let mut result = Vec::with_capacity(rows * num_cols);
    for r in 0..rows {
        let row_start = r * total_cols + start_col;
        result.extend_from_slice(&all_data[row_start..row_start + num_cols]);
    }
    backend.copy_from_host_f32(&result, &[rows, num_cols])
}
