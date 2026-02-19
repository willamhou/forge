//! Naive scaled dot-product attention using cuBLAS matmul + softmax kernel.
//!
//! This is the Level 1 attention from the design doc — correct but not optimized.
//! Will be replaced by FlashAttention in Task 16.

use forge_core::{Backend, Result, Tensor};

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

    assert_eq!(q_shape.len(), 4, "Q must be 4D");
    assert_eq!(k_shape.len(), 4, "K must be 4D");

    let batch = q_shape[0];
    let seq_len = q_shape[1];
    let num_heads = q_shape[2];
    let head_dim = q_shape[3];
    let kv_len = k_shape[1];
    let num_kv_heads = k_shape[2];

    // For now, process one head at a time by reshaping into 2D matmuls.
    // This is correct but slow — FlashAttention will replace this.

    // Flatten Q to [seq_len * num_heads, head_dim] for the batch=1 case
    let q_2d = backend.reshape(q, &[seq_len * num_heads, head_dim])?;

    // Handle GQA: if num_kv_heads < num_heads, repeat KV heads
    let heads_per_group = num_heads / num_kv_heads;

    // K^T: [kv_len * num_kv_heads, head_dim] -> transpose -> [head_dim, kv_len * num_kv_heads]
    let k_2d = backend.reshape(k, &[kv_len * num_kv_heads, head_dim])?;
    let k_t = backend.transpose(&k_2d, 0, 1)?; // [head_dim, kv_len * num_kv_heads]

    // V: [kv_len * num_kv_heads, head_dim]
    let v_2d = backend.reshape(v, &[kv_len * num_kv_heads, head_dim])?;

    // For each query head, compute attention against its corresponding KV head
    let mut head_outputs = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;

        // Extract Q slice for this head: rows [seq_len*h .. seq_len*(h+1)] of q_2d
        // For batch=1, each head has seq_len rows
        let q_head = extract_rows(backend, &q_2d, h * seq_len, seq_len, head_dim)?;

        // Extract K^T slice for this KV head: cols [kv_len*kv_h .. kv_len*(kv_h+1)]
        // K^T is [head_dim, kv_len*num_kv_heads], we need columns for this head
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
        head_outputs.push(head_out);
    }

    // Concatenate all heads: each is [seq_len, head_dim]
    // Result shape: [seq_len * num_heads, head_dim]
    let refs: Vec<&CudaTensor> = head_outputs.iter().collect();
    let concatenated = backend.cat(&refs, 0)?;

    // Reshape to [batch, seq_len, num_heads, head_dim]
    backend.reshape(&concatenated, &[batch, seq_len, num_heads, head_dim])
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
