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

    if k_shape[3] != head_dim || v_shape[3] != head_dim {
        return Err(ForgeError::InvalidArgument(format!(
            "Q/K/V head_dim must match: Q={}, K={}, V={}",
            head_dim, k_shape[3], v_shape[3]
        )));
    }

    // For now, process one head at a time by reshaping into 2D matmuls.
    // This is correct but slow — FlashAttention will replace this.

    if num_kv_heads == 0 || num_heads < num_kv_heads || num_heads % num_kv_heads != 0 {
        return Err(ForgeError::InvalidArgument(format!(
            "num_heads ({num_heads}) must be a positive multiple of num_kv_heads ({num_kv_heads})"
        )));
    }
    let heads_per_group = num_heads / num_kv_heads;

    // Q/K/V are [batch, seq_len, num_heads, head_dim] — token-major order in memory.
    // After reshape to 2D, layout is [t0h0, t0h1, ..., t1h0, t1h1, ...].
    // We read all data to CPU and extract per-head slices with stride.

    let q_data = backend.copy_to_host_f32(q)?;
    let k_data = backend.copy_to_host_f32(k)?;
    let v_data = backend.copy_to_host_f32(v)?;

    // Compute all head outputs first
    let mut head_outputs_data: Vec<Vec<f32>> = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;

        // Extract Q for this head: for each token t, take q_data[t * num_heads * head_dim + h * head_dim .. + head_dim]
        let q_head_data = extract_head(&q_data, seq_len, num_heads, head_dim, h);
        let q_head = backend.copy_from_host_f32(&q_head_data, &[seq_len, head_dim])?;

        // Extract K for this KV head
        let k_head_data = extract_head(&k_data, kv_len, num_kv_heads, head_dim, kv_h);
        let k_head = backend.copy_from_host_f32(&k_head_data, &[kv_len, head_dim])?;
        let k_t_head = backend.transpose(&k_head, 0, 1)?;

        // Extract V for this KV head
        let v_head_data = extract_head(&v_data, kv_len, num_kv_heads, head_dim, kv_h);
        let v_head = backend.copy_from_host_f32(&v_head_data, &[kv_len, head_dim])?;

        // scores = Q @ K^T * scale: [seq_len, head_dim] @ [head_dim, kv_len] = [seq_len, kv_len]
        let scores = backend.matmul(&q_head, &k_t_head)?;
        let scores = backend.mul_scalar(&scores, scale)?;

        // Apply causal mask for prefill (seq_len > 1): each query position
        // can only attend to positions at or before it.
        // During decode (seq_len == 1), the single query is at the latest
        // position and naturally attends to all cached KV entries — no mask needed.
        // NOTE: If multi-token decode (e.g. speculative) is added, revisit this guard.
        let scores = if seq_len > 1 {
            let mut scores_data = backend.copy_to_host_f32(&scores)?;
            for q_pos in 0..seq_len {
                let abs_pos = kv_len - seq_len + q_pos;
                for k_pos in (abs_pos + 1)..kv_len {
                    scores_data[q_pos * kv_len + k_pos] = f32::NEG_INFINITY;
                }
            }
            backend.copy_from_host_f32(&scores_data, &[seq_len, kv_len])?
        } else {
            scores
        };

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

/// Extract a single head's data from a flat [seq_len, num_heads, head_dim] buffer.
///
/// For each token `t`, copies `data[t * num_heads * head_dim + head * head_dim .. + head_dim]`.
/// Returns a contiguous `[seq_len, head_dim]` vector.
fn extract_head(
    data: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    head: usize,
) -> Vec<f32> {
    let stride = num_heads * head_dim;
    let mut out = Vec::with_capacity(seq_len * head_dim);
    for t in 0..seq_len {
        let start = t * stride + head * head_dim;
        out.extend_from_slice(&data[start..start + head_dim]);
    }
    out
}
