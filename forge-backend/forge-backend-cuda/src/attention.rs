//! Naive scaled dot-product attention using cuBLAS matmul + softmax kernel.
//!
//! GPU-native per-head attention — no CPU roundtrips. Uses Backend trait
//! methods (extract_head, apply_causal_mask, interleave_heads) which dispatch
//! to CUDA kernels on CudaBackend. Phase 2 will replace with fused PagedAttention.

use forge_core::{Backend, ForgeError, Result, Tensor};

use crate::backend::CudaBackend;
use crate::tensor::CudaTensor;

/// Compute naive scaled dot-product attention (causal).
///
/// Q: [batch, seq_len, num_heads, head_dim]
/// K: [batch, kv_len, num_kv_heads, head_dim]
/// V: [batch, kv_len, num_kv_heads, head_dim]
///
/// Returns: [batch, seq_len, num_heads, head_dim]
pub fn naive_attention(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
) -> Result<CudaTensor> {
    naive_attention_impl(backend, q, k, v, scale, true)
}

/// Compute naive scaled dot-product attention with explicit causal control.
pub fn naive_attention_causal(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
    is_causal: bool,
) -> Result<CudaTensor> {
    naive_attention_impl(backend, q, k, v, scale, is_causal)
}

fn naive_attention_impl(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
    is_causal: bool,
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

    if num_kv_heads == 0 || num_heads < num_kv_heads || num_heads % num_kv_heads != 0 {
        return Err(ForgeError::InvalidArgument(format!(
            "num_heads ({num_heads}) must be a positive multiple of num_kv_heads ({num_kv_heads})"
        )));
    }
    let heads_per_group = num_heads / num_kv_heads;

    if batch != 1 {
        return Err(ForgeError::InvalidArgument(format!(
            "naive attention only supports batch=1, got batch={batch}; use FlashAttention for batched inputs"
        )));
    }

    // Reshape to 3D for extract_head: strip batch dim (always 1)
    let q_3d = backend.reshape(q, &[seq_len, num_heads, head_dim])?;
    let k_3d = backend.reshape(k, &[kv_len, num_kv_heads, head_dim])?;
    let v_3d = backend.reshape(v, &[kv_len, num_kv_heads, head_dim])?;

    let mut head_outputs = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;

        // Extract per-head slices — GPU-native via CUDA kernels
        let q_head = backend.extract_head(&q_3d, seq_len, num_heads, head_dim, h)?;
        let k_head = backend.extract_head(&k_3d, kv_len, num_kv_heads, head_dim, kv_h)?;
        let v_head = backend.extract_head(&v_3d, kv_len, num_kv_heads, head_dim, kv_h)?;

        // scores = Q @ K^T * scale
        let k_t = backend.transpose(&k_head, 0, 1)?;
        let scores = backend.matmul(&q_head, &k_t)?;
        let scores = backend.mul_scalar(&scores, scale)?;

        // Apply causal mask for prefill (seq_len > 1)
        let scores = if is_causal && seq_len > 1 {
            backend.apply_causal_mask(&scores, seq_len, kv_len)?
        } else {
            scores
        };

        let attn = backend.softmax(&scores, -1)?;
        head_outputs.push(backend.matmul(&attn, &v_head)?);
    }

    // Interleave per-head outputs → [seq_len, num_heads * head_dim]
    let refs: Vec<&CudaTensor> = head_outputs.iter().collect();
    let interleaved = backend.interleave_heads(&refs, seq_len, head_dim)?;

    // Reshape back to 4D: [batch, seq_len, num_heads, head_dim]
    backend.reshape(&interleaved, &[batch, seq_len, num_heads, head_dim])
}
