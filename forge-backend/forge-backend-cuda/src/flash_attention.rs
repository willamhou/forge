//! FlashAttention integration with fallback to naive attention.
//!
//! When compiled with `--features flash-attn`, this module attempts to call
//! the FlashAttention library via FFI. If the FFI call fails or the feature
//! is disabled, it falls back to `naive_attention`.
//!
//! Phase 1 MVP: all tensors are f32, and FlashAttention requires fp16/bf16,
//! so the naive path is always taken. The infrastructure is in place for when
//! fp16 compute and FlashAttention are available.

use forge_core::Result;

use crate::attention::naive_attention;
use crate::backend::CudaBackend;
use crate::tensor::CudaTensor;

// FFI declarations for the FlashAttention wrapper.
// Only linked when the `flash-attn` feature is enabled.
#[cfg(feature = "flash-attn")]
#[allow(dead_code)]
unsafe extern "C" {
    fn forge_flash_attn_fwd(
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// Run attention using FlashAttention if available, otherwise naive attention.
///
/// Q: [batch, seq_len, num_heads, head_dim]
/// K: [batch, kv_len, num_kv_heads, head_dim]
/// V: [batch, kv_len, num_kv_heads, head_dim]
///
/// Returns: [batch, seq_len, num_heads, head_dim]
pub fn attention_fwd(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
    _is_causal: bool,
) -> Result<CudaTensor> {
    #[cfg(feature = "flash-attn")]
    {
        use forge_core::{DType, Tensor};
        // FlashAttention requires fp16 or bf16 tensors.
        // Phase 1 uses f32, so this path is not yet taken.
        if q.dtype() != DType::F32 {
            match try_flash_attention(backend, q, k, v, scale, _is_causal) {
                Ok(out) => return Ok(out),
                Err(e) => {
                    tracing::debug!("FlashAttention failed, falling back to naive: {e}");
                }
            }
        }
    }

    // Fallback: naive attention (always available)
    // TODO: pass is_causal to naive_attention when causal masking is implemented
    naive_attention(backend, q, k, v, scale)
}

/// Attempt to call FlashAttention via FFI.
///
/// This function handles the cudarc device pointer extraction and FFI call.
/// It will be fully implemented when fp16 compute is added and the
/// FlashAttention library is linked.
#[cfg(feature = "flash-attn")]
fn try_flash_attention(
    _backend: &CudaBackend,
    _q: &CudaTensor,
    _k: &CudaTensor,
    _v: &CudaTensor,
    _scale: f32,
    _is_causal: bool,
) -> Result<CudaTensor> {
    // TODO: Extract device pointers from CudaTensor, get stream handle,
    // and call forge_flash_attn_fwd(). Requires:
    // 1. fp16 tensor support in the forward pass
    // 2. FlashAttention library linked at build time
    // 3. cudarc DevicePtr/DevicePtrMut trait for raw pointer extraction
    Err(forge_core::ForgeError::Internal(
        "FlashAttention FFI not yet wired — awaiting fp16 compute support".into(),
    ))
}
