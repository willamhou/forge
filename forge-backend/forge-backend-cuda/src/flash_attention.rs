//! FlashAttention integration with fallback to naive attention.
//!
//! When compiled with `--features flash-attn`, this module attempts to call
//! the FlashAttention library via FFI. If the FFI call fails or the feature
//! is disabled, it falls back to `naive_attention`.

use forge_core::Result;

use crate::attention::naive_attention_causal;
use crate::backend::CudaBackend;
use crate::tensor::CudaTensor;

// FFI declarations for the FlashAttention wrapper.
// Only linked when the `flash-attn` feature is enabled.
#[cfg(feature = "flash-attn")]
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
    is_causal: bool,
) -> Result<CudaTensor> {
    #[cfg(feature = "flash-attn")]
    {
        use forge_core::{DType, Tensor};
        // FlashAttention requires fp16 or bf16 tensors.
        if q.dtype() != DType::F32 {
            match try_flash_attention(backend, q, k, v, scale, is_causal) {
                Ok(out) => return Ok(out),
                Err(e) => {
                    tracing::debug!("FlashAttention failed, falling back to naive: {e}");
                }
            }
        }
    }

    // Fallback: naive attention (always available).
    naive_attention_causal(backend, q, k, v, scale, is_causal)
}

/// Attempt to call FlashAttention via FFI.
///
/// Extracts raw device pointers from CudaTensor, gets the CUDA stream handle,
/// and calls `forge_flash_attn_fwd()`. Returns an error if the FFI call
/// returns non-zero (e.g., library not linked / stub returns -1).
#[cfg(feature = "flash-attn")]
fn try_flash_attention(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
    is_causal: bool,
) -> Result<CudaTensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use forge_core::{DType, ForgeError, Tensor};

    let q_shape = q.shape();
    let k_shape = k.shape();
    if q_shape.len() != 4 || k_shape.len() != 4 {
        return Err(ForgeError::InvalidArgument(
            "FlashAttention requires 4D tensors [batch, seq_len, heads, head_dim]".into(),
        ));
    }

    let batch_size = q_shape[0] as i32;
    let seqlen_q = q_shape[1] as i32;
    let num_heads = q_shape[2] as i32;
    let head_dim = q_shape[3] as i32;
    let seqlen_k = k_shape[1] as i32;
    let num_heads_k = k_shape[2] as i32;

    // Allocate output tensor with same dtype and shape as Q
    let out_numel = q_shape.iter().product::<usize>();
    let mut out = match q.dtype() {
        DType::F16 => {
            let data = backend
                .stream
                .alloc_zeros::<half::f16>(out_numel)
                .map_err(|e| ForgeError::Cuda(e.to_string()))?;
            CudaTensor::f16_data(data, q_shape.to_vec())
        }
        DType::BF16 => {
            let data = backend
                .stream
                .alloc_zeros::<half::bf16>(out_numel)
                .map_err(|e| ForgeError::Cuda(e.to_string()))?;
            CudaTensor::bf16_data(data, q_shape.to_vec())
        }
        other => {
            return Err(ForgeError::InvalidArgument(format!(
                "FlashAttention requires fp16 or bf16, got {:?}",
                other
            )));
        }
    };

    // Extract raw device pointers from cudarc CudaSlice.
    // The SyncOnDrop guards must be kept alive until after the kernel is scheduled.
    let stream = &backend.stream;
    let raw_stream = stream.cu_stream() as *mut std::ffi::c_void;

    let ret = match q.dtype() {
        DType::F16 => {
            let (q_ptr, _q_guard) = q.f16_slice()?.device_ptr(stream);
            let (k_ptr, _k_guard) = k.f16_slice()?.device_ptr(stream);
            let (v_ptr, _v_guard) = v.f16_slice()?.device_ptr(stream);
            let (out_ptr, _out_guard) = out.f16_slice_mut()?.device_ptr_mut(stream);

            unsafe {
                forge_flash_attn_fwd(
                    q_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    v_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    batch_size,
                    seqlen_q,
                    seqlen_k,
                    num_heads,
                    num_heads_k,
                    head_dim,
                    scale,
                    is_causal as i32,
                    raw_stream,
                )
            }
        }
        DType::BF16 => {
            let (q_ptr, _q_guard) = q.bf16_slice()?.device_ptr(stream);
            let (k_ptr, _k_guard) = k.bf16_slice()?.device_ptr(stream);
            let (v_ptr, _v_guard) = v.bf16_slice()?.device_ptr(stream);
            let (out_ptr, _out_guard) = out.bf16_slice_mut()?.device_ptr_mut(stream);

            unsafe {
                forge_flash_attn_fwd(
                    q_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    v_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    batch_size,
                    seqlen_q,
                    seqlen_k,
                    num_heads,
                    num_heads_k,
                    head_dim,
                    scale,
                    is_causal as i32,
                    raw_stream,
                )
            }
        }
        _ => unreachable!(), // Checked above
    };

    if ret != 0 {
        return Err(ForgeError::Internal(format!(
            "FlashAttention FFI returned error code {ret}"
        )));
    }

    Ok(out)
}
