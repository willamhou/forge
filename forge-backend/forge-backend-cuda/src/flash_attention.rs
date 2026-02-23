//! Attention dispatch entry point.
//!
//! Routes to FlashAttention v2 (when `flash-attn` feature is enabled)
//! or falls back to naive per-head GPU attention.

use forge_core::Result;

#[cfg(feature = "flash-attn")]
use forge_core::DType;

use crate::attention::naive_attention_causal;
use crate::backend::CudaBackend;
use crate::tensor::CudaTensor;

/// Run scaled dot-product attention.
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
        return flash_attn_dispatch(backend, q, k, v, scale, is_causal);
    }

    #[cfg(not(feature = "flash-attn"))]
    {
        naive_attention_causal(backend, q, k, v, scale, is_causal)
    }
}

#[cfg(feature = "flash-attn")]
fn flash_attn_dispatch(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
    is_causal: bool,
) -> Result<CudaTensor> {
    use cudarc::driver::DevicePtr;
    use forge_core::{Backend, ForgeError, Tensor};

    let q_shape = q.shape();
    let batch_size = q_shape[0] as i32;
    let seqlen_q = q_shape[1] as i32;
    let num_heads = q_shape[2] as i32;
    let head_dim = q_shape[3] as i32;
    let seqlen_k = k.shape()[1] as i32;
    let num_heads_k = k.shape()[2] as i32;

    // FA2 requires F16 or BF16 — auto-cast from F32 if needed
    let original_dtype = q.dtype();
    let (q, k, v, fa_dtype) = match original_dtype {
        DType::F16 => (
            q.clone(),
            k.clone(),
            v.clone(),
            forge_flash::FlashDType::F16,
        ),
        DType::BF16 => (
            q.clone(),
            k.clone(),
            v.clone(),
            forge_flash::FlashDType::BF16,
        ),
        DType::F32 => {
            let q = backend.cast(q, DType::F16)?;
            let k = backend.cast(k, DType::F16)?;
            let v = backend.cast(v, DType::F16)?;
            (q, k, v, forge_flash::FlashDType::F16)
        }
    };

    // Allocate output tensor (same shape + dtype as Q after cast)
    let out = backend.allocate(q.shape(), q.dtype())?;

    // Get raw device pointers
    let (q_ptr, _q_guard) = match &q.data {
        crate::tensor::TensorData::F16(s) => (s.device_ptr(&backend.stream).0, Some(())),
        crate::tensor::TensorData::BF16(s) => (s.device_ptr(&backend.stream).0, Some(())),
        _ => return Err(ForgeError::InvalidArgument("FA2 requires F16/BF16".into())),
    };
    let (k_ptr, _k_guard) = match &k.data {
        crate::tensor::TensorData::F16(s) => (s.device_ptr(&backend.stream).0, Some(())),
        crate::tensor::TensorData::BF16(s) => (s.device_ptr(&backend.stream).0, Some(())),
        _ => return Err(ForgeError::InvalidArgument("FA2 requires F16/BF16".into())),
    };
    let (v_ptr, _v_guard) = match &v.data {
        crate::tensor::TensorData::F16(s) => (s.device_ptr(&backend.stream).0, Some(())),
        crate::tensor::TensorData::BF16(s) => (s.device_ptr(&backend.stream).0, Some(())),
        _ => return Err(ForgeError::InvalidArgument("FA2 requires F16/BF16".into())),
    };
    let (out_ptr, _out_guard) = match &out.data {
        crate::tensor::TensorData::F16(s) => (s.device_ptr(&backend.stream).0, Some(())),
        crate::tensor::TensorData::BF16(s) => (s.device_ptr(&backend.stream).0, Some(())),
        _ => return Err(ForgeError::InvalidArgument("FA2 requires F16/BF16".into())),
    };

    // Default stream (0); TODO: extract from backend.stream
    let stream_ptr: u64 = 0;

    unsafe {
        forge_flash::flash_fwd(
            q_ptr, k_ptr, v_ptr, out_ptr, batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k,
            head_dim, scale, is_causal, fa_dtype, stream_ptr,
        );
    }

    // Cast back to original dtype if needed
    if original_dtype == DType::F32 {
        backend.cast(&out, DType::F32)
    } else {
        Ok(out)
    }
}
