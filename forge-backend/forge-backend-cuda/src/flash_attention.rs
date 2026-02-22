//! Attention dispatch entry point.
//!
//! Currently delegates to `naive_attention_causal` (GPU-native per-head attention).
//! Phase 2 will replace with a fused PagedAttention CUDA kernel.

use forge_core::Result;

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
    naive_attention_causal(backend, q, k, v, scale, is_causal)
}
