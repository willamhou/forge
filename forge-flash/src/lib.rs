//! FlashAttention v2 FFI bindings.
//!
//! Provides safe Rust wrappers around vendored FA2 C++ CUDA kernels.
//! Supports SM80 (Ampere) and SM90 (Hopper).

/// Data type for FlashAttention inputs.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum FlashDType {
    F16 = 0,
    BF16 = 1,
}

unsafe extern "C" {
    fn forge_flash_attn_fwd(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        scale: f32,
        is_causal: bool,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    );
}

/// Run FlashAttention v2 forward pass.
///
/// # Safety
/// - All device pointers must be valid CUDA allocations with correct sizes
/// - `stream` must be a valid `CUstream` (pass 0 for default stream)
/// - Q shape: `[batch_size, seqlen_q, num_heads, head_dim]` (contiguous)
/// - K shape: `[batch_size, seqlen_k, num_heads_k, head_dim]` (contiguous)
/// - V shape: `[batch_size, seqlen_k, num_heads_k, head_dim]` (contiguous)
/// - out shape: `[batch_size, seqlen_q, num_heads, head_dim]` (pre-allocated)
/// - Data must be F16 or BF16 (not F32)
pub unsafe fn flash_fwd(
    q: u64,
    k: u64,
    v: u64,
    out: u64,
    batch_size: i32,
    seqlen_q: i32,
    seqlen_k: i32,
    num_heads: i32,
    num_heads_k: i32,
    head_dim: i32,
    scale: f32,
    is_causal: bool,
    dtype: FlashDType,
    stream: u64,
) {
    unsafe {
        forge_flash_attn_fwd(
            q as *const core::ffi::c_void,
            k as *const core::ffi::c_void,
            v as *const core::ffi::c_void,
            out as *mut core::ffi::c_void,
            batch_size,
            seqlen_q,
            seqlen_k,
            num_heads,
            num_heads_k,
            head_dim,
            scale,
            is_causal,
            dtype as i32,
            stream as *mut core::ffi::c_void,
        );
    }
}
