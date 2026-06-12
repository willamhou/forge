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
        softmax_lse: *mut core::ffi::c_void,
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

    fn forge_flash_attn_fwd_kvcache(
        q: *const core::ffi::c_void,
        kcache: *const core::ffi::c_void,
        vcache: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        softmax_lse: *mut core::ffi::c_void,
        block_table: *const i32,
        cache_seqlens: *const i32,
        batch_size: i32,
        seqlen_q: i32,
        num_blocks: i32,
        max_num_blocks_per_seq: i32,
        page_block_size: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        scale: f32,
        is_causal: bool,
        dtype: i32,
        num_splits: i32,
        num_sm: i32,
        softmax_lseaccum: *mut core::ffi::c_void,
        oaccum: *mut core::ffi::c_void,
        num_splits_cap: i32,
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
/// - `softmax_lse` points to a `[batch_size, num_heads, seqlen_q]` f32 scratch
///   buffer the kernel always writes; it must hold at least that many floats and
///   stay valid for the launch. Forward-only inference never reads it back.
/// - Data must be F16 or BF16 (not F32)
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_fwd(
    q: u64,
    k: u64,
    v: u64,
    out: u64,
    softmax_lse: u64,
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
            softmax_lse as *mut core::ffi::c_void,
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

/// Run FlashAttention v2 forward pass against a *paged* KV cache.
///
/// Used by forge for decode: `q` is small per-step (typically `seqlen_q = 1`)
/// and the K/V live in a contiguous block pool indexed by `block_table`.
///
/// # Safety
/// - All device pointers must be valid CUDA allocations with the correct shape.
/// - `q`     : F16/BF16, `[batch_size, seqlen_q, num_heads, head_dim]` row-major contig.
/// - `kcache`/`vcache`: F16/BF16, `[num_blocks, page_block_size, num_heads_k, head_dim]`
///   row-major contig. The KV cache memory is shared across all sequences.
/// - `out`   : F16/BF16, same shape and stride as `q`, pre-allocated.
/// - `softmax_lse` : f32 `[batch_size, num_heads, seqlen_q]` scratch the kernel
///   always writes; caller-owned, persistent, never read back in forward-only mode.
/// - `block_table`  : i32 device pointer to `[batch_size, max_num_blocks_per_seq]`
///   row-major contig. Entry `[b, j]` holds the pool block index of seq `b`'s `j`-th block.
/// - `cache_seqlens`: i32 device pointer of length `batch_size` holding each
///   sequence's *current K length* (not cumulative).
/// - `softmax_lseaccum`/`oaccum`: f32 split-KV reduction scratch, sized for
///   `num_splits_cap` splits — see the C entry's comment for layout. Both
///   must stay valid for the launch.
/// - `num_splits = 0` selects FA2's heuristic (clamped to `num_splits_cap`).
/// - `page_block_size` MUST be a multiple of 256 (FA2 hard constraint).
/// - `num_heads % num_heads_k == 0`; GQA is fully supported.
/// - `stream` is a `cudaStream_t` (pass 0 for default).
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_fwd_kvcache(
    q: u64,
    kcache: u64,
    vcache: u64,
    out: u64,
    softmax_lse: u64,
    block_table: u64,
    cache_seqlens: u64,
    batch_size: i32,
    seqlen_q: i32,
    num_blocks: i32,
    max_num_blocks_per_seq: i32,
    page_block_size: i32,
    num_heads: i32,
    num_heads_k: i32,
    head_dim: i32,
    scale: f32,
    is_causal: bool,
    dtype: FlashDType,
    num_splits: i32,
    num_sm: i32,
    softmax_lseaccum: u64,
    oaccum: u64,
    num_splits_cap: i32,
    stream: u64,
) {
    unsafe {
        forge_flash_attn_fwd_kvcache(
            q as *const core::ffi::c_void,
            kcache as *const core::ffi::c_void,
            vcache as *const core::ffi::c_void,
            out as *mut core::ffi::c_void,
            softmax_lse as *mut core::ffi::c_void,
            block_table as *const i32,
            cache_seqlens as *const i32,
            batch_size,
            seqlen_q,
            num_blocks,
            max_num_blocks_per_seq,
            page_block_size,
            num_heads,
            num_heads_k,
            head_dim,
            scale,
            is_causal,
            dtype as i32,
            num_splits,
            num_sm,
            softmax_lseaccum as *mut core::ffi::c_void,
            oaccum as *mut core::ffi::c_void,
            num_splits_cap,
            stream as *mut core::ffi::c_void,
        );
    }
}
