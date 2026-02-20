// FlashAttention FFI wrapper implementation.
//
// This file is only compiled when the `flash-attn` feature is enabled.
// It links against the FlashAttention library headers and calls the
// optimized kernel. When the library is not available, the build will
// fail with a clear error pointing to the FlashAttention dependency.
//
// For Phase 1 MVP, this is a stub that returns an error — the actual
// FlashAttention linkage will be wired when the library is available
// on the build system.

#include "flash_attn_wrapper.h"
#include <stdio.h>

// TODO: Include FlashAttention headers when available:
// #include "flash_attn/flash_api.h"

extern "C" int forge_flash_attn_fwd(
    const void* q, const void* k, const void* v, void* out,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    float softmax_scale, int is_causal,
    void* stream
) {
    // Stub: FlashAttention library not linked yet.
    // The Rust side should check the return code and fall back to
    // naive attention when this returns non-zero.
    (void)q; (void)k; (void)v; (void)out;
    (void)batch_size; (void)seqlen_q; (void)seqlen_k;
    (void)num_heads; (void)num_heads_k; (void)head_dim;
    (void)softmax_scale; (void)is_causal; (void)stream;
    return -1; // Not implemented — use naive fallback
}
