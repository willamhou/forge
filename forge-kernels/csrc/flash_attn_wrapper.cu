// FlashAttention FFI wrapper implementation.
//
// When FLASH_ATTN_AVAILABLE is defined (set by build.rs when FLASH_ATTN_INCLUDE_DIR
// is provided), this calls the actual FlashAttention library. Otherwise it returns
// -1, signaling the Rust caller to fall back to naive attention.

#include "flash_attn_wrapper.h"

#ifdef FLASH_ATTN_AVAILABLE
#include "flash_attn/flash_api.h"
#endif

extern "C" int forge_flash_attn_fwd(
    const void* q, const void* k, const void* v, void* out,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    float softmax_scale, int is_causal,
    void* stream
) {
#ifdef FLASH_ATTN_AVAILABLE
    // TODO: Wire actual FlashAttention API call here when the library
    // headers/API are finalized. The call pattern depends on which
    // FlashAttention version (v2/v3) is linked.
    //
    // For now, even with headers available, return -1 until the
    // specific API call is verified against the installed version.
    (void)q; (void)k; (void)v; (void)out;
    (void)batch_size; (void)seqlen_q; (void)seqlen_k;
    (void)num_heads; (void)num_heads_k; (void)head_dim;
    (void)softmax_scale; (void)is_causal; (void)stream;
    return -1;
#else
    // FlashAttention library not linked — Rust side falls back to naive attention.
    (void)q; (void)k; (void)v; (void)out;
    (void)batch_size; (void)seqlen_q; (void)seqlen_k;
    (void)num_heads; (void)num_heads_k; (void)head_dim;
    (void)softmax_scale; (void)is_causal; (void)stream;
    return -1;
#endif
}
