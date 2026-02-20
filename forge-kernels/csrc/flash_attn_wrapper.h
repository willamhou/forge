// FlashAttention FFI wrapper for Forge.
//
// This header defines the C-compatible interface that Rust calls via FFI.
// The implementation links against the FlashAttention library (system install
// or submodule build). When the `flash-attn` feature is disabled, the Rust
// side falls back to naive attention — this header is not compiled.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Run FlashAttention forward pass (causal or non-causal).
///
/// All pointers are device pointers on the same CUDA stream.
///
/// q: [batch_size, seqlen_q, num_heads, head_dim] — fp16 or bf16
/// k: [batch_size, seqlen_k, num_heads_k, head_dim] — fp16 or bf16
/// v: [batch_size, seqlen_k, num_heads_k, head_dim] — fp16 or bf16
/// out: [batch_size, seqlen_q, num_heads, head_dim] — same dtype as q
///
/// Returns 0 on success, non-zero on error.
int forge_flash_attn_fwd(
    void* q, void* k, void* v, void* out,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    float softmax_scale, int is_causal,
    void* stream
);

#ifdef __cplusplus
}
#endif
