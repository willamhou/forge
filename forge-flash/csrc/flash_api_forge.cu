/******************************************************************************
 * flash_api_forge.cu -- Thin C-linkage wrapper around FlashAttention v2
 *
 * This file bypasses PyTorch entirely. It constructs Flash_fwd_params from
 * raw device pointers and scalar dimensions, then calls FA2's internal
 * run_mha_fwd_ dispatch (via FP16_SWITCH / HEADDIM_SWITCH / BOOL_SWITCH).
 *
 * Copyright (c) 2024, forge contributors.  FA2 code is Copyright (c) 2023-2024, Tri Dao.
 ******************************************************************************/

// Must be defined before any FA2 header so that ATen/c10 stubs are used.
#ifndef FORGE_NO_PYTORCH
#define FORGE_NO_PYTORCH
#endif

// We only need forward pass, and we disable dropout/alibi/softcap/local
// attention at compile time to reduce the number of template instantiations.
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_LOCAL

#include <cstring>   // memset
#include <cmath>     // M_LOG2E
#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>

#include "flash_attn/src/namespace_config.h"
#include "flash_attn/src/flash.h"
#include "flash_attn/src/static_switch.h"

namespace FLASH_NAMESPACE {

// ---------- run_mha_fwd dispatch (same as flash_api.cpp:243) ----------------
// This calls into the template specializations compiled from the individual
// flash_fwd_hdim*.cu files.

static void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                // For inference we never use the split-KV path (num_splits <= 1).
                run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}

} // namespace FLASH_NAMESPACE

// ---------- Helper -----------------------------------------------------------

static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// ---------- C-linkage entry point for Rust FFI --------------------------------

extern "C" void forge_flash_attn_fwd(
    void* q_ptr,      // [B, seqlen_q, num_heads, head_dim]  device ptr
    void* k_ptr,      // [B, seqlen_k, num_heads_k, head_dim]
    void* v_ptr,      // [B, seqlen_k, num_heads_k, head_dim]
    void* out_ptr,     // [B, seqlen_q, num_heads, head_dim]
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    int dtype,         // 0 = F16, 1 = BF16
    void* stream       // cudaStream_t
) {
    using namespace FLASH_NAMESPACE;

    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    // --- dtype ---
    params.is_bf16 = (dtype == 1);

    // --- pointers ---
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = out_ptr;

    // --- dimensions ---
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_dim;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;

    // Rounded dimensions used by the kernels.
    params.d_rounded = round_up(head_dim, head_dim <= 128 ? 32 : 64);
    params.seqlen_q_rounded = round_up(seqlen_q, 128);
    params.seqlen_k_rounded = round_up(seqlen_k, 128);

    // --- strides (row-major: [B, seqlen, num_heads, head_dim]) ---
    // Stride units are in *elements* (not bytes).
    params.q_batch_stride  = static_cast<int64_t>(seqlen_q) * num_heads   * head_dim;
    params.k_batch_stride  = static_cast<int64_t>(seqlen_k) * num_heads_k * head_dim;
    params.v_batch_stride  = static_cast<int64_t>(seqlen_k) * num_heads_k * head_dim;
    params.o_batch_stride  = static_cast<int64_t>(seqlen_q) * num_heads   * head_dim;

    params.q_row_stride    = static_cast<int64_t>(num_heads)   * head_dim;
    params.k_row_stride    = static_cast<int64_t>(num_heads_k) * head_dim;
    params.v_row_stride    = static_cast<int64_t>(num_heads_k) * head_dim;
    params.o_row_stride    = static_cast<int64_t>(num_heads)   * head_dim;

    params.q_head_stride   = head_dim;
    params.k_head_stride   = head_dim;
    params.v_head_stride   = head_dim;
    params.o_head_stride   = head_dim;

    // --- softmax scale ---
    params.scale_softmax      = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * static_cast<float>(M_LOG2E);

    // --- causal ---
    params.is_causal = is_causal;
    if (is_causal) {
        params.window_size_left  = -1;
        params.window_size_right =  0;
    } else {
        params.window_size_left  = -1;
        params.window_size_right = -1;
    }

    // --- no dropout (inference only) ---
    params.p_dropout               = 1.0f;   // probability of *keeping*
    params.p_dropout_in_uint8_t    = 255;
    params.rp_dropout              = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    // --- no split-KV ---
    params.num_splits = 1;

    // --- cumulative seqlens (nullptr = fixed-length batch) ---
    params.is_seqlens_k_cumulative = true;

    // --- softcap disabled ---
    params.softcap = 0.0f;

    // --- allocate scratch for softmax log-sum-exp (always written by kernel) ---
    // Shape: [batch_size, num_heads, seqlen_q]  (float32)
    void* softmax_lse = nullptr;
    size_t lse_bytes = static_cast<size_t>(batch_size) * num_heads * seqlen_q * sizeof(float);
    cudaMalloc(&softmax_lse, lse_bytes);
    params.softmax_lse_ptr = softmax_lse;

    // --- launch ---
    run_mha_fwd(params, static_cast<cudaStream_t>(stream));

    // Free the scratch buffer (synchronize on the stream first so the kernel
    // has finished writing to it before we free).
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    cudaFree(softmax_lse);
}
