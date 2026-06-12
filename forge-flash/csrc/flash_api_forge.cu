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

// SplitKV dispatch used by the paged-KV decode path. The split kernel reads
// `params.block_table` / `params.page_block_size` and (when configured) the
// per-split accumulators in `oaccum_ptr` / `softmax_lseaccum_ptr`.
static void run_mha_fwd_splitkv(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}

// num_splits_heuristic copied verbatim from flash_api.cpp:263 so we don't pull
// in the ATen TU. Used only when the caller passes num_splits <= 0.
static inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    if (num_SMs < max_splits) { max_splits = num_SMs; }
    if (num_n_blocks < max_splits) { max_splits = num_n_blocks; }
    if (max_splits < 1) { max_splits = 1; }
    float max_efficiency = 0.f;
    float effs[129] = {0.f};
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int ns = 1; ns <= max_splits; ++ns) {
        if (!is_split_eligible(ns)) { effs[ns] = 0.f; continue; }
        float n_waves = float(batch_nheads_mblocks * ns) / float(num_SMs);
        float eff = n_waves / ceilf(n_waves);
        if (eff > max_efficiency) { max_efficiency = eff; }
        effs[ns] = eff;
    }
    for (int ns = 1; ns <= max_splits; ++ns) {
        if (!is_split_eligible(ns)) { continue; }
        if (effs[ns] >= 0.85f * max_efficiency) { return ns; }
    }
    return 1;
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
    void* softmax_lse_ptr, // [B, num_heads, seqlen_q] f32 scratch (caller-owned, persistent)
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

    // --- softmax log-sum-exp scratch (always written by the kernel) ---
    // Shape: [batch_size, num_heads, seqlen_q] f32. The caller passes a
    // persistent, backend-owned buffer so we avoid a per-call cudaMalloc/cudaFree
    // — and the cudaStreamSynchronize they'd otherwise require to free the buffer
    // safely, which serialized every layer's attention and killed CPU/GPU overlap
    // during prefill. LSE is never read back in forward-only inference, and all
    // ops run on `stream`, so reusing this buffer across calls is safe.
    params.softmax_lse_ptr = softmax_lse_ptr;

    // --- launch ---
    run_mha_fwd(params, static_cast<cudaStream_t>(stream));
}

// ---------- Paged-KV decode entry point --------------------------------------
//
// Mirrors FA2's `mha_fwd_kvcache` (flash_api.cpp:1203) for the *paged* case
// only. Used by forge for decode: the active sequences each carry a small
// query (typically seqlen_q = 1) against a paged K/V cache stored as a
// contiguous block pool. The kernel always routes through the split-KV
// dispatch because that's the only path that understands `block_table`.
//
// Memory layout expected:
//   q   : [batch, seqlen_q, num_heads,   head_dim]            row-major contig
//   K/V : [num_blocks, page_block_size, num_heads_k, head_dim] row-major contig
//   out : same as q                                            (pre-allocated)
//   block_table  : i32 [batch, max_num_blocks_per_seq]         row-major contig
//   cache_seqlens: i32 [batch]                                 contig
//
// All accumulator scratch (`softmax_lse_ptr`, `softmax_lseaccum_ptr`,
// `oaccum_ptr`) is caller-owned and persistent: see comment on
// `forge_flash_attn_fwd` above for why we avoid per-call allocation.
//
// FA2 hard-requires `page_block_size % 256 == 0` for paged KV. The caller
// MUST gate on that before invoking — we don't re-check here.
extern "C" void forge_flash_attn_fwd_kvcache(
    void* q_ptr,                  // [batch, seqlen_q, num_heads, head_dim]
    void* kcache_ptr,             // [num_blocks, page_block_size, num_heads_k, head_dim]
    void* vcache_ptr,             // [num_blocks, page_block_size, num_heads_k, head_dim]
    void* out_ptr,                // [batch, seqlen_q, num_heads, head_dim]
    void* softmax_lse_ptr,        // f32 [batch, num_heads, seqlen_q] (persistent)
    int*  block_table_ptr,        // i32 [batch, max_num_blocks_per_seq]
    int*  cache_seqlens_ptr,      // i32 [batch] — per-seq actual K length
    int batch_size,
    int seqlen_q,
    int num_blocks,               // unused except for sanity (caller-checked)
    int max_num_blocks_per_seq,
    int page_block_size,
    int num_heads,
    int num_heads_k,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    int dtype,                    // 0 = F16, 1 = BF16
    int num_splits,               // 0 = auto via heuristic; caller sizes scratch for the cap
    int num_sm,                   // SM count for the heuristic
    void* softmax_lseaccum_ptr,   // f32 [num_splits_cap, batch, num_heads, seqlen_q] (persistent)
    void* oaccum_ptr,             // f32 [num_splits_cap, batch, num_heads, seqlen_q, d_rounded] (persistent)
    int num_splits_cap,           // upper bound on num_splits the scratch is sized for
    void* stream
) {
    using namespace FLASH_NAMESPACE;
    (void) num_blocks;            // for asserts only on the caller side

    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    // --- dtype + scalar scale ---
    params.is_bf16            = (dtype == 1);
    params.scale_softmax      = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * static_cast<float>(M_LOG2E);

    // --- pointers ---
    params.q_ptr           = q_ptr;
    params.k_ptr           = kcache_ptr;
    params.v_ptr           = vcache_ptr;
    params.o_ptr           = out_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;

    // --- dimensions ---
    // For paged KV, FA2 treats seqlen_k = max_num_blocks_per_seq * page_block_size
    // (the *padded* K length); the real per-seq length comes from cache_seqlens.
    const int seqlen_k_padded = max_num_blocks_per_seq * page_block_size;

    // ── seqlenq_ngroups_swapped optimization (FA2 mha_fwd_kvcache:1278) ──
    // For decode (seqlen_q=1) with GQA, remap q so the kernel sees
    // `ngroups` q-tokens per kv-head: `[B, 1, num_heads, D]` is
    // reinterpreted as `[B, ngroups, num_heads_k, D]` via stride remap (no
    // data movement). The splitkv kernel then processes ngroups q-rows per
    // (b, kv_head) tile, getting better MMA utilization vs the 1-row tile.
    // Forge's GQA layout is `q_head_idx = kv_head*ngroups + group` (the
    // standard interleaved-pair-per-kv-head order), which matches FA2's
    // assumption. Output is written directly into the row-major
    // `[B, 1, num_heads, D]` layout via the inverse stride remap — no
    // post-transpose needed.
    const int ngroups = num_heads / num_heads_k;
    const bool seqlenq_ngroups_swapped =
        (seqlen_q == 1) && (ngroups > 1) && (head_dim % 8 == 0);
    const int eff_seqlen_q = seqlenq_ngroups_swapped ? ngroups : seqlen_q;
    const int eff_num_heads = seqlenq_ngroups_swapped ? num_heads_k : num_heads;

    params.b                = batch_size;
    params.seqlen_q         = eff_seqlen_q;
    params.seqlen_k         = seqlen_k_padded;
    params.d                = head_dim;
    params.h                = eff_num_heads;
    params.h_k              = num_heads_k;
    params.h_h_k_ratio      = eff_num_heads / num_heads_k;
    params.d_rounded        = round_up(head_dim, head_dim <= 128 ? 32 : 64);
    params.seqlen_q_rounded = round_up(eff_seqlen_q, 128);
    params.seqlen_k_rounded = round_up(seqlen_k_padded, 128);
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;

    // --- strides (in elements, not bytes) ---
    if (seqlenq_ngroups_swapped) {
        // Q / out are physically `[B, 1, num_heads, D]` row-major, but the
        // kernel reads/writes them as `[B, ngroups, num_heads_k, D]` where
        // `(m, h)` maps to the original q_head index `h*ngroups + m`. The
        // required strides for this mapping are:
        //   batch stride = num_heads * D    (per-sequence span unchanged)
        //   row   stride = D                (adjacent m's are 1 q_head apart)
        //   head  stride = ngroups * D      (adjacent h's are ngroups q_heads apart)
        params.q_batch_stride = static_cast<int64_t>(num_heads) * head_dim;
        params.q_row_stride   = head_dim;
        params.q_head_stride  = static_cast<int64_t>(ngroups) * head_dim;
        params.o_batch_stride = params.q_batch_stride;
        params.o_row_stride   = params.q_row_stride;
        params.o_head_stride  = params.q_head_stride;
    } else {
        // Q / out: row-major [B, seqlen_q, H, D]
        params.q_batch_stride = static_cast<int64_t>(seqlen_q) * num_heads * head_dim;
        params.q_row_stride   = static_cast<int64_t>(num_heads) * head_dim;
        params.q_head_stride  = head_dim;
        params.o_batch_stride = params.q_batch_stride;
        params.o_row_stride   = params.q_row_stride;
        params.o_head_stride  = params.q_head_stride;
    }

    // K / V cache: row-major [num_blocks, page_block_size, H_k, D] — the
    // "batch" axis is the block index, so k_batch_stride is the per-block
    // stride, NOT the per-seq stride.
    const int64_t per_block_elems = static_cast<int64_t>(page_block_size) * num_heads_k * head_dim;
    params.k_batch_stride = per_block_elems;
    params.v_batch_stride = per_block_elems;
    params.k_row_stride   = static_cast<int64_t>(num_heads_k) * head_dim;
    params.v_row_stride   = static_cast<int64_t>(num_heads_k) * head_dim;
    params.k_head_stride  = head_dim;
    params.v_head_stride  = head_dim;

    // --- causal / window ---
    params.is_causal = is_causal;
    if (is_causal) {
        params.window_size_left  = -1;
        params.window_size_right =  0;
    } else {
        params.window_size_left  = -1;
        params.window_size_right = -1;
    }

    // --- no dropout (inference) ---
    params.p_dropout                = 1.0f;
    params.p_dropout_in_uint8_t     = 255;
    params.rp_dropout               = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    // --- softcap disabled ---
    params.softcap = 0.0f;

    // --- per-seq actual K length, NOT cumulative ---
    // FA2 treats `cu_seqlens_k[b]` as the used length of sequence b when
    // is_seqlens_k_cumulative=false (see flash.h:143-145 and the kvcache
    // branch in flash_api.cpp:1392-1394).
    params.cu_seqlens_k             = cache_seqlens_ptr;
    params.is_seqlens_k_cumulative  = false;

    // --- paged KV wiring ---
    params.block_table              = block_table_ptr;
    params.block_table_batch_stride = max_num_blocks_per_seq;
    params.page_block_size          = page_block_size;

    // --- split-KV scratch + num_splits ---
    // The split kernel writes per-split partials to oaccum/lseaccum; a second
    // reduction kernel collapses them into the final out/softmax_lse. Caller
    // sized these buffers for `num_splits_cap`, so we clamp.
    const int block_n = head_dim <= 64 ? 256 : (head_dim <= 128 ? 128 : 64);
    const int num_n_blocks = (seqlen_k_padded + block_n - 1) / block_n;
    // Heuristic input must use the *effective* (post-swap) dimensions: the
    // kernel grid is (num_m_blocks, num_splits>1?num_splits:b, num_splits>1?
    // b*h:h) so b*h*num_m_blocks counts the work the heuristic sees. With
    // the swap, h=num_heads_k and seqlen_q=ngroups → b*num_heads_k*1 entries
    // instead of b*num_heads*1; the heuristic naturally picks more splits to
    // keep occupancy up.
    const int num_m_blocks = (eff_seqlen_q + 64 - 1) / 64;
    int chosen_splits = num_splits;
    if (chosen_splits < 1) {
        chosen_splits = num_splits_heuristic(
            batch_size * eff_num_heads * num_m_blocks,
            num_sm * 2,
            num_n_blocks,
            num_splits_cap);
    }
    if (chosen_splits > num_splits_cap) { chosen_splits = num_splits_cap; }
    if (chosen_splits < 1) { chosen_splits = 1; }
    params.num_splits = chosen_splits;
    if (chosen_splits > 1) {
        params.softmax_lseaccum_ptr = softmax_lseaccum_ptr;
        params.oaccum_ptr           = oaccum_ptr;
    } else {
        params.softmax_lseaccum_ptr = nullptr;
        params.oaccum_ptr           = nullptr;
    }

    // --- launch (always splitkv for paged KV) ---
    run_mha_fwd_splitkv(params, static_cast<cudaStream_t>(stream));
}
