//! Paged attention CUDA kernels (decode, q_len = 1).
//!
//! Same algorithm as `decode_attention.rs` (two-pass online softmax against
//! a single query vector per (seq, head)) — the only difference is K/V
//! pointer computation: we index into a single pool tensor via per-seq
//! block tables instead of dereferencing per-seq K/V pointer arrays.
//!
//! Grid: (num_seqs, num_heads, 1)
//! Block: (THREADS_PER_BLOCK, 1, 1) — typically 128
//! Shared mem: `blockDim.x * sizeof(float) + head_dim * sizeof(float)`
//!   [0..blockDim.x]: reduction scratch
//!   [blockDim.x..+head_dim]: per-block output accumulator
//!
//! Pool layout: `[num_blocks, block_size, num_kv_heads * head_dim]` (matches
//! `forge_kvcache::PagedKvCache` allocator).
//! Block tables: `[num_seqs, max_blocks_per_seq]` i32, `-1` = padding (never
//! dereferenced because the loop bound is `kv_lens[seq]`).
//!
//! F16 / BF16 variants follow once F32 is validated.

pub const F32_SRC: &str = r#"
extern "C" __global__ void paged_attention_f32(
    float* out,                       // [num_seqs, num_heads, head_dim]
    const float* q,                   // [num_seqs, num_heads, head_dim]
    const float* k_pool,              // [num_blocks, block_size, num_kv_heads * head_dim]
    const float* v_pool,              // same shape as k_pool
    const int* block_tables,          // [num_seqs, max_blocks_per_seq]
    const int* kv_lens,               // [num_seqs]
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int kv_len = kv_lens[seq_idx];
    int heads_per_group = num_heads / num_kv_heads;
    int kv_head = head_idx / heads_per_group;
    int kv_dim = num_kv_heads * head_dim;

    const float* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    extern __shared__ float smem[];
    float* scratch = smem;
    float* s_out = smem + blockDim.x;

    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    __syncthreads();

    // Pass 1: compute Q@K^T scores, find local max per thread.
    float local_max = -1e30f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        int block_idx = t / block_size;
        int slot      = t - block_idx * block_size;
        int block_id  = my_block_table[block_idx];
        const float* k_t = k_pool
            + block_id * block_size * kv_dim
            + slot * kv_dim
            + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += q_ptr[d] * k_t[d];
        score *= scale;
        if (score > local_max) local_max = score;
    }

    // Block reduce to global max.
    scratch[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid])
            scratch[tid] = scratch[tid + s];
        __syncthreads();
    }
    float global_max = scratch[0];
    __syncthreads();

    // Pass 2: softmax weights + weighted V accumulation.
    float local_sum = 0.0f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        int block_idx = t / block_size;
        int slot      = t - block_idx * block_size;
        int block_id  = my_block_table[block_idx];
        const float* k_t = k_pool
            + block_id * block_size * kv_dim
            + slot * kv_dim
            + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += q_ptr[d] * k_t[d];
        score *= scale;
        float w = expf(score - global_max);
        local_sum += w;

        const float* v_t = v_pool
            + block_id * block_size * kv_dim
            + slot * kv_dim
            + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&s_out[d], w * v_t[d]);
    }
    __syncthreads();

    // Reduce softmax denominator.
    scratch[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float total_sum = scratch[0];
    __syncthreads();

    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    float* out_ptr = out + seq_idx * num_heads * head_dim + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x)
        out_ptr[d] = s_out[d] * inv_sum;
}

// Scatter rows of `src` into the paged pool at slots given by a DEVICE
// `slot_mapping` tensor. Unlike the host-loop memcpy_dtod in paged_write_kv,
// the write destination is computed from a device value at kernel runtime, so
// this op is safe to record inside a captured CUDA Graph: re-staging
// slot_mapping (and replaying) writes to the new slots without re-capture.
//
// Pool is contiguous [num_blocks, block_size, kv_dim]; a flat slot index
// `slot` addresses element block*block_size + slot_in_block, exactly what
// slot_mapping encodes (dst = slot * kv_dim). OOB / negative slots are skipped
// defensively (a kernel cannot return an error); the host validates first.
//
// Grid: (n_rows, 1, 1)   Block: (min(256, kv_dim), 1, 1)
extern "C" __global__ void scatter_kv_f32(
    float* pool,                  // [total_slots, kv_dim] (flattened pool)
    const float* src,             // [n_rows, kv_dim]
    const int* slot_mapping,      // [n_rows]
    unsigned int n_rows,
    unsigned int kv_dim,
    unsigned int total_slots) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;
    int slot = slot_mapping[row];
    if (slot < 0 || (unsigned int)slot >= total_slots) return;
    const float* s = src + (size_t)row * kv_dim;
    float* d = pool + (size_t)slot * kv_dim;
    for (unsigned int i = threadIdx.x; i < kv_dim; i += blockDim.x)
        d[i] = s[i];
}
"#;

/// F16 variant. Same algorithm; accumulation stays in f32 for numerical
/// stability (softmax exponents underflow fast at f16). Pool / q / out are
/// `__half`; intermediate scores, accumulator, and softmax denominator are
/// all f32 in shared memory and registers — `__half2float` on read,
/// `__float2half` on write.
pub const F16_SRC: &str = r#"
extern "C" __global__ void paged_attention_f16(
    __half* out,
    const __half* q,
    const __half* k_pool,
    const __half* v_pool,
    const int* block_tables,
    const int* kv_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int kv_len = kv_lens[seq_idx];
    int heads_per_group = num_heads / num_kv_heads;
    int kv_head = head_idx / heads_per_group;
    int kv_dim = num_kv_heads * head_dim;

    const __half* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    extern __shared__ float smem[];
    float* scratch = smem;
    float* s_out = smem + blockDim.x;

    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    __syncthreads();

    float local_max = -1e30f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        int block_idx = t / block_size;
        int slot      = t - block_idx * block_size;
        int block_id  = my_block_table[block_idx];
        const __half* k_t = k_pool
            + block_id * block_size * kv_dim
            + slot * kv_dim
            + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += __half2float(q_ptr[d]) * __half2float(k_t[d]);
        score *= scale;
        if (score > local_max) local_max = score;
    }

    scratch[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid])
            scratch[tid] = scratch[tid + s];
        __syncthreads();
    }
    float global_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        int block_idx = t / block_size;
        int slot      = t - block_idx * block_size;
        int block_id  = my_block_table[block_idx];
        const __half* k_t = k_pool
            + block_id * block_size * kv_dim
            + slot * kv_dim
            + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += __half2float(q_ptr[d]) * __half2float(k_t[d]);
        score *= scale;
        float w = expf(score - global_max);
        local_sum += w;

        const __half* v_t = v_pool
            + block_id * block_size * kv_dim
            + slot * kv_dim
            + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&s_out[d], w * __half2float(v_t[d]));
    }
    __syncthreads();

    scratch[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float total_sum = scratch[0];
    __syncthreads();

    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    __half* out_ptr = out + seq_idx * num_heads * head_dim + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x)
        out_ptr[d] = __float2half(s_out[d] * inv_sum);
}

// ───────────────────────────────────────────────────────────────────────
// Split-KV flash-decoding (the fast decode attention path).
//
// The naive `paged_attention_f16` above launches only (num_seqs, num_heads)
// blocks — at batch=1 that's 32 blocks on a GB10 with far more SMs, so the
// device sits idle while a handful of blocks each scan the *entire* ~1400-token
// KV serially (and twice: once for max, once for softmax). It also accumulates
// the weighted-V via `atomicAdd` into shared memory (head_dim-wide contention).
//
// Split-KV fixes occupancy by adding a 3rd grid dim `num_splits`: the KV range
// of each (seq, head) is partitioned into `num_splits` contiguous chunks, and a
// separate block reduces each chunk independently (single-pass online softmax),
// writing a partial {out, running-max m, running-sum l} per split into scratch.
// A tiny combine kernel then merges the `num_splits` partials per (seq, head)
// into the final output with the usual log-sum-exp rescaling. With num_splits up
// to 16 the main kernel launches (num_seqs * num_heads * num_splits) blocks —
// enough to saturate the device even at batch=1.
//
// Block layout for the SPLIT kernel: `head_dim` threads (== 128 here, 4 warps).
// Thread `d` owns output element `acc[d]` in a register — NO atomicAdd. The per
// token score `s = scale * Σ_d q[d]*k[d]` is a block reduction over those
// threads (warp shuffle + 1 shared word per warp), broadcast back, then each
// thread does its own `acc[d] = acc[d]*alpha + w*v[d]` online update. q is loaded
// once into a per-thread register (q[d]) since it's constant across the split.
// half2 vectorization isn't used for the dot because the per-thread layout
// already gives one multiply per thread per token (coalesced k/v reads across
// threads); the win here is occupancy + single-pass + no atomics.
//
// Partial scratch layout (f32, contiguous, allocated by the host launch):
//   partial_out: [num_seqs, num_heads, num_splits, head_dim]
//   partial_m:   [num_seqs, num_heads, num_splits]   (running max, -inf if empty)
//   partial_l:   [num_seqs, num_heads, num_splits]   (running sum of exp)
// Indexing matches the grid: split block (seq, head, s) writes row
//   base = ((seq*num_heads + head)*num_splits + s).

extern "C" __global__ void paged_attention_f16_split(
    float* partial_out,               // [num_seqs, num_heads, num_splits, head_dim]
    float* partial_m,                 // [num_seqs, num_heads, num_splits]
    float* partial_l,                 // [num_seqs, num_heads, num_splits]
    const __half* q,
    const __half* k_pool,
    const __half* v_pool,
    const int* block_tables,
    const int* kv_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_splits
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int split_idx = blockIdx.z;
    int tid = threadIdx.x;            // 0..head_dim-1, one thread per output dim

    int kv_len = kv_lens[seq_idx];
    int heads_per_group = num_heads / num_kv_heads;
    int kv_head = head_idx / heads_per_group;
    int kv_dim = num_kv_heads * head_dim;

    // This split owns KV tokens [t_start, t_end).
    int split_len = (kv_len + num_splits - 1) / num_splits;
    int t_start = split_idx * split_len;
    int t_end = t_start + split_len;
    if (t_end > kv_len) t_end = kv_len;

    long part_row = ((long)(seq_idx * num_heads + head_idx) * num_splits + split_idx);

    // `blockDim.x` is head_dim rounded up to a multiple of 32 (so warp shuffles
    // see full warps); threads with `tid >= head_dim` are padding — they join the
    // dot-product reduction with a 0 contribution but own no output element.
    bool active = (tid < head_dim);

    // Empty split (kv_len smaller than num_splits*split positions): emit identity.
    if (t_start >= t_end) {
        if (active) partial_out[part_row * head_dim + tid] = 0.0f;
        if (tid == 0) {
            partial_m[part_row] = -1e30f;
            partial_l[part_row] = 0.0f;
        }
        return;
    }

    const __half* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Per-thread registers: q[d] (constant across split) and the online accumulator.
    float q_d = active ? __half2float(q_ptr[tid]) : 0.0f;
    float acc = 0.0f;     // running Σ exp(s-m) * v[d]
    float m_run = -1e30f; // running max score
    float l_run = 0.0f;   // running Σ exp(s-m)

    // Shared scratch for the per-token block reduction of the dot product:
    // one float per warp plus a broadcast slot. `num_warps = blockDim.x / 32`.
    extern __shared__ float red[];        // size: (blockDim.x/32 + 1) floats
    int warp = tid >> 5;
    int lane = tid & 31;
    int num_warps = blockDim.x >> 5;

    for (int t = t_start; t < t_end; t++) {
        int block_idx = t / block_size;
        int slot      = t - block_idx * block_size;
        int block_id  = my_block_table[block_idx];
        const __half* k_t = k_pool
            + (long)block_id * block_size * kv_dim
            + (long)slot * kv_dim
            + kv_head * head_dim;

        // Per-thread partial product, warp-reduce, then block-reduce.
        float prod = active ? (q_d * __half2float(k_t[tid])) : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            prod += __shfl_down_sync(0xffffffff, prod, off);
        if (lane == 0) red[warp] = prod;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < num_warps) ? red[lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0) red[num_warps] = v * scale;
        }
        __syncthreads();
        float score = red[num_warps];
        __syncthreads();

        // Online-softmax update (rescale running acc when a new max appears).
        float m_new = (score > m_run) ? score : m_run;
        float alpha = __expf(m_run - m_new);
        float w = __expf(score - m_new);
        if (active) {
            const __half* v_t = v_pool
                + (long)block_id * block_size * kv_dim
                + (long)slot * kv_dim
                + kv_head * head_dim;
            acc = acc * alpha + w * __half2float(v_t[tid]);
        }
        l_run = l_run * alpha + w;
        m_run = m_new;
    }

    // Write this split's partial (unnormalized acc + m + l).
    if (active) partial_out[part_row * head_dim + tid] = acc;
    if (tid == 0) {
        partial_m[part_row] = m_run;
        partial_l[part_row] = l_run;
    }
}

// Combine kernel: merge the num_splits partials for each (seq, head) into the
// final normalized output. Grid (num_seqs, num_heads), head_dim threads (thread
// d owns out[d]). Pass 1 (single thread / shared) finds global max over the
// splits' running-max and the total rescaled denominator; pass 2 sums the
// rescaled per-split acc for dim d and divides.
extern "C" __global__ void paged_attention_f16_combine(
    __half* out,                      // [num_seqs, num_heads, head_dim]
    const float* partial_out,         // [num_seqs, num_heads, num_splits, head_dim]
    const float* partial_m,           // [num_seqs, num_heads, num_splits]
    const float* partial_l,           // [num_seqs, num_heads, num_splits]
    int num_heads,
    int head_dim,
    int num_splits
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;            // 0..head_dim-1 (block launched with exactly head_dim threads)

    long base = (long)(seq_idx * num_heads + head_idx) * num_splits;

    // Shared: global max + global denominator, computed once by thread 0.
    __shared__ float g_max;
    __shared__ float g_denom;
    if (tid == 0) {
        float gm = -1e30f;
        for (int s = 0; s < num_splits; s++) {
            float m = partial_m[base + s];
            if (m > gm) gm = m;
        }
        float denom = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float m = partial_m[base + s];
            float l = partial_l[base + s];
            if (l > 0.0f) denom += l * __expf(m - gm);
        }
        g_max = gm;
        g_denom = denom;
    }
    __syncthreads();

    float gm = g_max;
    float denom = g_denom;
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    // Each thread sums its own out dim across the rescaled splits.
    float acc = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float l = partial_l[base + s];
        if (l <= 0.0f) continue;       // empty split contributes nothing
        float m = partial_m[base + s];
        float r = __expf(m - gm);      // rescale this split's acc to global max
        acc += partial_out[(base + s) * head_dim + tid] * r;
    }

    __half* out_ptr = out + seq_idx * num_heads * head_dim + head_idx * head_dim;
    out_ptr[tid] = __float2half(acc * inv);
}

// F16 sibling of scatter_kv_f32. Pure element copy (no arithmetic), so no
// f32 intermediate is needed — copy __half words directly. See the F32 doc.
extern "C" __global__ void scatter_kv_f16(
    __half* pool,
    const __half* src,
    const int* slot_mapping,
    unsigned int n_rows,
    unsigned int kv_dim,
    unsigned int total_slots) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;
    int slot = slot_mapping[row];
    if (slot < 0 || (unsigned int)slot >= total_slots) return;
    const __half* s = src + (size_t)row * kv_dim;
    __half* d = pool + (size_t)slot * kv_dim;
    for (unsigned int i = threadIdx.x; i < kv_dim; i += blockDim.x)
        d[i] = s[i];
}
"#;
