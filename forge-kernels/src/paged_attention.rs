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
"#;
