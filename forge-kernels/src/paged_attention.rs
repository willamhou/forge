//! Paged decode attention CUDA kernels.
//!
//! Same two-pass algorithm as `decode_attention`, but reads KV data from
//! a block pool via block table indirection instead of contiguous per-seq
//! pointers. This eliminates the need to gather KV into contiguous tensors
//! during decode.
//!
//! Grid: (num_seqs, num_heads, 1)
//! Block: (THREADS_PER_BLOCK, 1, 1) -- e.g. 128
//! Each thread block handles one (seq_idx, head_idx) pair.
//!
//! Shared memory: blockDim.x * sizeof(float) + head_dim * sizeof(float)
//!   [0..blockDim.x]: scratch for reductions
//!   [blockDim.x..blockDim.x+head_dim]: output accumulator

pub const F32_SRC: &str = r#"
extern "C" __global__ void paged_decode_attention_f32(
    float* out,
    const float* q,
    const float* k_pool,
    const float* v_pool,
    const int* block_tables,
    const int* kv_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int kv_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int kv_len = kv_lens[seq_idx];
    int heads_per_group = num_heads / num_kv_heads;
    int kv_head = head_idx / heads_per_group;

    const float* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    extern __shared__ float smem[];
    float* scratch = smem;
    float* s_out = smem + blockDim.x;

    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    __syncthreads();

    // Pass 1: Compute Q*K scores, find local max
    float local_max = -__FLT_MAX__;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        int block_idx = t / block_size;
        int slot = t % block_size;
        int physical_block = seq_block_table[block_idx];
        int k_offset = (physical_block * block_size + slot) * kv_dim + kv_head * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += q_ptr[d] * k_pool[k_offset + d];
        score *= scale;
        if (score > local_max) local_max = score;
    }

    // Block reduce to find global max
    scratch[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid])
            scratch[tid] = scratch[tid + s];
        __syncthreads();
    }
    float global_max = scratch[0];
    __syncthreads();

    // Pass 2: Compute softmax weights and accumulate weighted V
    float local_sum = 0.0f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        int block_idx = t / block_size;
        int slot = t % block_size;
        int physical_block = seq_block_table[block_idx];
        int k_offset = (physical_block * block_size + slot) * kv_dim + kv_head * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += q_ptr[d] * k_pool[k_offset + d];
        score *= scale;
        float w = expf(score - global_max);
        local_sum += w;

        int v_offset = (physical_block * block_size + slot) * kv_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&s_out[d], w * v_pool[v_offset + d]);
    }
    __syncthreads();

    // Reduce sum across threads
    scratch[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float total_sum = scratch[0];
    __syncthreads();

    // Normalize and write output
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    float* out_ptr = out + seq_idx * num_heads * head_dim + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x)
        out_ptr[d] = s_out[d] * inv_sum;
}
"#;

pub const F16_SRC: &str = r#"
extern "C" __global__ void paged_decode_attention_f16(
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
    int max_blocks_per_seq,
    int kv_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int kv_len = kv_lens[seq_idx];
    int heads_per_group = num_heads / num_kv_heads;
    int kv_head = head_idx / heads_per_group;

    const __half* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    extern __shared__ float smem[];
    float* scratch = smem;
    float* s_out = smem + blockDim.x;

    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    __syncthreads();

    float local_max = -__FLT_MAX__;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        int block_idx = t / block_size;
        int slot = t % block_size;
        int physical_block = seq_block_table[block_idx];
        int k_offset = (physical_block * block_size + slot) * kv_dim + kv_head * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += __half2float(q_ptr[d]) * __half2float(k_pool[k_offset + d]);
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
        int slot = t % block_size;
        int physical_block = seq_block_table[block_idx];
        int k_offset = (physical_block * block_size + slot) * kv_dim + kv_head * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += __half2float(q_ptr[d]) * __half2float(k_pool[k_offset + d]);
        score *= scale;
        float w = expf(score - global_max);
        local_sum += w;

        int v_offset = (physical_block * block_size + slot) * kv_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&s_out[d], w * __half2float(v_pool[v_offset + d]));
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
"#;
