//! Batched decode attention CUDA kernels.
//!
//! Grid: (num_seqs, num_heads, 1)
//! Block: (THREADS_PER_BLOCK, 1, 1) -- e.g. 128
//! Each thread block handles one (seq_idx, head_idx) pair.
//!
//! Two-pass algorithm:
//!   Pass 1: Compute Q*K scores, find global max via block reduction
//!   Pass 2: Compute exp(score - max), accumulate weighted V, reduce sum
//!
//! Shared memory: blockDim.x * sizeof(float) + head_dim * sizeof(float)
//!   [0..blockDim.x]: scratch for reductions
//!   [blockDim.x..blockDim.x+head_dim]: output accumulator

pub const F32_SRC: &str = r#"
extern "C" __global__ void batched_decode_attention_f32(
    float* out,
    const float* q,
    const float* const* k_ptrs,
    const float* const* v_ptrs,
    const int* kv_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int kv_len = kv_lens[seq_idx];
    int heads_per_group = num_heads / num_kv_heads;
    int kv_head = head_idx / heads_per_group;

    const float* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const float* k_cache = k_ptrs[seq_idx];
    const float* v_cache = v_ptrs[seq_idx];

    extern __shared__ float smem[];
    float* scratch = smem;
    float* s_out = smem + blockDim.x;

    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    __syncthreads();

    // Pass 1: Compute Q*K scores, find local max
    float local_max = -1e30f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        const float* k_t = k_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += q_ptr[d] * k_t[d];
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
        const float* k_t = k_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += q_ptr[d] * k_t[d];
        score *= scale;
        float w = expf(score - global_max);
        local_sum += w;

        const float* v_t = v_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&s_out[d], w * v_t[d]);
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
extern "C" __global__ void batched_decode_attention_f16(
    __half* out,
    const __half* q,
    const __half* const* k_ptrs,
    const __half* const* v_ptrs,
    const int* kv_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int kv_len = kv_lens[seq_idx];
    int heads_per_group = num_heads / num_kv_heads;
    int kv_head = head_idx / heads_per_group;

    const __half* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const __half* k_cache = k_ptrs[seq_idx];
    const __half* v_cache = v_ptrs[seq_idx];

    extern __shared__ float smem[];
    float* scratch = smem;
    float* s_out = smem + blockDim.x;

    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    __syncthreads();

    float local_max = -1e30f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        const __half* k_t = k_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
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
        const __half* k_t = k_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += __half2float(q_ptr[d]) * __half2float(k_t[d]);
        score *= scale;
        float w = expf(score - global_max);
        local_sum += w;

        const __half* v_t = v_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
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
"#;
