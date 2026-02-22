//! Positional CUDA kernels: rope, embedding.

pub const F32_SRC: &str = r#"
extern "C" __global__ void embedding_f32(
    float* out,
    const float* weight,
    const unsigned int* indices,
    unsigned int num_indices,
    unsigned int embedding_dim,
    unsigned int vocab_size
) {
    unsigned int idx = blockIdx.x;
    if (idx >= num_indices) return;

    unsigned int token_id = indices[idx];
    if (token_id >= vocab_size) return;

    const float* src = weight + token_id * embedding_dim;
    float* dst = out + idx * embedding_dim;

    for (unsigned int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void rope_f32(
    float* out,
    const float* input,
    const float* cos_freqs,
    const float* sin_freqs,
    unsigned int batch,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_dim = head_dim / 2;
    unsigned int total = batch * seq_len * num_heads * half_dim;
    if (tid >= total) return;

    unsigned int h = tid % half_dim;
    unsigned int rem = tid / half_dim;
    unsigned int head = rem % num_heads;
    rem = rem / num_heads;
    unsigned int pos = rem % seq_len;

    unsigned int base = (rem / seq_len) * seq_len * num_heads * head_dim
                      + pos * num_heads * head_dim
                      + head * head_dim;

    float x0 = input[base + h];
    float x1 = input[base + h + half_dim];
    float cos_val = cos_freqs[pos * half_dim + h];
    float sin_val = sin_freqs[pos * half_dim + h];

    out[base + h] = x0 * cos_val - x1 * sin_val;
    out[base + h + half_dim] = x0 * sin_val + x1 * cos_val;
}
"#;

pub const F16_SRC: &str = r#"
extern "C" __global__ void embedding_f16(
    __half* out,
    const __half* weight,
    const unsigned int* indices,
    unsigned int num_indices,
    unsigned int embedding_dim,
    unsigned int vocab_size
) {
    unsigned int idx = blockIdx.x;
    if (idx >= num_indices) return;

    unsigned int token_id = indices[idx];
    if (token_id >= vocab_size) return;

    const __half* src = weight + token_id * embedding_dim;
    __half* dst = out + idx * embedding_dim;

    for (unsigned int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void rope_f16(
    __half* out,
    const __half* input,
    const float* cos_freqs,
    const float* sin_freqs,
    unsigned int batch,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_dim = head_dim / 2;
    unsigned int total = batch * seq_len * num_heads * half_dim;
    if (tid >= total) return;

    unsigned int h = tid % half_dim;
    unsigned int rem = tid / half_dim;
    unsigned int head = rem % num_heads;
    rem = rem / num_heads;
    unsigned int pos = rem % seq_len;

    unsigned int base = (rem / seq_len) * seq_len * num_heads * head_dim
                      + pos * num_heads * head_dim
                      + head * head_dim;

    float x0 = __half2float(input[base + h]);
    float x1 = __half2float(input[base + h + half_dim]);
    float cos_val = cos_freqs[pos * half_dim + h];
    float sin_val = sin_freqs[pos * half_dim + h];

    out[base + h] = __float2half(x0 * cos_val - x1 * sin_val);
    out[base + h + half_dim] = __float2half(x0 * sin_val + x1 * cos_val);
}
"#;
