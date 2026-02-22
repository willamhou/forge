//! Memory/data CUDA kernels: transpose, cast, split_qkv.

pub const F32_SRC: &str = r#"
extern "C" __global__ void transpose_f32(
    float* out, const float* in_data,
    unsigned int rows, unsigned int cols
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    out[c * rows + r] = in_data[r * cols + c];
}

extern "C" __global__ void split_qkv_f32(
    float* q_out, float* k_out, float* v_out,
    const float* qkv, unsigned int rows,
    unsigned int q_cols, unsigned int kv_cols
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;
    if (row >= rows) return;
    unsigned int total_cols = q_cols + kv_cols + kv_cols;
    const float* src = qkv + row * total_cols;
    for (unsigned int c = col; c < q_cols; c += blockDim.x)
        q_out[row * q_cols + c] = src[c];
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        k_out[row * kv_cols + c] = src[q_cols + c];
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        v_out[row * kv_cols + c] = src[q_cols + kv_cols + c];
}
"#;

pub const F16_SRC: &str = r#"
extern "C" __global__ void transpose_f16(
    __half* out, const __half* in_data,
    unsigned int rows, unsigned int cols
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    out[c * rows + r] = in_data[r * cols + c];
}

extern "C" __global__ void split_qkv_f16(
    __half* q_out, __half* k_out, __half* v_out,
    const __half* qkv, unsigned int rows,
    unsigned int q_cols, unsigned int kv_cols
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;
    if (row >= rows) return;
    unsigned int total_cols = q_cols + kv_cols + kv_cols;
    const __half* src = qkv + row * total_cols;
    for (unsigned int c = col; c < q_cols; c += blockDim.x)
        q_out[row * q_cols + c] = src[c];
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        k_out[row * kv_cols + c] = src[q_cols + c];
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        v_out[row * kv_cols + c] = src[q_cols + kv_cols + c];
}

extern "C" __global__ void cast_f16_to_f32(
    float* out, const __half* input, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half2float(input[i]);
    }
}

extern "C" __global__ void cast_f32_to_f16(
    __half* out, const float* input, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(input[i]);
    }
}
"#;
