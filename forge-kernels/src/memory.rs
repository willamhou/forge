//! Memory/data CUDA kernels: transpose, cast.

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
