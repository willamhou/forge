//! Element-wise CUDA kernels: add, mul, mul_scalar, silu.

pub const F32_SRC: &str = r#"
extern "C" __global__ void add_f32(float* out, const float* a, const float* b, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void mul_f32(float* out, const float* a, const float* b, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

extern "C" __global__ void mul_scalar_f32(float* out, const float* a, float scalar, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * scalar;
    }
}

extern "C" __global__ void silu_f32(float* out, const float* a, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void fused_silu_mul_f32(
    float* out, const float* gate, const float* up, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
"#;

pub const F16_SRC: &str = r#"
extern "C" __global__ void add_f16(__half* out, const __half* a, const __half* b, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hadd(a[i], b[i]);
    }
}

extern "C" __global__ void mul_f16(__half* out, const __half* a, const __half* b, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hmul(a[i], b[i]);
    }
}

extern "C" __global__ void mul_scalar_f16(__half* out, const __half* a, float scalar, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(__half2float(a[i]) * scalar);
    }
}

extern "C" __global__ void silu_f16(__half* out, const __half* a, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __half2float(a[i]);
        out[i] = __float2half(x / (1.0f + expf(-x)));
    }
}

extern "C" __global__ void fused_silu_mul_f16(
    __half* out, const __half* gate, const __half* up, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        out[i] = __float2half((g / (1.0f + expf(-g))) * __half2float(up[i]));
    }
}
"#;
