//! Normalization CUDA kernels: rms_norm, softmax.

pub const F32_SRC: &str = r#"
extern "C" __global__ void rms_norm_f32(
    float* out,
    const float* input,
    const float* weight,
    float eps,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input + row * cols;
    float* o = out + row * cols;

    extern __shared__ float shared[];

    float local_ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        local_ss += val * val;
    }

    shared[threadIdx.x] = local_ss;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)cols + eps);

    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        o[i] = x[i] * rms * weight[i];
    }
}

extern "C" __global__ void fused_residual_rms_norm_f32(
    float* norm_out,
    float* residual_out,
    const float* x,
    const float* residual_in,
    const float* weight,
    float eps,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const float* xi = x + row * cols;
    const float* ri = residual_in + row * cols;
    float* ro = residual_out + row * cols;
    float* no = norm_out + row * cols;

    extern __shared__ float shared[];

    float local_ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float tmp = xi[i] + ri[i];
        ro[i] = tmp;
        local_ss += tmp * tmp;
    }

    shared[threadIdx.x] = local_ss;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)cols + eps);

    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        no[i] = ro[i] * rms * weight[i];
    }
}

extern "C" __global__ void softmax_f32(
    float* out,
    const float* input,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input + row * cols;
    float* o = out + row * cols;

    extern __shared__ float shared[];

    // Find max for numerical stability
    float local_max = -1e30f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        if (val > local_max) local_max = val;
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared[threadIdx.x + s] > shared[threadIdx.x]) {
                shared[threadIdx.x] = shared[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float max_val = shared[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(x[i] - max_val);
        o[i] = val;
        local_sum += val;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum = shared[0];

    // Normalize
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        o[i] = o[i] / sum;
    }
}
"#;

pub const F16_SRC: &str = r#"
extern "C" __global__ void rms_norm_f16(
    __half* out,
    const __half* input,
    const __half* weight,
    float eps,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x = input + row * cols;
    __half* o = out + row * cols;

    extern __shared__ float shared[];

    float local_ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x[i]);
        local_ss += val * val;
    }

    shared[threadIdx.x] = local_ss;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)cols + eps);

    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        o[i] = __float2half(__half2float(x[i]) * rms * __half2float(weight[i]));
    }
}

extern "C" __global__ void fused_residual_rms_norm_f16(
    __half* norm_out,
    __half* residual_out,
    const __half* x,
    const __half* residual_in,
    const __half* weight,
    float eps,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const __half* xi = x + row * cols;
    const __half* ri = residual_in + row * cols;
    __half* ro = residual_out + row * cols;
    __half* no = norm_out + row * cols;

    extern __shared__ float shared[];

    float local_ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float tmp = __half2float(xi[i]) + __half2float(ri[i]);
        ro[i] = __float2half(tmp);
        local_ss += tmp * tmp;
    }

    shared[threadIdx.x] = local_ss;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)cols + eps);

    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        no[i] = __float2half(__half2float(ro[i]) * rms * __half2float(weight[i]));
    }
}

extern "C" __global__ void softmax_f16(
    __half* out,
    const __half* input,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x = input + row * cols;
    __half* o = out + row * cols;

    extern __shared__ float shared[];

    float local_max = -1e30f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x[i]);
        if (val > local_max) local_max = val;
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared[threadIdx.x + s] > shared[threadIdx.x]) {
                shared[threadIdx.x] = shared[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float max_val = shared[0];

    float local_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(__half2float(x[i]) - max_val);
        o[i] = __float2half(val);
        local_sum += val;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum = shared[0];

    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        o[i] = __float2half(__half2float(o[i]) / sum);
    }
}
"#;
