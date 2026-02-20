use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use forge_core::{Backend, DType, ForgeError, Result, Tensor};

use crate::tensor::CudaTensor;

const ELEMENTWISE_PTX_SRC: &str = r#"
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

const F16_PTX_SRC: &str = r#"
#include <cuda_fp16.h>

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

struct KernelFunctions {
    add_f32: CudaFunction,
    mul_f32: CudaFunction,
    mul_scalar_f32: CudaFunction,
    silu_f32: CudaFunction,
    rms_norm_f32: CudaFunction,
    softmax_f32: CudaFunction,
    embedding_f32: CudaFunction,
    rope_f32: CudaFunction,
    transpose_f32: CudaFunction,
    // FP16 variants
    add_f16: CudaFunction,
    mul_f16: CudaFunction,
    mul_scalar_f16: CudaFunction,
    silu_f16: CudaFunction,
    rms_norm_f16: CudaFunction,
    softmax_f16: CudaFunction,
    embedding_f16: CudaFunction,
    rope_f16: CudaFunction,
    transpose_f16: CudaFunction,
    cast_f16_to_f32: CudaFunction,
    cast_f32_to_f16: CudaFunction,
}

// CudaBackend is Clone for sharing with components like NaiveKvCache,
// but cuBLAS/kernel calls are NOT thread-safe for concurrent use.
// The engine must ensure single-threaded access to the backend.
#[allow(dead_code)]
#[derive(Clone)]
pub struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) blas: Arc<CudaBlas>,
    kernels: Arc<KernelFunctions>,
    _module_f32: Arc<CudaModule>,
    _module_f16: Arc<CudaModule>,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx =
            CudaContext::new(ordinal).map_err(|e| ForgeError::Cuda(format!("context: {e}")))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| ForgeError::Cuda(format!("cublas: {e}")))?;

        // Compile F32 kernels
        let ptx_f32 = compile_ptx(ELEMENTWISE_PTX_SRC)
            .map_err(|e| ForgeError::Cuda(format!("nvrtc f32: {e}")))?;
        let module_f32 = ctx
            .load_module(ptx_f32)
            .map_err(|e| ForgeError::Cuda(format!("module load f32: {e}")))?;

        // Compile F16 kernels — requires cuda_fp16.h from CUDA toolkit
        let cuda_include = Self::find_cuda_include()?;
        let ptx_f16 = cudarc::nvrtc::compile_ptx_with_opts(
            F16_PTX_SRC,
            cudarc::nvrtc::CompileOptions {
                use_fast_math: Some(true),
                include_paths: vec![cuda_include],
                ..Default::default()
            },
        )
        .map_err(|e| ForgeError::Cuda(format!("nvrtc f16: {e}")))?;
        let module_f16 = ctx
            .load_module(ptx_f16)
            .map_err(|e| ForgeError::Cuda(format!("module load f16: {e}")))?;

        let load_f32 = |name: &str| -> Result<CudaFunction> {
            module_f32
                .load_function(name)
                .map_err(|e| ForgeError::Cuda(format!("load {name}: {e}")))
        };
        let load_f16 = |name: &str| -> Result<CudaFunction> {
            module_f16
                .load_function(name)
                .map_err(|e| ForgeError::Cuda(format!("load {name}: {e}")))
        };

        let kernels = KernelFunctions {
            add_f32: load_f32("add_f32")?,
            mul_f32: load_f32("mul_f32")?,
            mul_scalar_f32: load_f32("mul_scalar_f32")?,
            silu_f32: load_f32("silu_f32")?,
            rms_norm_f32: load_f32("rms_norm_f32")?,
            softmax_f32: load_f32("softmax_f32")?,
            embedding_f32: load_f32("embedding_f32")?,
            rope_f32: load_f32("rope_f32")?,
            transpose_f32: load_f32("transpose_f32")?,
            // F16 kernels
            add_f16: load_f16("add_f16")?,
            mul_f16: load_f16("mul_f16")?,
            mul_scalar_f16: load_f16("mul_scalar_f16")?,
            silu_f16: load_f16("silu_f16")?,
            rms_norm_f16: load_f16("rms_norm_f16")?,
            softmax_f16: load_f16("softmax_f16")?,
            embedding_f16: load_f16("embedding_f16")?,
            rope_f16: load_f16("rope_f16")?,
            transpose_f16: load_f16("transpose_f16")?,
            cast_f16_to_f32: load_f16("cast_f16_to_f32")?,
            cast_f32_to_f16: load_f16("cast_f32_to_f16")?,
        };

        Ok(Self {
            ctx,
            stream,
            blas: Arc::new(blas),
            kernels: Arc::new(kernels),
            _module_f32: module_f32,
            _module_f16: module_f16,
        })
    }

    /// Locate the CUDA toolkit include directory containing `cuda_fp16.h`.
    ///
    /// Search order: `$CUDA_HOME/include`, `$CUDA_PATH/include`, `/usr/local/cuda/include`.
    fn find_cuda_include() -> Result<String> {
        let candidates: Vec<std::path::PathBuf> = std::env::var("CUDA_HOME")
            .into_iter()
            .chain(std::env::var("CUDA_PATH"))
            .map(|p| std::path::PathBuf::from(p).join("include"))
            .chain(std::iter::once(std::path::PathBuf::from(
                "/usr/local/cuda/include",
            )))
            .collect();

        for path in &candidates {
            if path.join("cuda_fp16.h").exists() {
                return Ok(path.to_string_lossy().into_owned());
            }
        }

        Err(ForgeError::Cuda(format!(
            "cuda_fp16.h not found; searched: {}. Set CUDA_HOME to your CUDA toolkit.",
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }
}

/// Round up to the next power of 2 (minimum 32 for warp size).
fn next_power_of_2(n: u32) -> u32 {
    let n = n.max(32);
    1u32 << (32 - (n - 1).leading_zeros())
}

fn validate_shape(data_len: usize, shape: &[usize]) -> Result<()> {
    let expected: usize = shape.iter().product();
    if data_len != expected {
        return Err(ForgeError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![data_len],
        });
    }
    Ok(())
}

fn validate_same_len(a: &CudaTensor, b: &CudaTensor) -> Result<()> {
    if a.len() != b.len() {
        return Err(ForgeError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

impl CudaBackend {
    /// Cast an F16 tensor to F32 on the GPU, returning a host Vec<f32>.
    fn cast_f16_to_f32_host(&self, tensor: &CudaTensor) -> Result<Vec<f32>> {
        let n = tensor.len() as u32;
        let mut out = self
            .stream
            .alloc_zeros::<f32>(n as usize)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        let mut builder = self.stream.launch_builder(&self.kernels.cast_f16_to_f32);
        builder.arg(&mut out);
        builder.arg(tensor.f16_slice()?);
        builder.arg(&n);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n))
                .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        }

        self.stream
            .memcpy_dtov(&out)
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }
}

impl Backend for CudaBackend {
    type Tensor = CudaTensor;

    fn name(&self) -> &str {
        "cuda"
    }

    fn device_count(&self) -> usize {
        CudaContext::device_count().unwrap_or(0) as usize
    }

    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<CudaTensor> {
        let numel = CudaTensor::numel_from_shape(shape);
        match dtype {
            DType::F32 => {
                let data = self
                    .stream
                    .alloc_zeros::<f32>(numel)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(CudaTensor::f32_data(data, shape.to_vec()))
            }
            DType::F16 => {
                let data = self
                    .stream
                    .alloc_zeros::<half::f16>(numel)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(CudaTensor::f16_data(data, shape.to_vec()))
            }
            DType::BF16 => {
                let data = self
                    .stream
                    .alloc_zeros::<half::bf16>(numel)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(CudaTensor::bf16_data(data, shape.to_vec()))
            }
            _ => Err(ForgeError::UnsupportedDtype(dtype)),
        }
    }

    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<CudaTensor> {
        self.allocate(shape, dtype)
    }

    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<CudaTensor> {
        validate_shape(data.len(), shape)?;
        let slice = self
            .stream
            .memcpy_stod(data)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        Ok(CudaTensor::f32_data(slice, shape.to_vec()))
    }

    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<CudaTensor> {
        validate_shape(data.len(), shape)?;
        let slice = self
            .stream
            .memcpy_stod(data)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        Ok(CudaTensor::f16_data(slice, shape.to_vec()))
    }

    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<CudaTensor> {
        validate_shape(data.len(), shape)?;
        let slice = self
            .stream
            .memcpy_stod(data)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        Ok(CudaTensor::bf16_data(slice, shape.to_vec()))
    }

    fn copy_to_host_f32(&self, tensor: &CudaTensor) -> Result<Vec<f32>> {
        match tensor.dtype() {
            DType::F32 => self
                .stream
                .memcpy_dtov(tensor.f32_slice()?)
                .map_err(|e| ForgeError::Cuda(e.to_string())),
            DType::F16 => self.cast_f16_to_f32_host(tensor),
            other => Err(ForgeError::InvalidArgument(format!(
                "copy_to_host_f32 not supported for {:?}",
                other
            ))),
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    fn matmul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "matmul requires 2D tensors".into(),
            ));
        }
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        if b_shape[0] != k {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![k, n],
                got: b_shape.to_vec(),
            });
        }

        if a.dtype() != b.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "matmul dtype mismatch: {:?} vs {:?}",
                a.dtype(),
                b.dtype()
            )));
        }

        match a.dtype() {
            DType::F32 => {
                let mut c = self
                    .stream
                    .alloc_zeros::<f32>(m * n)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                unsafe {
                    self.blas
                        .gemm(
                            GemmConfig {
                                transa: cublasOperation_t::CUBLAS_OP_N,
                                transb: cublasOperation_t::CUBLAS_OP_N,
                                m: n as i32,
                                n: m as i32,
                                k: k as i32,
                                alpha: 1.0f32,
                                lda: n as i32,
                                ldb: k as i32,
                                beta: 0.0f32,
                                ldc: n as i32,
                            },
                            b.f32_slice()?,
                            a.f32_slice()?,
                            &mut c,
                        )
                        .map_err(|e| ForgeError::Cuda(format!("gemm f32: {e}")))?;
                }

                Ok(CudaTensor::f32_data(c, vec![m, n]))
            }
            DType::F16 => {
                let mut c = self
                    .stream
                    .alloc_zeros::<half::f16>(m * n)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                // cudarc safe Gemm<half::f16> uses cublasGemmEx with F32 accumulation
                unsafe {
                    self.blas
                        .gemm(
                            GemmConfig {
                                transa: cublasOperation_t::CUBLAS_OP_N,
                                transb: cublasOperation_t::CUBLAS_OP_N,
                                m: n as i32,
                                n: m as i32,
                                k: k as i32,
                                alpha: half::f16::from_f32(1.0),
                                lda: n as i32,
                                ldb: k as i32,
                                beta: half::f16::from_f32(0.0),
                                ldc: n as i32,
                            },
                            b.f16_slice()?,
                            a.f16_slice()?,
                            &mut c,
                        )
                        .map_err(|e| ForgeError::Cuda(format!("gemm f16: {e}")))?;
                }

                Ok(CudaTensor::f16_data(c, vec![m, n]))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn add(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        validate_same_len(a, b)?;
        let n = a.len() as u32;

        match a.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.add_f16);
                builder.arg(&mut out);
                builder.arg(a.f16_slice()?);
                builder.arg(b.f16_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, a.shape.clone()))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.add_f32);
                builder.arg(&mut out);
                builder.arg(a.f32_slice()?);
                builder.arg(b.f32_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, a.shape.clone()))
            }
        }
    }

    fn mul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        validate_same_len(a, b)?;
        let n = a.len() as u32;

        match a.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.mul_f16);
                builder.arg(&mut out);
                builder.arg(a.f16_slice()?);
                builder.arg(b.f16_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, a.shape.clone()))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.mul_f32);
                builder.arg(&mut out);
                builder.arg(a.f32_slice()?);
                builder.arg(b.f32_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, a.shape.clone()))
            }
        }
    }

    fn mul_scalar(&self, a: &CudaTensor, scalar: f32) -> Result<CudaTensor> {
        let n = a.len() as u32;

        match a.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.mul_scalar_f16);
                builder.arg(&mut out);
                builder.arg(a.f16_slice()?);
                builder.arg(&scalar);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, a.shape.clone()))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.mul_scalar_f32);
                builder.arg(&mut out);
                builder.arg(a.f32_slice()?);
                builder.arg(&scalar);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, a.shape.clone()))
            }
        }
    }

    fn silu(&self, a: &CudaTensor) -> Result<CudaTensor> {
        let n = a.len() as u32;

        match a.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.silu_f16);
                builder.arg(&mut out);
                builder.arg(a.f16_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, a.shape.clone()))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.silu_f32);
                builder.arg(&mut out);
                builder.arg(a.f32_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, a.shape.clone()))
            }
        }
    }

    fn rms_norm(&self, x: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        let shape = x.shape();
        let cols = *shape.last().unwrap();
        if weight.len() != cols {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![cols],
                got: weight.shape().to_vec(),
            });
        }
        let rows = x.len() / cols;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        let block_dim = next_power_of_2(256u32.min(cols as u32));
        // Shared memory uses f32 for both F16 and F32 paths (reduction in f32)
        let shared_mem = block_dim * 4;

        match x.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.rms_norm_f16);
                builder.arg(&mut out);
                builder.arg(x.f16_slice()?);
                builder.arg(weight.f16_slice()?);
                builder.arg(&eps);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, shape.to_vec()))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.rms_norm_f32);
                builder.arg(&mut out);
                builder.arg(x.f32_slice()?);
                builder.arg(weight.f32_slice()?);
                builder.arg(&eps);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, shape.to_vec()))
            }
        }
    }

    fn rope(
        &self,
        x: &CudaTensor,
        freqs_cos: &CudaTensor,
        freqs_sin: &CudaTensor,
    ) -> Result<CudaTensor> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(ForgeError::InvalidArgument(
                "rope expects 4D tensor [batch, seq_len, heads, head_dim]".into(),
            ));
        }
        let batch = shape[0] as u32;
        let seq_len = shape[1] as u32;
        let num_heads = shape[2] as u32;
        let head_dim = shape[3] as u32;
        if head_dim % 2 != 0 {
            return Err(ForgeError::InvalidArgument(
                "rope requires even head_dim".into(),
            ));
        }
        let half_dim = head_dim / 2;
        let expected_freq_len = (seq_len * half_dim) as usize;
        if freqs_cos.len() < expected_freq_len || freqs_sin.len() < expected_freq_len {
            return Err(ForgeError::InvalidArgument(format!(
                "freq tensors need at least {} elements, got cos={} sin={}",
                expected_freq_len,
                freqs_cos.len(),
                freqs_sin.len()
            )));
        }
        let total = batch * seq_len * num_heads * half_dim;

        match x.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(x.len())
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                // RoPE F16 kernel reads input as __half, freqs as float (f32)
                let mut builder = self.stream.launch_builder(&self.kernels.rope_f16);
                builder.arg(&mut out);
                builder.arg(x.f16_slice()?);
                builder.arg(freqs_cos.f32_slice()?);
                builder.arg(freqs_sin.f32_slice()?);
                builder.arg(&batch);
                builder.arg(&seq_len);
                builder.arg(&num_heads);
                builder.arg(&head_dim);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, shape.to_vec()))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(x.len())
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.rope_f32);
                builder.arg(&mut out);
                builder.arg(x.f32_slice()?);
                builder.arg(freqs_cos.f32_slice()?);
                builder.arg(freqs_sin.f32_slice()?);
                builder.arg(&batch);
                builder.arg(&seq_len);
                builder.arg(&num_heads);
                builder.arg(&head_dim);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, shape.to_vec()))
            }
        }
    }

    fn softmax(&self, x: &CudaTensor, dim: i32) -> Result<CudaTensor> {
        let shape = x.shape();
        let ndim = shape.len() as i32;
        let normalized_dim = if dim < 0 { ndim + dim } else { dim };
        if normalized_dim != ndim - 1 {
            return Err(ForgeError::InvalidArgument(format!(
                "softmax only supports last dimension (got dim={dim}, ndim={ndim})"
            )));
        }

        let cols = *shape.last().unwrap();
        let rows = x.len() / cols;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        let block_dim = next_power_of_2(256u32.min(cols as u32));
        let shared_mem = block_dim * 4;

        match x.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.softmax_f16);
                builder.arg(&mut out);
                builder.arg(x.f16_slice()?);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, shape.to_vec()))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.softmax_f32);
                builder.arg(&mut out);
                builder.arg(x.f32_slice()?);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, shape.to_vec()))
            }
        }
    }

    fn embedding(&self, weight: &CudaTensor, indices: &[u32]) -> Result<CudaTensor> {
        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "embedding weight must be 2D [vocab_size, embedding_dim]".into(),
            ));
        }
        let vocab_size = weight_shape[0];
        let embedding_dim = weight_shape[1];
        let num_indices = indices.len();

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(ForgeError::InvalidArgument(format!(
                    "embedding index {idx} out of range (vocab_size={vocab_size})"
                )));
            }
        }

        let num_indices_u32 = num_indices as u32;
        let embedding_dim_u32 = embedding_dim as u32;
        let vocab_size_u32 = vocab_size as u32;

        let indices_dev = self
            .stream
            .memcpy_stod(indices)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        match weight.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(num_indices * embedding_dim)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.embedding_f16);
                builder.arg(&mut out);
                builder.arg(weight.f16_slice()?);
                builder.arg(&indices_dev);
                builder.arg(&num_indices_u32);
                builder.arg(&embedding_dim_u32);
                builder.arg(&vocab_size_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (num_indices as u32, 1, 1),
                            block_dim: (256.min(embedding_dim as u32), 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, vec![num_indices, embedding_dim]))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(num_indices * embedding_dim)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.embedding_f32);
                builder.arg(&mut out);
                builder.arg(weight.f32_slice()?);
                builder.arg(&indices_dev);
                builder.arg(&num_indices_u32);
                builder.arg(&embedding_dim_u32);
                builder.arg(&vocab_size_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (num_indices as u32, 1, 1),
                            block_dim: (256.min(embedding_dim as u32), 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, vec![num_indices, embedding_dim]))
            }
        }
    }

    fn reshape(&self, x: &CudaTensor, shape: &[usize]) -> Result<CudaTensor> {
        let numel: usize = shape.iter().product();
        if numel != x.len() {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: x.shape.clone(),
            });
        }
        Ok(CudaTensor {
            data: x.data.clone(),
            shape: shape.to_vec(),
            dtype: x.dtype,
        })
    }

    fn transpose(&self, x: &CudaTensor, dim0: usize, dim1: usize) -> Result<CudaTensor> {
        let shape = x.shape();
        if shape.len() != 2 || !((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0)) {
            return Err(ForgeError::InvalidArgument(
                "transpose currently only supports 2D tensors with dims (0,1)".into(),
            ));
        }
        let rows = shape[0];
        let cols = shape[1];
        let n = (rows * cols) as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        match x.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.transpose_f16);
                builder.arg(&mut out);
                builder.arg(x.f16_slice()?);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, vec![cols, rows]))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.transpose_f32);
                builder.arg(&mut out);
                builder.arg(x.f32_slice()?);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, vec![cols, rows]))
            }
        }
    }

    fn cat(&self, tensors: &[&CudaTensor], dim: usize) -> Result<CudaTensor> {
        if tensors.is_empty() {
            return Err(ForgeError::InvalidArgument("empty tensor list".into()));
        }
        if dim != 0 {
            return Err(ForgeError::InvalidArgument(
                "cat currently only supports dim=0".into(),
            ));
        }

        let ndim = tensors[0].shape().len();
        let inner_size: usize = if ndim > 1 {
            tensors[0].shape()[1..].iter().product()
        } else {
            1
        };

        for t in tensors.iter().skip(1) {
            if t.shape().len() != ndim {
                return Err(ForgeError::ShapeMismatch {
                    expected: tensors[0].shape().to_vec(),
                    got: t.shape().to_vec(),
                });
            }
            for d in 1..ndim {
                if t.shape()[d] != tensors[0].shape()[d] {
                    return Err(ForgeError::ShapeMismatch {
                        expected: tensors[0].shape().to_vec(),
                        got: t.shape().to_vec(),
                    });
                }
            }
        }

        let mut total_first_dim = 0usize;
        for t in tensors {
            total_first_dim += t.shape()[0];
        }
        let total_elems = total_first_dim * inner_size;

        let mut out_shape = tensors[0].shape().to_vec();
        out_shape[0] = total_first_dim;

        match tensors[0].dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(total_elems)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut offset = 0usize;
                for t in tensors {
                    let len = t.len();
                    let src = t.f16_slice()?;
                    self.stream
                        .memcpy_dtod(src, &mut out.slice_mut(offset..offset + len))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    offset += len;
                }

                Ok(CudaTensor::f16_data(out, out_shape))
            }
            _ => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(total_elems)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut offset = 0usize;
                for t in tensors {
                    let len = t.len();
                    let src = t.f32_slice()?;
                    self.stream
                        .memcpy_dtod(src, &mut out.slice_mut(offset..offset + len))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    offset += len;
                }

                Ok(CudaTensor::f32_data(out, out_shape))
            }
        }
    }

    fn cast(&self, x: &CudaTensor, dtype: DType) -> Result<CudaTensor> {
        if x.dtype() == dtype {
            return Ok(x.clone());
        }
        let n = x.len() as u32;
        let shape = x.shape().to_vec();

        match (x.dtype(), dtype) {
            (DType::F16, DType::F32) => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.cast_f16_to_f32);
                builder.arg(&mut out);
                builder.arg(x.f16_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, shape))
            }
            (DType::F32, DType::F16) => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.cast_f32_to_f16);
                builder.arg(&mut out);
                builder.arg(x.f32_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, shape))
            }
            (from, to) => Err(ForgeError::InvalidArgument(format!(
                "cast from {:?} to {:?} not supported",
                from, to
            ))),
        }
    }
}
