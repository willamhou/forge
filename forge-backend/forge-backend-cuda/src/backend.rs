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
}

#[allow(dead_code)]
pub struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) blas: CudaBlas,
    kernels: KernelFunctions,
    _module: Arc<CudaModule>,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx =
            CudaContext::new(ordinal).map_err(|e| ForgeError::Cuda(format!("context: {e}")))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| ForgeError::Cuda(format!("cublas: {e}")))?;

        let ptx = compile_ptx(ELEMENTWISE_PTX_SRC)
            .map_err(|e| ForgeError::Cuda(format!("nvrtc: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| ForgeError::Cuda(format!("module load: {e}")))?;

        let load = |name: &str| -> Result<CudaFunction> {
            module
                .load_function(name)
                .map_err(|e| ForgeError::Cuda(format!("load {name}: {e}")))
        };

        let kernels = KernelFunctions {
            add_f32: load("add_f32")?,
            mul_f32: load("mul_f32")?,
            mul_scalar_f32: load("mul_scalar_f32")?,
            silu_f32: load("silu_f32")?,
            rms_norm_f32: load("rms_norm_f32")?,
            softmax_f32: load("softmax_f32")?,
            embedding_f32: load("embedding_f32")?,
            rope_f32: load("rope_f32")?,
            transpose_f32: load("transpose_f32")?,
        };

        Ok(Self {
            ctx,
            stream,
            blas,
            kernels,
            _module: module,
        })
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
        let slice = tensor.f32_slice()?;
        self.stream
            .memcpy_dtov(slice)
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    fn matmul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        // A: [M, K] row-major, B: [K, N] row-major -> C: [M, N]
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

        let mut c = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        // cuBLAS is column-major. For row-major C = A * B:
        // We compute C^T = B^T * A^T, which in col-major is: C = B * A
        // with m=N, n=M, k=K, lda=N, ldb=K, ldc=N
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
                .map_err(|e| ForgeError::Cuda(format!("gemm: {e}")))?;
        }

        Ok(CudaTensor::f32_data(c, vec![m, n]))
    }

    fn add(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        validate_same_len(a, b)?;
        let n = a.len() as u32;
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

    fn mul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        validate_same_len(a, b)?;
        let n = a.len() as u32;
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

    fn mul_scalar(&self, a: &CudaTensor, scalar: f32) -> Result<CudaTensor> {
        let n = a.len() as u32;
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

    fn silu(&self, a: &CudaTensor) -> Result<CudaTensor> {
        let n = a.len() as u32;
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

        let mut out = self
            .stream
            .alloc_zeros::<f32>(rows * cols)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        let block_dim = next_power_of_2(256u32.min(cols as u32));
        let shared_mem = block_dim * 4; // sizeof(f32) per thread

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

        let mut out = self
            .stream
            .alloc_zeros::<f32>(rows * cols)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        let block_dim = next_power_of_2(256u32.min(cols as u32));
        let shared_mem = block_dim * 4;

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

        // Validate indices are in range (CPU-side check)
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

        let mut out = self
            .stream
            .alloc_zeros::<f32>(rows * cols)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
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

        // Validate all non-concat dimensions match element-wise
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

        let mut out_shape = tensors[0].shape().to_vec();
        out_shape[0] = total_first_dim;

        Ok(CudaTensor::f32_data(out, out_shape))
    }
}
