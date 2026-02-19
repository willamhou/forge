use crate::tensor::Tensor;
use crate::{DType, Result};

pub trait Backend: Send + Sync + 'static {
    type Tensor: Tensor;

    fn name(&self) -> &str;
    fn device_count(&self) -> usize;

    // Allocation
    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;
    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;

    // Data transfer
    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_to_host_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>>;

    // Synchronization
    fn synchronize(&self) -> Result<()>;

    // Core ops
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn mul_scalar(&self, a: &Self::Tensor, scalar: f32) -> Result<Self::Tensor>;
    fn silu(&self, a: &Self::Tensor) -> Result<Self::Tensor>;
    fn rms_norm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor>;
    fn rope(
        &self,
        x: &Self::Tensor,
        freqs_cos: &Self::Tensor,
        freqs_sin: &Self::Tensor,
    ) -> Result<Self::Tensor>;
    fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor>;
    fn embedding(&self, weight: &Self::Tensor, indices: &[u32]) -> Result<Self::Tensor>;
    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor>;
    fn transpose(&self, x: &Self::Tensor, dim0: usize, dim1: usize) -> Result<Self::Tensor>;
    fn cat(&self, tensors: &[&Self::Tensor], dim: usize) -> Result<Self::Tensor>;
}
