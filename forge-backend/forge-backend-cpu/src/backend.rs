use forge_core::{Backend, DType, ForgeError, Result, Tensor};

use crate::tensor::CpuTensor;

/// CPU backend for Forge inference engine.
///
/// All data lives in host memory as `Vec<f32>` wrapped in `Arc`.
/// Allocation and transfer ops are implemented; compute ops are stubs.
#[derive(Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
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

fn validate_same_len(a: &CpuTensor, b: &CpuTensor) -> Result<()> {
    if a.len() != b.len() {
        return Err(ForgeError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;

    fn name(&self) -> &str {
        "cpu"
    }

    fn device_count(&self) -> usize {
        1
    }

    // ── Allocation ──────────────────────────────────────────────

    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<CpuTensor> {
        match dtype {
            DType::F32 => {
                let numel: usize = shape.iter().product();
                Ok(CpuTensor::new(vec![0.0; numel], shape.to_vec()))
            }
            _ => Err(ForgeError::UnsupportedDtype(dtype)),
        }
    }

    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<CpuTensor> {
        self.allocate(shape, dtype)
    }

    // ── Data transfer ───────────────────────────────────────────

    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        Ok(CpuTensor::new(data.to_vec(), shape.to_vec()))
    }

    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        Ok(CpuTensor::new(f32_data, shape.to_vec()))
    }

    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        Ok(CpuTensor::new(f32_data, shape.to_vec()))
    }

    fn copy_to_host_f32(&self, tensor: &CpuTensor) -> Result<Vec<f32>> {
        Ok(tensor.data().to_vec())
    }

    // ── Synchronization ─────────────────────────────────────────

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    // ── Compute ops (stubs) ─────────────────────────────────────

    fn matmul(&self, _a: &CpuTensor, _b: &CpuTensor) -> Result<CpuTensor> {
        todo!("Task 3: matmul via CBLAS sgemm")
    }

    fn add(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        validate_same_len(a, b)?;
        todo!("Task 3: elementwise add")
    }

    fn mul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        validate_same_len(a, b)?;
        todo!("Task 3: elementwise mul")
    }

    fn mul_scalar(&self, _a: &CpuTensor, _scalar: f32) -> Result<CpuTensor> {
        todo!("Task 3: scalar mul")
    }

    fn silu(&self, _a: &CpuTensor) -> Result<CpuTensor> {
        todo!("Task 4: silu activation")
    }

    fn rms_norm(
        &self,
        _x: &CpuTensor,
        _weight: &CpuTensor,
        _eps: f32,
    ) -> Result<CpuTensor> {
        todo!("Task 4: rms_norm")
    }

    fn rope(
        &self,
        _x: &CpuTensor,
        _freqs_cos: &CpuTensor,
        _freqs_sin: &CpuTensor,
    ) -> Result<CpuTensor> {
        todo!("Task 4: rope")
    }

    fn softmax(&self, _x: &CpuTensor, _dim: i32) -> Result<CpuTensor> {
        todo!("Task 4: softmax")
    }

    fn embedding(&self, _weight: &CpuTensor, _indices: &[u32]) -> Result<CpuTensor> {
        todo!("Task 4: embedding lookup")
    }

    fn reshape(&self, _x: &CpuTensor, _shape: &[usize]) -> Result<CpuTensor> {
        todo!("Task 5: reshape")
    }

    fn transpose(&self, _x: &CpuTensor, _dim0: usize, _dim1: usize) -> Result<CpuTensor> {
        todo!("Task 5: transpose")
    }

    fn cat(&self, _tensors: &[&CpuTensor], _dim: usize) -> Result<CpuTensor> {
        todo!("Task 5: cat")
    }
}
