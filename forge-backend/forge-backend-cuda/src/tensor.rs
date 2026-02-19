use cudarc::driver::CudaSlice;
use forge_core::{DType, ForgeError, Result, Tensor};

#[derive(Debug, Clone)]
pub(crate) enum TensorData {
    F32(CudaSlice<f32>),
    F16(CudaSlice<half::f16>),
    BF16(CudaSlice<half::bf16>),
}

#[derive(Debug, Clone)]
pub struct CudaTensor {
    pub(crate) data: TensorData,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

impl CudaTensor {
    pub(crate) fn numel_from_shape(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    pub(crate) fn f32_data(data: CudaSlice<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: TensorData::F32(data),
            shape,
            dtype: DType::F32,
        }
    }

    pub(crate) fn f16_data(data: CudaSlice<half::f16>, shape: Vec<usize>) -> Self {
        Self {
            data: TensorData::F16(data),
            shape,
            dtype: DType::F16,
        }
    }

    pub(crate) fn bf16_data(data: CudaSlice<half::bf16>, shape: Vec<usize>) -> Self {
        Self {
            data: TensorData::BF16(data),
            shape,
            dtype: DType::BF16,
        }
    }

    pub(crate) fn f32_slice(&self) -> Result<&CudaSlice<f32>> {
        match &self.data {
            TensorData::F32(s) => Ok(s),
            _ => Err(ForgeError::InvalidArgument(format!(
                "expected f32 tensor, got {:?}",
                self.dtype
            ))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn f16_slice(&self) -> Result<&CudaSlice<half::f16>> {
        match &self.data {
            TensorData::F16(s) => Ok(s),
            _ => Err(ForgeError::InvalidArgument(format!(
                "expected f16 tensor, got {:?}",
                self.dtype
            ))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn bf16_slice(&self) -> Result<&CudaSlice<half::bf16>> {
        match &self.data {
            TensorData::BF16(s) => Ok(s),
            _ => Err(ForgeError::InvalidArgument(format!(
                "expected bf16 tensor, got {:?}",
                self.dtype
            ))),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.data {
            TensorData::F32(s) => s.len(),
            TensorData::F16(s) => s.len(),
            TensorData::BF16(s) => s.len(),
        }
    }
}

impl Tensor for CudaTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}
