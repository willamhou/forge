use cudarc::driver::CudaSlice;
use forge_core::{DType, ForgeError, Result, Tensor};

#[derive(Debug, Clone)]
pub(crate) enum TensorData {
    F32(CudaSlice<f32>),
    F16(CudaSlice<half::f16>),
    BF16(CudaSlice<half::bf16>),
    /// Raw block-quantized bytes (e.g. Q8_0). Layout is dtype-defined; the
    /// quantized GEMV kernels reinterpret these bytes per block. `len()`
    /// reports the byte count, not the logical element count.
    Quant(CudaSlice<u8>),
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

    /// Build a block-quantized tensor from raw device bytes. `shape` is the
    /// logical (dequantized) element shape; `dtype` must be a quantized dtype
    /// (`DType::is_quantized()`), which defines the byte layout of `bytes`.
    pub(crate) fn quant_data(bytes: CudaSlice<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            data: TensorData::Quant(bytes),
            shape,
            dtype,
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

    pub(crate) fn f32_slice_mut(&mut self) -> Result<&mut CudaSlice<f32>> {
        match &mut self.data {
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

    #[allow(dead_code)]
    pub(crate) fn f16_slice_mut(&mut self) -> Result<&mut CudaSlice<half::f16>> {
        match &mut self.data {
            TensorData::F16(s) => Ok(s),
            _ => Err(ForgeError::InvalidArgument(format!(
                "expected f16 tensor, got {:?}",
                self.dtype
            ))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn bf16_slice_mut(&mut self) -> Result<&mut CudaSlice<half::bf16>> {
        match &mut self.data {
            TensorData::BF16(s) => Ok(s),
            _ => Err(ForgeError::InvalidArgument(format!(
                "expected bf16 tensor, got {:?}",
                self.dtype
            ))),
        }
    }

    /// Raw quantized byte slice. Errors unless the tensor is block-quantized.
    pub(crate) fn quant_slice(&self) -> Result<&CudaSlice<u8>> {
        match &self.data {
            TensorData::Quant(s) => Ok(s),
            _ => Err(ForgeError::InvalidArgument(format!(
                "expected quantized tensor, got {:?}",
                self.dtype
            ))),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.data {
            TensorData::F32(s) => s.len(),
            TensorData::F16(s) => s.len(),
            TensorData::BF16(s) => s.len(),
            // NOTE: byte count, not logical element count, for quantized data.
            TensorData::Quant(s) => s.len(),
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
