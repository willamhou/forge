use cudarc::driver::CudaSlice;
use forge_core::{DType, Tensor};

#[derive(Debug, Clone)]
pub struct CudaTensor {
    pub(crate) data_f32: Option<CudaSlice<f32>>,
    pub(crate) data_f16: Option<CudaSlice<half::f16>>,
    pub(crate) data_bf16: Option<CudaSlice<half::bf16>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

impl CudaTensor {
    pub(crate) fn numel_from_shape(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    pub(crate) fn f32_slice(&self) -> &CudaSlice<f32> {
        self.data_f32.as_ref().expect("expected f32 tensor")
    }

    pub(crate) fn f32_slice_mut(&mut self) -> &mut CudaSlice<f32> {
        self.data_f32.as_mut().expect("expected f32 tensor")
    }

    pub(crate) fn f16_slice(&self) -> &CudaSlice<half::f16> {
        self.data_f16.as_ref().expect("expected f16 tensor")
    }

    pub(crate) fn bf16_slice(&self) -> &CudaSlice<half::bf16> {
        self.data_bf16.as_ref().expect("expected bf16 tensor")
    }

    pub(crate) fn len(&self) -> usize {
        match self.dtype {
            DType::F32 => self.data_f32.as_ref().map_or(0, |s| s.len()),
            DType::F16 => self.data_f16.as_ref().map_or(0, |s| s.len()),
            DType::BF16 => self.data_bf16.as_ref().map_or(0, |s| s.len()),
            _ => 0,
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
