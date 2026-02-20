use std::sync::Arc;

use forge_core::{DType, Tensor};

#[derive(Clone, Debug)]
pub struct CpuTensor {
    pub(crate) data: Arc<Vec<f32>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            dtype: DType::F32,
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Tensor for CpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}
