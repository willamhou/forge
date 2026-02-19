use crate::DType;

pub trait Tensor: Clone + Send + Sync + std::fmt::Debug {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }
    fn size_bytes(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }
}
