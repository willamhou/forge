//! CPU backend for Forge inference engine.

mod backend;
pub mod tensor;

pub use backend::CpuBackend;
pub use tensor::CpuTensor;
