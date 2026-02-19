//! Forge core types, traits, and error definitions.

pub mod backend;
pub mod error;
pub mod tensor;
pub mod types;

pub use backend::Backend;
pub use error::{ForgeError, Result};
pub use tensor::Tensor;
pub use types::*;
