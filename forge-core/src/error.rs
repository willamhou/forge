use thiserror::Error;

use crate::types::DType;

#[derive(Error, Debug)]
pub enum ForgeError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    #[error("Model load error: {0}")]
    ModelLoad(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Sequence not found: {0}")]
    SeqNotFound(u64),

    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(DType),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, ForgeError>;
