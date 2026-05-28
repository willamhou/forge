//! Forge transformer models (Llama / Qwen2 / Qwen3 family).
//!
//! A single generic [`TransformerModel`] (decoder-only transformer with GQA +
//! RoPE) serves the whole Llama family; per-architecture differences are
//! optional attention features (QKV bias, QK-norm). [`registry::load_model`]
//! dispatches by HuggingFace architecture string.

pub mod layers;
pub mod loader;
pub mod model;
pub mod persistent_buffers;
pub mod quantized_linear;
pub mod registry;
pub mod rope;

pub use loader::load_transformer;
pub use model::TransformerModel;
pub use persistent_buffers::{LlamaDecodeState, LlamaPersistentBuffers, StagedDecodeMeta};
pub use registry::load_model;
