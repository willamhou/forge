//! Model registry: dispatch by HuggingFace `architectures` to the right model
//! builder. This is the single place that declares which architectures Forge
//! supports and how each is constructed — adding a model means adding an arm
//! here (plus any new layer feature it needs).
//!
//! The Llama / Qwen2 / Qwen3 family all build the same generic
//! [`TransformerModel`]; they differ only in optional attention features (QKV
//! bias for Qwen2, per-head QK-norm for Qwen3), which the loader picks up from
//! the weight set. Architectures that diverge structurally would get their own
//! builder rather than sharing `load_transformer`.

use forge_core::{Backend, ForgeError, ModelConfig, Result};
use forge_loader::SafeTensorsLoader;

use crate::loader::load_transformer;
use crate::model::TransformerModel;

/// Architectures Forge can serve. All currently map to the shared
/// decoder-transformer builder.
pub const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "MistralForCausalLM",
];

/// Build a model for the given HF architecture string (`config.architectures`).
/// `arch = None` (architecture absent from config.json) is treated as a
/// Llama-family model. Unknown architectures are rejected.
///
/// `quantize` enables the Q8_0 quantized-decode path (an extra Q8_0 weight copy
/// for decode GEMV; prefill stays FP16). With it false the load is byte-for-byte
/// identical to the unquantized path.
pub fn load_model<B: Backend + Clone>(
    arch: Option<&str>,
    loader: &SafeTensorsLoader,
    config: ModelConfig,
    backend: &B,
    quantize: bool,
) -> Result<TransformerModel<B>> {
    match arch {
        None
        | Some(
            "LlamaForCausalLM" | "Qwen2ForCausalLM" | "Qwen3ForCausalLM" | "MistralForCausalLM",
        ) => load_transformer(loader, config, backend, quantize),
        Some(other) => Err(ForgeError::InvalidArgument(format!(
            "unsupported model architecture {other:?}; supported: {SUPPORTED_ARCHITECTURES:?}"
        ))),
    }
}
