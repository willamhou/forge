use crate::kvcache::KvCache;
use crate::tensor::Tensor;
use crate::{ModelConfig, ModelInput, Result};

pub struct ModelOutput<T: Tensor> {
    /// Logits tensor with shape [batch, vocab_size].
    pub logits: T,
}

pub trait Model: Send + Sync {
    type T: Tensor;

    fn forward(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = Self::T>,
    ) -> Result<ModelOutput<Self::T>>;

    fn config(&self) -> &ModelConfig;
}
