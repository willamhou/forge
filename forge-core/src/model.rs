use crate::kvcache::KvCache;
use crate::tensor::Tensor;
use crate::{ModelConfig, ModelInput, Result};

pub struct ModelOutput<T: Tensor> {
    /// Logits tensor with shape [batch, vocab_size].
    pub logits: T,
}

pub trait Model: Send + Sync {
    type T: Tensor;

    /// Persistent per-batch decode state: pre-allocated activation buffers
    /// plus the per-step input tensors that [`Model::stage_decode`] refreshes
    /// and [`Model::compute_decode`] reads. Allocated once per batch shape
    /// (one per CUDA-Graph bucket) so every device pointer a captured graph
    /// bakes stays stable across replays.
    type DecodeState;

    fn forward(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = Self::T>,
    ) -> Result<ModelOutput<Self::T>>;

    fn config(&self) -> &ModelConfig;

    /// Allocate decode state for a fixed `batch_size`.
    ///
    /// Used by the engine's CUDA-Graph decode path (one state per bucket).
    fn make_decode_state(&self, batch_size: usize) -> Result<Self::DecodeState>;

    /// Stage one decode step's inputs into the state's persistent input
    /// tensors (token ids, positions→RoPE freqs, paged block tables, KV
    /// lengths, KV-write slot mapping) via host→device copies.
    ///
    /// Must run on **every** decode step (capture or replay) and **outside**
    /// any captured region — the copies write fresh per-step data into the
    /// stable device buffers the captured [`Model::compute_decode`] graph
    /// reads. Block-table width is fixed (not per-step max) so the staged
    /// buffer's shape is stable across steps.
    fn stage_decode(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = Self::T>,
        state: &mut Self::DecodeState,
    ) -> Result<()>;

    /// Capture-stable decode forward: a pure kernel sequence that reads the
    /// pre-staged inputs in `state` and writes logits into the state. Issues
    /// no host→device copies and touches only stable device pointers, so it
    /// can be recorded into a CUDA Graph and replayed. Must be preceded by
    /// [`Model::stage_decode`] for the same step.
    fn compute_decode(
        &self,
        kv_cache: &mut dyn KvCache<T = Self::T>,
        state: &mut Self::DecodeState,
    ) -> Result<()>;

    /// Borrow the logits tensor (`[batch, vocab_size]`) from the decode state
    /// after [`Model::compute_decode`].
    fn decode_logits<'a>(&self, state: &'a Self::DecodeState) -> &'a Self::T;
}
