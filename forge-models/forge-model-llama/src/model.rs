use forge_core::{Backend, KvCache, Model, ModelConfig, ModelInput, ModelOutput, Result};

use crate::layers::{LlamaDecoderLayer, RMSNorm};
use crate::rope::RopeFreqs;

pub struct LlamaModel<B: Backend> {
    config: ModelConfig,
    embed_tokens: B::Tensor,
    layers: Vec<LlamaDecoderLayer<B>>,
    norm: RMSNorm<B>,
    lm_head: B::Tensor,
    rope_freqs: RopeFreqs<B>,
    backend: B,
}

impl<B: Backend> LlamaModel<B> {
    pub fn new(
        config: ModelConfig,
        embed_tokens: B::Tensor,
        layers: Vec<LlamaDecoderLayer<B>>,
        norm: RMSNorm<B>,
        lm_head: B::Tensor,
        rope_freqs: RopeFreqs<B>,
        backend: B,
    ) -> Self {
        Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope_freqs,
            backend,
        }
    }
}

impl<B: Backend + Clone> Model for LlamaModel<B> {
    type T = B::Tensor;

    fn forward(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = B::Tensor>,
    ) -> Result<ModelOutput<B::Tensor>> {
        // For Phase 1, we process one sequence at a time.
        // Batch > 1 support will come with the continuous batching engine.
        let seq_meta = &input.seq_metadata[0];
        let token_ids = &input.token_ids[0];

        // 1. Token embedding lookup
        let hidden = self.backend.embedding(&self.embed_tokens, token_ids)?;
        // hidden shape: [seq_len, hidden_size]

        // 2. Run through each decoder layer
        let mut hidden = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden,
                &self.rope_freqs,
                kv_cache,
                seq_meta.seq_id,
                i,
                &self.backend,
            )?;
        }

        // 3. Final RMS norm
        hidden = self.norm.forward(&hidden, &self.backend)?;

        // 4. LM head projection → logits [seq_len, vocab_size]
        let logits = self.backend.matmul(&hidden, &self.lm_head)?;

        Ok(ModelOutput { logits })
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
