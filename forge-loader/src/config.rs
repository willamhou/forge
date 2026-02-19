use forge_core::{DType, ModelConfig};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub head_dim: Option<usize>,
}

fn default_rms_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10000.0
}

impl LlamaConfig {
    pub fn to_model_config(&self) -> ModelConfig {
        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        ModelConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            dtype: DType::F16,
        }
    }
}
