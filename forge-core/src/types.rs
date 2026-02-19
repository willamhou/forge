use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
    Q4K,
    Q8K,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 => 1,
            DType::I4 => 1, // packed: 2 values per byte, but min alloc is 1 byte
            DType::Q4K | DType::Q8K => 1, // block-quantized, varies
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
pub struct ModelInput {
    pub token_ids: Vec<Vec<u32>>,
    pub positions: Vec<Vec<u32>>,
    pub seq_metadata: Vec<SeqMetadata>,
}

#[derive(Debug, Clone)]
pub struct SeqMetadata {
    pub seq_id: u64,
    pub prompt_len: usize,
    pub generated_len: usize,
    pub is_prefill: bool,
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            min_p: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            max_tokens: 256,
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    MaxTokens,
    EosToken,
    StopString,
    Cancelled,
}
