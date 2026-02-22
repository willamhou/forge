# Data Models Codemap

> Freshness: 2026-02-21 | Branch: feat/phase1-mvp

## Core Types (`forge-core`)

### Model Configuration
```rust
// forge-core/src/model.rs
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}
```

### Inference Request
```rust
// forge-core/src/scheduler.rs
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
}
```

### Sampling Parameters
```rust
// forge-core/src/sampling.rs
pub struct SamplingParams {
    pub temperature: f32,           // default 1.0
    pub top_p: f32,                 // default 1.0
    pub top_k: Option<usize>,      // default None
    pub max_tokens: usize,         // default 256
    pub seed: Option<u64>,
    pub repetition_penalty: f32,    // default 1.0
    pub presence_penalty: f32,      // default 0.0
    pub frequency_penalty: f32,     // default 0.0
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
}
```

### Scheduler Types
```rust
// forge-core/src/scheduler.rs
pub struct RequestHandle { pub request_id: String, pub seq_id: u64 }

pub struct ScheduleBatch {
    pub prefill_seqs: Vec<ScheduledSeq>,
    pub decode_seqs: Vec<ScheduledSeq>,
    pub rejected_seq_ids: Vec<u64>,
}

pub struct ScheduledSeq {
    pub seq_id: u64,
    pub token_ids: Vec<u32>,
    pub position_offset: usize,
    pub sampling_params: SamplingParams,
    pub is_prefill: bool,
}

pub struct CacheUsage {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub block_size: usize,
}

pub enum FinishReason { EosToken, MaxTokens, StopString, Cancelled }
```

### Model I/O
```rust
// forge-core/src/model.rs
pub struct ModelInput {
    pub token_ids: Vec<Vec<u32>>,
    pub positions: Vec<Vec<u32>>,
    pub seq_metadata: Vec<SeqMetadata>,
}

pub struct SeqMetadata {
    pub seq_id: u64,
    pub prompt_len: usize,
    pub generated_len: usize,
    pub is_prefill: bool,
}

pub struct ModelOutput<T: Tensor> {
    pub logits: T,   // [batch * seq_len, vocab_size]
}
```

### Error Types
```rust
// forge-core/src/error.rs
pub enum ForgeError {
    Cuda(String),
    InvalidArgument(String),
    SeqNotFound(u64),
    OutOfMemory(String),
    Internal(String),
    Tokenizer(String),
    Io(std::io::Error),
}
```

## API Types (`forge-server`)

### Request
```rust
// forge-server/src/api/types.rs
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub json_schema: Option<serde_json::Value>,  // structured output
    pub regex: Option<String>,                    // structured output
}
```

### Response
```rust
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,  // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

pub struct ChatCompletionChunk {  // streaming
    pub id: String,
    pub object: &'static str,  // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}
```

## Engine Events
```rust
// forge-runtime/src/engine.rs
pub enum EngineEvent {
    Token { seq_id: u64, token_id: u32, text: Option<String> },
    Finish { seq_id: u64, reason: FinishReason },
    Error { seq_id: u64, error: String },
}

pub struct EngineRequest {
    pub inference_req: InferenceRequest,
    pub event_tx: mpsc::Sender<EngineEvent>,
    pub constraint: Option<Box<dyn FsmConstraint>>,
}
```

## Loader Types
```rust
// forge-loader/src/lib.rs
pub struct LlamaConfig {  // mirrors HuggingFace config.json
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: Option<f64>,
    pub rope_theta: Option<f64>,
    pub max_position_embeddings: Option<usize>,
    pub head_dim: Option<usize>,
}

pub struct SafeTensorsLoader { /* path, file handles */ }
// Methods: new(path), load_tensor(name, backend), list_tensors()
```
