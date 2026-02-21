# Runtime & Server Codemap

> Freshness: 2026-02-21 | Branch: feat/phase1-mvp

## forge-runtime — Engine, Sampling, Constraints

### Files
| File | Purpose |
|------|---------|
| `src/engine.rs` | `Engine<B,M>` — main inference loop |
| `src/sampling.rs` | `CpuSampler` — temperature, top-k/p, penalties |
| `src/constraints/mod.rs` | Module exports |
| `src/constraints/fsm.rs` | `FsmConstraint` trait, `TokenFsmIndex`, `TokenVocab` |
| `src/constraints/regex.rs` | Regex pattern → DFA → token-level FSM |
| `src/constraints/json_schema.rs` | JSON Schema → regex → DFA → token-level FSM |

### Engine (`engine.rs`)
```rust
pub struct Engine<B: Backend, M: Model<T = B::Tensor>> {
    model, backend, scheduler, kv_cache, sampler,
    request_rx: mpsc::Receiver<EngineRequest>,
    event_senders: HashMap<u64, mpsc::Sender<EngineEvent>>,
    constraints: HashMap<u64, SeqConstraint>,
    decode_fn: Option<DecodeFn>,
}
// Key methods:
//   run() — async main loop (drain requests → schedule → process)
//   process_sequence(seq) — forward + sample + emit events
//   with_decode_fn(f) — enable stop_strings checking
```

### Sampling (`sampling.rs`)
```rust
pub struct CpuSampler;
// Methods:
//   sample(logits, params, generated) -> SampleResult
//   sample_with_constraint(logits, params, generated, constraint) -> SampleResult
//
// Pipeline: repetition_penalty → temperature → top_k → top_p → multinomial
// Constraint: zero out disallowed token logits before sampling
```

### FSM Constraints (`constraints/`)

#### `FsmConstraint` trait
```rust
pub trait FsmConstraint: Send + Sync {
    fn initial_state(&self) -> u32;
    fn next_state(&self, state: u32, token_id: u32) -> Option<u32>;
    fn allowed_tokens(&self, state: u32) -> Option<Vec<u32>>;
    fn is_final_state(&self, state: u32) -> bool;
}
```

#### `TokenFsmIndex` — implements `FsmConstraint`
- Built from DFA + token vocabulary
- Maps each (DFA state, token) → next DFA state
- Pre-computes allowed token sets per state

#### `TokenVocab`
```rust
pub struct TokenVocab { tokens: Vec<(u32, String)> }
// Built via: TokenVocab::from_decode_fn(vocab_size, decode_fn)
```

#### JSON Schema Pipeline
```
JSON Schema string
  → schema_to_regex() → regex pattern string
  → build_regex_fsm(pattern, vocab) → TokenFsmIndex
```
Supports: string, integer, number, boolean, null, array, object, enum, const, anyOf/oneOf

#### Regex Pipeline
```
Regex pattern string
  → build_regex_fsm(pattern, vocab) → TokenFsmIndex
```

## forge-scheduler — Continuous Batching

### Files
| File | Purpose |
|------|---------|
| `src/lib.rs` | Module exports |
| `src/continuous.rs` | `ContinuousBatchingScheduler` |
| `src/sequence.rs` | `SequenceState`, `SeqState` enum |

### `Scheduler` trait (`forge-core`)
```rust
pub trait Scheduler: Send + Sync {
    fn enqueue(request) -> Result<RequestHandle>
    fn cancel(seq_id) -> Result<()>
    fn schedule(cache_usage) -> Result<ScheduleBatch>
    fn finish(seq_id, reason) -> Result<()>
    fn append_token(seq_id, token_id) -> Result<()>
    fn get_generated_tokens(seq_id) -> Result<Vec<u32>>
}
```

### `ContinuousBatchingScheduler`
- FCFS queue (`VecDeque<u64>`) for waiting sequences
- Running set (`Vec<u64>`) for active decode
- Config: `max_batch_size` (256), `max_prefill_tokens` (4096)
- Rejects prompts exceeding `max_prefill_tokens` (added to `rejected_seq_ids`)
- Cache-aware: checks `free_blocks` with committed block accounting

## forge-server — HTTP API

### Files
| File | Purpose |
|------|---------|
| `src/main.rs` | CLI (`clap`), model loading, engine spawn, axum server |
| `src/lib.rs` | Module exports |
| `src/api/mod.rs` | API module |
| `src/api/openai.rs` | `chat_completions`, `list_models`, `health` handlers |
| `src/api/types.rs` | Request/response serde types |
| `src/tokenizer.rs` | `ForgeTokenizer`, `IncrementalDecoder` |
| `src/chat_template.rs` | `ChatTemplate` (minijinja), ChatML default |

### `AppState`
```rust
pub struct AppState {
    pub model_name: String,
    pub tokenizer: Arc<ForgeTokenizer>,
    pub chat_template: ChatTemplate,
    pub request_tx: mpsc::Sender<EngineRequest>,
    pub token_vocab: Option<Arc<TokenVocab>>,
}
```

### Endpoints
| Method | Path | Handler |
|--------|------|---------|
| POST | `/v1/chat/completions` | `chat_completions` — streaming + non-streaming |
| GET | `/v1/models` | `list_models` |
| GET | `/forge/v1/health` | `health` |

### Streaming
- SSE via `axum::response::sse::Sse`
- `IncrementalDecoder` handles multi-byte UTF-8 across BPE token boundaries
- Flushes pending bytes on stream finish

## forge-models/forge-model-llama — Llama Model

### Files
| File | Purpose |
|------|---------|
| `src/lib.rs` | `LlamaModel<B>`, `load_llama_model()` |
| `src/attention.rs` | Multi-head attention with GQA, RoPE |
| `src/layers.rs` | `TransformerBlock`, `RMSNorm`, `FeedForward` |

### `Model` trait (`forge-core`)
```rust
pub trait Model: Send + Sync {
    type T: Tensor;
    fn forward(input: &ModelInput, kv_cache: &mut dyn KvCache<T=Self::T>) -> Result<ModelOutput<Self::T>>
    fn config(&self) -> &ModelConfig
}
```

### `LlamaModel<B: Backend>`
- Embedding table + N transformer blocks + final RMSNorm + LM head
- Supports FP32 and FP16 compute paths
- GQA (grouped query attention) with configurable kv_heads

## forge-loader — Weight Loading

### Files
| File | Purpose |
|------|---------|
| `src/lib.rs` | `SafeTensorsLoader`, `LlamaConfig` |

### `SafeTensorsLoader`
```rust
pub fn new(path: &Path) -> Result<Self>
pub fn load_tensor<B: Backend>(name: &str, backend: &B) -> Result<B::Tensor>
pub fn list_tensors() -> Vec<String>
```
- Loads SafeTensors format (`.safetensors` files)
- Supports F32, F16, BF16 (auto-converts BF16→F16)
- `LlamaConfig` deserializes HuggingFace `config.json`
