# Forge: Rust LLM Inference Framework Design

> Design document for Forge, a high-performance LLM inference serving framework written in Rust.
> Target: vLLM / SGLang level capability with Rust's performance and safety guarantees.

## 1. Overview

### Goals

| Dimension | Decision |
|-----------|----------|
| Name | **Forge** |
| Positioning | Production-grade LLM serving framework, targeting vLLM/SGLang |
| Core objectives | Extreme performance + Multi-backend + Extensible + Edge deployment |
| Model architecture | Generic Transformer abstraction layer; initial: Llama + DeepSeek |
| GPU backend | CUDA first, future: Metal / ROCm / CPU |
| Kernel strategy | Hybrid: Rust orchestration + FlashAttention/FlashInfer FFI + custom CUDA C++ kernels |
| API | OpenAI-compatible + Forge extension API |
| Scheduling | Continuous Batching + PagedAttention + RadixAttention/Prefix Caching |
| Weight formats | SafeTensors + GGUF |
| Quantization | FP16/BF16 + INT8/INT4 (GPTQ/AWQ) + GGUF quantization formats (Q4_K_M, etc.) |

### Architecture: Layered Monolith with Distributed Extension Points

Primary architecture is a single-binary layered monolith (Phase 1), with clean abstraction boundaries that allow transparent evolution to disaggregated deployment (Phase 3+).

```
+-------------------------------------+
|           API Layer (axum)          |  OpenAI compat + extension API
+-------------------------------------+
|         Scheduler / Router          |  Continuous Batching + request routing
+-------------------------------------+
|    Sequence Manager + KV Cache      |  PagedAttention + RadixAttention
+-------------------------------------+
|       Model Runtime (trait)         |  Generic Transformer abstraction
|  +----------+  +--------------+    |
|  |  Llama   |  |  DeepSeek    |    |  Pluggable model implementations
|  +----------+  +--------------+    |
+-------------------------------------+
|       Backend Abstraction           |  Unified trait: Tensor/Kernel ops
|  +-------+ +-------+ +------+     |
|  | CUDA  | | Metal | | CPU  |     |  Pluggable backends
|  +-------+ +-------+ +------+     |
+-------------------------------------+
|    Kernel Layer (FFI bindings)      |  FlashAttention + CUTLASS + custom
+-------------------------------------+
```

## 2. Project Structure

```
forge/
├── Cargo.toml                    # workspace root
├── forge-server/                 # Binary entry, API layer (axum)
├── forge-core/                   # Core type definitions, trait abstractions
├── forge-scheduler/              # Scheduler: Continuous Batching + request management
├── forge-kvcache/                # KV Cache: PagedAttention + RadixAttention
├── forge-models/                 # Model implementations
│   ├── forge-model-llama/
│   └── forge-model-deepseek/
├── forge-runtime/                # Model execution engine, forward pass orchestration
├── forge-backend/                # Backend abstraction trait
│   ├── forge-backend-cuda/       # CUDA backend (FFI to kernels)
│   ├── forge-backend-cpu/        # CPU backend (future)
│   └── forge-backend-metal/      # Metal backend (future)
├── forge-kernels/                # CUDA C++ kernel source + FlashAttention bindings
├── forge-loader/                 # Weight loading: SafeTensors + GGUF
├── forge-quantize/               # Quantization: GPTQ/AWQ/GGUF formats
└── forge-transport/              # Communication abstraction (distributed extension point)
```

Key decisions:
- **forge-transport**: Distributed extension point. Initial: `InProcessTransport` (zero-overhead function calls). Future: `GrpcTransport` / `TcpTransport` for scheduler-worker separation without changing scheduling logic.
- Each crate supports Cargo feature flags for conditional compilation. Edge deployment can compile only `cpu` backend + `gguf` loader for minimal binary size.

## 3. Core Trait Design

### 3.1 Backend + Tensor

```rust
pub trait Backend: Send + Sync + 'static {
    type Tensor: Tensor;
    type Stream: Stream;

    fn name(&self) -> &str;
    fn device_count(&self) -> usize;
    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;
    fn synchronize(&self) -> Result<()>;
    fn comm(&self) -> Option<&dyn CommOps<Tensor = Self::Tensor>> { None }
}

pub trait Tensor: Clone + Send + Sync {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    fn to_backend<B: Backend>(&self, backend: &B) -> Result<B::Tensor>;
    fn matmul(&self, rhs: &Self) -> Result<Self>;
    fn add(&self, rhs: &Self) -> Result<Self>;
    fn mul(&self, rhs: &Self) -> Result<Self>;
    fn softmax(&self, dim: i32) -> Result<Self>;
    fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self>;
    fn rope(&self, freqs: &Self) -> Result<Self>;
    fn silu(&self) -> Result<Self>;
}

pub enum DType {
    F32, F16, BF16, I8, I4, Q4K, Q8K,
}
```

### 3.2 Model

```rust
pub trait Model: Send + Sync {
    fn forward(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache,
    ) -> Result<ModelOutput>;
    fn config(&self) -> &ModelConfig;
    fn supported_dtypes(&self) -> &[DType];
}
```

`Model::forward` is **synchronous** by design. GPU kernel launches are inherently async (CPU returns immediately), and the real async boundary is at the runtime loop and transport layer. This holds even for tensor parallelism: NCCL all-reduce is a blocking API from the CPU caller's perspective.

### 3.3 KV Cache

```rust
pub trait KvCache: Send + Sync {
    fn allocate(&mut self, seq_id: u64, max_len: usize) -> Result<()>;
    fn append(&mut self, seq_id: u64, layer: usize, key: &dyn Tensor, value: &dyn Tensor) -> Result<()>;
    fn get_kv(&self, seq_id: u64, layer: usize) -> Result<(Box<dyn Tensor>, Box<dyn Tensor>)>;
    fn free(&mut self, seq_id: u64) -> Result<()>;
    fn match_prefix(&self, token_ids: &[u32]) -> Option<PrefixMatch>;
    fn usage(&self) -> CacheUsage;
    fn rollback(&mut self, seq_id: u64, num_tokens: usize) -> Result<()>;
}
```

Built-in `match_prefix` for RadixAttention and `rollback` for speculative decoding from day one.

### 3.4 Scheduler

```rust
pub trait Scheduler: Send + Sync {
    fn enqueue(&self, request: InferenceRequest) -> Result<RequestHandle>;
    fn cancel(&self, handle: &RequestHandle) -> Result<()>;
    fn schedule(&mut self) -> Result<ScheduleBatch>;
    fn finish(&mut self, seq_id: u64, reason: FinishReason) -> Result<()>;
}
```

### 3.5 Transport (Distributed Extension Point)

```rust
#[async_trait]
pub trait Transport: Send + Sync {
    async fn send_batch(&self, batch: ScheduleBatch) -> Result<BatchResult>;
    async fn health_check(&self) -> Result<WorkerStatus>;
}

#[async_trait]
pub trait KvTransport: Send + Sync {
    async fn send_kv(&self, seq_id: u64, kv_data: &KvTransferPayload, dst: WorkerId) -> Result<()>;
    async fn recv_kv(&self, seq_id: u64, src: WorkerId) -> Result<KvTransferPayload>;
    fn capabilities(&self) -> TransportCapabilities;
}
```

Initial: `InProcessTransport` (zero overhead). Future: `GrpcTransport`, `NvLinkKvTransport`, `RdmaKvTransport`, `TcpKvTransport`.

### 3.6 Tensor Parallelism

Extends `Backend` with `CommOps`, not `Model`:

```rust
pub trait CommOps: Send + Sync {
    type Tensor;
    fn all_reduce_sum(&self, tensor: &Self::Tensor) -> Result<Self::Tensor>;
    fn send(&self, tensor: &Self::Tensor, dst_rank: usize) -> Result<()>;
    fn recv(&self, shape: &[usize], dtype: DType, src_rank: usize) -> Result<Self::Tensor>;
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
}
```

Model code calls `backend.comm().all_reduce_sum()` where needed. Single-GPU: `comm()` returns `None`, zero overhead. Multi-GPU: NCCL implementation. Weight sharding happens at load time via `ParallelConfig { tp_size, tp_rank, pp_size, pp_rank }`.

## 4. Scheduler Design

### 4.1 Two-Level Queue

```
Request arrives → Waiting Queue (priority-sorted)
                       │ free KV Cache blocks available
                       ▼
                  Running Pool
                  ├── Prefill Set (this step's prefill sequences)
                  └── Decode Set  (this step's decode sequences)
                       │ cache pressure
                       ▼
                  Preemption (swap out / recompute)
```

### 4.2 Continuous Batching with Chunked Prefill

Long prefill sequences are split into chunks to prevent decode latency spikes:

```
Without chunking:
  Step N:   [prefill 4096 tokens] + [decode seq_A] + [decode seq_B]  ~50ms
  → decode latency spike

With chunking (chunk_size=512):
  Step N:   [prefill chunk_0 512t] + [decode seq_A] + [decode seq_B]  ~8ms
  Step N+1: [prefill chunk_1 512t] + [decode seq_A] + [decode seq_B]  ~8ms
  → stable decode latency
```

Sequence state machine: `Waiting → Prefilling (multi-step) → Decoding → Finished`

Each chunk's attention sees all previously cached KV + current chunk KV. FlashInfer's `append_attention` natively supports this pattern.

Token budget system: `max_batch_tokens` limits total tokens per step. Decode sequences get priority (1 token each), remaining budget goes to prefill chunks.

Configuration knobs:
- `max_batch_tokens`: total token budget per step (e.g., 4096)
- `prefill_chunk_size`: max tokens per prefill chunk (e.g., 512)
- `decode_priority`: decode sequences always scheduled first (default: true)

### 4.3 Preemption

When KV cache is exhausted:
- **Recompute**: evict sequence, recompute later (saves memory)
- **Swap out**: move KV to CPU memory (saves compute)

## 5. Attention Implementation

### 5.1 Attention Kernel Abstraction

```rust
pub trait AttentionKernel: Send + Sync {
    fn prefill_attention(&self, q, k, v, mask, scale) -> Result<Tensor>;
    fn decode_attention(&self, q, k_cache, v_cache, block_table, seq_lens, scale) -> Result<Tensor>;
    fn chunked_prefill_attention(&self, q, k_new, v_new, k_cache, v_cache, block_table, cached_len, scale) -> Result<Tensor>;
}
```

### 5.2 Three-Level Implementation

| Level | Implementation | Purpose |
|-------|---------------|---------|
| 3 | FlashAttention / FlashInfer (C++ FFI) | Production, best performance |
| 2 | Custom CUDA C++ kernels (FFI) | Special requirements / experiments |
| 1 | Naive Rust (pure Tensor ops) | Development, debugging, CPU backend |

### 5.3 PagedAttention

Block-based KV cache eliminates memory fragmentation:

```
Block Pool: [B0][B1][B2][B3][B4][B5]...
seq_0 → page table: [B0, B3, B5]    arbitrary allocation, no fragmentation
seq_1 → page table: [B1, B4]
```

Block size: 16 tokens per block (typical).

### 5.4 RadixAttention (Prefix Caching)

Radix tree indexes shared prefix KV cache blocks:

```
Multiple requests sharing system prompt:
    "You are a helpful assistant." → [B0, B1, B2] (shared, copy-on-write)
     ├── "What is Rust?" → [B3, B4]
     ├── "Explain CUDA." → [B5]
     └── "How to cook?"  → [B6]
```

LRU eviction on leaf nodes when cache pressure increases.

## 6. Prefill-Decode Disaggregation (PD Separation)

### 6.1 Motivation

Prefill (compute-bound) and decode (memory-bound) have fundamentally different resource profiles. Mixing them causes mutual interference. Industry trend: PD disaggregation is now standard in vLLM, SGLang, NVIDIA Dynamo, Mooncake.

### 6.2 Architecture

```
                    Global Scheduler
                         |
               +---------+---------+
               |                   |
         Prefill Pool         Decode Pool
         (high FLOPS)         (high bandwidth)
               |                   |
               +--- KV Transfer ---+
                  RDMA / NVLink / TCP
```

### 6.3 KV Cache Transfer

| Method | Bandwidth | GPU SM usage | Latency |
|--------|-----------|-------------|---------|
| NCCL send/recv | High | Yes (launches kernel) | Medium |
| GPU-Direct RDMA | High | Zero | Low |
| Host staging | Medium (via CPU) | Low | High |
| NVLink (same node) | Highest | Zero | Lowest |

### 6.4 Deployment Modes

Same binary, different startup parameters:

```
forge serve                                      → Standalone (default)
forge serve --mode scheduler --prefill/decode ... → Scheduler
forge serve --mode prefill --scheduler ...        → Prefill Worker
forge serve --mode decode --scheduler ...         → Decode Worker
```

Model, Backend, KvCache code is fully reused. Only scheduler and transport layers differ.

## 7. API Layer

### 7.1 Routes

**OpenAI-compatible:**
- `POST /v1/chat/completions` — chat completion (streaming/non-streaming)
- `POST /v1/completions` — text completion
- `GET /v1/models` — model list

**Forge extension:**
- `POST /forge/v1/generate` — low-level generation with full control
- `POST /forge/v1/batch` — batch requests (offline throughput optimization)
- `GET /forge/v1/cache/stats` — RadixAttention cache hit statistics
- `POST /forge/v1/cache/clear` — manual prefix cache clear
- `GET /forge/v1/metrics` — Prometheus metrics
- `GET /forge/v1/health` — health check + GPU status
- `WS /forge/v1/stream` — WebSocket streaming (low-latency)

### 7.2 Extension Parameters

Passed via `extra` field in OpenAI API or natively in Forge API:
- `prefix_cache`: Auto / Force / Disable
- `sampling`: Advanced sampling controls
- `priority`: Low / Normal / High
- `logprobs`: Token-level log probabilities
- `json_schema`: Guided generation with JSON Schema constraint
- `regex`: Guided generation with regex constraint

### 7.3 Streaming

SSE-based streaming with `tokio::sync::mpsc` channel per request. Incremental token decoding handles UTF-8 boundary issues (multi-byte characters spanning tokens).

### 7.4 Metrics (Prometheus)

Request-level: total, active, latency. Token-level: TTFT, ITL, tokens/sec. Engine-level: batch size, KV cache usage, prefix cache hit rate, queue depth, preemption count. GPU-level: utilization, memory.

## 8. Sampling

### 8.1 Pipeline Architecture

```
logits [batch, vocab]
  → Logit Processor Chain (composable, ordered):
    1. Repetition penalty
    2. Presence penalty
    3. Frequency penalty
    4. Temperature scaling
    5. Top-K filter
    6. Top-P (nucleus) filter
    7. Min-P (dynamic threshold)
    8. Grammar constraint (JSON Schema / Regex FSM)
  → Sampler: Multinomial / Greedy
  → Stop condition check: EOS token / max_tokens / stop strings
  → token_id
```

### 8.2 GPU Sampling (Production Optimization)

CPU sampling requires transferring full logits from GPU (~125 MB/step at batch=256, vocab=128k). GPU sampling eliminates this:

| | CPU Sampling | GPU Sampling |
|---|---|---|
| Transfer/step | ~125 MB | ~1 KB (token IDs only) |
| Sampling latency/step | ~5.5ms | ~0.3ms |
| Implementation complexity | Low | High (CUDA kernel) |
| Structured output | Native | Allowed-token mask approach |
| Use case | Development / small batch | Production / large batch |

Hybrid approach: penalties, temperature, top-k, top-p, multinomial sampling on GPU. Stop string detection, FSM state advancement on CPU. FSM computes allowed-token masks on CPU, uploads to GPU (16 KB/sequence), applied before sampling. Pipeline overlap: FSM mask computation for step N+1 runs concurrently with GPU forward of step N+1.

Unified via `SamplerBackend` trait with `CpuSampler`, `GpuSampler`, and `Auto` (threshold-based switching).

### 8.3 Structured Output

JSON Schema and regex constraints compiled to finite state machines (FSM). Each step, FSM produces allowed-token set. Compatible with both CPU and GPU sampling via mask approach.

## 9. Tokenizer & Chat Template

### 9.1 Tokenizer

Uses `tokenizers` crate (HuggingFace's official Rust implementation). Zero overhead, native support for all major tokenizer formats. Batch encoding is internally parallelized.

Incremental decoding for streaming: handles UTF-8 boundary issues where characters span multiple tokens. Maintains `DecodeState` per sequence with pending token buffer.

### 9.2 Chat Template

Uses `minijinja` crate (pure Rust Jinja2 implementation) to render HuggingFace chat templates. Each model's `tokenizer_config.json` contains its Jinja2 template.

GGUF compatibility: extracts tokenizer vocabulary and merges from GGUF metadata. Falls back to ChatML format when no chat template is present.

### 9.3 Stop String Detection

Rolling buffer approach for cross-token stop string matching. Supports partial match pending (holds output until confirmed match or non-match).

## 10. Speculative Decoding

### 10.1 Mechanism

Draft engine quickly generates K candidate tokens. Target model verifies all K tokens in a single forward pass (treated as prefill). Rejection sampling ensures output distribution is mathematically identical to standard decoding.

### 10.2 Draft Strategies

| Strategy | Draft source | Overhead | Acceptance rate | Use case |
|----------|-------------|----------|----------------|----------|
| Independent model | Small model (1-7B) | Extra VRAM | 60-80% | Server, sufficient VRAM |
| N-gram prompt lookup | Pattern match in context | Zero compute | Task-dependent | Translation, summarization, code |
| Self-speculative | Target model early exit | Zero extra VRAM | Lower | VRAM-constrained (future) |

Pluggable via `DraftEngine` trait.

### 10.3 Adaptive K

Tracks per-sequence acceptance rate with sliding window. Adjusts speculation length: `K = ceil(1 / (1 - acceptance_rate))`, clamped to [1, 8]. High acceptance → more speculation; low acceptance → less waste.

### 10.4 KV Cache Rollback

Rejected draft tokens require KV cache rollback. `PagedKvCache::rollback()` supports partial block deallocation and RadixTree truncation.

## 11. Implementation Phases

### Phase 1: MVP — Single GPU, Full Pipeline

- [ ] Project scaffolding (Cargo workspace, CI)
- [ ] Core traits (`Backend`, `Tensor`, `Model`, `KvCache`, `Scheduler`)
- [ ] CUDA backend with cudarc + FFI kernels (RMSNorm, RoPE, SiLU)
- [ ] FlashAttention integration (FFI binding)
- [ ] SafeTensors weight loader
- [ ] Llama model implementation
- [ ] Naive continuous batching scheduler (no chunked prefill)
- [ ] CPU sampling with logit processor pipeline
- [ ] Tokenizer + chat template (tokenizers + minijinja)
- [ ] OpenAI-compatible API (axum, SSE streaming)
- [ ] Basic PagedAttention KV cache
- [ ] Benchmarks: TTFT, ITL, throughput vs vLLM baseline

### Phase 2: Production Features

- [ ] DeepSeek model implementation (MoE)
- [ ] Chunked prefill scheduler
- [ ] RadixAttention / prefix caching
- [ ] GGUF weight loader + quantization (Q4_K_M, etc.)
- [ ] INT8/INT4 quantization (GPTQ/AWQ)
- [ ] GPU sampling kernel
- [ ] Structured output (JSON Schema / Regex FSM)
- [ ] Prometheus metrics + health endpoints
- [ ] Forge extension API
- [ ] Speculative decoding (independent model + N-gram)

### Phase 3: Scaling

- [ ] Single-node tensor parallelism (NCCL)
- [ ] Single-node PD disaggregation (NVLink KV transfer)
- [ ] Multi-node PD disaggregation (RDMA KV transfer)
- [ ] Multi-node tensor parallelism
- [ ] Expert parallelism (DeepSeek MoE)
- [ ] Pipeline parallelism

### Phase 4: Multi-Backend & Edge

- [ ] Metal backend (Apple Silicon)
- [ ] CPU backend (edge deployment)
- [ ] ROCm backend (AMD GPUs)
- [ ] Minimal binary compilation (feature flags)
- [ ] ARM/RISC-V edge optimization

## 12. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Rust | Memory safety, zero-cost abstractions, single binary |
| HTTP framework | axum | Async, performant, Rust ecosystem standard |
| Async runtime | tokio | Industry standard |
| CUDA interaction | cudarc + custom C++ FFI | cudarc for driver API, C++ for optimized kernels |
| Attention kernels | FlashAttention / FlashInfer | Battle-tested, best-in-class performance |
| Tokenizer | tokenizers crate | HuggingFace official Rust implementation |
| Chat template | minijinja | Pure Rust Jinja2, compatible with HF templates |
| Serialization | serde + serde_json | Rust standard |
| Weight format | safetensors crate + custom GGUF parser | Native Rust support |
| Metrics | prometheus crate | Standard observability |
| Communication | NCCL (FFI) + custom transport | Proven collective comm library |
