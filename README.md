# Forge

A from-scratch LLM inference server written in Rust with CUDA acceleration.

## Features

- **OpenAI-compatible API** — `/v1/chat/completions` with streaming (SSE) and non-streaming modes
- **Continuous batching** with chunked prefill for stable decode latency
- **Structured output** — JSON Schema → regex → DFA → token-level FSM constrained generation
- **N-gram speculative decoding** with adaptive draft length
- **Paged KV cache** with block-level memory management
- **GGUF model loading** with Q4_K_M and Q8_0 dequantization
- **CUDA + CPU backends** — feature-gated CUDA for CPU-only builds
- **Llama model family** support (Llama 2/3, TinyLlama, etc.)

## Architecture

```
forge-core          Shared traits (Backend, Model, Scheduler, KvCache), error types
forge-backend       Backend trait abstraction
forge-backend-cpu   CPU backend (OpenBLAS)
forge-backend-cuda  CUDA backend (cuBLAS + custom kernels)
forge-kernels       CUDA kernels (RMSNorm, SiLU, Softmax, RoPE, Embedding)
forge-kvcache       Naive and paged KV cache implementations
forge-loader        SafeTensors + GGUF model loaders
forge-model-llama   Llama model (attention, FFN, RoPE, GQA)
forge-runtime       Engine loop, sampling, FSM constraints, speculative decoding
forge-scheduler     Continuous batching scheduler
forge-server        Axum HTTP server, OpenAI-compatible API
forge-transport     In-process channel transport
forge-quantize      Quantization scaffolding
```

## Quick Start

### Prerequisites

- Rust stable (see `rust-toolchain.toml`)
- CUDA toolkit 12.x+ (for GPU backend)
- A Llama-family model in SafeTensors or GGUF format

### Build

```bash
# With CUDA (default)
cargo build --release

# CPU-only
cargo build --release --no-default-features
```

### Run

```bash
cargo run --release -- --model-path /path/to/model --port 8080
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | required | Path to SafeTensors model directory |
| `--port` | 8080 | HTTP server port |
| `--backend` | cuda | Backend: `cuda` or `cpu` |
| `--kv-cache` | paged | KV cache: `paged` or `naive` |
| `--max-batch-size` | 256 | Max concurrent sequences |
| `--max-prefill-tokens` | 4096 | Max prefill tokens per step |
| `--device` | 0 | CUDA device ordinal |
| `--block-size` | 16 | Paged cache block size (tokens) |
| `--num-blocks` | 2048 | Total KV cache blocks |

## API

### Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": false
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Structured Output (JSON Schema)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "json_schema": {
      "type": "object",
      "properties": {
        "answer": {"type": "integer"},
        "explanation": {"type": "string"}
      },
      "required": ["answer", "explanation"]
    }
  }'
```

### Other Endpoints

- `GET /v1/models` — List available models
- `GET /forge/v1/health` — Health check

## Testing

```bash
# Unit tests (149 tests)
cargo test --workspace

# Integration test with a model
bash scripts/test_server.sh /path/to/model

# Benchmark (TTFT, ITL, throughput)
bash scripts/benchmark.sh /path/to/model
```

## License

Apache-2.0
