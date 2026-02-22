# Contributing to Forge

## Prerequisites

- **Rust** stable (managed via `rust-toolchain.toml`)
- **CUDA toolkit** 12.x+ (for `forge-backend-cuda` and `forge-kernels`)
- **Python 3** (for integration test and benchmark scripts)
- **curl** (for E2E tests)

## Development Setup

```bash
git clone <repo-url>
cd forge

# Build the workspace
cargo build --workspace

# Run all tests
cargo test --workspace
```

## Project Structure

Forge is a Rust workspace with 13 crates:

| Crate | Purpose |
|-------|---------|
| `forge-core` | Core traits, types, error definitions |
| `forge-server` | HTTP API (axum), binary entry point |
| `forge-backend` | Backend abstraction trait |
| `forge-backend-cuda` | CUDA backend (cudarc 0.17, FP32 + FP16) |
| `forge-backend-cpu` | CPU backend (OpenBLAS, for testing) |
| `forge-runtime` | Engine runtime loop, sampling, FSM constraints |
| `forge-scheduler` | Continuous batching scheduler |
| `forge-kvcache` | KV cache (naive + paged block manager) |
| `forge-models/forge-model-llama` | Llama model implementation |
| `forge-loader` | SafeTensors weight loader (F32, F16, BF16→F16) |
| `forge-kernels` | CUDA C++ kernels + FFI bindings |
| `forge-transport` | Communication abstraction (in-process, future: gRPC) |
| `forge-quantize` | Quantization support (Phase 2+) |

## Available Scripts

| Script | Usage | Description |
|--------|-------|-------------|
| `scripts/test_server.sh` | `bash scripts/test_server.sh /path/to/model [port]` | E2E integration test: health, models, non-streaming, streaming |
| `scripts/benchmark.sh` | `bash scripts/benchmark.sh /path/to/model [num_requests] [max_tokens] [port]` | Performance benchmark: TTFT, ITL, throughput |

## Build Commands

```bash
# Check workspace (fast, no codegen)
cargo check --workspace

# Build release
cargo build --release --workspace

# Run tests
cargo test --workspace

# Run the server (CUDA backend, paged KV cache)
cargo run --release -p forge-server -- --model-path /path/to/model --port 8080

# Run with CPU backend
cargo run --release -p forge-server -- --model-path /path/to/model --backend cpu --kv-cache naive
```

## Server CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to model directory (SafeTensors + config.json + tokenizer.json) |
| `--port` | `8080` | HTTP listen port |
| `--backend` | `cuda` | Backend: `cuda` or `cpu` |
| `--device` | `0` | CUDA device ordinal |
| `--kv-cache` | `paged` | KV cache type: `paged` or `naive` |
| `--block-size` | `16` | Tokens per KV cache block (paged only) |
| `--num-blocks` | `2048` | Total KV cache blocks (paged only) |
| `--max-batch-size` | `256` | Max sequences in a batch |
| `--max-prefill-tokens` | `4096` | Max prefill tokens per scheduling step |

## Testing

### Unit Tests

```bash
cargo test --workspace
```

### Integration Tests (requires a model)

```bash
bash scripts/test_server.sh /path/to/tinyllama-1.1b
```

### Benchmarks (requires a model)

```bash
bash scripts/benchmark.sh /path/to/model 10 128
```

## Code Style

- Edition 2024, stable toolchain
- `cargo fmt` and `cargo clippy` before committing
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `perf:`, `docs:`, `chore:`

## Architecture

See [docs/plans/2026-02-19-forge-design.md](plans/2026-02-19-forge-design.md) for the full architecture document.

See [docs/plans/2026-02-19-forge-phase1-plan.md](plans/2026-02-19-forge-phase1-plan.md) for the Phase 1 MVP implementation plan.
