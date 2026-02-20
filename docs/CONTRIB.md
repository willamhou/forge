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

Forge is a Rust workspace with 12 crates:

| Crate | Purpose |
|-------|---------|
| `forge-core` | Core traits, types, error definitions |
| `forge-server` | HTTP API (axum), binary entry point |
| `forge-backend` | Backend abstraction trait |
| `forge-backend-cuda` | CUDA backend (cudarc 0.17) |
| `forge-runtime` | Engine runtime loop, forward pass orchestration |
| `forge-scheduler` | Continuous batching scheduler |
| `forge-kvcache` | PagedAttention block manager |
| `forge-models/forge-model-llama` | Llama model implementation |
| `forge-loader` | SafeTensors/GGUF weight loader |
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

# Build with FlashAttention FFI (requires libflash_attn)
cargo build -p forge-backend-cuda --features flash-attn

# Run tests
cargo test --workspace

# Run the server
cargo run --release -- serve --model /path/to/model --port 8080
```

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

## Feature Flags

| Flag | Crate | Description |
|------|-------|-------------|
| `flash-attn` | `forge-backend-cuda` | Enable FlashAttention FFI (requires library headers + linked `.so`) |

## Architecture

See [docs/plans/2026-02-19-forge-design.md](plans/2026-02-19-forge-design.md) for the full architecture document.

See [docs/plans/2026-02-19-forge-phase1-plan.md](plans/2026-02-19-forge-phase1-plan.md) for the Phase 1 MVP implementation plan.
