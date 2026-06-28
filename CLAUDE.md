# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Forge is a from-scratch LLM inference server in Rust with CUDA acceleration: OpenAI-compatible `/v1/chat/completions`, continuous batching with chunked prefill, paged KV cache, n-gram speculative decoding, JSON-Schema / regex constrained decoding, and a Llama-family model implementation. Edition 2024, stable toolchain.

## Common Commands

```bash
# Build
cargo build --workspace                                # debug, CUDA on by default
cargo build --release                                  # release, CUDA on
cargo build --release --no-default-features            # CPU-only (no CUDA toolchain needed)
cargo check --workspace                                # fast type-check pass

# Test
cargo test --workspace                                 # full unit + integration suite
cargo test -p forge-runtime                            # tests for one crate
cargo test -p forge-runtime sampling::                 # tests matching a path/module prefix
cargo test -p forge-runtime sampling::greedy -- --nocapture  # single test, see stdout
cargo test -p forge-server --test test_e2e             # an integration test file

# Run
cargo run --release -p forge-server -- --model-path /path/to/model --port 8080
cargo run --release -p forge-server -- --model-path /path/to/model --backend cpu --kv-cache naive
RUST_LOG=debug cargo run --release -p forge-server -- --model-path /path/to/model

# E2E + benchmark (need a downloaded model on disk)
bash scripts/test_server.sh /path/to/model [port]
bash scripts/benchmark.sh /path/to/model [num_requests] [max_tokens] [port]
```

See `docs/CONTRIB.md` for the full flag list and `docs/RUNBOOK.md` for ops/troubleshooting (OOM, prefill rejection, CUDA build errors).

## Architecture (big picture)

The workspace is layered top-to-bottom; each layer only depends on the ones below it.

```
forge-server   HTTP/SSE (axum) → tokenize → ChatTemplate → push EngineRequest into mpsc
forge-runtime  Engine loop: drain → schedule → forward → sample → emit EngineEvent
forge-scheduler ContinuousBatchingScheduler: FCFS, cache-aware, chunked prefill
forge-models/forge-transformer  Decoder-transformer (Llama / Qwen2 / Qwen3 / Mistral) generic over Backend; one parameterized impl, optional QKV bias + QK-norm detected from weights
forge-kvcache  Naive (CPU vectors) and Paged (GPU block manager) implementations
forge-backend-cuda / forge-backend-cpu  Tensor compute (Backend trait)
forge-kernels  CUDA C++ kernel source strings, compiled to PTX via NVRTC at backend init
forge-flash   FlashAttention v2 C++ FFI (vendored, built via build.rs + cc)
forge-loader  SafeTensors + GGUF (Q4_K_M, Q8_0) weight loading
forge-core    Backend / Tensor / Model / KvCache / Scheduler traits + shared types
```

Key request flow (`docs/codemaps/architecture.md` has the full diagram):

```
POST /v1/chat/completions
  → api::openai::chat_completions (tokenize, build SamplingParams, optional FSM)
  → Engine::request_tx.send(EngineRequest)
  → Engine::run loop, per scheduling step:
      drain_requests → scheduler.enqueue
      scheduler.schedule(cache_usage) → ScheduleBatch{ prefill_seqs, decode_seqs, rejected }
      kv_cache.allocate(prefill seqs)
      per-seq: model.forward → backend.copy_to_host_f32 → sampler.sample_with_constraint
                → scheduler.append_token → FSM advance → stop check → emit EngineEvent
  → SSE stream (incremental UTF-8-safe decode) or buffered response
```

### Trait surfaces to know

- `Backend` (`forge-core/src/backend.rs`): dual F32/F16 paths (`matmul` + `matmul_f16`, `rms_norm` + `rms_norm_f16`, etc.). Implementors must support both; add new ops to all three of `Backend`, `CudaBackend`, `CpuBackend`.
- `Tensor` (`forge-core/src/tensor.rs`): shape/dtype/reshape/slice/contiguous. `CudaTensor` stores `CudaSlice<u8>` (type-erased) — use the typed accessors (`f32_slice`, `f16_slice`, `f16_slice_mut`, `bf16_slice_mut`).
- `Model` (`forge-core/src/model.rs`): `forward(&ModelInput, &mut dyn KvCache<T=Self::T>) -> ModelOutput<T>`; logits returned as `[batch * seq_len, vocab_size]`.
- `KvCache` (`forge-core/src/kvcache.rs`): `allocate / append / get_kv / get_block_table / get_seq_len / free / usage / can_allocate`. The scheduler reads `CacheUsage` to make admission decisions.
- `Scheduler` (`forge-core/src/scheduler.rs`): `enqueue / cancel / schedule / append_token / finish / get_generated_tokens`. `ScheduleBatch.rejected_seq_ids` is how prompts that exceed `max_prefill_tokens` come back.
- `FsmConstraint` (`forge-runtime/src/constraints/fsm.rs`): JSON Schema and regex both compile down through `regex-automata` DFAs into a `TokenFsmIndex` keyed on the model's vocabulary. Constraints zero out disallowed-token logits before sampling.

### CUDA kernel pipeline

`forge-kernels` exposes kernel source code as `&str` constants per family (elementwise, norm, positional, memory, attention). `CudaBackend::new()` concatenates them and compiles to PTX via NVRTC. To add a kernel: drop the CUDA source into the appropriate `forge-kernels/src/*.rs` module, expose it from `lib.rs`, include it in the backend's PTX bundle, then wire the launch in `forge-backend-cuda`.

`forge-flash` is the FA2 path: vendored `flash_attn` + `cutlass` under `csrc/`, built by `build.rs` (the `cc` crate). The CUDA attention impl dispatches to FA2 when shape constraints allow and falls back to the naive kernel otherwise — keep both paths working when touching attention.

## Conventions worth following

- Commits use conventional prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `perf:`, `docs:`, `chore:`); recent history is consistent — match it.
- `cargo fmt` and `cargo clippy` before committing.
- When you add a Backend op, add the F32 *and* F16 variants; the Llama model and many tests exercise both.
- `Engine::with_decode_fn` enables `stop_strings` checking — the server wires it up in `main.rs`; engine tests that need stop-string behavior must too.
- Long prompts beyond `--max-prefill-tokens` are rejected, not truncated — the scheduler surfaces them via `rejected_seq_ids` and the engine emits an error event. Don't silently truncate.

## Design and plan docs

- `docs/codemaps/{architecture,runtime,backend,data}.md` — current code map (per-crate file/type index, freshness-dated).
- `docs/plans/` — design + implementation plans, dated. Most recent: Flash Attention v2 (`2026-02-23-*`) and Phase 2 perf (`2026-02-22-*`).
- `docs/CONTRIB.md`, `docs/RUNBOOK.md` — contributor setup and operational guidance.
