# pegainfer vs forge — Comparison

**Date:** 2026-05-24
**Subject repo:** `github.com/xiaguan/pegainfer` (~339 stars, 9 contributors, default branch `main`)
**Method:** code-spelunked the public repo + workspace `Cargo.toml`s + `build.rs` files; cross-checked against forge at `feat/phase1-mvp`.

## 1. pegainfer profile

- **Scale**: ~84 K Rust LOC (~5× forge's ~18 K), ~19 K self-written CUDA C++/CUH, ~10 K Python for Triton / TileLang AOT kernel generation.
- **Workspace**: 30 crates, including a 14-crate `pegainfer-comm/*` subtree for EP all-to-all over RDMA. Per-model crate boundary — each architecture owns its own `scheduler`, `executor`, `kernel_plan`, `batch_decode`, `prefill`, `unified_forward` (see `pegainfer-qwen3-4b/src/`).
- **Models**: Qwen3-4B/8B, Qwen3.5 (hybrid linear + full attn), DeepSeek-V2-Lite, DeepSeek-V4-Flash (MoE + MLA + FP8/FP4 sparse attention), Kimi-K2. Each feature-gated.
- **Multi-GPU**: NCCL via cudarc, plus custom PPLX-style EP all-to-all over IB Verbs + GDRCopy. MP8 DSv4 across `cuda:0..7`, DP/EP for Kimi.
- **Runtime quantization**: Marlin W4/INT4 (Kimi), FP8/FP4 TileLang-generated (DSv4).
- **Activity**: daily commits, ~9 contributors. forge: 122 commits, single author.

## 2. CUDA integration: pegainfer is NOT pure Rust

Despite the README's "Pure Rust + CUDA" tagline, the actual story:

- `pegainfer-kernels/build.rs` shells out to **`nvcc`** to compile 28+ `.cu` files under `pegainfer-kernels/csrc/` at every build, with a parallel job pool gated by `PEGAINFER_NVCC_JOBS`. Many files are lifted from vLLM (`kimi_k2/vllm_marlin/*`).
- Build-time **requires Python + Triton** (`PEGAINFER_TRITON_PYTHON`) to AOT-compile `tools/triton/{gated_delta_rule_chunkwise_kernels, flash_attention_prefill_hd256_kernel}.py` into PTX, then links the PTX into the binary.
- `--features deepseek-v4` additionally requires **Python + TileLang** (`PEGAINFER_TILELANG_PYTHON`) to generate FP8/FP4 CUDA via `tools/tilelang/deepseek_v4/generate.py`.
- Vendored `pegainfer-kernels/third_party/flashinfer/` C++ headers (the FlashInfer paged-attention kernel suite).
- **`bindgen 0.72.1`** is a workspace dep used by `pegainfer-cupti/build.rs` and `cuda-sys` / `cudart-sys` / `libibverbs-sys` / `gdrapi-sys` (the EP all-to-all FFI).
- One **LibTorch cxx bridge** in `pegainfer-comm/crates/pegainfer-comm-torch-lib/` for benchmark interop; feature-gated (`hw-cuda`, default off). Inference runtime does not link it.
- The HTTP frontend is **vLLM's**: `vllm-engine-core-client`, `vllm-server`, `vllm-text`, `vllm-tokenizer` pulled from `github.com/vllm-project/vllm@65b7a812`. Pegainfer doesn't own its OpenAI surface — it speaks vLLM's ZeroMQ MQ protocol.

**Forge by contrast** is genuinely close to pure-Rust at the runtime build/dep level:

- `cudarc 0.17.8` (CUDA Driver API bindings), no `bindgen`, no `cuda-sys`.
- **NVRTC at backend init** for all custom kernels — no `nvcc` step in `cargo build`.
- One `cc::Build` in `forge-flash/build.rs` for vendored FlashAttention v2 (~30 min nvcc compile on first build, cached after).
- No Python. No PyTorch. No vLLM. No NCCL.
- Own axum-based OpenAI-compatible HTTP layer.

The trade-off is exactly what you'd expect: forge `cargo build` works on a fresh box with just the CUDA Toolkit; pegainfer needs CUDA Toolkit + `uv` + Python + `pip install torch+triton` + optional TileLang + `CUDA_HOME` / `PEGAINFER_TRITON_PYTHON` env vars. The cost is much smaller kernel coverage on forge's side.

## 3. Axis-by-axis verdict

🟢 = forge ahead · 🟡 = roughly equivalent · 🔴 = forge behind

| Axis | Verdict | Notes |
|---|---|---|
| Workspace layering | 🟡 | forge by concern (core/runtime/scheduler/kvcache/backend), pegainfer by model crate + shared kernels |
| CUDA integration purity | 🟢 | see Section 2 |
| Build complexity | 🟢 | `cargo build` vs nvcc + Python + uv + env vars |
| Custom kernel surface | 🔴 | pegainfer ~20 families vs forge ~7 |
| KV cache design | 🟡 | both device-paged. pegainfer's `KvPool::padding_permit` (`pegainfer-core/src/kv_pool.rs:64`) is exactly the bucket-padding trick forge's Task 5 needs |
| Attention kernel coverage | 🔴 | pegainfer: FlashInfer paged, hd128/hd256 prefill, gated-delta-rule linear, DSv4 sparse + indexer, Kimi MLA. forge: FA2 + paged + naive, F32/F16 |
| Model coverage | 🔴 | 1 (Llama) vs 4 architectures incl. MoE/MLA/linear |
| Quantization runtime | 🔴 | Marlin W4 / FP8 / FP4 vs none (GGUF load-time only) |
| Multi-GPU / TP / EP | 🔴 | 14-crate `pegainfer-comm` over IB Verbs+GDRCopy vs none |
| Scheduler | 🟡 | forge has chunked prefill admission gate; pegainfer has CUDA-graph bucket planning |
| FSM constrained decoding | 🟢 | forge: JSON Schema + regex via `regex-automata` DFA. pegainfer: temp/top-k/top-p only (18-line sampler) |
| Speculative decoding | 🟢 | forge has n-gram; pegainfer has none |
| HTTP server / observability | 🔴 | pegainfer reuses vLLM frontend + nvtx + fastrace + jemalloc + `bench_serving` binary |
| Test discipline | 🔴 | pegainfer: HF-output parity per model + criterion benches + vllm-baseline crate. forge: ~18 test files, no HF parity |
| Documentation | 🟡 | both maintained; different shapes |
| Activity | 🔴 | daily / 9 contributors vs single author |

## 4. What forge could borrow

- **Per-model executor crate boundary** (`pegainfer-qwen3-4b/src/{scheduler,executor,kernel_plan}.rs`) — adopt this when adding Qwen3, don't dump it into `forge-model-llama`.
- **`KvPool::padding_permit`** (`pegainfer-core/src/kv_pool.rs:64`) — concrete pattern for making CUDA-Graph capture work across variable batch sizes. **This is exactly what's blocking forge's Task 5** (persistent buffers for graph capture).
- **`tikv-jemallocator` global allocator** (`pegainfer-server/src/main.rs:11`) — drop-in throughput win.
- **AOT Triton kernel pattern** (`pegainfer-kernels/build.rs` + `tools/triton/`) — if forge ever wants linear attention, this is the cleanest precedent. But note it pulls Python into the build.
- **`kernel-call-trace` feature flag** — ship-able diagnostics gated by a feature.

## 5. What pegainfer could borrow from forge

- FSM constrained decoding (`forge-runtime/src/constraints/fsm.rs`) — pegainfer's 18-line sampler is a glaring prod-readiness gap.
- N-gram speculative (`forge-runtime/src/speculative/`) — orthogonal to model, ~free throughput.
- Chunked prefill admission gate (`forge-scheduler/src/continuous.rs` → `rejected_seq_ids`) — pegainfer's schedulers don't bound prefill cost per step.
- `Backend` trait with F32/F16 dual paths — pegainfer is BF16-only, makes CPU testing impossible.

## 6. Bottom line

**Pegainfer is ~5× ahead in scope** (4 model families incl. MoE/MLA/linear, multi-GPU EP, runtime quantization, daily commits from a real team).
**Forge is ahead in runtime feature polish** (FSM-constrained decoding, n-gram speculative, chunked prefill, F32/F16 dtype parity) and **genuinely cleaner build story** (`cudarc` + NVRTC + one cc-built FA2, no Python).

**Highest-leverage gap to close**: add a second model architecture — Qwen3 is the obvious pick — and adopt pegainfer's per-model crate boundary at the same time. Estimated 2–3 weeks for one engineer given forge's existing `Backend` trait + `PagedKvCache`. Multi-GPU TP/EP is a bigger lift (pegainfer needed 14 crates + IB Verbs/GDRCopy bindings) and should wait until the dual-DGX-Spark link is live.

**ROADMAP.md doesn't need a direction change**, but Phase A priorities should be re-ordered: finish CUDA Graphs (in progress) → **add Qwen3 + per-model crate split** → W4A16 quantization → multi-GPU.
