# CUDA Graphs for Batched Decode — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> Design doc: [2026-05-22-cuda-graphs-design.md](2026-05-22-cuda-graphs-design.md)

**Goal:** Capture decode forward as CUDA Graph, replay on every step that fits a batch-size bucket. Target ≥ 15% TPOT improvement at batch_size=8.

**Strategy:** Bottom-up. Land the prerequisites (stream fix, paged attention kernel) before touching graph capture. Every task is independently shippable, gated behind `--cuda-graph` (default off until Task 8 flips it on).

**Order of operations:** 0 → 1 (spike, then continue or pivot) → 2 → **2.5** → 3 → 4 → 5 → 6 → 7 → 8. Task 2.5 was added 2026-05-23 after the Task 1 spike (commit 6ecf839) surfaced that Backend's per-op `alloc_zeros` makes capture impossible regardless of the rest of the plan.

---

## Task 0: Stream plumbing fix

The FA2 wrapper currently passes `stream_ptr = 0` (default stream). For CUDA Graph capture to include FA2 ops, FA2 must run on the same stream we capture on. This is a one-line semantic fix gated only by exposing `CudaStream` as a raw `CUstream` (cudarc 0.17 supports it via `cu_stream()`).

**Files:**
- Modify: `forge-backend/forge-backend-cuda/src/flash_attention.rs:109-115`
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — pub(crate) accessor for stream raw pointer if not already present
- Modify: `forge-flash/src/lib.rs` — confirm `stream_ptr: u64` is wired through

**Steps:**
1. Add `pub(crate) fn cu_stream(&self) -> u64` on `CudaBackend` returning the cudarc stream's raw `CUstream`.
2. In `flash_attention.rs`, replace `let stream_ptr: u64 = 0;` with `let stream_ptr = backend.cu_stream();`.
3. Run `cargo test -p forge-backend-cuda --features flash-attn` — all FA2 tests must still pass.
4. Commit: `fix: pass backend stream to FA2 (prereq for CUDA Graphs)`.

**Definition of done:** All existing FA2 tests pass; FA2 ops execute on `backend.stream`, not stream 0.

---

## Task 1: cudarc CUDA Graph API spike (1 day, hard time-box)

Before committing to the plan, confirm cudarc 0.17 exposes everything we need. If it doesn't, this task pivots the design (raw FFI fallback or wait for cudarc upgrade).

**What to verify:**
- `stream.begin_capture(mode)` + `stream.end_capture(flags) -> Result<Option<CudaGraph>>` works (cudarc 0.17.8 folds instantiation into `end_capture` — no separate `CudaGraphExec` type)
- `CudaGraph::launch()` re-runs the captured kernels (no stream arg — the graph remembers the stream it was captured on)
- A captured graph that includes a cuBLAS GEMM (via `CudaBlas` on the same stream) actually replays the GEMM
- Persistent device buffer + memcpy from host *before* graph launch updates the contents seen by the next replay

**Deliverable:** `forge-backend-cuda/examples/graph_spike.rs` — a minimal program that captures `add → mul → norm` on a 1024-elem buffer, replays 100 times, and verifies output matches direct launches. Discard the example after the plan completes (or fold into tests).

**Decision gates:**
- ✅ All four checks pass → proceed to Task 2
- ⚠️ cudarc lacks one item but raw FFI is accessible via `cudarc::driver::sys::*` → proceed with raw FFI shim
- ❌ Fundamental gap → stop, write a 1-pager on alternatives, return to design

**No commit unless decision gate ✅ or ⚠️.** If the spike fails, file the example as `examples/graph_spike.rs.disabled` and update this plan.

**Findings absorbed from the actual spike (2026-05-23, commit 6ecf839):** decision gate was ✅ GREEN with two non-obvious gotchas that Task 2/3 must apply:

1. `ctx.default_stream()` returns the NULL/legacy stream (literally `cu_stream: null_mut()`); CUDA Graphs **cannot capture the NULL stream**. Task 2's first change is `CudaBackend::new` → `ctx.new_stream()`. Every call site that depends on default-stream's implicit device-wide sync needs an explicit `synchronize()` audit.
2. cudarc's safe `launch_builder.arg(&mut CudaSlice)` auto-records `cuStreamWaitEvent` deps in multi-stream mode. These reference events from outside the capture region and trip `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`. Call `unsafe { ctx.disable_event_tracking() }` before any capture-bound op (per-context, permanent — multi-stream callers lose cudarc's auto-sync and must order ops via the graph topology themselves).

Additional issues found by review (Claude + Codex) that the spike does not in fact cover and Task 2/3 must address rather than inherit the GREEN gate uncritically: (a) the spike pre-allocates all buffers outside capture but production `Backend` ops call `alloc_zeros` per op — cudaMalloc-in-capture is illegal, so Task 2.5 below adds a persistent-output-buffer migration; (b) the spike uses `CU_STREAM_CAPTURE_MODE_RELAXED`, not `GLOBAL`, so capture-mode strictness is not yet validated; (c) FA2 (`forge_flash::flash_fwd`) is not captured at all in the spike and may issue its own cudaMalloc/cudaEventRecord; Task 2 should add a FA2-in-capture sub-spike before relying on attention being graph-compatible.

---

## Task 2: Paged attention kernel (prerequisite for graph reuse)

Replace the current "gather K/V then run FA2" path with a kernel that reads `(kv_pool, block_tables, kv_lens)` directly. Without this, captured graphs can't replay because gathered K/V live at fresh device addresses each step.

**Files:**
- Create: `forge-backend/forge-backend-cuda/src/paged_attention.rs`
- Create: `forge-kernels/src/paged_attention.rs` — only if we don't use FA2 varlen
- Modify: `forge-core/src/backend.rs` — add `paged_attention` trait method
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — kernel function handle + `paged_attention` impl
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs` — naive default impl over already-gathered K/V (no perf concern, just correctness)
- Modify: `forge-models/forge-model-llama/src/attention.rs` — switch decode path to paged_attention
- Test: `forge-backend/forge-backend-cuda/tests/test_paged_attention.rs`

**Implementation choices (pick one — design doc recommends A):**

- **A: Wrap FA2 varlen with a gather inside the kernel.** Build `cu_seqlens` + a packed `K`/`V` from blocks in one fused gather kernel, then call FA2 varlen. Gather kernel is in-graph, packed buffers are persistent device memory. Reuses FA2 (no new attention math).
- **B: Write a from-scratch paged attention kernel.** Decode-only (q_len=1), GQA-aware, reads blocks directly. ~300 LOC CUDA. More invasive but no FA2 dependency for paged path.

Default to A. B is the fallback if FA2 varlen turns out to require contiguous KV.

**Steps:**
1. Add `Backend::paged_attention` trait method:
   ```rust
   fn paged_attention(
       &self,
       q: &Self::Tensor,                  // [batch, num_heads * head_dim] (decode: q_len=1)
       kv_pool: &Self::Tensor,            // [total_blocks, block_size, num_kv_heads * head_dim * 2]
       block_tables: &Self::Tensor,       // [batch, max_blocks_per_seq] i32
       kv_lens: &Self::Tensor,            // [batch] i32
       num_heads: usize,
       num_kv_heads: usize,
       head_dim: usize,
       scale: f32,
   ) -> Result<Self::Tensor>;
   ```
2. CPU default impl: gather to contiguous K/V, call existing naive attention. Slow but correct.
3. CUDA impl (option A): gather kernel + FA2 varlen. Gather kernel signature in `forge-kernels/src/memory.rs` or new `paged_gather.rs`.
4. Wire `LlamaAttention::forward` (decode branch) through `backend.paged_attention(...)`. Prefill stays on existing `multi_head_attention`.
5. Test: deterministic random weights, compare paged-attention output against the existing gather-then-FA2 path, tolerance `1e-3` (FP16/BF16).
6. Run full workspace tests + E2E.
7. Commit: `feat: paged attention kernel reading block tables directly`.

**Definition of done:** Decode goes through `paged_attention`; logits match the current path within tolerance; benchmark shows no regression.

---

## Task 2.5: Capture-safe Backend ops (persistent output pool + cuBLAS workspace)

`CudaMalloc` is illegal during stream capture. Today every `Backend` op (`matmul`, `rms_norm`, `add`, `mul`, ~60 sites in `forge-backend-cuda/src/backend.rs`) ends with `self.stream.alloc_zeros::<T>(numel)` to produce its output tensor. Wrapping decode forward in `begin_capture/end_capture` therefore fails at the first op, regardless of whether Task 3 / 4 are correct.

Likewise, cuBLAS GEMM allocates a per-handle workspace lazily on first call. The Task 1 spike got away with capturing a GEMM only because Stage A had pre-warmed the workspace — production capture-first runs would fail with `STREAM_CAPTURE_UNSUPPORTED` on a `cudaMalloc` inside cuBLAS.

**Files:**
- Modify: `forge-core/src/backend.rs` — extend `Backend` trait so hot ops accept a caller-owned `out` tensor (e.g. add `matmul_into`, `add_into`, etc., or a single `with_output` adapter). Keep the existing alloc-returning variants for code outside capture; mark them `#[deprecated]` to surface remaining alloc-in-hot-path call sites at compile time.
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — add `cublasSetWorkspace` on a backend-owned `CudaSlice<u8>` (size from `cublasGetCudartVersion`-keyed table; 32 MiB is a safe default for modern Ampere/Blackwell) in `CudaBackend::new`. Implement the `_into` variants by skipping the trailing `alloc_zeros` and writing directly to the caller's tensor.
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs` — naive `_into` impls (just a copy into the out tensor). No perf concern.
- Modify: `forge-models/forge-model-llama/src/{lib,attention,mlp}.rs` — migrate decode-path call sites to `_into` variants, threading through a per-layer set of persistent intermediates owned by `LlamaForward`. Prefill / one-shot paths can stay on the alloc-returning variants for now.
- Modify: `forge-backend/forge-backend-cuda/src/flash_attention.rs` — same treatment for `multi_head_attention` and FA2 output buffer. The FA2 C++ side may need a workspace patch; investigate as part of this task.
- Test: `forge-backend/forge-backend-cuda/tests/test_capture_safe_ops.rs` — wrap each migrated op inside `begin_capture/end_capture` and assert `end_capture` returns `Ok(Some(_))` with no allocator panic.

**Steps:**
1. Land `cublasSetWorkspace` in `CudaBackend::new` (single commit, no semantics change for non-capture path).
2. Add the `_into` trait methods and CUDA/CPU impls for the **decode** hot set: `matmul`, `matmul_f16`, `rms_norm{_f16}`, `add{_f16}`, `mul{_f16}`, `softmax{_f16}`, `silu{_f16}`, `rope{_f16}`, `multi_head_attention`. Keep prefill ops on the legacy alloc-returning path.
3. Migrate `forge-model-llama` decode forward to use `_into` variants. `LlamaForward` (new or extended) owns the per-layer persistent activation buffers; allocated once at first decode step.
4. Per-op capture test for each migrated op.
5. Commit: `feat: capture-safe Backend ops (persistent output buffers + cuBLAS workspace)`.

**Definition of done:** every Backend op the decode hot path touches can be issued inside `begin_capture/end_capture` on a real `CudaBackend` without invalidating the capture or hitting `STREAM_CAPTURE_UNSUPPORTED`. Test file demonstrates this op-by-op.

**Risk:** medium — invasive to the model code, but the migration is mechanical and each op is testable in isolation. Estimate 3–4 days. **This task gates Task 3 and Task 4 entirely; doing Task 3 first would just hit `cudaMalloc-during-capture` and have to be rolled back.**

---

## Task 3: `CudaGraphCache` infrastructure

Capture/replay framework. No engine integration yet — purely a library.

**Files:**
- Create: `forge-backend/forge-backend-cuda/src/cuda_graph.rs`
- Modify: `forge-backend/forge-backend-cuda/src/lib.rs` — `pub mod cuda_graph`
- Modify: `forge-backend/forge-backend-cuda/Cargo.toml` — no new deps

**API sketch (aligned to cudarc 0.17.8's actual surface — `CudaGraph` already holds both the captured graph and its instantiated exec; no separate `CudaGraphExec` type):**

```rust
pub struct CudaGraphCache {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    graphs: HashMap<u32, CapturedGraph>,
}

struct CapturedGraph {
    graph: CudaGraph,    // cudarc 0.17.8 — already owns cu_graph + cu_graph_exec, RAII drop
    // Persistent input/output buffers, owned by the cache.
    // Populated in Task 5; placeholder here.
}

impl CudaGraphCache {
    pub fn new(device: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self;

    pub fn has(&self, bucket: u32) -> bool;

    /// Run the closure; if no graph exists for this bucket, capture it.
    pub fn run_or_capture<F: FnOnce() -> Result<()>>(
        &mut self, bucket: u32, fwd: F
    ) -> Result<()>;

    /// Replay only (panics if not present). Used in benchmarks / tests.
    pub fn replay(&self, bucket: u32) -> Result<()>;
}
```

**Steps:**
1. Implement `run_or_capture`: on cache miss, `stream.begin_capture(CU_STREAM_CAPTURE_MODE_GLOBAL)` → run `fwd` → `stream.end_capture(flags)?.ok_or(...)?` (returns `CudaGraph`, already instantiated) → insert into `graphs` → `graph.launch()`. On hit, just `graph.launch()`. **Precondition** (per Task 1 findings): `ctx.disable_event_tracking()` and `ctx.new_stream()` must already be in effect from `CudaBackend::new`; if not, capture invalidates. For the `flags` argument cudarc exposes no `_NONE` variant; pass `CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD` (a pure latency optimization, no semantic effect) or call `cudarc::driver::result::graph::instantiate` directly with `0u32`.
2. Implement `Drop` for `CapturedGraph` if extra fields need cleanup; the inner `CudaGraph` already RAII-releases cu_graph and cu_graph_exec (see cudarc-0.17.8/src/driver/safe/graph.rs:21).
3. Test (`tests/test_cuda_graph.rs`): capture a simple add+mul kernel sequence, replay 5 times, compare output vs uncaptured launches. Test with cuBLAS GEMM inside capture.
4. Commit: `feat: CudaGraphCache for per-bucket graph capture and replay`.

**Definition of done:** Unit tests pass; can capture and replay a sequence that includes cuBLAS GEMM + custom kernels.

---

## Task 4: `DecodeGraphRunner` and engine integration

Wire `process_decode_batch` to dispatch through the graph cache. Initially behind `--cuda-graph` flag, default `false` (so all existing tests stay on the uncaptured path).

**Files:**
- Create: `forge-runtime/src/decode_graph.rs`
- Modify: `forge-runtime/src/engine.rs` — call `DecodeGraphRunner` from `process_decode_batch`
- Modify: `forge-runtime/src/lib.rs` — export `DecodeGraphRunner`
- Modify: `forge-server/src/main.rs` — `--cuda-graph` CLI flag (default `false` in this task; default flips to `true` in Task 8 after perf gate passes)

**Steps:**
1. `DecodeGraphRunner` holds `CudaGraphCache` + bucket config `Vec<u32>`. (For backends without graph support — CPU — `DecodeGraphRunner` is a no-op shim.)
2. `engine.rs::process_decode_batch`: after `active_seqs` filter, call `runner.dispatch(active_seqs, |seqs| forward_pass(seqs))`. Inside `dispatch`:
   - If the smallest bucket `B ≥ active_seqs.len()` exists in config AND `active_seqs.len() == B` (exact match): call `cache.run_or_capture(B, || forward_pass(active_seqs))`.
   - Otherwise (non-bucket batch size, or larger than max bucket): **fall back to the uncaptured forward path** by invoking the closure directly with the real `active_seqs`. Log at `debug!` level once per (bucket-miss reason, seen-size) pair, then stay silent.
   - **Do not panic.** Continuous batching produces batch sizes 3, 5, 7, etc. all the time — a panic here would crash the server on ordinary traffic. Task 7 lifts this restriction by padding non-bucket sizes up to the nearest bucket.
3. Tests (`tests/test_decode_graph.rs`):
   - With `--cuda-graph=true`, run 5 decode steps at fixed batch_size=4 (exact bucket hit). Assert logits match `--cuda-graph=false` within `1e-4`.
   - With `--cuda-graph=true`, run 5 decode steps at batch_size=3 (non-bucket). Assert no panic, and logits match `--cuda-graph=false`. This guards the fallback path.
   - With `--cuda-graph=true`, run a mixed sequence of batch sizes (1, 3, 4, 5, 8) interleaved. Assert no panic across bucket misses and bucket hits.
4. Commit: `feat: DecodeGraphRunner integrating CudaGraphCache into engine`.

**Definition of done:** `cargo test --workspace` passes both with `--cuda-graph=true` and `--cuda-graph=false`. E2E test under `--cuda-graph=true` produces sensible output at batch_sizes ∈ {1, 3, 4, 5, 8}. **Server never panics on non-bucket batch sizes — they fall through to the uncaptured path.**

---

## Task 5: Persistent input/output buffers

The captured graph reads from fixed device addresses. This task allocates those buffers and wires input staging.

**Files:**
- Modify: `forge-runtime/src/decode_graph.rs` — `PersistentBuffers` per bucket
- Modify: `forge-runtime/src/engine.rs::build_batch_input` — write into persistent buffers instead of building fresh tensors
- Modify: `forge-models/forge-model-llama/src/lib.rs` — model accepts pre-allocated logits output buffer (already returns one; needs to *reuse* a persistent one when graph-capturing)

**Sizing the block-table buffer (critical):**

The captured graph reads the block table at a fixed device address with fixed shape `[batch, max_blocks_per_seq]`. `max_blocks_per_seq` must be the **per-sequence ceiling**, not the average — a single low-concurrency long-context request can legitimately own far more blocks than `num_blocks / max_batch`. Sizing by the average lets one such request truncate its own block table and read the wrong KV blocks.

Anchor `max_blocks_per_seq` to a **per-sequence KV ceiling**:

```
max_blocks_per_seq = ceil(max_seq_len / block_size)
```

where `max_seq_len` comes from a new CLI flag `--max-seq-len` (defaults to the model's `max_position_embeddings`). The scheduler must reject any prompt whose `prompt_tokens + max_tokens > max_seq_len` with a clear error; the graph dispatcher must `fail closed` (return an error / fall back to uncaptured) if a real sequence at dispatch time would need more than `max_blocks_per_seq` blocks. **Never silently truncate the block table.**

**Buffer set per bucket:**

| Buffer | Shape | Dtype | Notes |
|---|---|---|---|
| `token_ids_in` | `[batch]` | u32 | One token per seq (decode) |
| `positions_in` | `[batch]` | u32 | Per-seq position |
| `block_tables_in` | `[batch, max_blocks_per_seq]` | i32 | KV block indices, -1 = unused; `max_blocks_per_seq = ceil(max_seq_len / block_size)` |
| `kv_lens_in` | `[batch]` | i32 | Current KV length per seq |
| `logits_out` | `[batch, vocab_size]` | f32 | Read by host after replay |

**Files (delta vs Task 4):**
- Modify: `forge-runtime/src/decode_graph.rs` — `PersistentBuffers` per bucket
- Modify: `forge-runtime/src/engine.rs::build_batch_input` — write into persistent buffers instead of building fresh tensors
- Modify: `forge-models/forge-model-llama/src/lib.rs` — model accepts pre-allocated logits output buffer (already returns one; needs to *reuse* a persistent one when graph-capturing)
- Modify: `forge-server/src/main.rs` — add `--max-seq-len` flag (defaults to `config.max_position_embeddings`)
- Modify: `forge-scheduler/src/continuous.rs` — reject prompts where `prompt_tokens + sampling_params.max_tokens > max_seq_len`
- Modify: `forge-core/src/error.rs` — add `SequenceTooLong { needed_blocks, max_blocks }` variant for the fail-closed path

**Steps:**
1. Add `--max-seq-len` flag plumbed into `Engine` config. Default it to the loaded model's `max_position_embeddings`.
2. Plumb a per-seq KV ceiling through the scheduler so over-long requests are rejected at enqueue time (consistent with current `max_prefill_tokens` rejection pattern).
3. Allocate persistent buffers lazily on first capture; size `block_tables_in` as `[bucket, ceil(max_seq_len / block_size)]`.
4. `dispatch(active_seqs)`:
   - Before staging, assert `active_seqs.iter().all(|s| seq_blocks(s) <= max_blocks_per_seq)`. If any seq violates this, log `warn!` once and **fall back to the uncaptured path** for this step (do not truncate, do not panic).
   - H2D memcpy of `token_ids`, `positions`, `block_tables`, `kv_lens` *before* graph launch.
   - Launch graph.
   - D2H memcpy of `logits_out[:active_seqs.len()]`.
5. Adjust model forward signature so the same persistent `logits_out` is reused. May require a `Model::forward_into(input, kv_cache, out: &mut Self::T)` variant. Decide and document in the task PR.
6. Tests (`tests/test_decode_graph.rs`):
   - Steady-state allocation: 20 decode steps, no fresh device allocations (track via counter in `CudaTensor::new`).
   - Long-context fail-closed: construct a sequence needing `max_blocks_per_seq + 1` blocks, dispatch with `--cuda-graph=true`, assert the runner falls back to uncaptured (logs warning) and the sequence completes correctly. **Must not crash or return wrong logits.**
   - Scheduler-level rejection: submit a request with `prompt_tokens + max_tokens > max_seq_len`, assert a clear error returned at enqueue time.
7. Commit: `feat: persistent decode buffers for graph replay`.

**Definition of done:**
- No fresh device allocations during steady-state decode (verified by allocation counter).
- Block-table buffer is sized from `max_seq_len`, not `num_blocks / max_batch`.
- Over-long sequences are either rejected at enqueue or run via the uncaptured fallback — never via a truncated block table.

---

## Task 6: CLI surface & defaults

Make the feature configurable and documented. Default still off.

**Files:**
- Modify: `forge-server/src/main.rs` — flags
- Modify: `docs/RUNBOOK.md` — short section
- Modify: `README.md` — table row

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--cuda-graph` | `false` (flips in Task 8) | Enable CUDA Graph capture for decode |
| `--cuda-graph-buckets` | `1,2,4,8,16,32` | Comma-separated batch-size buckets |

**Auto-disable rules (documented + implemented):**
- `--backend cpu` → graph flag is a no-op (warn once)
- `--kv-cache naive` → graph flag is a no-op (warn once)

**Steps:**
1. Add `clap` flags in `main.rs`.
2. Plumb into `Engine::new(...)` config struct.
3. Update RUNBOOK with "Disabling CUDA Graphs for debugging" subsection.
4. Add row to README features and CLI table.
5. Commit: `feat: --cuda-graph CLI flag and buckets`.

---

## Task 7: Padding and bucket promotion

Smooth over batch sizes that don't exactly match a bucket. Currently Task 4 only handles exact matches.

**Files:**
- Modify: `forge-runtime/src/decode_graph.rs` — `pad_to_bucket`
- Modify: `forge-kvcache/src/paged_cache.rs` — reserve block 0 (or a configurable index) as the "dummy KV slot"; never allocated to real sequences

**Steps:**
1. Reserve `dummy_block_id` at `PagedKvCache::new`. Document. Add an assertion that the block manager never returns it from `alloc_blocks`.
2. `pad_to_bucket(active_seqs, bucket)`:
   - For real seqs: copy their real `(token_id, position, block_table, kv_len)` rows into the persistent buffers.
   - For padding slots: write `(token_id=0, position=0, block_table=[dummy_block_id, ...], kv_len=1)`. They compute a throwaway logits row that the host discards.
3. Test: batch_size=5 promoted to bucket 8 — assert first 5 logits rows match a direct batch_size=5 forward pass.
4. Test: batch_size=1 promoted to bucket 1 (no padding) — fast path, equivalent to direct.
5. Commit: `feat: bucket padding with dummy KV slot for variable batch sizes`.

**Definition of done:** Mixed batch sizes (1, 3, 5, 7, 9, 17) across decode steps all run through graph cache; logits match per-row against uncaptured forward.

---

## Task 8: Perf validation and default flip

Confirm we hit the perf target before flipping the default.

**Files:**
- Modify: `forge-server/src/main.rs` — default `--cuda-graph=true`
- Modify: `scripts/benchmark_vllm.sh` (or successor) — add `--cuda-graph={true,false}` columns
- Modify: `docs/RUNBOOK.md` — note default change

**Steps:**
1. Run `bash scripts/benchmark_vllm.sh` (from companion PR) with three configurations on the same model+prompts:
   - forge `--cuda-graph=false`
   - forge `--cuda-graph=true`
   - vLLM 0.18 baseline
2. Compute TPOT improvement of (graph-on) vs (graph-off) at batch_size ∈ {1, 8, 32}.
3. Gate: if improvement at batch_size=8 ≥ 15%, flip default to `true` and commit. If not, file an investigation issue with the numbers and pause. **Do not flip the default to mask a missing gain.**
4. Add a short perf table to RUNBOOK.
5. Commit: `perf: enable CUDA Graphs by default after perf validation` (only if gate passes).

**Definition of done:** Default `--cuda-graph=true`; RUNBOOK has the perf table; benchmark script reports the comparison.

---

## Summary

| Task | Depends on | Estimate | Risk |
|---|---|---|---|
| 0. Stream plumbing fix | — | 0.5 day | Low |
| 1. cudarc Graph API spike | 0 | 1 day | High (gating) |
| 2. Paged attention kernel | 0 | 4–5 days | Medium |
| 2.5. Capture-safe Backend ops | 2 | 3–4 days | Medium (gates 3 & 4) |
| 3. `CudaGraphCache` | 1, 2.5 | 2 days | Low |
| 4. `DecodeGraphRunner` + engine wiring | 2.5, 3 | 2 days | Medium |
| 5. Persistent buffers | 4 | 2 days | Medium |
| 6. CLI surface | 5 | 0.5 day | Low |
| 7. Padding & bucket promotion | 5 | 1.5 days | Medium |
| 8. Perf validation + default flip | 7, vLLM benchmark script | 1 day | Low (gates on data) |

**Total:** ~3 weeks (was ~2.5 before Task 2.5 was added). The hard time-box on Task 1 keeps the plan honest.

**Rollout strategy:** Default off through Tasks 2–7. Only Task 8 changes user-facing behavior, and only if perf gate passes. This keeps `main` shippable at every commit.
