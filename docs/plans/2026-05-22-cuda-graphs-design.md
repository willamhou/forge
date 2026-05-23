# CUDA Graphs for Batched Decode

> Status: design | Author: Phase 2 perf track | Branch: feat/phase1-mvp
>
> Companion plan: [2026-05-22-cuda-graphs-plan.md](2026-05-22-cuda-graphs-plan.md)

## Context

After FlashAttention v2 merged (PR #3), the next biggest decode-path overhead is **kernel launch latency**, not compute. A Llama-3.2-1B forward pass with 16 layers issues ~60+ kernel launches per token (RMSNorm × 2, QKV proj, RoPE, FA2, O proj, gate/up proj, SiLU+mul, down proj, residual adds, sampling D→H). At batch_size=8, decode is bandwidth-bound — pegainfer's scheduler doc puts it at "~91% of decode time is GEMV/MLP at 82–87% DRAM utilization." Launch overhead becomes a hard floor on TPOT.

CUDA Graphs let us capture a fixed kernel sequence once and replay it with a single launch. pegainfer reports this is their main lever for decode performance and the design pattern is well-documented (vLLM ships the same).

**Goal:** Capture the decode-batch forward pass once per `(batch_size_bucket)` and replay on every subsequent decode step that fits the bucket. Prefill stays as-is (variable shape, not graph-friendly).

**Non-goals (this design):**
- Capturing prefill (shapes too variable)
- Capturing sampling / D→H copies (we still copy logits to host)
- Graph-based multi-stream parallelism

## Performance hypothesis

| Path | Current | With CUDA Graphs | Source |
|---|---|---|---|
| Per-layer launch overhead (decode) | ~6–10 μs × ~60 kernels × 16 layers ≈ 5–10 ms | ~10 μs single graph launch | NVIDIA blog + pegainfer doc |
| TPOT at batch_size=8 | TBD (need vLLM baseline) | ~10–30% lower | pegainfer scheduler doc |

Concrete target: at batch_size=8, Llama-3.2-1B BF16, **TPOT improvement ≥ 15%** vs current main. Verified via the `benchmark_vllm.sh` matrix.

## High-level architecture

Two new layers:

1. **`CudaGraphCache` in `forge-backend-cuda`** — owns captured `CudaGraph` and `CudaGraphExec` instances, keyed by bucket. Capture happens on first miss; replay on every hit.
2. **`DecodeGraphRunner` in `forge-runtime`** — wraps `process_decode_batch`. Decides bucket from `active_seqs.len()` and dispatches to either a captured graph or a normal forward pass (graph cold).

### Decode step flow

```
process_decode_batch(active_seqs):
  bucket = round_up_to_bucket(active_seqs.len())     # e.g. 1, 2, 4, 8, 16, 32

  if bucket > MAX_GRAPH_BUCKET:                       # batch too large
    return uncaptured_forward(active_seqs)            # fallback

  # Pad active_seqs up to bucket size with dummy seqs (zero-token contribution)
  padded = pad_to_bucket(active_seqs, bucket)

  # Stage inputs into pinned/persistent device buffers
  graph_inputs.copy_from(padded)                      # token_ids, positions, block_tables, kv_lens

  if cache.has(bucket):
    cache.replay(bucket)                              # single launch, ~10 μs
  else:
    cache.capture(bucket, || forward(padded))         # first-time cost ~one extra forward

  # logits is read from the persistent output buffer at the captured device address
  copy_to_host(graph_outputs.logits[: active_seqs.len()])
  per-seq sampling (CPU, unchanged)
```

The critical move is that **all device pointers passed into the captured kernels are fixed**. The only data that changes between steps is the *contents* of those buffers (token IDs, KV block tables, KV cache cells). cudarc + CUDA 12 supports this pattern via `CUgraphExec` updates of input *contents* without re-capture.

## Bucket strategy

Single-axis bucketing by `batch_size`:

```
BUCKETS = [1, 2, 4, 8, 16, 32]   // configurable via CLI: --cuda-graph-buckets
```

- `batch_size == 0` → no decode work, skip
- `batch_size > 32` → fallback to uncaptured forward (warns once, then silent)
- `batch_size = 5` → padded to bucket 8, runs the bucket-8 graph, sampling reads only first 5 rows

Pros: small bucket set → small memory overhead (one persistent KV input buffer + logit buffer per bucket).

Cons: padded sequences burn compute. At batch_size=5 in bucket-8, ~37% of attention/MLP work is wasted. Mitigation: padded "dummy" sequences point to a shared zero-length KV slot; their logits are discarded host-side. Compute cost is still proportional to bucket size, but it's the price for graph reuse.

Trade-offs explicitly rejected:
- **Per-(batch_size, max_kv_len) buckets** — combinatorial explosion. Variable KV length is handled via the paged design below, not bucketing.
- **No bucketing, capture every shape** — capture itself takes ~one full forward to complete; the cost dominates if it happens often.

## Paged KV compatibility (the hard part)

The captured graph must see **fixed device addresses** for KV reads. Our `PagedKvCache::get_kv(seq_id, layer)` currently materializes a contiguous `(K, V)` tensor by gathering blocks — the resulting device addresses are different each call. That breaks graph replay.

Design: **block-table-driven attention kernel.** The graph captures attention as a kernel that takes:

```
attention_kernel(
    q_ptr,              // fixed: persistent input buffer
    kv_pool_ptr,        // fixed: PagedKvCache.block_pool base ptr
    block_tables_ptr,   // fixed: persistent device buffer, contents updated per step
    kv_lens_ptr,        // fixed: persistent device buffer, contents updated per step
    out_ptr,            // fixed: persistent output buffer
    ...
)
```

- `kv_pool_ptr` is the base of the global block pool allocated once at startup — never moves.
- `block_tables_ptr` is a persistent `[max_batch_size, max_blocks_per_seq] i32` buffer owned by `CudaGraphCache`. Each decode step, `DecodeGraphRunner` writes the current sequences' block tables into this buffer (`cudaMemcpyAsync H2D` *before* graph launch — counts as graph input update, not a kernel re-capture).
- **Sizing `max_blocks_per_seq`:** must equal the *per-sequence* block ceiling, not the *average* (`num_blocks / max_batch`). A long-context request running at low concurrency can legitimately own most of the pool. We anchor to a CLI-configured per-seq KV ceiling: `max_blocks_per_seq = ceil(max_seq_len / block_size)`, where `--max-seq-len` defaults to `config.max_position_embeddings`. The scheduler rejects prompts where `prompt_tokens + max_tokens > max_seq_len` at enqueue. At dispatch, if a real sequence somehow exceeds this width (e.g. someone bypassed the scheduler check), the runner **fails closed** by falling back to the uncaptured path — it never silently truncates the block table.
- `kv_lens_ptr` is `[max_batch_size] i32`, updated the same way.
- New KV cells from the current step are written into the pool by an `append_kv` kernel that is *inside* the captured graph, reading from the persistent input buffer.

This matches how FlashInfer and vLLM handle paged attention + graphs. It requires us to extend our attention path to read from `(kv_pool, block_tables, kv_lens)` rather than from gathered `(K, V)` tensors. FA2's `flash_attn_varlen_fwd` already takes `cu_seqlens` and works with non-contig layouts; we'll need a small shim or move to the FlashInfer-style paged kernel.

**This is the gating dependency.** If we keep gather-then-FA2, graphs won't help. Options ranked:

1. **Add a paged attention kernel that reads block tables directly** (preferred long-term, ~1 week of work, aligns us with vLLM/FlashInfer convention). Pulls in the work pegainfer points at as "FlashInfer for paged attention."
2. **Pre-allocate a single contiguous KV scratch per bucket** (~1 day, but breaks the point of paging when KV grows beyond bucket capacity). Stopgap only.
3. **Skip CUDA Graphs for now, do batched decode attention kernel (Task 4) first** (defers the problem; Task 4 is also blocked by attention layout, so this just delays).

Recommendation: do option 1 as part of this work. The plan doc lays it out as a prerequisite task.

## When to capture / when to invalidate

| Event | Action |
|---|---|
| Server start | No capture; lazy |
| First decode at bucket B | Capture graph for bucket B, then replay |
| Subsequent decodes at bucket B | Replay |
| Model weights change | N/A (immutable post-load) |
| KV pool resize | Invalidate all graphs (rare; can panic for now) |
| `--cuda-graph=false` flag | Skip capture entirely; pure uncaptured forward |

Capture cost is bounded: one full forward per bucket, amortized over thousands of decode steps. With buckets `[1, 2, 4, 8, 16, 32]` that's 6 warm-up forwards spread across the first few requests.

## Crate-level changes

```
forge-backend-cuda/
  src/
    cuda_graph.rs        # NEW: CudaGraphCache, CudaGraphExec wrapper
    paged_attention.rs   # NEW: block-table-driven attention (FA2 varlen wrapper)
    backend.rs           # +stream/graph capture hooks
forge-core/
  src/
    backend.rs           # +trait method: paged_attention(qkv, block_tables, kv_lens, ...)
forge-runtime/
  src/
    engine.rs            # process_decode_batch dispatches via DecodeGraphRunner
    decode_graph.rs      # NEW: bucket selection, input staging, fallback
forge-server/
  src/
    main.rs              # +CLI: --cuda-graph, --cuda-graph-buckets, --max-seq-len
forge-scheduler/
  src/
    continuous.rs        # +reject prompts where prompt_tokens + max_tokens > max_seq_len
forge-core/
  src/
    error.rs             # +SequenceTooLong variant for fail-closed paths
docs/RUNBOOK.md          # +sections on disabling graphs and on --max-seq-len
```

`forge-scheduler` does pick up changes (the per-sequence KV ceiling is enforced at enqueue). `forge-kvcache::block_manager` stays untouched — its pool-base-pointer is already stable. `forge-flash` stays untouched — FA2 is fine; we'll call `flash_attn_varlen_fwd` from the new paged path.

## Risks

1. **cudarc CUDA Graph API surface.** cudarc 0.17 exposes `CudaGraph`, `CudaGraphExec`, and stream `begin_capture`/`end_capture`. Need to confirm `cudaGraphInstantiate` flags we want (`CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` etc.) are reachable; if not, drop to raw FFI through cudarc's `Driver`. **Mitigation:** spike a 1-day prototype before committing to the plan.

2. **FA2 vendored sources may not expose varlen with paged block tables.** Our `forge-flash/csrc/` is FA2 v2 source; `flash_attn_varlen_fwd` takes `cu_seqlens` but expects contiguous KV per sequence within a packed layout — not block-paged. We may need to write a thin "gather-then-FA2" inside the graph (gather kernel + FA2 launch, both captured), or pull FlashInfer in. **Mitigation:** prototype Task 0 (paged attention kernel) before Task 1 of the plan.

3. **Non-determinism between captured and uncaptured paths.** Sampling already runs on CPU, so identity should hold. We must assert this in tests with seeded sampling.

4. **Padded dummy sequences leak state.** Dummy seqs need a fixed "scratch" KV slot that's never read by real sequences. Plan reserves block 0 as the dummy slot, written but never appended to a real seq.

5. **Stream ownership.** `flash_attention.rs:109` still has the `// TODO: extract from backend.stream` — we need to pass our real `CudaStream` into FA2 before capture, otherwise FA2 ops won't be in the captured graph. **Must fix as a prerequisite.**

## Open questions

- **Q1: Buckets per CLI flag, or fixed?** Default is `1,2,4,8,16,32`. Configurable means more user-visible knobs but lets us iterate without recompiling. *Lean: configurable, default keeps current behavior.*
- **Q2: Capture failure policy?** If `cudaGraphInstantiate` fails (e.g. unsupported kernel inside capture region), do we fall back silently or hard-error? *Lean: log warn + permanently disable graphs for this bucket; the request still completes.*
- **Q3: When to free graphs?** Server lifetime, ref-counted with model? Cleanest is "drop with backend." Memory cost per bucket: one `CUgraphExec` + one set of persistent buffers (~few MB for batch_size=32, KV pool already allocated). *Lean: server-lifetime, drop on backend Drop.*
- **Q4: Interaction with `--kv-cache naive`?** Naive cache uses CPU vectors, no fixed pool pointer. Probably just disable graphs when `kv-cache=naive`. *Lean: yes, document this.*

## Test strategy

**Unit (`forge-backend-cuda/tests/test_cuda_graph.rs`):**
- Capture a trivial kernel sequence, replay, compare to direct launch — must match bit-for-bit (deterministic kernels).
- Capture, mutate input buffer contents, replay — output reflects new inputs.
- Capture, attempt replay on different stream — graceful error.

**Integration (`forge-models/forge-model-llama/tests/test_decode_graph.rs`):**
- Build TinyLlama-like model with random weights, run 10 decode steps batched-size=4.
- Compare logits between `--cuda-graph=true` and `--cuda-graph=false` paths — must match within `1e-4` (FP32 accumulator non-determinism).
- Vary batch sizes across runs to exercise bucket promotion (5 → bucket 8).

**E2E (`scripts/test_server.sh`):**
- Add a `--cuda-graph=false` invocation to confirm parity with current main.

**Perf gate:** Phase 2 wrap-up benchmark must show ≥ 15% TPOT improvement at batch_size=8 on Llama-3.2-1B BF16, otherwise the design is wrong somewhere.

## Summary

| Component | Description |
|---|---|
| `CudaGraphCache` (`forge-backend-cuda`) | Per-bucket `CudaGraphExec` + persistent input/output buffers |
| `DecodeGraphRunner` (`forge-runtime`) | Bucket selection, padding, input staging, **fallback for non-bucket batch sizes and over-long sequences** |
| Paged attention kernel | Reads `(kv_pool, block_tables, kv_lens)` directly — prerequisite |
| Stream fix in FA2 | Pass real `CudaStream`, replace the TODO at `flash_attention.rs:109` |
| Per-sequence KV ceiling | `--max-seq-len` flag; `max_blocks_per_seq = ceil(max_seq_len / block_size)`. Scheduler rejects over-long prompts at enqueue; dispatcher fails closed at runtime |
| CLI | `--cuda-graph=true\|false`, `--cuda-graph-buckets=1,2,4,8,16,32`, `--max-seq-len=N` |
| Expected gain | ≥ 15% TPOT at batch_size=8 |

The two non-obvious pieces are (a) paged attention kernel as a prerequisite, not a follow-up, and (b) the persistent-buffer pattern for variable KV (sized from per-sequence KV ceiling, not pool average) — everything else is mechanical graph capture wiring.
