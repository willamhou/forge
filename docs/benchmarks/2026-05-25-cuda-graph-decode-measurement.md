# CUDA Graph decode capture — measurement (TinyLlama-1.1B / GB10)

Date: 2026-05-25
Harness: `forge-server/examples/graph_decode_spike.rs`
Model: TinyLlama-1.1B-Chat, F16, 22 layers, batch=1 decode (q_len=1)
Hardware: NVIDIA GB10 (DGX Spark, Blackwell sm_121)

## What was measured

Whether a full multi-layer `LlamaModel::forward_into` can be captured into a
CUDA Graph and replayed, and the per-decode-step latency of eager vs replay.

Capture was made possible by routing the 5 in-forward `memcpy_htod` uploads
(embedding indices, RoPE cos/sin, paged-attention block_tables + kv_lens)
through persistent backend-owned HOST mirror buffers, so the captured memcpy
nodes' baked host source pointers stay valid across replay (commit 3abce86).

## Result — capture correctness

```
[baseline forward_into] argmax=29889 (14.078)   # eager
[capture] SUCCEEDED      argmax=29889 (14.148)
[replay]                 argmax=29889 (14.148)  max|Δ vs captured| = 0.0000
```

Capture + replay produce logits identical to the captured launch (Δ=0).

## Result — latency

| mode | eager forward_into | captured replay | speedup | saved |
|------|--------------------|-----------------|---------|-------|
| pipelined (sync at end) | 13.35 ms (75 tok/s) | 9.98 ms (100 tok/s) | 1.34× | 3.4 ms/step |
| **per-call-sync (engine-realistic)** | **18.03 ms (55 tok/s)** | **10.21 ms (98 tok/s)** | **1.77×** | **7.8 ms/step** |

The engine syncs every token (logits d2h → host sampling), so the
per-call-sync row reflects production: launch overhead cannot pipeline across
the sync barrier, making the graph win larger (1.77× vs 1.34×).

## Interpretation

- **GPU compute floor ≈ 10 ms/step.** The captured replay is a single launch
  of pure GPU work; 10 ms ≈ the LPDDR5X memory roofline (~2.2 GB weights /
  ~273 GB/s ≈ 8 ms). The forward is already near-optimal — the addressable
  slack is launch overhead, not kernel efficiency.
- **CUDA Graphs removes ~8 ms/step of launch overhead** in the engine-realistic
  (sync-per-token) regime → 1.77× on the forward.
- **The forward is a minority of end-to-end per-token latency.** Single-stream
  benchmark was ~38 ms/token, but raw forward-with-sync is ~18 ms. The
  remaining ~20 ms/step is non-forward engine/server overhead: per-token
  logits d2h + host sampling, scheduler, SSE serialization, tokenizer
  incremental decode, mpsc hops, HTTP.

## Implication for prioritization

- CUDA Graphs end-to-end (the planned Task 6 / "approach B") is worth ~1.77×
  on the forward → estimated ~38 ms → ~30 ms single-stream (~+25% throughput).
  Real, but not dramatic on its own.
- The ~20 ms/step non-forward overhead is the larger target. GPU-side sampling
  (avoid the full 32k-vocab logits d2h + sync every token) would both cut that
  overhead AND let the forward pipeline (removing the sync barrier), which
  compounds with CUDA Graphs.

## Caveats

- Measured at batch=1, short context (prefill 6 tokens). Launch-overhead share
  shrinks at larger batch (more compute per launch); graph win is largest at
  small batch / single stream.
- The eager timing loop grows the KV cache, which reallocates the kv_lens /
  block_tables scratches and invalidates the captured graph (version bump).
  Production must recapture on scratch grow — already designed via the
  `*_scratch_version` accessors + DecodeGraphRunner invalidation.
