# Benchmark Baseline — TinyLlama-1.1B on GB10 (pre-CUDA-Graphs)

**Date:** 2026-05-25
**Hardware:** NVIDIA GB10 (DGX Spark, Blackwell, sm_121)
**Model:** TinyLlama-1.1B-Chat-v1.0, F16 weights (BF16→F16 at load)
**Config:** `--backend cuda --kv-cache paged --num-blocks 1024 --block-size 16 --max-batch-size 64`
**Build:** `cargo build --release` (CUDA on)
**Path under test:** paged attention decode, **no CUDA Graphs** (engine calls
the allocating `model.forward()`; `forward_into` + graph capture not yet wired).

This is the reference point for the CUDA Graphs end-to-end work. Re-run the
same harness after graph capture lands to quantify the win.

## Numbers

| Metric | Value |
|---|---|
| **Single-stream decode** | **38.2 ms/token (26.2 tok/s)** |
| Concurrent throughput, C=4 | 179 tok/s |
| Concurrent throughput, C=16 | 531 tok/s |
| Concurrent throughput, C=32 | **730 tok/s** |
| TTFT (C=4, short prompt) | ~148 ms |

Single-stream decode measured by the differential method: stream the same
prompt at `max_tokens` 32 and 256 (greedy), take min-of-3 each, subtract to
cancel prefill — `(256-32) tok / (t256 - t32)`. This avoids the urllib
line-buffering artifact that corrupts naive per-chunk TTFT/ITL timing on a
single fast stream (see "Measurement notes").

Concurrent throughput = total decoded tokens / wall-clock, firing C streams
simultaneously (Python threads), max_tokens=128 each.

## Interpretation

1. **38 ms/token single-stream is slow for a 1.1B model on Blackwell**
   (should be sub-10 ms). The decode hot path issues ~220 kernel launches
   per step (22 layers × ~10 kernels/layer: matmul ×4, rms_norm ×2,
   silu_mul, paged_attention, rope, add, …) plus per-op output allocations.
   Launch latency + allocation churn dominate at batch=1. **This is exactly
   what CUDA Graphs targets** — collapse the ~220 launches into one graph
   replay.

2. **Concurrent throughput scales near-linearly** (179 → 531 → 730 tok/s for
   C = 4 → 16 → 32). Continuous batching + paged attention are working;
   C=32 throughput is ~28× single-stream, so the GPU is well-utilized under
   batching and the single-step bottleneck is launch overhead, not FLOPs.

3. **Expected CUDA Graphs win**: single-stream TPOT should drop substantially
   (launch overhead amortized into one replay). Concurrent throughput may
   improve less (already GPU-bound) but should still benefit from removing
   per-step CPU launch work.

## Measurement notes / caveats

- **urllib SSE buffering artifact**: a naive `for line in resp` over a single
  fast SSE stream buffers the whole body, so per-chunk TTFT/ITL timing is
  bogus (TTFT≈total, ITL≈0). Only wall-clock-total metrics are trustworthy
  for single streams. The differential method sidesteps this.
- **Non-streaming long generations 500**: `stream:false` + `max_tokens=256`
  returns HTTP 500 (`event channel full, cancelling slow consumer`). The
  non-streaming handler does not drain the bounded engine event channel
  concurrently, so long generations overflow it. Streaming works fine. This
  is a separate forge bug (fixed in a follow-up); the benchmark uses
  streaming throughout.
- Numbers are single-run (min-of-3 for the differential); not averaged over
  many trials. Treat as order-of-magnitude baseline, not a precise figure.

## Repro

```bash
# Start server
./target/release/forge-server \
  --model-path /path/to/tinyllama-1.1b-chat \
  --backend cuda --kv-cache paged --num-blocks 1024 --block-size 16 \
  --max-batch-size 64 --port 8110

# Differential single-stream decode + concurrent throughput
python bench.py http://127.0.0.1:8110 128
```
