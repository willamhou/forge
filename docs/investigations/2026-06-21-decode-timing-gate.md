# Async decode pipeline gate (host work is 0.6% of step time)

**Date:** 2026-06-21
**Branch:** `feat/phase1-mvp`
**Question:** is the 7.89 ms / step gap on Qwen3-8B C=8 long-context
(forge 98.4 ms vs pegainfer 82.1 ms TPOT) dominated by host overhead,
in which case an architectural async decode pipeline would close it?

## TL;DR

**No.** Direct instrumentation of `process_decode_batch` shows host
work is **0.65 ms median / 0.67 ms mean per step (0.6% of step time)**
at C=8 with prompt=4096. The remaining 100 ms is GPU forward + sync.
Architectural async-pipeline plan retired before spec — it would have
delivered <1 ms gain at multi-week cost.

This is the third "cheap measurement saves a multi-week project" gate
in this codebase (BF16 widening, attention path probe, now async
pipeline). The pattern is documented in memory
`forge-workflow-playbook`.

## Why this was on the table

The 2026-06-19 investigation (`docs/investigations/2026-06-19-batch-gap-8b-nsys.md`)
attributed ~5.5 ms of the 7.89 ms gap to "diffuse host overhead" by
nsys subtraction (TPOT − GPU-busy from kernel histogram). That
attribution was a back-of-envelope estimate, not a direct measurement.
Memory `forge-decode-host-bound` from 2026-05-28 had directly measured
host work at ~1.57 ms on a different workload (C=1 short prompt), so
the 5.5 ms figure would have required host work to grow 3.5× at C=8
long context — plausible but not yet measured.

A multi-week architectural change deserved one hour of measurement
before commitment.

## Method

Added env-gated `FORGE_DECODE_TIMING=1` instrumentation to
`process_decode_batch` (see `forge-runtime/src/engine.rs`). Three
`std::time::Instant` markers:

- `t_gpu` — from function entry through `stage_decode` +
  `compute_decode` + `backend.synchronize()`
- `t_sample` — device-side argmax / on-device sampling / `copy_to_host`
  of ids or full logits
- `t_emit` — per-sequence emit loop (token append, FSM advance, stop
  check, detok, send_event)

`tracing::info!` once per decode step under
`target=forge_runtime::decode_timing`. Env lookup costs ~10 ns per step;
no measurable overhead on production runs (verified — TPOT unchanged
within noise).

## Workload

- Qwen3-8B BF16 (cast to F16 at load) on GB10
- forge default flags: `--num-blocks 192 --max-prefill-tokens 4096`
  (FA2 paged, block_size auto, eager mode)
- `scripts/bench_concurrent.py http://localhost:8951 qwen3-8b 50 4096 8 1`
  — prompt~4096, out=50, C=8, 1 run
- 81 steady-state n=8 steps after skipping 3 ramp-up steps where active
  sequences are still entering decode

## Results

| phase | median | mean | min | max | % of step |
|---|---|---|---|---|---|
| `t_gpu` (stage + compute + sync) | **101.17 ms** | 97.45 ms | 82.73 | 110.07 | **99.4%** |
| `t_sample` (argmax + D2H ids) | 0.63 ms | 0.66 ms | — | — | 0.6% |
| `t_emit` (per-seq host loop) | 0.017 ms | 0.016 ms | — | — | 0.02% |
| **HOST** (`t_sample` + `t_emit`) | **0.65 ms** | 0.67 ms | — | — | **0.6%** |

Step total median is ~101 ms, matching the wall-clock TPOT of ~98–100 ms
on this workload (within noise).

## Interpretation

The host phases (sample + emit) sit at 0.6% of step time. An async
decode pipeline that hides 100% of host work behind the next step's
GPU forward could remove at most 0.65 ms per step. That is below the
3 ms threshold we agreed on for an architectural commitment.

Where does the 5.5 ms gap actually live? It must be inside `t_gpu` —
inside the GPU forward proper. Plausible internal contributors:

1. **Batched paged-attention scaling.** At C=8 with KV length ~4400
   the `flash_fwd_splitkv` work is ~8× a single-stream step plus
   per-batch overhead. pegainfer's flashinfer `BatchDecodeWithPagedKVCacheKernel`
   is one fused kernel where forge launches splitkv + combine separately.
2. **proj-GEMM constant terms** at hidden=4096 batch=8. forge's
   `nvjet_sm121_hsh_mma_*` and pega's `cutlass_80_wmma_bf16` both sit
   on the cuBLAS ceiling for M=8 (within 1% in our probe), but the
   handful of microseconds per launch × 36 layers can add up.
3. **scatter_kv batched at C=8** + KV append cost across longer
   sequences.

All three live inside `t_gpu`, all three are kernel-level rather than
host-level, and all three have small per-step leverage. None justify
the architectural cost of an async pipeline.

## Verdict — retire the async pipeline plan

The plan stays in memory as a *retired option* with a documented
reason. Future readers see "host work is 0.6% of step time, async
pipeline doesn't pay" and don't re-evaluate from scratch.

The instrumentation that produced this evidence is shipped as a
permanent dev tool (`FORGE_DECODE_TIMING=1`, RUNBOOK section "When to
use"). Any future "is host work the bottleneck?" question can be
answered in 30 minutes by re-running.

## Productive directions instead

The 7.89 ms gap is inside `t_gpu` and dominated by kernel-level factors
on the cuBLAS / FA2 ceiling. The actionable directions are:

1. **Q8 8B productisation** — `--quantize-decode` gives 1.49× single
   stream Q8 over pegainfer native on 8B (long context). Already
   documented; broader sanity coverage (multi-family, NLL drift)
   gates default-on.
2. **14B / 32B baseline** — verify the GEMM-ceiling result holds at
   larger scale on GB10's 119 GB unified memory.
3. **Long-context FA2 paged improvements** — splitkv + combine
   fusion (single kernel a la flashinfer), or a half2-vectorised path
   for the load loop. Per-launch gain measured in low microseconds;
   per-step gain measured in ones of ms. Not zero, not architectural.

The async pipeline is no longer in the candidate list.

## Artifacts

- `/tmp/bench-2026-06-21/gate.log` — raw forge log with per-step timing
  rows (FORGE_DECODE_TIMING=1)
- `forge-runtime/src/engine.rs` — committed instrumentation, env-gated,
  zero measured overhead on production runs

## Related

- `docs/RUNBOOK.md` — "FORGE_DECODE_TIMING=1 decode-step instrumentation"
- `docs/investigations/2026-06-19-batch-gap-8b-nsys.md` — the
  predecessor that flagged ~5.5 ms host overhead by nsys subtraction
  (now corrected by direct measurement)
- memory `forge-decode-host-bound` — the 2026-05-28 single-stream
  measurement (1.57 ms) that this run extends to batched long context
- memory `forge-workflow-playbook` — "cheap sanity saves multi-week
  projects" pattern, now with a third datapoint
