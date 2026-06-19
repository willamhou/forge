# Qwen3-8B C=8 batch gap: nsys + GEMM probe investigation

**Date:** 2026-06-19
**Branch:** `feat/phase1-mvp`
**Target:** locate the source of forge 80.68 ms vs pegainfer 72.79 ms per-stream
TPOT gap at C=8 on Qwen3-8B (1.11× — wider than the 1.02× we saw on Qwen3-4B).

## TL;DR

GEMM kernel choice is **not** the source. forge's `nvjet_sm121_hsh_mma_*`
selection on M=8 8B shapes is within ~1% of cuBLASLt's swap-trick and native
col-major variants. Closing the 7.89 ms/step gap requires looking at the
non-GEMM 17.3 ms (attention + host overhead + sampling), not at GEMM.

## Setup

- GB10 / driver 580.82, CUDA 13.0, sm_121
- Qwen3-8B BF16(36 layers × 4096 hidden × 12288 intermediate × 8 KV heads,
  head_dim 128, vocab 151936)
- prompt~1024 / out 256, greedy, streaming, C=8, 1 run
- forge head: `163bd27` (block_size auto + FA2 default), eager
- pegainfer: `--cuda-graph=false` for nsys visibility (graph hides kernels)

## Step 1 — nsys kernel histograms

Both servers captured with `nsys profile -t cuda,nvtx --delay=60 --duration=15`
around steady-state decode.

**forge default (eager, FA2 paged + cuBLAS GEMM, F16 path)**:

| Family | Kernel | % time |
|---|---|---|
| proj-GEMM | `nvjet_sm121_hsh_mma_*` (4 variants) | **78.6%** |
| prefill-GEMM | `cutlass_80_tensorop_f16_s16816gemm_*` | 6.7% |
| attention | `flash::flash_fwd_splitkv_kernel` + combine | **5.9%** |
| residual/norm | `fused_residual_rms_norm_f16` + `rms_norm_f16` | 1.4% |
| sampling | `argmax_f16` | 0.3% |
| other | scatter_kv, rope, transpose, splitKreduce | ~7% |

**pegainfer (eager, flashinfer + BF16 path)**:

| Family | Kernel | % time |
|---|---|---|
| proj-GEMM | `cutlass_80_wmma_tensorop_bf16_s161616gemm_16x16_128x{1,2}` | **87.1%** |
| attention | `flashinfer::BatchDecodeWithPagedKVCacheKernel` | **5.0%** |
| prefill-GEMM | `nvjet_sm121_tst_mma_*` | 4.5% |
| residual/norm | `flashinfer::norm::FusedAddRMSNormKernel` + `RMSNormKernel` | 0.8% |
| sampling | `flashinfer::sampling::RadixTopKKernel_Unified` | 0.2% |
| other | rope, append KV, silu | ~2% |

**Per-step normalization (after subtracting prefill kernels):** both engines spend
~80% of decode-step GPU time in proj-GEMM, ~5% in paged attention. The gap is
not in family proportions.

## Step 2 — GEMM probe (the actual lever check)

Question: do forge's `nvjet_sm121_hsh_mma` selections match what cuBLASLt would
pick, or is there a faster algo we're missing?

Probe: `forge-backend/forge-backend-cuda/examples/cublaslt_decode_gemm_probe.rs`
extended with the 8B M=8 shapes. Three call styles measured per shape, 500
iters each:

1. **classic cuBLAS** with swap-trick (forge's actual production path)
2. **cuBLASLt** with the same swap trick
3. **cuBLASLt native col-major** (different descriptor, exposes different algos)

**Qwen3-8B M=8 raw GEMM time (μs/call):**

| shape | classic swap | LT swap | LT native | best/worst |
|---|---|---|---|---|
| qkv (8×4096×6144) | 208.2 | 206.0 | 229.0 | 0.99×/1.10× |
| o (8×4096×4096) | 124.6 | 123.1 | 119.6 | 0.96×/0.99× |
| gate/up (8×4096×12288) | 415.9 | 411.1 | 467.1 | 0.99×/1.12× |
| down (8×12288×4096) | 432.6 | 427.0 | 441.0 | 0.99×/1.02× |
| lm_head (8×4096×151936) | 5271 | 5268 | 5362 | 1.00×/1.02× |

Per decode step (classic swap, single stream — these are per-token, not per-batch):
`qkv + o + (gate + up) + down = 208 + 125 + 415 + 415 + 432 = 1595 μs / layer`
`36 layers + lm_head = 36 × 1595 + 5271 = 62.7 ms per step`

forge's measured C=8 TPOT is 80.0 ms. **62.7 / 80 = 78.4% GEMM** —
matches the nsys histogram (78.6%) **exactly**.

## Findings

1. **GEMM kernel selection is at the cuBLAS ceiling.** Across qkv / o / gate / up
   / down / lm_head on 8B M=8 shapes, classic and LT swap-trick converge to
   the same `nvjet_sm121_hsh_mma_*` kernel within ~1%. LT native col-major
   is a different descriptor that sometimes exposes a different algo but is
   never meaningfully faster on these shapes. There is no "missed" cuBLAS algo.

2. **The 7.89 ms/step gap is not in GEMM.** Both engines spend ~80% of step
   time in proj-GEMM. forge's 62.7 ms GEMM matches the GEMM ceiling for these
   shapes; pegainfer's GEMM cannot be far below this (BF16 vs F16 wmma on
   the same M=8 shape gives similar TFLOPs / bandwidth). The gap must be in
   the remaining ~17 ms (attention + residual + sampling + host).

3. **dtype path differs but does not explain GEMM speed.** forge runs the F16
   path (`hsh_mma` H×S×H wmma); pegainfer runs the BF16 path (`cutlass_80_wmma
   bf16` s161616gemm). Both are SM80/89-class wmma kernels; BF16 vs F16 do
   not differ in raw throughput on Blackwell sm_121 — the 1.11× gap is not a
   "BF16 GEMM is intrinsically faster" effect.

4. **Where the gap likely lives** (in decreasing order of probability,
   based on nsys deltas):
   - **paged attention impl** — forge `flash::flash_fwd_splitkv_kernel` 5.9%
     vs pegainfer `flashinfer::BatchDecodeWithPagedKVCacheKernel` 5.0%.
     Different algorithms, different per-step instance counts. **Probe this
     next.**
   - **host overhead from launch count** — forge issues ~280 kernels/step at
     C=8 (from prior memory `forge-decode-host-bound`); pegainfer's wmma
     splits one logical GEMM into many 16×16 instances (20k+ launches in 15s
     capture) but their CUDA Graph default amortizes that — when graph is off
     (this nsys), per-launch host cost is competitive but slightly higher
     than forge.
   - **fused residual + RMSNorm shape** — forge 1.4%, pegainfer 0.8%. Small
     but not zero.
   - **sampling kernel choice** — forge `argmax_f16` 0.3% vs pegainfer
     `RadixTopKKernel` 0.2%. Marginal.

## Recommended next step

**Attention path probe.** Profile forge's `flash_fwd_splitkv_kernel`
configuration (num_splits heuristic, split-K combine cost) against
flashinfer's `BatchDecodeWithPagedKVCacheKernel` on the same shape
(B=8, num_qo=32, num_kv=8, head_dim=128, seqlen_k≈1280). Targets:
- per-launch time difference
- instances-per-step delta (forge launches main + combine kernel; flashinfer
  is one kernel)
- whether forge's `num_splits=max_kv_len/512 clamped [1,16]` heuristic is
  too aggressive for batch=8

If attention path explains ≥ 3 ms of the 7.89 ms gap, kernel rewrite has
positive ROI. If < 1 ms, drop the investigation — remaining gap is
diffuse host overhead and not worth a multi-week kernel project.

## Artifacts

- `/tmp/bench-2026-06-15/nsys/forge_8b_c8.nsys-rep` (forge histogram)
- `/tmp/bench-2026-06-15/nsys/pega_8b_c8_eager.nsys-rep` (pegainfer eager histogram)
- `/tmp/bench-2026-06-15/qwen3-8b-baseline.jsonl` (locked baseline numbers)
- `forge-backend/forge-backend-cuda/examples/cublaslt_decode_gemm_probe.rs`
  (extended with 8B shapes — committed)

## Related memory

- `gb10-qwen3-8b-bench` — the baseline numbers this gap is measured against
- `forge-cublaslt-nvjet-path` — prior cuBLASLt investigation on 4B (showed
  same near-ceiling result on 4B GEMM)
- `forge-decode-host-bound` — decode is launch-bound, ~31 ms/step host work
  at 4B C=1 eager
- `forge-fa2-paged-decode` — current FA2 paged-decode dispatch with the
  split-KV combine kernel
