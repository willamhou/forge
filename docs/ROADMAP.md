# Forge Roadmap

**Date:** 2026-06-24 (major revision; prior 2026-05-24 draft retained
in git history)
**Status:** Forward-looking. Not a commitment, a direction. Bench
evidence and retired-plan reasons trump roadmap ordering.

This is the strategic backlog. Implementation-level plans live in
`docs/plans/<date>-<topic>.md`; gate evidence lives in
`docs/investigations/<date>-<topic>.md`; this file is the *why* and the
*ordering*. It is opinionated about what **not** to do as much as
what to do — see §"Non-goals" with rejection reasons.

## Where we are today

Measured baselines (GB10 sm_121 / Qwen3-8B BF16 → F16 cast at load
/ greedy / 3 runs per concurrency, locked in
`/tmp/bench-2026-06-{09,15,19,21}/*.jsonl`):

| workload | forge default | forge `--quantize-decode` | pegainfer | forge_def vs pega |
|---|---|---|---|---|
| 4B prompt 1024 C=1 | 42.5 ms | **30.5 ms** | 47.3 ms | **0.90× (forge faster)** |
| 4B prompt 1024 C=8 | 56.5 ms | 56.8 ms | 45.4 ms | 1.24× |
| 8B prompt 1024 C=1 | 72.8 ms | **47.4 ms** | 73.1 ms | flat |
| 8B prompt 1024 C=8 | 80.7 ms | 79.7 ms | 72.8 ms | 1.11× |
| 8B prompt 4096 C=1 | 74.0 ms | **49.3 ms** | 73.1 ms | flat |
| 8B prompt 4096 C=8 | 98.4 ms | 98.1 ms | 82.1 ms | **1.20× (worst case)** |

Q8 single-stream is currently forge's strongest result vs pega native:
**1.49–1.54× faster** on 8B short and long context.

### What works (and what shipped this quarter)

- **block_size auto + FA2 paged decode default** (PR #11) — zero-config
  users now hit the FA2 path on supported geometries
- **Q8 weight quantisation, m-dispatch at decode** — m=1 → Q8 GEMV,
  m>1 → f16 GEMM, with zero batch regression
- **`FORGE_FA2_PAGED=0` kill switch + `FORGE_DECODE_TIMING=1`
  instrumentation** — env-gated escape hatches, both opt-in
- **Token-parallel split-KV attention** (commit `1667e12`) — removed
  per-token barriers in the F16 paged-attention kernel
- **cuBLASLt nvjet path** (commit `aaa38cb`) — col-major descriptor +
  tiled transpose for the decode GEMM path
- **8 docs/investigations reports** preserving negative results +
  reproducer scripts (`scripts/q8_sanity.py`, `cublaslt_decode_gemm_probe.rs`)

### Functional strengths (carried from May draft, still true)

- Clear layering (server → runtime → scheduler → model → kvcache →
  backend → kernels), 14 crates, 27.6k Rust LOC
- F32 + F16 dispatch consistent across the `Backend` surface
- Continuous batching, chunked prefill, n-gram speculative decode,
  JSON Schema / regex constrained sampling
- Custom CUDA kernels via NVRTC, no PyTorch runtime
- Documentation discipline: codemaps, dated plans, dated
  investigations, three different memory threads

### Engineering-process strength (newly demonstrated this quarter)

Three multi-week projects retired by cheap-measurement gates within
six weeks:

- **BF16 end-to-end widening** retired by an 8-prompt sanity check
  (forge cast vs pega native: 5/8 character-identical greedy);
  memory `forge-fa2-paged-decode`
- **Attention path kernel rewrite** retired by reading nsys per-launch
  averages (forge `flash_fwd_splitkv` 120.6 μs + combine 2.9 μs vs
  pega `BatchDecodeWithPagedKVCache` 105 μs → 0.85 ms gap vs 3 ms
  gate); PR #12 / `docs/investigations/2026-06-19-batch-gap-8b-nsys.md`
- **Async decode pipeline** retired by env-gated `Instant`
  instrumentation showing host work = 0.65 ms / 0.6% of step time on
  the worst case; PR #15 /
  `docs/investigations/2026-06-21-decode-timing-gate.md`

The pattern is documented in memory `forge-workflow-playbook`; same
harness used for future gates. **One hour of measurement saves ~150
engineer-hours.**

### What is missing for "real" production inference

- Quantization only at *load* (GGUF Q4_K_M / Q8_0) — Q8_0 decode shipped
  as opt-in; no W4A16 / W8A8 / FP8 inference path
- Single-GPU only — no TP / PP / EP. The dual DGX-Spark link is
  imminent; the gap shows the moment they're connected.
- Llama-family only — no MoE, no MLA, no Mamba, no encoder-decoder
- **Prefix cache: radix exists in `forge-kvcache` but is not wired
  into the paged decode path** (still true since May)
- FCFS scheduler — no priority, fairness, preemption, SLO-aware
  admission
- Sampler covers top-p / top-k / min-p / repetition + Gumbel — no
  typical, no Mirostat, no grammar-aware top-p
- No metrics / Prometheus / OpenTelemetry
- No LoRA, no draft-model speculative (only n-gram), no Medusa heads
- Known: FA2 cuBLAS handle leak on GB10 across multi-fixture tests

## Strategic stance: two product lines

Server line and edge line are deliberately separate. Different
binaries, different optimisation budgets, different test fixtures.
Mixing them has historically been a category error.

- **Server line** competes against pegainfer / vLLM / SGLang on
  8B–32B Blackwell / Hopper serving. Phases A → B → C below.
- **Edge line** is Apple Silicon / ARM / WebGPU single-user
  inference. Phase D, parallel.

## Phase A — Single-GPU production readiness (now → +6 weeks)

Goal: take what already works to "default for the workloads where it's
a measured win", and close the only kernel-internal gap with a
positive ROI.

| Item | Gate / exit | Anchor |
|---|---|---|
| **Q8 broader sanity → default-on** | NLL drift vs FP16 within 0.5% on WikiText / C4 slice across Qwen3 / Llama-3 / Qwen2.5 × {~1B, ~8B}; identity rate ≥ existing 8-prompt level on the same; failure-mode catalogue. Then flip `--quantize-decode` default. | `docs/investigations/2026-06-19-q8-vs-fp16-sanity.md` |
| **flashinfer vendoring** | `BatchDecodeWithPagedKVCacheKernel` lands behind a feature flag; Qwen3-8B C=8 long-context within 5% of pega TPOT. Keep FA2 path; dispatch by shape if both stay competitive. | nsys evidence in PR #12 (pega uses flashinfer; forge uses FA2 splitkv + combine) |
| **14B / 32B baseline** | `/tmp/bench-2026-06-XX/*.jsonl` style; locked into memory `gb10-qwen3-NB-bench`. | retired Phase 1 item from prior draft |
| **`FORGE_DECODE_TIMING` t_d2h split** | currently lumped into t_sample; separation unblocks future gates. | PR #15 |
| **Finish CUDA Graphs handling** | **deferred to non-goals** based on GB10 evidence; see below. | retired item |
| **FA2 GB10 test-fixture leak** | single-test workaround → CI tightening. | Task #4, still open |
| **Paged attention BF16** | unblock pega-style BF16 native path on demand. Defer to spec round if user pull doesn't appear. | latent demand only |

**Why this comes first:** Q8 default-on is a marketing-equivalent
upgrade for users without code change; flashinfer vendoring directly
attacks the only measurable forge weakness at scale; the larger model
baselines unblock Phase B sizing decisions.

## Phase B — Scheduler features + multi-GPU and big models (months 2-5)

Goal: parity with SGLang / vLLM on the scheduling features that decide
real production wins independent of raw kernel speed. Multi-GPU
arrives in parallel anchored on the dual DGX-Spark setup.

### B.1 Scheduler features (months 2-4)

| Item | Gate / exit |
|---|---|
| **Prefix cache (RadixAttention-style)** | shared system-prompt / few-shot prefix dedup; chat / RAG bench shows ≥ 1.5× throughput on a representative prompt distribution. Spec round before code (KV cache reference-counting is a structural change). |
| **Speculative decoding (n-gram → EAGLE)** | n-gram is table-stakes; EAGLE is where the C=8 batch win lives. Target ≥ 1.4× single-stream and ≥ 1.2× C=8 on Qwen3-8B. |
| **Mixed prefill + decode batches** | scheduler already supports chunked prefill; this admits a prefill chunk and decode tokens into the same forward call. Pega and vLLM both do this by default. |
| **Speculative + constrained-decode interaction** | constrained sampling must invalidate draft tokens correctly. Subtle; matters once both ship in prod. |
| **LoRA serving (single adapter first)** | gated on user demand; S-LoRA-style unified KV + LoRA weights. |

**Why now and not Phase A:** none of these features lower forge's raw
TPOT in isolation; they multiply throughput on real workloads. Phase A
gives a faster engine; Phase B makes the engine deliver the win at the
service level.

### B.2 Multi-GPU and big models (anchored on dual DGX-Spark; months 3-5)

| Item | Why it matters |
|---|---|
| **Tensor Parallelism (TP)** | Required for ≥30B at FP16 on consumer / Spark hardware. Row/col-linear shard + NCCL all-reduce. Start single-machine, then RDMA cross-machine. |
| **MLA + GQA generalization** | DeepSeek-V2 / V3 / V4 are MLA. Qwen2.5 ≥7B is GQA-only at scale. GQA partially in place; MLA needs new attention kernel + tensor layout. |
| **FP8 KV cache (Blackwell)** | Halves KV footprint → 2× concurrent sequences at the same memory. Hopper / Blackwell only; pair with the paged-attention BF16 work. |
| **Cross-machine KV migration** | Foundation for disaggregated prefill / decode in Phase C. |

Model coverage ladder remains the one from
`docs/research/2026-05-23-deepseek-v4-flash.md`: Llama → Qwen2.5 →
DeepSeek-V2-Lite → DeepSeek-V3 → DeepSeek-V4-Flash.

## Phase C — Production-grade serving + MoE / VLM (months 5-8)

Goal: ops-credible serving and model-family expansion **without** the
per-model-crate explosion that brought pegainfer to 89.3k LOC (3.2×
forge's 27.6k).

| Item | Gate / exit |
|---|---|
| **MoE (DeepSeek-V2-Lite first, V3 second)** | Marlin-style expert GEMM dispatch + token routing kernel. `pegainfer-deepseek-v2-lite` parity within 1.2× TPOT on equivalent setup. |
| **VLM (Qwen2.5-VL first)** | Image encoder mostly off-the-shelf; main work is the OpenAI vision message shape in `forge-server`. |
| **Disaggregated prefill / decode** | DistServe / vLLM 1.0 pattern; relies on Phase B.2 cross-machine KV. Big TPOT win at scale. |
| **Scheduler upgrade** | Priority queues, preemption, SLO-aware admission, per-tenant fairness. Today's FCFS will not survive contention. |
| **Observability** | Prometheus metrics (TTFT, TPOT, queue depth, cache hit, KV usage), structured logs, OpenTelemetry traces. Block on this before any external user. |
| **MTP / Medusa speculative upgrades** | n-gram is a floor; MTP (DeepSeek-style) and Medusa heads raise the ceiling. |

**Pega comparison once more:** pega has separate crates per model
(`pegainfer-{qwen3-4b,qwen35-4b,deepseek-v4,kimi-k2,...}`, 89.3k LOC
total). forge **does not** adopt this architecture. A single generic
transformer crate with architecture-specific attention / projection /
KV layout adapters is the forge property to preserve. Adding a model
should be ~ a few hundred LOC, not a new crate.

## Phase D — Edge line (parallel, months 3-9)

Independent product line. Does **not** share its bench harness, build
profile, or feature priorities with the server line.

| Item | Gate / exit |
|---|---|
| **`forge-backend-metal` crate** | Apple Silicon (Mac / iOS). MLX or direct Metal Performance Shaders. The `Backend` trait already isolates this work to one crate. Llama-3.2-1B / Qwen3-0.6B running end-to-end. |
| **GGUF Q4_K_M / Q4_0 decode path** | Edge needs 4-bit, not 8-bit. Loader nominally supports GGUF; production decode path today is FP16 / Q8. |
| **Embedded library API** | Remove HTTP server entry; expose `Forge::generate(prompt, params) -> Stream<Token>`. Same model code, no server process. |
| **`forge-backend-cpu` SIMD + Q4 dequant** | Currently 527 LOC F32 reference; NEON / AVX2 / Q4_K_M dequant brings it to "actual edge fallback" rather than "test reference". |
| **WebGPU / `wgpu` backend** | After Metal proves the abstraction works. WebGPU is the cross-platform target; Android, web, Windows benefit from one backend. |

What does **not** belong on the edge line:

- Continuous batching (single-user inference)
- Prefix cache / speculative decoding (server-line features; revisit
  if a measured win appears on long-prompt edge use)
- FA2 (`flash_fwd_kvcache` is sm_80+; edge GPUs have their own
  architectures and want their own attention impl)

## Cross-cutting concerns

### Kernel-maintenance debt

Hand-written CUDA C++ in `forge-kernels` has been fine at small scale.
As surface grows (paged, MoE, FP8, FP4), consider:

- **Triton** for high-level kernels (dispatch + autotune for free,
  easier to read), keeping CUTLASS / hand-CUDA for the bottom-tier
  critical paths.
- **CUTLASS templates** rather than from-scratch for new GEMM
  variants.
- **flashinfer vendoring** (Phase A) is the first real test of the
  "vendor over write" stance.

### Quantization matrix

The target matrix to keep in mind:

| Weights | Activations | KV | Use case |
|---|---|---|---|
| FP16 | FP16 | FP16 | baseline today |
| **Q8_0** | FP16 | FP16 | **shipped opt-in (PR #11), default-on gate in Phase A** |
| W4A16 (AWQ) | FP16 | FP16 | Phase B — throughput, server line |
| W4A16 | FP16 | FP8 | Phase B — concurrency |
| W4A4 / FP4 | FP8 | FP8 | Phase C — Blackwell-native |
| FP8 (per-tensor) | FP8 | FP8 | Phase B — large-model production |
| Q4_K_M | FP16 | FP16 | Phase D — edge line |

DeepSeek-V4-Flash points at FP4 experts as the future; align
research direction with that. See
`docs/research/2026-05-23-deepseek-v4-flash.md`.

### Standing infrastructure (enables everything above)

| Item | Gate / exit |
|---|---|
| Standing benchmark CI | Nightly Qwen3-4B / 8B locked-bench against `feat/phase1-mvp` head; regression alert at ±2% TPOT. Replaces "run bench by hand every PR". |
| `scripts/nsys_profile.sh` harness | Wraps server start + warmup + capture + clean shutdown. Used in PR #12 / #15 ad-hoc; codify. |
| Codex review SOP | spec → push → Codex `--resume` → fix → re-review until clean. Continue as documented in `forge-workflow-playbook`. |

## Non-goals (explicit, with retraction evidence)

Items considered and rejected, with reasons. Future readers see why.

- **BF16 end-to-end widening.** Sanity (2026-06-15) shows forge's
  BF16 → F16 cast at load is effectively lossless; forge cast vs pega
  BF16 native gave 5/8 character-identical greedy outputs.
  Multi-week engineering for no measurable user-visible win.
  Memory `forge-fa2-paged-decode` records the retraction.
- **Async decode pipeline.** Direct instrumentation
  (2026-06-21, PR #15) shows host work = 0.65 ms / 0.6% of step time
  on the worst case (8B C=8 long context). A full pipeline could
  recover at most 0.65 ms per step. Below the 3 ms gate threshold.
- **Per-model crate split (pegainfer-style).** Doubles LOC without
  doubling capability; loses the architectural property that keeps
  forge maintainable at 27.6k vs pega 89.3k.
- **CUDA Graph on GB10.** Net +3 ms per step
  (memory `forge-decode-host-bound`, multiple bench passes).
  GB10-specific driver behaviour; `cuGraphLaunch` loses the
  host-GPU overlap that eager mode gets from interleaving
  submission with execution. Not a forge bug, not actionable
  forge-side. Different on Hopper / future hardware; revisit then.
- **Custom small-M GEMM kernel.** Two probes (memory
  `forge-decode-host-bound`, May 29) show forge already at the
  cuBLAS ceiling for M=8 hidden=4096; hand-written wmma loses to
  cuBLAS on every shape tried.
- **Reflection-based model loading.** Hard-coded registry covers four
  architectures cleanly; reflective loading adds complexity and
  removes compile-time checking.
- **OpenAI Python client wrapper.** Upstream `openai` SDK works
  against our OpenAI-compatible HTTP API. Carried from May.
- **Web UI / playground.** Orthogonal to the inference core. Carried
  from May.
- **Training.** This is an inference framework. Carried from May.
- **Generic CPU-only production serving.** CPU backend is correctness
  oracle + dev loop. Edge line treats it as fallback; production
  serving is GPU. Carried from May.

## Risks

- **flashinfer compatibility on GB10 sm_121.** flashinfer is
  Hopper-first; the vendored copy may need patches similar to FA2
  on Blackwell consumer silicon. Mitigation: keep FA2 path as
  fallback (already the existing design).
- **Prefix cache + KV cache invariant conflicts.** Today's KV cache
  assumes per-sequence ownership; prefix cache needs reference
  counting. Spec round before code; do not retrofit by patching.
- **Speculative decoding doubles model memory.** Draft + target both
  resident. Constrains 14B+ to fewer concurrent users on consumer
  GPUs. Document clearly when shipping.
- **Edge work cannibalises server attention.** Different focus weeks
  or different operators; do not let one product line block the other.
- **Q8 default-on requires honest NLL drift evidence.** Eight prompts
  is not coverage. If the broader study shows > 0.5% NLL increase on
  any large slice, ship as opt-in indefinitely. The default flip is
  a gate, not a goal.

## Open questions

- A hosted reference deployment running `feat/phase1-mvp` head against
  a public benchmark dashboard? Today the comparison vs pegainfer /
  vLLM is reproducible but private.
- Edge work: upstream to this repo, or a separate `forge-mobile` repo
  depending on `forge-core` + `forge-models`?
- Public benchmark slice — MLPerf inference, HELM efficiency, or a
  forge-specific one? Each shapes optimisation priorities.

## Honest framing

Forge today is closer to "early vLLM (2023)" than to "vLLM 0.6 /
SGLang 0.4 (2025)" in coverage, but with **cleaner foundations and a
proven engineering process**. The May draft estimated 6–12 months of
focused single-engineer work to close that gap; that estimate stands.
What has changed since May: the engineering-discipline strengths above
make the estimate more credible, not less. Three multi-week traps
already avoided this quarter — each was a ~ 1-month potential
sink — is direct evidence the same harness applies to the
roadmap items below.

For the personal-use case (your own workloads on your own hardware),
forge is already useful and the trajectory is healthy. For external
users, **Phase A + Phase C observability are the minimum bar**.

## How to read this document

- Phase numbers are sequencing, not deadlines. Each phase blocks the
  next where dependencies are real (flashinfer vendoring before
  speculative decoding because we want the fast attention path in
  place when measuring speculative throughput).
- Bench evidence trumps roadmap order. If 14B baseline turns up a
  surprise — forge becomes the C=8 winner at 14B for unrelated
  reasons — Phase A/B priorities flip to defending that win.
- Anything not in this document does not get implementation effort
  without a spec round. Codex spec review then Codex implementation
  review, both per `forge-workflow-playbook`.
- Items moved to "Non-goals" stay there until new evidence appears.
  Re-litigating retired plans without new data is wasted cycles;
  the docs/investigations/ trail is the audit log.
