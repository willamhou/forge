# Forge Roadmap

**Date:** 2026-05-24
**Status:** Forward-looking — not a commitment, a direction.

This is the strategic backlog. Implementation-level plans live in
`docs/plans/<date>-<topic>.md`; this file is the why and the ordering.

## Where we are today

Forge is a clean, layered Rust inference framework with CUDA + CPU dual-path
support, continuous batching + chunked prefill, paged KV cache, n-gram
speculative decoding, JSON Schema / regex constrained decoding, FA2-vendored
attention, and a Llama-family model implementation.

Comparable to **early vLLM / SGLang in functional surface area, but with a
more principled foundation**: F32/F16 first-class dispatch, no PyTorch deps,
NVRTC PTX compilation, explicit Backend trait that keeps CUDA-specific code
out of the model layer.

**What works well today:**

- Clear layering (server → runtime → scheduler → model → kvcache → backend → kernels)
- F32 + F16 dispatch consistent across the Backend surface
- Continuous batching, chunked prefill, scheduler that rejects over-long
  prompts cleanly (no silent truncation)
- n-gram speculative decode + JSON / regex FSM constraint
- Custom CUDA kernels via NVRTC, no PyTorch runtime
- Documentation discipline (codemaps, dated plan docs, this file)

**What is missing for "real" production inference:**

- Quantization only at *load* time (GGUF Q4_K_M / Q8_0) — no runtime W4A16
  / W8A8 / FP8 inference path
- Single-GPU only — no TP / PP / EP. (Two DGX-Spark boxes link is imminent
  per project context; the gap shows the moment they're connected.)
- Llama-family only — no MoE, no MLA, no Mamba, no encoder-decoder
- FCFS scheduler — no priority, fairness, preemption, SLO-aware admission
- Radix tree exists in `forge-kvcache` but prefix-cache hits not yet wired
  into the paged decode path
- Sampler is basics only (top-p / top-k / min-p / repetition) — no typical,
  Mirostat, grammar-aware top-p
- No metrics / Prometheus / OpenTelemetry
- No LoRA, no draft-model speculative (only n-gram), no Medusa heads
- Known: FA2 cuBLAS handle leak on GB10 across multi-fixture tests (Task #4)

---

## Phase A — Single-GPU production readiness (~1-2 months)

Goal: a server you would actually deploy to serve one model on one box, end
to end, with competitive TPOT.

| Item | Why it matters | Notes |
|---|---|---|
| Finish CUDA Graphs (Tasks 1-8) | Decode TPOT typically drops 30-50% on small-batch | In progress; plan in `docs/plans/2026-05-22-cuda-graphs-plan.md` |
| Paged attention F16 / BF16 | Real serving never runs F32. Today's pool is F32-only | Follows Task 2.2 CUDA kernel landing |
| W4A16 (AWQ / GPTQ) runtime inference | ~2× decode throughput on the same hardware; near-mandatory at ≥30B | Dequant-on-the-fly kernels; reuse `forge-kernels` pattern |
| Prefix cache hit on paged path | Long system prompts / few-shot → 60-90% hit; visible TTFT win | Radix exists in `forge-kvcache`; wire to scheduler + paged path |
| FP8 KV cache (Blackwell) | Halves KV footprint → 2× concurrent sequences at the same memory | Hopper/Blackwell only; align with paged attn F16 work |
| FA2 GB10 test-fixture leak | Currently single-test workaround; blocks CI tightening | Task #4 |

## Phase B — Multi-GPU and big models (~2-3 months, anchored on the dual DGX-Spark setup)

Goal: serve 30B-100B+ models across two boxes; unlock MoE / MLA.

| Item | Why it matters | Notes |
|---|---|---|
| Tensor Parallelism (TP) | Required for ≥30B at FP16 on consumer GPUs / Spark | Row/col-linear shard + NCCL all-reduce; start single-machine, then RDMA cross-machine |
| MLA + GQA generalization | DeepSeek-V2/V3/V4 are MLA; Qwen2.5 ≥7B is GQA-only at scale | Already partially there for GQA; MLA needs new attention kernel |
| MoE inference (Expert Parallelism) | DeepSeek, Mixtral, Qwen2 MoE — top open models are MoE | Grouped GEMM + EP comms; revisit `forge-flash` for batched-expert FA2 |
| FP8 activation (Blackwell) | Cuts compute time on H100/B100 by ~1.7×; needed to feed FP8 KV | Pairs with W8A8 quant story |
| Cross-machine KV migration | Required for disaggregated prefill/decode (Phase C) | Foundation laid here, payoff in Phase C |

Model coverage ladder is in `docs/research/2026-05-23-deepseek-v4-flash.md`:
Llama → Qwen2.5 → DeepSeek-V2-Lite → DeepSeek-V3 → DeepSeek-V4-Flash.

## Phase C — Production-grade serving (ongoing)

Goal: ops-credible. Things that distinguish a research server from one an
SRE will tolerate.

| Item | Why it matters | Notes |
|---|---|---|
| Scheduler upgrade | Priority queues, preemption, SLO-aware admission, per-tenant fairness | Today's FCFS will not survive contention |
| Observability | Prometheus metrics (TTFT, TPOT, queue depth, cache hit, KV usage), structured logs, OpenTelemetry traces | Block on this before any external user |
| Speculative decode upgrades | MTP (DeepSeek-style), draft model, Medusa heads | n-gram is a floor, not a ceiling |
| LoRA / multi-LoRA serving | S-LoRA-style unified KV + LoRA weights | One model, many adapters served per request |
| Disaggregated prefill / decode | DistServe / vLLM 1.0 pattern; separate prefill cluster from decode cluster | Big TPOT win at scale, needs Phase B cross-machine KV |
| Speculative + constrained-decode interaction | Constrained sampling has to invalidate draft tokens correctly | Subtle; matters once both are enabled in prod |

---

## Cross-cutting concerns

**Kernel-maintenance debt.** Hand-written CUDA C++ in `forge-kernels` has
been fine at small scale. As kernel surface grows (paged, MoE, FP8, FP4),
consider:
- **Triton** for high-level kernels (dispatch + autotune for free, easier to
  read), keeping CUTLASS / hand-CUDA for the bottom-tier critical paths.
- **CUTLASS templates** rather than from-scratch for new GEMM variants.

**Quantization story.** The matrix to keep in mind:

| Weights | Activations | KV | Use case |
|---|---|---|---|
| FP16 | FP16 | FP16 | baseline today |
| W4A16 (AWQ) | FP16 | FP16 | Phase A — throughput |
| W4A16 | FP16 | FP8 | Phase B — concurrency |
| W4A4 / FP4 | FP8 | FP8 | Phase B+ — Blackwell-native |
| FP8 (per-tensor) | FP8 | FP8 | Phase B — large-model production |

DeepSeek-V4-Flash points at FP4 experts as the future; align research
direction with that. See `docs/research/2026-05-23-deepseek-v4-flash.md`.

**Multi-modal.** Out of scope for the current arc. Revisit after Phase B.

---

## Non-goals (intentionally not on the roadmap)

- A drop-in OpenAI Python client wrapper — already provided by upstream
  `openai` SDK against our OpenAI-compatible HTTP API.
- A web UI / playground — orthogonal to the inference core; let users bring
  their own (open-webui, etc.).
- Training. This is an inference framework; fine-tuning belongs elsewhere.
- Generic CPU-only production serving. CPU backend is a correctness oracle
  and dev-loop convenience, not a deployment target.

---

## Honest framing

Forge today is closer to "early vLLM (2023)" than "vLLM 0.6 / SGLang 0.4
(2025)" in coverage, but with cleaner foundations. Closing that gap is
roughly **6-12 months of single full-time engineer** of focused work to be
genuinely competitive on common workloads. The architecture won't be the
bottleneck; functional completeness and benchmark coverage will be.

For the personal-use case (your own workloads on your own hardware), Forge
is already useful and the trajectory is healthy. For external users, Phase A
+ observability from Phase C are the minimum bar.
