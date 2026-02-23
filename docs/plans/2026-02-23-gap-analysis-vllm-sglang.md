# Forge vs vLLM/SGLang Gap Analysis

> Last updated: 2026-02-23

## Overview

This document compares Forge's current architecture against vLLM and SGLang, the two leading open-source LLM inference engines. The goal is to identify gaps and prioritize future work.

**Forge status:** V1 complete with OpenAI-compatible API, continuous batching, structured output, speculative decoding, FP16 compute, and FlashAttention v2 (pending GPU verification).

---

## Feature Parity Matrix

### Already Aligned

| Capability | Forge | vLLM | SGLang | Notes |
|-----------|-------|------|--------|-------|
| Continuous Batching | Y | Y | Y | FCFS scheduler, chunked prefill |
| OpenAI-compatible API | Y | Y | Y | `/v1/chat/completions`, streaming SSE |
| Grouped Query Attention | Y | Y | Y | Configurable num_heads / num_kv_heads |
| Structured Output | Y | Y | Y | JSON Schema + Regex FSM constraint |
| Chunked Prefill | Y | Y | Y | Configurable budget, stable decode latency |
| Speculative Decoding (N-gram) | Y | Y | Y | Zero-cost n-gram drafter, adaptive length |
| GGUF Quantized Loading | Y | Y | Y | Q8_0, Q4_K_M dequantization |
| FP16 Compute | Y | Y | Y | cuBLAS GemmEx, custom FP16 kernels |
| Fused Kernels | Partial | Y | Y | Fused SiLU-mul, fused residual+RMSNorm |
| FlashAttention v2 | Scaffolded | Y | Y | Code complete, pending GPU verification |

---

## Key Gaps

### P0 - Performance Critical (production blockers)

#### 1. PagedAttention GPU Kernel

**Current state:** KV cache blocks are organized on CPU side (`NaiveKvCache` / `PagedKvCache`). Each attention step reconstructs device tensors from CPU data.

**vLLM/SGLang approach:** Block tables live on GPU. The PagedAttention CUDA kernel reads KV data directly via block index lookup - zero CPU involvement during inference.

**Impact:** Eliminates CPU-GPU data transfer bottleneck for KV cache. This is vLLM's foundational innovation.

**Work estimate:** Large. Requires new CUDA kernel + GPU-resident block manager + block table upload.

#### 2. FlashAttention v2 End-to-End Verification

**Current state:** `forge-flash` crate complete with vendored FA2 sources, build.rs NVCC compilation, Rust FFI bindings, and feature-gated dispatch. Not yet verified on GPU.

**Impact:** 2-4x prefill speedup, IO-aware tiling reduces HBM reads.

**Work estimate:** Small. Run `cargo test --workspace --features flash-attn` on SM80+ GPU.

#### 3. CUDA Graph Capture

**Current state:** Not implemented. Every decode step issues individual kernel launches.

**vLLM/SGLang approach:** Capture the entire decode forward pass as a CUDA graph (fixed shape). Replay eliminates kernel launch overhead and CPU-side scheduling.

**Impact:** ~2x decode throughput improvement (reported by vLLM). Critical for high-batch decode-bound workloads.

**Work estimate:** Medium. Requires static shape padding, graph capture/replay logic, invalidation on shape change.

#### 4. Prefix Caching (Automatic Prefix Caching)

**Current state:** Radix tree (`forge-kvcache/src/radix.rs`) is implemented with insert, match, evict, and ref-counting. But block sharing is not wired into the attention path.

**vLLM approach:** APC (Automatic Prefix Caching) - hash KV blocks by token content, share across sequences with matching prefixes.

**SGLang approach:** RadixAttention - radix tree on token sequences, reuse KV blocks for common prefixes.

**Impact:** Eliminates redundant prefill computation for shared system prompts and few-shot examples. Significant for multi-turn chat and agentic workloads.

**Work estimate:** Medium. Radix tree exists; need to wire block sharing into scheduler + attention dispatch.

---

### P1 - Scale (required for large models)

#### 5. Tensor Parallelism (TP)

**Current state:** Single GPU only. Device ordinal selectable via CLI.

**vLLM/SGLang approach:** NCCL-based all-reduce / all-gather across GPUs. Shard QKV weights column-wise, MLP row/column-wise. Each GPU holds 1/N of model.

**Impact:** Required for 70B+ models. The machine has 8x RTX 4090 (24GB each) - with TP=4 or TP=8, can serve Llama 70B.

**Work estimate:** Large. Requires NCCL integration, weight sharding, distributed KV cache, all-reduce kernels.

#### 6. Pipeline Parallelism (PP)

**Current state:** Not implemented.

**vLLM approach:** Distribute transformer layers across GPU groups. Combined with TP for very large models (e.g., Llama 405B across nodes).

**Impact:** Enables models that don't fit in a single node's GPU memory.

**Work estimate:** Large. Requires inter-node communication, micro-batching, bubble minimization.

#### 7. Multi-LoRA Serving

**Current state:** Not supported.

**vLLM approach:** Load multiple LoRA adapters sharing base model weights. S-LoRA kernel for batched LoRA computation across sequences with different adapters.

**Impact:** Serve many fine-tuned variants from a single base model deployment.

**Work estimate:** Medium. Requires LoRA weight loading, adapter selection per request, fused LoRA kernel.

#### 8. Disaggregated Prefill/Decode

**Current state:** Not implemented. Prefill and decode share the same GPU.

**SGLang v0.4+ approach:** Separate GPU pools for prefill (compute-bound) and decode (memory-bound). Transfer KV cache between pools.

**Impact:** Better GPU utilization - prefill GPUs run at high FLOPS, decode GPUs optimize for memory bandwidth.

**Work estimate:** Large. Requires KV cache transfer protocol, separate scheduling, load balancing.

---

### P2 - Throughput / Latency Optimization

#### 9. Quantized Inference (FP8 / INT8 / INT4)

**Current state:** Only dequantization on model load (Q8_0, Q4_K_M). Inference runs in F32 or F16.

**vLLM approach:** GPTQ, AWQ, FP8 W8A8 quantized inference with specialized CUDA kernels. INT8 KV cache.

**Impact:** 2x throughput for W8A8, 50% memory savings for KV cache with FP8.

**Work estimate:** Large. Requires quantized GEMM kernels (or integration with CUTLASS/cuBLAS INT8), calibration support.

#### 10. FlashInfer / Advanced Attention Variants

**Current state:** Naive per-head attention + FlashAttention v2 (feature-gated).

**SGLang approach:** FlashInfer library with multiple attention backends - decode attention, prefill attention, paged attention, sliding window attention - all fused and optimized.

**Impact:** Flexible attention dispatch for different workload patterns.

**Work estimate:** Medium-Large. Could vendor FlashInfer or implement specialized kernels.

#### 11. Dynamic Split-Fuse

**Current state:** Prefill and decode are processed in separate batches.

**SGLang approach:** Mix prefill tokens and decode tokens in the same batch. Use remaining compute capacity from decode to process prefill chunks.

**Impact:** Better GPU utilization, reduced time-to-first-token under load.

**Work estimate:** Medium. Requires mixed-batch attention kernel and scheduler changes.

#### 12. Compute/Communication Overlap

**Current state:** Not applicable (single GPU).

**vLLM/SGLang approach:** In TP mode, overlap all-reduce communication with next layer's computation using CUDA streams.

**Impact:** Hides communication latency in TP setups.

**Work estimate:** Small (once TP is implemented). Requires multi-stream scheduling.

---

### P3 - Ecosystem & Model Coverage

#### 13. Model Architecture Coverage

**Current state:** Llama family only (Llama 2, Llama 3, TinyLlama).

**vLLM:** 70+ model architectures (Mistral, Qwen, Gemma, DeepSeek, Phi, Falcon, GPT-NeoX, MPT, ...).

**SGLang:** 30+ model architectures with growing coverage.

**Work estimate per model:** Small-Medium. Most share transformer backbone; differences are in attention pattern, normalization, activation, positional encoding.

**Priority candidates:**
1. Qwen2/2.5 (strong multilingual, very popular in China)
2. Mistral/Mixtral (MoE architecture)
3. DeepSeek-V3 (MLA attention, MoE)
4. Gemma 2

#### 14. Vision Language Models (VLM)

**Current state:** Text-only.

**vLLM/SGLang:** Support LLaVA, Qwen-VL, Pixtral, InternVL, etc.

**Impact:** Multimodal is a major trend. VLMs need image encoder integration + cross-attention or image-token interleaving.

**Work estimate:** Large. Requires image preprocessing pipeline, vision encoder, multimodal fusion.

#### 15. Tool Calling / Function Calling

**Current state:** Not supported.

**vLLM/SGLang:** Structured tool call generation with OpenAI-compatible function calling API.

**Impact:** Required for agentic use cases.

**Work estimate:** Medium. Mostly prompt engineering + structured output constraint for tool call JSON format.

#### 16. KV Cache Quantization

**Current state:** KV cache stored in F32/F16.

**vLLM approach:** FP8 KV cache reduces memory by ~50%, enabling longer contexts or larger batches.

**Impact:** Significant memory savings with minimal quality loss.

**Work estimate:** Medium. Requires FP8 quantize/dequantize kernels and attention kernel FP8 input support.

#### 17. Embedding / Reward Model Endpoints

**Current state:** Not supported.

**vLLM:** Supports `/v1/embeddings` endpoint for embedding models.

**Impact:** Enables RAG pipelines and reward model scoring.

**Work estimate:** Small. Reuse transformer forward pass, skip LM head, pool hidden states.

---

## Recommended Roadmap

```
Phase 2 (Current): Performance
  |
  |-- FA2 GPU verification              [Small]   <- immediate
  |-- PagedAttention GPU kernel          [Large]   <- biggest perf win
  |-- CUDA Graph for decode              [Medium]  <- ~2x decode speed
  |-- Prefix Caching wiring             [Medium]  <- common prefix reuse
  |
Phase 3: Scale
  |
  |-- Tensor Parallelism (TP)           [Large]   <- 70B on 8x 4090
  |-- FP8/INT8 quantized inference      [Large]   <- 2x throughput
  |-- Multi-LoRA serving                [Medium]
  |
Phase 4: Ecosystem
  |
  |-- More model architectures          [Medium]  <- Qwen, Mistral, DeepSeek
  |-- Vision Language Models            [Large]   <- multimodal
  |-- Tool calling                      [Medium]  <- agentic
  |-- KV Cache quantization             [Medium]  <- memory savings
  |-- Embedding endpoints               [Small]
```

---

## Architecture Comparison Summary

| Dimension | Forge | vLLM | SGLang |
|-----------|-------|------|--------|
| **Language** | Rust | Python + C++/CUDA | Python + C++/CUDA |
| **Attention** | Naive + FA2 (optional) | FlashAttention + PagedAttention | FlashInfer (multiple backends) |
| **KV Cache** | CPU-side paged blocks | GPU-resident paged blocks | GPU-resident radix tree |
| **Parallelism** | Single GPU | TP + PP + expert parallelism | TP + DP + disaggregated P/D |
| **Scheduling** | FCFS continuous batching | FCFS + priority + preemption | Radix-aware + chunked prefill |
| **Models** | Llama | 70+ architectures | 30+ architectures |
| **Quantization** | Load-time dequant only | GPTQ, AWQ, FP8, INT8 | GPTQ, AWQ, FP8 |
| **Speculative** | N-gram drafter | Draft model + N-gram | Eagle + N-gram |
| **Structured Output** | JSON Schema + Regex FSM | Outlines + xgrammar | xgrammar + compressed FSM |
| **Maturity** | V1 prototype | Production (1000+ contributors) | Production (400+ contributors) |

---

## Key Forge Advantages

Despite the gaps, Forge has structural advantages worth preserving:

1. **Rust safety** - Memory safety without GC, no Python GIL, zero-cost abstractions
2. **Clean trait system** - Backend trait enables CPU/CUDA/future backends with same model code
3. **Minimal dependencies** - No PyTorch runtime, direct cudarc + cuBLAS
4. **Compilation speed** - Incremental Rust builds are fast; only NVCC compilation is slow
5. **Single binary** - Deploy as one static binary, no Python environment management
