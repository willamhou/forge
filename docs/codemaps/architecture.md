# Architecture Codemap

> Freshness: 2026-02-21 | 116 tests passing | Branch: feat/phase1-mvp

## Workspace Overview

Forge is a Rust workspace (edition 2024) with 13 crates organized in a layered architecture:

```
HTTP Layer       forge-server (axum, CLI, tokenizer, chat template)
                      │
Engine Layer     forge-runtime (engine loop, sampling, FSM constraints)
                      │
Scheduling       forge-scheduler (continuous batching)
                      │
Model Layer      forge-models/forge-model-llama (Llama transformer)
                      │
Storage          forge-kvcache (naive CPU-side, paged GPU block manager)
                      │
Compute          forge-backend-cuda │ forge-backend-cpu
                      │                    │
Kernels          forge-kernels (CUDA C++ kernels + FFI)
                      │
Loading          forge-loader (SafeTensors, config parsing)
                      │
Core             forge-core (traits, types, errors)
                      │
Other            forge-transport (stub) │ forge-quantize (stub)
```

## Crate Dependency Graph

```
forge-server ──→ forge-runtime, forge-loader, forge-model-llama,
                 forge-backend-cuda, forge-backend-cpu, forge-scheduler,
                 forge-kvcache, forge-transport, forge-core

forge-runtime ──→ forge-core, regex-automata, serde_json

forge-scheduler ──→ forge-core

forge-model-llama ──→ forge-core, forge-kernels

forge-kvcache ──→ forge-core

forge-backend-cuda ──→ forge-core, forge-kernels, cudarc, half

forge-backend-cpu ──→ forge-core, half

forge-loader ──→ forge-core, safetensors, half, serde, serde_json

forge-kernels ──→ forge-core, cudarc, half

forge-core ──→ thiserror, half, serde
```

## Request Flow

```
HTTP POST /v1/chat/completions
  → openai::chat_completions (tokenize, build SamplingParams, optional FSM constraint)
  → Engine::request_tx.send(EngineRequest)
  → Engine::run() loop:
      1. drain_requests → scheduler.enqueue
      2. scheduler.schedule(cache_usage) → ScheduleBatch
      3. kv_cache.allocate (prefill seqs)
      4. for each seq: process_sequence
         a. build_input → ModelInput
         b. model.forward(input, kv_cache) → ModelOutput
         c. backend.copy_to_host_f32(logits)
         d. sampler.sample_with_constraint(logits, params, generated, fsm)
         e. scheduler.append_token
         f. FSM state advance
         g. stop condition check (EOS, max_tokens, stop_strings, FSM final)
         h. emit EngineEvent::Token / EngineEvent::Finish
  → event_rx → SSE stream or collected response
```

## Key External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `cudarc` | 0.17 | CUDA runtime, cuBLAS, NVRTC |
| `axum` | 0.8 | HTTP server + WebSocket |
| `tokenizers` | 0.21 | HuggingFace tokenizer |
| `minijinja` | 2.12 | Jinja2 chat templates |
| `safetensors` | (latest) | Weight loading |
| `half` | 2 | f16/bf16 types |
| `regex-automata` | (latest) | DFA for structured output FSM |
| `clap` | 4 | CLI argument parsing |
