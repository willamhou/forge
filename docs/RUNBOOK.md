# Forge Runbook

## Starting the Server

```bash
# CUDA backend with paged KV cache (default)
cargo run --release -p forge-server -- --model-path /path/to/model --port 8080

# CPU backend with naive KV cache
cargo run --release -p forge-server -- --model-path /path/to/model --backend cpu --kv-cache naive

# Custom KV cache sizing
cargo run --release -p forge-server -- --model-path /path/to/model --num-blocks 4096 --block-size 32
```

The server exposes:
- `GET /forge/v1/health` — health check
- `GET /v1/models` — list loaded models
- `POST /v1/chat/completions` — OpenAI-compatible chat completions (streaming + non-streaming)

## Health Check

```bash
curl -s http://localhost:8080/forge/v1/health
# Expected: {"status":"ok"}
```

## Verifying the API

Run the E2E integration test:

```bash
bash scripts/test_server.sh /path/to/model 8080
```

This tests health, model listing, non-streaming, and streaming endpoints.

## Performance Benchmarking

```bash
bash scripts/benchmark.sh /path/to/model 10 128 8080
```

Reports TTFT (avg/p50/p99), ITL (avg), and throughput (tokens/s).

### Comparative benchmark vs vLLM

```bash
bash scripts/benchmark_vllm.sh /path/to/model
```

Boots forge (port 8080) and vLLM 0.18 (port 8000) against the same model,
runs a matrix of `prompt_len × concurrency` cells, and writes a side-by-side
markdown report to `.reports/vllm-comparison-<timestamp>.md`.

Override the matrix via env vars:

```bash
PROMPT_LENS=128,1024,4096 CONCURRENCIES=1,8,32,64 REQUESTS_PER_CELL=32 \
    bash scripts/benchmark_vllm.sh /path/to/model
```

Prereq: `pip install vllm==0.18.*`. Server logs land in `.reports/forge-server.log`
and `.reports/vllm-server.log` for triage.

Validate the readiness/identity verifier without launching anything:

```bash
SELFTEST=1 bash scripts/benchmark_vllm.sh
```

Runs unit-style checks against `verify_models_response` — catches regressions
in the `/v1/models` shape verifier (including stdin-routing bugs that would
silently make every server look unhealthy).

## API Features

### Structured Output (JSON Schema / Regex)

Force the model to generate output matching a JSON schema or regex pattern:

```bash
# JSON Schema constraint
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Give me a person"}],
    "json_schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]},
    "max_tokens": 50
  }'

# Regex constraint
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "regex": "[0-9]+",
    "max_tokens": 10
  }'
```

`json_schema` and `regex` are mutually exclusive.

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | `1.0` | 0 = greedy, >0 = multinomial |
| `top_p` | `1.0` | Nucleus sampling threshold |
| `top_k` | `null` | Top-k filtering |
| `max_tokens` | `256` | Max generation length |
| `seed` | `null` | Deterministic sampling seed |
| `repetition_penalty` | `1.0` | Penalize repeated tokens |
| `presence_penalty` | `0.0` | Penalize tokens that have appeared |
| `frequency_penalty` | `0.0` | Penalize tokens by frequency |
| `stop` | `[]` | Stop strings (matched via `contains`; stop token suppressed from output per OpenAI semantics) |

## Common Issues

### Server fails to start

**Symptom:** `cargo run` exits immediately or panics.

**Check:**
1. Model path exists and contains valid SafeTensors files
2. CUDA toolkit is installed and `nvidia-smi` works
3. Sufficient GPU memory for the model

### CUDA errors at runtime

**Symptom:** `ForgeError::Cuda(...)` in logs.

**Check:**
1. GPU driver version is compatible with CUDA toolkit
2. `nvidia-smi` shows the GPU is available
3. No other process is consuming all GPU memory

### Out of memory

**Symptom:** `ForgeError::OutOfMemory(...)`.

**Actions:**
1. Reduce `--num-blocks` (default `auto` ≈ 32k token capacity) to lower KV cache memory
2. Reduce `max_tokens` in requests
3. Use a smaller model
4. Reduce `--max-batch-size` to limit concurrent sequences

### When to override `--block-size`

The default is `auto`, which resolves to 256 on CUDA + FA2-eligible
models (head_dim ∈ {32, 64, 96, 128, 192, 256}, F16) and to 16
otherwise. The startup log shows the resolved value:
`Paged KV cache: block_size=256 (auto), num_blocks=128 (auto), ...`.

Set the flag explicitly when:

- **CPU backend debugging.** `--block-size 16` keeps per-sequence KV
  allocation small; 256 is wasteful when there is no FA2 to consume it.
- **Short-context benchmarking.** At `block_size=256` an active sequence
  can carry up to 255 tokens of internal fragmentation in its tail block
  (vs 15 at the old default). For very short generations the working-set
  waste can matter; pin `--block-size 16`.
- **Reproducing FA2 vs split-KV comparisons.** Lock 16 and 256
  explicitly rather than relying on the auto default.

### When to use `--quantize-decode` (Q8_0)

Holds a Q8_0 copy of every linear weight; single-stream decode
(m==1 batch step) routes through a warp-per-row GEMV that reads
half the weight bytes per token. Batched decode (m>1) auto-dispatches
to the original f16 GEMM, so there is no regression at C≥2.

**Measured speedup vs forge default (FP16 cast) on GB10:**

| model / context | C=1 FP16 | C=1 Q8 | speedup | C=8 FP16 | C=8 Q8 |
|---|---|---|---|---|---|
| Qwen3-4B prompt 1024 | 42.5 ms | 30.5 ms | **1.40×** | 56.5 ms | 56.5 ms (no change) |
| Qwen3-8B prompt 1024 | 72.8 ms | 47.4 ms | **1.54×** | 80.7 ms | 79.7 ms (no change) |
| Qwen3-8B prompt 4096 | 74.0 ms | 49.3 ms | **1.49×** | 98.4 ms | 98.1 ms (no change) |

Single-stream Q8 on Qwen3-8B beats pegainfer's native BF16 path
(73 ms) by **1.49–1.54×** — currently forge's strongest result on
8B-class models.

**Precision tradeoff.** Q8_0 effective precision is ~8 mantissa bits
(32-element block sharing a FP16 scale + 32 INT8 values), vs FP16's
10 mantissa bits — a ~2-bit loss. This loss is enough to flip top-1
greedy token decisions when the top-2 candidates are close to a tie.
A token-level sanity check on Qwen3-8B (8 prompts, greedy 256
tokens) showed:

- 3/8 prompts produced **character-identical** completions vs FP16
- 5/8 diverged mid-reply (8% to 99% into the reply) but stayed
  semantically equivalent — same answer, paraphrased

**Recommended for:**

- Throughput-sensitive deployments where exact token-level reproducibility
  is not required: chat assistants, summarization, code completion,
  RAG output generation.
- Memory-bound workloads (8B+ on consumer hardware) where single-stream
  TPOT dominates user experience.

**Avoid when:**

- Output reproducibility is contractual: evaluation pipelines, scientific
  summarization, regulated-domain outputs where token-level audit matters.
- Numerical-correctness benchmarks (lm-eval-harness, GSM8K, etc.) without
  comparing against a Q8-specific baseline.

### Why is `--quantize-decode` not the default?

Q8 has no measured per-stream downside vs FP16 (batch auto-falls-back
to f16 GEMM), and the single-stream win is large. But shipping it
default requires broader sanity coverage than the current 8-prompt
spot check:

- Multi-family sanity: at minimum Qwen2.5, Qwen3, Llama-3 across
  small (~1B) and mid (~8B) sizes.
- Quantitative quality signal: NLL / perplexity drift vs FP16 on a
  standard slice (WikiText, C4), not just identity-rate on hand-picked
  prompts.
- Failure-mode catalog: characterise what kinds of prompts produce
  larger Q8 divergence so operators can pin `--quantize-decode false`
  for those workloads.

Until that evidence is in, ship as opt-in. The flag costs nothing
to flip and the documentation already calls out when to reach for it.

### `FORGE_DECODE_TIMING=1` decode-step instrumentation

Per-step three-phase timing of `process_decode_batch` for diagnosing
whether the GPU/host balance leaves room for an async pipeline. Logs
one line per decode step at `target=forge_runtime::decode_timing` info
level:

```text
decode n=8 t_gpu=101.165ms t_sample=0.630ms t_emit=0.017ms total=101.812ms
```

- `t_gpu` — stage + `compute_decode` + `synchronize`. The full GPU
  forward including the post-launch sync wait.
- `t_sample` — device argmax / on-device sampling / D2H of ids or
  full logits. Includes any logits memcpy.
- `t_emit` — per-sequence host emit loop (token append, FSM advance,
  stop check, detok, send_event).
- `n` — active sequence count in this batch.

Enable with:

```bash
FORGE_DECODE_TIMING=1 RUST_LOG=info,forge_runtime::decode_timing=info \
  ./target/release/forge-server ...
```

Disable: unset the env var (or set `=0` / `=false`). The check is one
env lookup per step; no measured perf overhead in production.

**When to use:** before sinking time into an async decode pipeline or
any host-side restructuring, confirm `t_sample + t_emit` is actually
large enough to be worth pursuing. On GB10 / Qwen3-8B / C=8 /
prompt~4096 the host phases sum to ~0.65 ms out of a 101 ms step
(0.6%) — well below any reasonable architectural-change threshold.

See `docs/investigations/2026-06-21-decode-timing-gate.md` for the
gate analysis that retired the multi-week async-pipeline plan.

### `FORGE_FA2_PAGED=0` kill switch

The FA2 paged-decode dispatch path has no runtime fallback. If a new
model geometry triggers a CUDA error from inside `flash_fwd_kvcache`,
set `FORGE_FA2_PAGED=0` (or `=false`, case-insensitive) to force the
split-KV fallback without changing the block size.

Note the semantics flip from the original opt-in: any value other than
`0` / `false` (including `=1`, `=on`, `=enabled`, or absence) leaves FA2
**enabled**. The only disabling values are `=0` and `=false`.

### Request rejected with "prompt exceeds max_prefill_tokens"

**Symptom:** API returns an error for long prompts.

**Cause:** The prompt token count exceeds `--max-prefill-tokens` (default 4096). The scheduler rejects prompts that can never fit in the prefill budget.

**Actions:**
1. Increase `--max-prefill-tokens` (e.g., `--max-prefill-tokens 8192`)
2. Shorten the prompt or system message
3. Ensure `--max-prefill-tokens` is at least as large as your longest expected prompt

### Build fails with cudarc errors

**Symptom:** Compilation errors in `forge-backend-cuda` or `forge-kernels`.

**Check:**
1. CUDA toolkit 12.x+ is installed
2. `nvcc` is on PATH
3. `CUDA_PATH` environment variable is set correctly

## Monitoring

Phase 1 uses `tracing` for structured logging. Set the log level via:

```bash
RUST_LOG=info cargo run --release -p forge-server -- --model-path /path/to/model
```

Log levels: `error`, `warn`, `info`, `debug`, `trace`.

## Rollback

Since Forge is deployed as a single binary:

1. Stop the running server process
2. Build the previous git revision: `git checkout <previous-sha> && cargo build --release`
3. Restart with the same model path and port
