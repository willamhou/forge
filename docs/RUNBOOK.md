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
1. Reduce `--num-blocks` (default 2048) to lower KV cache memory
2. Reduce `max_tokens` in requests
3. Use a smaller model
4. Reduce `--max-batch-size` to limit concurrent sequences

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

### FlashAttention feature fails to compile or link

**Symptom:** Build error when using `--features flash-attn`.

**Check:**
1. CUDA toolkit is installed and `nvcc` is on PATH (needed by `build.rs` to compile the wrapper `.cu` file)
2. The `cc` crate can find a working C++ compiler
3. If linking the FlashAttention library: verify headers and `.so` are available

**Note:** When the `flash-attn` feature is enabled, `build.rs` compiles `forge-kernels/csrc/flash_attn_wrapper.cu`. At runtime, if the FFI call fails the backend falls back to naive attention automatically.

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
