# Forge Runbook

## Starting the Server

```bash
cargo run --release -- serve --model /path/to/model --port 8080
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
1. Reduce `max_tokens` in requests
2. Use a smaller model
3. Reduce KV cache block count (when configurable)

### Build fails with cudarc errors

**Symptom:** Compilation errors in `forge-backend-cuda` or `forge-kernels`.

**Check:**
1. CUDA toolkit 12.x+ is installed
2. `nvcc` is on PATH
3. `CUDA_PATH` environment variable is set correctly

### FlashAttention feature fails to link

**Symptom:** Linker error when building with `--features flash-attn`.

**Note:** In Phase 1, the FlashAttention FFI is a stub. The `.cu`/`.h` files are infrastructure for Phase 2. Do not enable `--features flash-attn` in production until the library is linked in `build.rs`.

## Monitoring

Phase 1 uses `tracing` for structured logging. Set the log level via:

```bash
RUST_LOG=info cargo run --release -- serve --model /path/to/model
```

Log levels: `error`, `warn`, `info`, `debug`, `trace`.

## Rollback

Since Forge is deployed as a single binary:

1. Stop the running server process
2. Build the previous git revision: `git checkout <previous-sha> && cargo build --release`
3. Restart with the same model path and port
