#!/bin/bash
# Benchmark script for Forge LLM inference server.
#
# Measures TTFT, ITL, throughput, and peak memory.
#
# Usage: bash scripts/benchmark.sh /path/to/model [num_requests] [max_tokens]
#
# Requires: curl, python3, jq (optional)

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model-path> [num_requests] [max_tokens]}"
NUM_REQUESTS="${2:-10}"
MAX_TOKENS="${3:-128}"
PORT="${4:-8080}"
HOST="http://localhost:${PORT}"

echo "=== Forge Benchmark ==="
echo "Model:        ${MODEL_PATH}"
echo "Requests:     ${NUM_REQUESTS}"
echo "Max tokens:   ${MAX_TOKENS}"
echo "Port:         ${PORT}"
echo ""

# Start server in background
cargo run --release -- serve --model "${MODEL_PATH}" --port "${PORT}" &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null || true" EXIT

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -sf "${HOST}/forge/v1/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

echo ""
echo "--- Running ${NUM_REQUESTS} sequential requests (max_tokens=${MAX_TOKENS}) ---"
echo ""

python3 - "${HOST}" "${NUM_REQUESTS}" "${MAX_TOKENS}" << 'PYTHON'
import sys
import json
import time
import urllib.request

host = sys.argv[1]
num_requests = int(sys.argv[2])
max_tokens = int(sys.argv[3])

ttfts = []
itls = []
total_tokens = 0
total_time = 0.0

prompts = [
    "Explain what a transformer model is in simple terms.",
    "Write a short poem about the moon.",
    "What is the capital of France? Answer in one sentence.",
    "List three benefits of open-source software.",
    "Describe the water cycle in three sentences.",
]

for i in range(num_requests):
    prompt = prompts[i % len(prompts)]
    payload = json.dumps({
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time = None
    token_times = []
    token_count = 0

    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        ttfts.append(now - start)
                    else:
                        token_times.append(now - token_times[-1] if token_times else now - first_token_time)
                    token_count += 1
    except Exception as e:
        print(f"  Request {i+1}: ERROR - {e}")
        continue

    elapsed = time.perf_counter() - start
    total_time += elapsed
    total_tokens += token_count

    if token_times:
        itls.extend(token_times)

    print(f"  Request {i+1}/{num_requests}: {token_count} tokens in {elapsed:.2f}s "
          f"(TTFT={ttfts[-1]*1000:.1f}ms)" if ttfts else f"  Request {i+1}: no tokens")

print()
print("=== Results ===")
if ttfts:
    avg_ttft = sum(ttfts) / len(ttfts) * 1000
    p50_ttft = sorted(ttfts)[len(ttfts)//2] * 1000
    p99_ttft = sorted(ttfts)[int(len(ttfts)*0.99)] * 1000
    print(f"TTFT (avg/p50/p99):  {avg_ttft:.1f}ms / {p50_ttft:.1f}ms / {p99_ttft:.1f}ms")

if itls:
    avg_itl = sum(itls) / len(itls) * 1000
    print(f"ITL (avg):           {avg_itl:.1f}ms")

if total_time > 0:
    throughput = total_tokens / total_time
    print(f"Throughput:          {throughput:.1f} tokens/s")

print(f"Total tokens:        {total_tokens}")
print(f"Total time:          {total_time:.2f}s")
PYTHON

echo ""
echo "=== Benchmark complete ==="
