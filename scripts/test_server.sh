#!/bin/bash
# End-to-end integration test for Forge LLM inference server.
#
# Usage: bash scripts/test_server.sh /path/to/model
#
# Starts the server with the given model, sends requests to the
# OpenAI-compatible API, and verifies response format.

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model-path>}"
PORT="${2:-8080}"
HOST="http://localhost:${PORT}"

echo "=== Forge Integration Test ==="
echo "Model: ${MODEL_PATH}"
echo "Port:  ${PORT}"
echo ""

# Start server in background
cargo run --release -- serve --model "${MODEL_PATH}" --port "${PORT}" &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null || true" EXIT

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -sf "${HOST}/forge/v1/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Server failed to start within 30s" >&2
        exit 1
    fi
    sleep 1
done

# Health check
echo ""
echo "--- Health Check ---"
HEALTH=$(curl -sf "${HOST}/forge/v1/health")
echo "${HEALTH}" | python3 -c "import sys,json; r=json.load(sys.stdin); assert r['status']=='ok'; print('PASS: health check')"

# List models
echo ""
echo "--- List Models ---"
MODELS=$(curl -sf "${HOST}/v1/models")
echo "${MODELS}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
assert r['object'] == 'list'
assert len(r['data']) > 0
print(f'PASS: models list ({len(r[\"data\"])} model(s))')
"

# Non-streaming chat completion
echo ""
echo "--- Non-Streaming Chat Completion ---"
RESPONSE=$(curl -sf "${HOST}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "temperature": 0
    }')
echo "${RESPONSE}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
assert 'choices' in r, f'Missing choices: {r}'
assert len(r['choices']) > 0, 'Empty choices'
assert 'message' in r['choices'][0], 'Missing message in choice'
assert 'usage' in r, 'Missing usage'
assert r['usage']['prompt_tokens'] > 0, 'Zero prompt tokens'
print(f'PASS: non-streaming (completion_tokens={r[\"usage\"][\"completion_tokens\"]})')
"

# Streaming chat completion
echo ""
echo "--- Streaming Chat Completion ---"
STREAM=$(curl -sf -N "${HOST}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "stream": true
    }' 2>&1 || true)

echo "${STREAM}" | python3 -c "
import sys
lines = [l.strip() for l in sys.stdin if l.strip().startswith('data:')]
assert len(lines) >= 2, f'Expected at least 2 SSE data lines, got {len(lines)}'
assert lines[-1] == 'data: [DONE]', f'Last line should be [DONE], got {lines[-1]}'
print(f'PASS: streaming ({len(lines)} SSE events)')
"

echo ""
echo "=== All integration tests passed! ==="
