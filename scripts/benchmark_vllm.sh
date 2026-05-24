#!/bin/bash
# Comparative benchmark: forge vs vLLM 0.18 (serial, single-GPU-safe).
#
# Runs the two engines back-to-back, NEVER simultaneously: starts forge, runs
# the benchmark matrix against it, kills forge, waits for GPU memory to fall
# back to baseline, then starts vLLM and repeats. The two JSON fragments are
# merged into one markdown report.
#
# Usage:
#   bash scripts/benchmark_vllm.sh <hf-model-path>
#
# Example:
#   bash scripts/benchmark_vllm.sh /models/Llama-3.2-1B-Instruct
#
# Both engines are configured with the same served-model-name (the basename
# of the model path) so the /v1/models identity check works for either.
#
# Env knobs:
#   FORGE_PORT, VLLM_PORT
#   PROMPT_LENS, CONCURRENCIES, REQUESTS_PER_CELL, MAX_TOKENS
#   GPU_IDLE_MIB           — memory threshold (MiB) below which the GPU is "idle"; default 500
#   GPU_IDLE_TIMEOUT_S     — seconds to wait for GPU idle between runs; default 60
#   CUDA_VISIBLE_DEVICES   — recorded in the report; passes through unchanged
#   SKIP_FORGE=1 / SKIP_VLLM=1  — benchmark only one engine
#   ALLOW_ERRORS=1         — keep going even when a cell has request errors
#
# Prereqs:
#   - forge built (cargo build --release -p forge-server)
#   - vLLM installed: `pip install vllm==0.18.*`
#   - python3 (stdlib only)
#   - nvidia-smi (used for GPU-idle polling)
#   - ss or nc (used for port preflight)

set -euo pipefail

# In SELFTEST mode we run unit-test-style checks against the helpers
# without launching a server. The model path is therefore optional in that
# mode; otherwise it is required.
if [ "${SELFTEST:-0}" = "1" ]; then
    MODEL_PATH="${1:-}"
else
    MODEL_PATH="${1:?Usage: $0 <hf-model-path> (or SELFTEST=1 $0 to run self-tests)}"
fi

# forge derives model_name from basename(model_path) and ignores any client-
# supplied model field; we point vLLM at the same basename so both engines
# answer /v1/models with the same id and the readiness check is unambiguous.
SERVED_NAME="$(basename "${MODEL_PATH%/}")"

FORGE_PORT="${FORGE_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"

FORGE_HOST="http://localhost:${FORGE_PORT}"
VLLM_HOST="http://localhost:${VLLM_PORT}"

PROMPT_LENS="${PROMPT_LENS:-128,1024}"
CONCURRENCIES="${CONCURRENCIES:-1,8,32}"
REQUESTS_PER_CELL="${REQUESTS_PER_CELL:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"
GPU_IDLE_MIB="${GPU_IDLE_MIB:-500}"
GPU_IDLE_TIMEOUT_S="${GPU_IDLE_TIMEOUT_S:-60}"

SKIP_FORGE="${SKIP_FORGE:-0}"
SKIP_VLLM="${SKIP_VLLM:-0}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/.reports"
mkdir -p "${RESULTS_DIR}"
STAMP="$(date +%Y%m%d-%H%M%S)"
FORGE_JSON="${RESULTS_DIR}/forge-${STAMP}.json"
VLLM_JSON="${RESULTS_DIR}/vllm-${STAMP}.json"
REPORT_FILE="${RESULTS_DIR}/vllm-comparison-${STAMP}.md"
FORGE_LOG="${RESULTS_DIR}/forge-server-${STAMP}.log"
VLLM_LOG="${RESULTS_DIR}/vllm-server-${STAMP}.log"

GPU_INFO="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"

echo "=== forge vs vLLM benchmark (serial) ==="
echo "Model:        ${MODEL_PATH}"
echo "Served name:  ${SERVED_NAME}"
echo "Prompt lens:  ${PROMPT_LENS}"
echo "Concurrency:  ${CONCURRENCIES}"
echo "Req/cell:     ${REQUESTS_PER_CELL}"
echo "Max tokens:   ${MAX_TOKENS}"
echo "GPU:          ${GPU_INFO:-unknown}"
echo "CUDA dev:     ${CUDA_VISIBLE_DEVICES:-default}"
echo "Report:       ${REPORT_FILE}"
echo ""

# --- Helpers ---------------------------------------------------------------

current_engine_pid=""

cleanup() {
    if [ -n "${current_engine_pid}" ] && kill -0 "${current_engine_pid}" 2>/dev/null; then
        echo ""
        echo "Stopping current engine (PID ${current_engine_pid})..."
        kill "${current_engine_pid}" 2>/dev/null || true
        wait "${current_engine_pid}" 2>/dev/null || true
        current_engine_pid=""
    fi
}
trap cleanup EXIT

gpu_mem_used_mib() {
    # Reports max over all visible GPUs to avoid races on multi-GPU machines.
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | sort -n | tail -n1
}

ensure_port_free() {
    # Refuses to launch when something is already listening on the target
    # port — otherwise the benchmark could end up curling the stale process
    # and reporting its numbers instead of the freshly-spawned engine's.
    local port="$1" label="$2"
    if command -v ss > /dev/null 2>&1; then
        if ss -ltn "sport = :${port}" 2>/dev/null | tail -n +2 | grep -q .; then
            echo "ERROR: port ${port} is already in use; refusing to launch ${label}" >&2
            echo "  Diagnose: ss -ltnp 'sport = :${port}'" >&2
            return 1
        fi
    elif command -v nc > /dev/null 2>&1; then
        if nc -z localhost "${port}" 2>/dev/null; then
            echo "ERROR: port ${port} is already in use; refusing to launch ${label}" >&2
            return 1
        fi
    else
        echo "WARN: neither ss nor nc available; skipping port-${port} preflight" >&2
    fi
}

# Python verifier as a single-quoted -c program. Reads /v1/models JSON from
# stdin, takes the expected served-model-name as argv[1], exits 0 iff the
# response is well-formed and lists the expected id.
#
# IMPORTANT: keep this as `python3 -c '<program>' "<arg>" <<<"<body>"`. An
# earlier version used `python3 - "<arg>" <<'PY' ... PY` which lets the
# heredoc claim stdin, silently discarding the piped body — the verifier
# then sees empty stdin and exits 1 on every healthy response.
VERIFY_MODELS_PROGRAM='
import json, sys
try:
    r = json.load(sys.stdin)
except Exception as e:
    sys.stderr.write(f"verify: invalid JSON: {e}\n")
    sys.exit(1)
expected = sys.argv[1]
data = r.get("data") if isinstance(r, dict) else None
if not isinstance(data, list) or not data:
    sys.stderr.write("verify: response has no models\n")
    sys.exit(1)
ids = [m.get("id", "") for m in data if isinstance(m, dict)]
if expected not in ids:
    sys.stderr.write(f"verify: served-model-name mismatch: expected {expected!r}, got {ids}\n")
    sys.exit(1)
'

verify_models_response() {
    # $1 = body, $2 = expected model id. Returns 0 on success, non-zero
    # otherwise. stderr from the verifier is suppressed unless VERBOSE=1.
    local body="$1" expected="$2"
    if [ "${VERBOSE:-0}" = "1" ]; then
        python3 -c "${VERIFY_MODELS_PROGRAM}" "${expected}" <<<"${body}"
    else
        python3 -c "${VERIFY_MODELS_PROGRAM}" "${expected}" <<<"${body}" 2>/dev/null
    fi
}

wait_for_health() {
    # Readiness check that:
    #   (a) aborts immediately if the spawned PID exits,
    #   (b) requires /v1/models to list the expected served-model-name,
    # so that a stale forge/vLLM on the same port can't satisfy the probe.
    local url="$1" label="$2" expected_model="$3" pid="$4" timeout="${5:-180}"
    echo "Waiting for ${label} at ${url} (PID ${pid}, expecting model='${expected_model}', up to ${timeout}s)..."
    for i in $(seq 1 "${timeout}"); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${label} process (PID ${pid}) exited before becoming healthy" >&2
            return 1
        fi
        local body
        # Bound curl explicitly: without --connect-timeout/--max-time a server
        # that accepts the TCP connection but stalls on /v1/models would block
        # the command substitution forever, bypassing the outer timeout loop
        # and the spawned-PID liveness check.
        body="$(curl -sf --connect-timeout 1 --max-time 2 "${url}" 2>/dev/null || true)"
        if [ -n "${body}" ] && verify_models_response "${body}" "${expected_model}"; then
            echo "  ${label} ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: ${label} failed to become healthy within ${timeout}s" >&2
    return 1
}

selftest_verify_models() {
    # Internal self-test to catch regressions in the readiness verifier
    # (including any future stdin-routing mistakes). Run with SELFTEST=1.
    local passed=0 failed=0
    local good_body bad_id_body bad_json_body empty_data_body

    good_body='{"object":"list","data":[{"id":"Llama-3.2-1B-Instruct","object":"model"}]}'
    bad_id_body='{"object":"list","data":[{"id":"other-model","object":"model"}]}'
    bad_json_body='not json at all'
    empty_data_body='{"object":"list","data":[]}'

    check() {
        local name="$1" expect_exit="$2" body="$3" expected="$4"
        verify_models_response "${body}" "${expected}"
        local rc=$?
        if [ "${rc}" -eq "${expect_exit}" ]; then
            echo "  PASS: ${name}"
            passed=$((passed + 1))
        else
            echo "  FAIL: ${name} (expected exit ${expect_exit}, got ${rc})" >&2
            failed=$((failed + 1))
        fi
    }

    echo "Running selftest_verify_models..."
    check "matching id passes" 0 "${good_body}" "Llama-3.2-1B-Instruct"
    check "wrong id fails"     1 "${bad_id_body}" "Llama-3.2-1B-Instruct"
    check "bad json fails"     1 "${bad_json_body}" "Llama-3.2-1B-Instruct"
    check "empty data fails"   1 "${empty_data_body}" "Llama-3.2-1B-Instruct"
    check "empty body fails"   1 "" "Llama-3.2-1B-Instruct"
    echo "selftest_verify_models: ${passed} passed, ${failed} failed"
    [ "${failed}" -eq 0 ]
}

wait_for_gpu_idle() {
    local threshold="${1:-${GPU_IDLE_MIB}}" timeout="${2:-${GPU_IDLE_TIMEOUT_S}}"
    if ! command -v nvidia-smi > /dev/null; then
        echo "  nvidia-smi missing; skipping GPU-idle wait (results may be confounded)"
        sleep 5
        return 0
    fi
    echo "Waiting for GPU memory to fall below ${threshold} MiB (timeout ${timeout}s)..."
    for i in $(seq 1 "${timeout}"); do
        local used
        used="$(gpu_mem_used_mib || echo 0)"
        if [ -n "${used}" ] && [ "${used}" -lt "${threshold}" ]; then
            echo "  GPU idle (${used} MiB used) after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "WARN: GPU still busy after ${timeout}s; proceeding anyway." >&2
    return 0
}

stop_engine() {
    if [ -n "${current_engine_pid}" ]; then
        echo "Stopping engine (PID ${current_engine_pid})..."
        kill "${current_engine_pid}" 2>/dev/null || true
        wait "${current_engine_pid}" 2>/dev/null || true
        current_engine_pid=""
        wait_for_gpu_idle
    fi
}

run_engine() {
    local engine="$1" host="$2" json_out="$3"
    echo ""
    echo "--- Benchmarking ${engine} ---"
    local allow_errors_flag=""
    if [ "${ALLOW_ERRORS:-0}" = "1" ]; then
        allow_errors_flag="--allow-errors"
    fi
    # If the python script fails closed (exit 2), propagate the failure so
    # the merge step never sees a half-built fragment.
    python3 "${REPO_ROOT}/scripts/benchmark_vllm.py" \
        --mode benchmark \
        --engine "${engine}" \
        --host "${host}" \
        --served-name "${SERVED_NAME}" \
        --prompt-lens "${PROMPT_LENS}" \
        --concurrencies "${CONCURRENCIES}" \
        --requests-per-cell "${REQUESTS_PER_CELL}" \
        --max-tokens "${MAX_TOKENS}" \
        --out-json "${json_out}" \
        ${allow_errors_flag}
}

# --- SELFTEST gate ---------------------------------------------------------

if [ "${SELFTEST:-0}" = "1" ]; then
    if selftest_verify_models; then
        echo ""
        echo "=== SELFTEST passed ==="
        exit 0
    else
        echo ""
        echo "=== SELFTEST FAILED ===" >&2
        exit 1
    fi
fi

# --- forge run -------------------------------------------------------------

if [ "${SKIP_FORGE}" != "1" ]; then
    if [ ! -x "${REPO_ROOT}/target/release/forge-server" ]; then
        echo "ERROR: ${REPO_ROOT}/target/release/forge-server missing. Run \`cargo build --release -p forge-server\` first." >&2
        exit 1
    fi

    wait_for_gpu_idle

    if ! ensure_port_free "${FORGE_PORT}" "forge"; then
        exit 1
    fi

    echo "Starting forge..."
    "${REPO_ROOT}/target/release/forge-server" \
        --model-path "${MODEL_PATH}" \
        --port "${FORGE_PORT}" \
        > "${FORGE_LOG}" 2>&1 &
    current_engine_pid=$!

    if ! wait_for_health "${FORGE_HOST}/v1/models" "forge" "${SERVED_NAME}" "${current_engine_pid}"; then
        echo "  See ${FORGE_LOG}" >&2
        exit 1
    fi

    run_engine forge "${FORGE_HOST}" "${FORGE_JSON}"
    stop_engine
fi

# --- vLLM run --------------------------------------------------------------

if [ "${SKIP_VLLM}" != "1" ]; then
    echo ""
    if ! ensure_port_free "${VLLM_PORT}" "vLLM"; then
        exit 1
    fi

    echo "Starting vLLM..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_NAME}" \
        --port "${VLLM_PORT}" \
        --disable-log-requests \
        > "${VLLM_LOG}" 2>&1 &
    current_engine_pid=$!

    if ! wait_for_health "${VLLM_HOST}/v1/models" "vLLM" "${SERVED_NAME}" "${current_engine_pid}" 300; then
        echo "  See ${VLLM_LOG}" >&2
        exit 1
    fi

    run_engine vLLM "${VLLM_HOST}" "${VLLM_JSON}"
    stop_engine
fi

# --- Merge -----------------------------------------------------------------

echo ""
echo "--- Merging fragments into report ---"
fragments=()
[ -f "${FORGE_JSON}" ] && fragments+=("${FORGE_JSON}")
[ -f "${VLLM_JSON}" ] && fragments+=("${VLLM_JSON}")

if [ ${#fragments[@]} -eq 0 ]; then
    echo "ERROR: no fragments produced; nothing to merge" >&2
    exit 1
fi

python3 "${REPO_ROOT}/scripts/benchmark_vllm.py" \
    --mode merge \
    --fragments "${fragments[@]}" \
    --report "${REPORT_FILE}" \
    --gpu "${GPU_INFO}" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-}"

echo ""
echo "=== Report written to ${REPORT_FILE} ==="
