#!/usr/bin/env python3
"""TTFT / TPOT decode benchmark for any OpenAI-compatible endpoint, separating
prefill from decode the way pegainfer / `vllm bench serve` do.

Usage:
    python3 scripts/bench_openai.py URL [model] [out_tokens] [prompt_tokens] [runs]
    e.g. python3 scripts/bench_openai.py http://localhost:8950 qwen3-4b 256 1024 5

Streams the response and measures, per request (greedy, temperature=0):
  - TTFT   = time to first content token (tokenize + prefill + first decode)
  - TPOT   = mean inter-token gap over the steady state (pure decode/token)
  - decode tok/s = 1000 / TPOT_ms   (the comparable "decode throughput")
  - e2e tok/s    = output_tokens / total_wall   (includes prefill)

`prompt_tokens` builds a synthetic prompt of ~that many tokens (≈4 chars each)
to mimic pegainfer's decode_heavy (prompt_len=1024). A long, open-ended prompt
keeps generation going to `out_tokens` (no early EOS) so TPOT is steady-state.
"""
import json, sys, time, urllib.request, statistics

host = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8080"
model = sys.argv[2] if len(sys.argv) > 2 else "x"
out_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256
prompt_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
runs = int(sys.argv[5]) if len(sys.argv) > 5 else 5

# ~prompt_tokens tokens of filler, then an open-ended instruction so the model
# keeps generating to the token cap instead of stopping early.
_filler = ("The history of computing spans many decades and countless "
           "innovations across hardware and software. ")
prompt = (_filler * ((prompt_tokens // 12) + 1))[: prompt_tokens * 4] + \
    "\n\nContinue this essay in exhaustive detail, never stopping early."


def one():
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(f"{host}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    stamps = []  # arrival time of each content token
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            p = line[6:]
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            # Count any generated-text delta: `content`, or `reasoning`
            # (pegainfer/OpenAI put thinking-mode tokens in `reasoning`).
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning"):
                now = time.perf_counter()
                if first is None:
                    first = now
                stamps.append(now)
    ttft = (first - t0) * 1000 if first else 0.0
    n = len(stamps)
    # steady-state TPOT: mean gap between consecutive tokens after the first.
    gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, n)]
    tpot = statistics.mean(gaps) if gaps else 0.0
    e2e = (stamps[-1] - t0) if stamps else 0.0
    return ttft, tpot, n, e2e


def main():
    print(f"endpoint={host} model={model} prompt~{prompt_tokens}tok "
          f"out={out_tokens} runs={runs}")
    def safe_one():
        for _ in range(3):  # retry transient 5xx
            try:
                return one()
            except Exception as e:
                print(f"    (retry after error: {e})")
                time.sleep(2)
        return None

    for _ in range(2):
        safe_one()
    ttfts, tpots, e2es, ns = [], [], [], []
    i = 0
    while len(tpots) < runs:
        i += 1
        if i > runs * 3:
            break
        r = safe_one()
        if r is None:
            continue
        ttft, tpot, n, e2e = r
        if n < 2:
            continue
        ttfts.append(ttft); tpots.append(tpot); e2es.append(e2e); ns.append(n)
        print(f"  run {len(tpots)}: TTFT={ttft:6.1f}ms  TPOT={tpot:5.2f}ms  "
              f"decode={1000/tpot if tpot else 0:5.1f} tok/s  ({n} tok, e2e {e2e:.2f}s)")
    if not tpots:
        print("no successful runs"); return
    avg_tpot = statistics.mean(tpots)
    print(f"=> TTFT(prefill) {statistics.mean(ttfts):.1f} ms | "
          f"TPOT {avg_tpot:.2f} ms | decode {1000/avg_tpot:.1f} tok/s | "
          f"e2e {sum(ns)/sum(e2es):.1f} tok/s")


if __name__ == "__main__":
    main()
