#!/usr/bin/env python3
"""Concurrent decode-throughput benchmark for an OpenAI-compatible endpoint.

Fires `concurrency` streaming requests simultaneously (same ~prompt_tokens
prompt, greedy, out_tokens each) and measures how decode behaves under a batch:

  - per-stream TPOT  = mean inter-token gap within one stream (decode latency)
  - aggregate decode tok/s = total decoded tokens / steady-state window
        (window = from when the LAST stream produced its first token
         to when the FIRST stream produced its last token — the interval
         during which all streams are concurrently decoding)

The aggregate number is the batch-decode throughput lever: if the engine
batches decode well, aggregate tok/s should grow with concurrency even as
per-stream TPOT rises.

Usage:
    python3 bench_concurrent.py URL model out_tokens prompt_tokens concurrency [runs]
"""
import json, sys, time, urllib.request, statistics
from concurrent.futures import ThreadPoolExecutor

host = sys.argv[1].rstrip("/")
model = sys.argv[2] if len(sys.argv) > 2 else "x"
out_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256
prompt_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
concurrency = int(sys.argv[5]) if len(sys.argv) > 5 else 1
runs = int(sys.argv[6]) if len(sys.argv) > 6 else 3

_filler = ("The history of computing spans many decades and countless "
           "innovations across hardware and software. ")
prompt = (_filler * ((prompt_tokens // 12) + 1))[: prompt_tokens * 4] + \
    "\n\nContinue this essay in exhaustive detail, never stopping early."


def one_stream():
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(f"{host}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    stamps = []
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
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning"):
                stamps.append(time.perf_counter())
    return stamps


def one_batch():
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda _: one_stream(), range(concurrency)))
    results = [s for s in results if len(s) >= 2]
    if not results:
        return None
    # per-stream TPOT
    tpots = []
    for s in results:
        gaps = [(s[i] - s[i - 1]) * 1000 for i in range(1, len(s))]
        tpots.append(statistics.mean(gaps))
    # steady-state window: all streams concurrently decoding
    first_tokens = [s[0] for s in results]
    last_tokens = [s[-1] for s in results]
    win_start = max(first_tokens)   # last stream to start
    win_end = min(last_tokens)      # first stream to finish
    total_tokens = sum(len(s) for s in results)
    # tokens produced inside the steady-state window, across all streams
    in_win = sum(sum(1 for t in s if win_start <= t <= win_end) for s in results)
    window = win_end - win_start
    agg_steady = in_win / window if window > 0 else 0.0
    # also a simpler aggregate: all tokens / full span
    span = max(last_tokens) - min(first_tokens)
    agg_span = total_tokens / span if span > 0 else 0.0
    return statistics.mean(tpots), agg_steady, agg_span, total_tokens


def main():
    print(f"endpoint={host} model={model} prompt~{prompt_tokens}tok "
          f"out={out_tokens} concurrency={concurrency} runs={runs}")
    one_batch()  # warmup
    tpots, aggs, spans = [], [], []
    for r in range(runs):
        res = one_batch()
        if res is None:
            print(f"  run {r+1}: FAILED"); continue
        tpot, agg_steady, agg_span, tot = res
        tpots.append(tpot); aggs.append(agg_steady); spans.append(agg_span)
        print(f"  run {r+1}: per-stream TPOT={tpot:6.2f}ms  "
              f"agg(steady)={agg_steady:6.1f} tok/s  agg(span)={agg_span:6.1f} tok/s  "
              f"({tot} tok total)")
    if tpots:
        print(f"=> C={concurrency}: TPOT {statistics.mean(tpots):.2f}ms | "
              f"agg-steady {statistics.mean(aggs):.1f} tok/s | "
              f"agg-span {statistics.mean(spans):.1f} tok/s")


if __name__ == "__main__":
    main()
