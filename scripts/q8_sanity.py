#!/usr/bin/env python3
"""Q8_0 decode vs FP16 token-level sanity for forge.

Spins two passes against a forge endpoint (operator restarts the server
between passes — FP16 default then `--quantize-decode`) and compares
greedy completions character-by-character on the same prompt set.

The intent is to detect *quality* regressions in the Q8 path, not to
demand byte-equivalence: Q8_0 effective precision is ~8 mantissa bits
versus FP16's 10, so greedy top-1 will occasionally flip when the
top-2 candidates are close to a tie. Character divergence is expected;
semantic equivalence is not.

Usage:

    # 1. Start forge in FP16 mode, then collect:
    python3 scripts/q8_sanity.py collect \\
        --url http://localhost:8951 --model qwen3-8b \\
        --out /tmp/q8_sanity/fp16.json

    # 2. Restart forge with --quantize-decode, then collect:
    python3 scripts/q8_sanity.py collect \\
        --url http://localhost:8951 --model qwen3-8b \\
        --out /tmp/q8_sanity/q8.json

    # 3. Compare:
    python3 scripts/q8_sanity.py compare \\
        --fp16 /tmp/q8_sanity/fp16.json --q8 /tmp/q8_sanity/q8.json

Custom prompt set: pass `--prompts <path>` pointing at a JSON list of
strings (the default eight-prompt fixture is embedded for quick checks).

Background and locked sanity numbers:
- docs/investigations/2026-06-19-q8-vs-fp16-sanity.md
- docs/RUNBOOK.md ("When to use --quantize-decode")
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_PROMPTS = [
    "Explain the principle of operation of FlashAttention v2 in one paragraph.",
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    "What are the first ten primes? List them.",
    "Summarize what a CUDA stream is, in two sentences.",
    "Write a Python function that returns the nth Fibonacci number using memoization.",
    "1 + 1 = ?",
    "Give me three uses of paged attention in LLM serving.",
    "What's the time complexity of FlashAttention's online softmax?",
]


def greedy_completion(url: str, model: str, prompt: str, n_tok: int) -> str:
    """Hit /v1/chat/completions with greedy decode and return the full text.

    Forge returns the thinking trace inline inside `content`; pegainfer
    splits it into a separate `reasoning` field. Normalize both shapes
    and strip <think>...</think> wrappers so the comparison is on
    actual content.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_tok,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    s = (msg.get("reasoning") or "") + (msg.get("content") or "")
    return s.replace("<think>", "").replace("</think>", "").lstrip()


def collect(url: str, model: str, prompts: list[str], n_tok: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {p[:70]}", file=sys.stderr)
        rows.append(greedy_completion(url, model, p, n_tok))
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path} ({len(rows)} completions)")


def compare_pair(a: str, b: str) -> tuple[str, int, int]:
    """Return (status, shared_prefix_chars, max_len)."""
    n = min(len(a), len(b))
    prefix = next((i for i in range(n) if a[i] != b[i]), n)
    matches_full = prefix == len(a) == len(b)
    return ("IDENTICAL" if matches_full else "DIVERGE", prefix, max(len(a), len(b)))


def compare(fp16_path: Path, q8_path: Path) -> int:
    a = json.loads(fp16_path.read_text())
    b = json.loads(q8_path.read_text())
    if len(a) != len(b):
        print(f"ERROR: prompt counts differ ({len(a)} vs {len(b)})", file=sys.stderr)
        return 2

    identical = 0
    divergent = 0
    diverge_pcts: list[float] = []
    for i, (x, y) in enumerate(zip(a, b), 1):
        status, prefix, max_len = compare_pair(x, y)
        pct = (prefix / max_len * 100) if max_len else 100
        if status == "IDENTICAL":
            identical += 1
            print(f"[{i}] IDENTICAL  |fp16={len(x)} q8={len(y)}|")
        else:
            divergent += 1
            diverge_pcts.append(pct)
            print(f"[{i}] diverge@{prefix}({pct:.0f}%)  |fp16={len(x)} q8={len(y)}|")
        print(f"    fp16: {x[:80]!r}")
        print(f"    q8  : {y[:80]!r}")

    print()
    print(f"=== {identical}/{len(a)} identical, "
          f"{divergent}/{len(a)} divergent ===")
    if diverge_pcts:
        avg = sum(diverge_pcts) / len(diverge_pcts)
        print(f"    average first divergence at {avg:.0f}% into reply")
    # Exit code is informational, not a gate — quality judgement belongs
    # to the operator reading the output.
    return 0


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    with open(path) as f:
        loaded = json.load(f)
    if not (isinstance(loaded, list) and all(isinstance(s, str) for s in loaded)):
        raise SystemExit(f"{path} must be a JSON list of strings")
    return loaded


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="hit a forge endpoint and save outputs")
    c.add_argument("--url", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--prompts", help="optional JSON list of prompts (default: 8 embedded)")

    cmp = sub.add_parser("compare", help="diff two collected output sets character-by-character")
    cmp.add_argument("--fp16", required=True, type=Path)
    cmp.add_argument("--q8", required=True, type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "collect":
        prompts = load_prompts(args.prompts)
        collect(args.url, args.model, prompts, args.max_tokens, args.out)
        return 0
    if args.cmd == "compare":
        return compare(args.fp16, args.q8)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
