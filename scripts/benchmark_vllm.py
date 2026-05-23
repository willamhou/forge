#!/usr/bin/env python3
"""Comparative benchmark client for forge and vLLM.

Two modes:

1. `--mode benchmark` (default): drive concurrent SSE requests against ONE
   engine endpoint, emit cell results to a JSON fragment. Designed to be
   invoked twice by `benchmark_vllm.sh` — once per engine, with the *other*
   engine torn down — so the two runs do not compete for the same GPU.

2. `--mode merge`: combine two JSON fragments into a markdown report.

Stdlib-only (urllib + threading). No pip dependencies.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Optional


# --- Prompt corpus ---------------------------------------------------------

# Neutral seed paragraph; we repeat it to hit a target prompt length.
# Rough token-to-char ratio for English ≈ 4:1 (Llama-family tokenizers).
SEED = (
    "The history of computing is a story of relentless miniaturization, "
    "shifting abstractions, and recurring battles between generality and "
    "specialization. Early machines filled rooms; today, more compute fits "
    "on a single die than was available globally a generation ago. "
)


def make_prompt(target_tokens: int) -> str:
    chars_per_token = 4
    target_chars = max(target_tokens * chars_per_token, len(SEED))
    repeats = max(1, target_chars // len(SEED))
    body = SEED * repeats
    return body + "\n\nSummarize the passage above in one paragraph."


# --- In-flight tracker -----------------------------------------------------


class InFlightTracker:
    """Threadsafe counter for active concurrent requests; tracks peak."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self._peak = 0

    def enter(self) -> None:
        with self._lock:
            self._current += 1
            if self._current > self._peak:
                self._peak = self._current

    def exit(self) -> None:
        with self._lock:
            self._current -= 1

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak


# --- Per-request measurement -----------------------------------------------


@dataclass
class RequestResult:
    ttft_s: float
    itls_s: list[float]
    completion_tokens: int
    prompt_tokens: int
    wallclock_s: float
    error: Optional[str] = None


def stream_one(
    host: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    tracker: Optional[InFlightTracker] = None,
) -> RequestResult:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()

    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    if tracker:
        tracker.enter()

    start = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    itls: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        else:
                            itls.append(now - (last_token_time or now))
                        last_token_time = now
                        completion_tokens += 1

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        if tracker:
            tracker.exit()
        return RequestResult(
            ttft_s=0.0, itls_s=[], completion_tokens=0, prompt_tokens=0,
            wallclock_s=time.perf_counter() - start, error=str(exc),
        )

    if tracker:
        tracker.exit()

    wallclock = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time is not None else 0.0
    return RequestResult(
        ttft_s=ttft, itls_s=itls, completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens, wallclock_s=wallclock,
    )


# --- Cell aggregation ------------------------------------------------------


@dataclass
class CellStats:
    engine: str
    prompt_len: int
    concurrency: int
    requests: int
    ttft_avg_ms: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    itl_avg_ms: float
    throughput_tok_s: float
    errors: int
    actual_prompt_tokens_avg: int
    completion_tokens_total: int
    wallclock_s: float
    peak_in_flight: int

    extras: dict = field(default_factory=dict)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    idx = min(int(len(xs) * p), len(xs) - 1)
    return sorted(xs)[idx]


def run_cell(
    engine: str,
    host: str,
    model: str,
    prompt_len: int,
    concurrency: int,
    requests: int,
    max_tokens: int,
    timeout: float,
) -> CellStats:
    if requests < concurrency:
        raise ValueError(
            f"requests ({requests}) must be >= concurrency ({concurrency}); "
            "otherwise the cell cannot exercise the declared concurrency"
        )

    prompt = make_prompt(prompt_len)

    # Warm-up at *target concurrency*: warm caches, capture any CUDA Graphs
    # for this bucket, and bring the engine to steady-state before timing.
    # We need at least `concurrency` concurrent in-flight requests to
    # exercise the bucket; we also use shorter generations to keep warm-up
    # cheap.
    warm_tracker = InFlightTracker()
    with ThreadPoolExecutor(max_workers=concurrency) as warm:
        warm_results = list(warm.map(
            lambda _: stream_one(host, model, prompt, 16, timeout, warm_tracker),
            range(concurrency),
        ))
    warm_errors = sum(1 for r in warm_results if r.error is not None)
    if warm_tracker.peak < concurrency:
        print(
            f"  WARN: warm-up peak in-flight = {warm_tracker.peak} < target {concurrency}; "
            "cell may not be exercising the declared concurrency.",
            file=sys.stderr, flush=True,
        )

    tracker = InFlightTracker()
    results: list[RequestResult] = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_one, host, model, prompt, max_tokens, timeout, tracker)
            for _ in range(requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    wallclock = time.perf_counter() - start

    ttfts = [r.ttft_s for r in results if r.error is None and r.ttft_s > 0]
    itls = [t for r in results if r.error is None for t in r.itls_s]
    timed_errors = sum(1 for r in results if r.error is not None)
    completion_total = sum(r.completion_tokens for r in results if r.error is None)
    prompt_tokens_seen = [r.prompt_tokens for r in results if r.error is None and r.prompt_tokens > 0]
    actual_prompt_avg = int(statistics.mean(prompt_tokens_seen)) if prompt_tokens_seen else 0

    return CellStats(
        engine=engine,
        prompt_len=prompt_len,
        concurrency=concurrency,
        requests=requests,
        ttft_avg_ms=(statistics.mean(ttfts) * 1000) if ttfts else 0.0,
        ttft_p50_ms=(pct(ttfts, 0.50) * 1000),
        ttft_p99_ms=(pct(ttfts, 0.99) * 1000),
        itl_avg_ms=(statistics.mean(itls) * 1000) if itls else 0.0,
        throughput_tok_s=(completion_total / wallclock) if wallclock > 0 else 0.0,
        # Total errors = warm-up errors + timed-window errors. Cell validity
        # checks downstream treat any non-zero value as a hard failure.
        errors=timed_errors + warm_errors,
        actual_prompt_tokens_avg=actual_prompt_avg,
        completion_tokens_total=completion_total,
        wallclock_s=wallclock,
        peak_in_flight=tracker.peak,
        extras={"warm_errors": warm_errors, "timed_errors": timed_errors},
    )


# --- Reporting -------------------------------------------------------------


def render_markdown(cells: list[CellStats], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# forge vs vLLM benchmark\n")

    total_errors = sum(c.errors for c in cells)
    invalid_cells = sum(1 for c in cells if validate_cell(c))

    if total_errors > 0 or invalid_cells > 0:
        lines.append(
            f"> ⚠️ **UNTRUSTED**: {invalid_cells} of {len(cells)} cells failed validation "
            f"(total errors: {total_errors}). This report was generated with `--allow-errors` "
            f"and MUST NOT be used as a perf gate.\n"
        )

    for k in ("forge_host", "vllm_host", "model", "generated", "gpu", "cuda_visible_devices", "max_tokens"):
        if k in meta and meta[k] not in (None, ""):
            lines.append(f"- {k.replace('_', ' ')}: `{meta[k]}`")
    lines.append("")

    grouped: dict[tuple[int, int], dict[str, CellStats]] = {}
    for c in cells:
        grouped.setdefault((c.prompt_len, c.concurrency), {})[c.engine] = c

    lines.append("| prompt_len (target/actual) | concurrency (decl/peak) | engine | requests | TTFT avg/p50/p99 (ms) | ITL avg (ms) | throughput (tok/s) | errors |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (plen, conc), engines in sorted(grouped.items()):
        actual = next((e.actual_prompt_tokens_avg for e in engines.values() if e.actual_prompt_tokens_avg), 0)
        for engine in ("forge", "vLLM"):
            c = engines.get(engine)
            if not c:
                continue
            lines.append(
                f"| {plen}/{actual} | {conc}/{c.peak_in_flight} | {engine} | {c.requests} | "
                f"{c.ttft_avg_ms:.1f}/{c.ttft_p50_ms:.1f}/{c.ttft_p99_ms:.1f} | "
                f"{c.itl_avg_ms:.2f} | {c.throughput_tok_s:.1f} | {c.errors} |"
            )

    lines.append("")
    lines.append("## Notes\n")
    lines.append("- Engines were benchmarked **serially**, with the other engine torn down between runs. Each run owned the GPU exclusively.")
    lines.append("- `prompt_len`: target token count for synthetic prompts; `actual` reflects what the server tokenized.")
    lines.append("- `concurrency (decl/peak)`: declared client concurrency vs the peak in-flight request count actually observed during the timed window. If `peak < decl`, the cell is not honoring the declared concurrency.")
    lines.append("- `TTFT`: time from request send to first streamed token.")
    lines.append("- `ITL`: inter-token latency between subsequent streamed tokens, averaged.")
    lines.append("- `throughput`: aggregate completion tokens / wallclock for the cell.")
    lines.append("- Warm-up: `concurrency` parallel short requests per cell before timing; results discarded.")
    lines.append("- Temperature: 0 (greedy).")
    return "\n".join(lines)


# --- Main ------------------------------------------------------------------


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def validate_cell(c: CellStats) -> list[str]:
    """Return a list of reasons this cell should be treated as invalid.

    Empty list = cell is trustworthy. Non-empty = the cell contributes
    misleading data and the caller should fail closed.
    """
    reasons: list[str] = []
    if c.errors > 0:
        reasons.append(f"errors={c.errors}")
    if c.completion_tokens_total == 0:
        reasons.append("completion_tokens_total=0 (no successful generation)")
    if c.actual_prompt_tokens_avg == 0:
        reasons.append("actual_prompt_tokens_avg=0 (no usage reported)")
    # Declared concurrency must actually have been exercised — otherwise the
    # cell can claim "batch_size=32" while never reaching more than a few
    # in-flight requests, which is exactly the scenario CUDA Graph buckets
    # exist to measure. An underloaded cell silently mislabels the bucket.
    if c.peak_in_flight < c.concurrency:
        reasons.append(
            f"peak_in_flight={c.peak_in_flight} < declared concurrency={c.concurrency} "
            "(cell did not exercise the bucket it claims to measure)"
        )
    return reasons


def cmd_benchmark(args: argparse.Namespace) -> int:
    if not args.host or not args.engine:
        print("ERROR: --host and --engine are required in benchmark mode", file=sys.stderr)
        return 2

    prompt_lens = parse_int_list(args.prompt_lens)
    concurrencies = parse_int_list(args.concurrencies)

    for conc in concurrencies:
        if args.requests_per_cell < conc:
            print(
                f"ERROR: --requests-per-cell={args.requests_per_cell} < concurrency={conc}; "
                f"set --requests-per-cell to at least {max(concurrencies)} (or 4x highest concurrency).",
                file=sys.stderr,
            )
            return 2

    cells: list[CellStats] = []
    for plen in prompt_lens:
        for conc in concurrencies:
            print(
                f"[{args.engine}] prompt_len={plen} concurrency={conc} requests={args.requests_per_cell}...",
                flush=True,
            )
            cell = run_cell(
                engine=args.engine,
                host=args.host,
                model=args.served_name,
                prompt_len=plen,
                concurrency=conc,
                requests=args.requests_per_cell,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            cells.append(cell)
            print(
                f"  TTFT avg/p50/p99 = {cell.ttft_avg_ms:.1f}/{cell.ttft_p50_ms:.1f}/{cell.ttft_p99_ms:.1f} ms | "
                f"ITL avg = {cell.itl_avg_ms:.2f} ms | thr = {cell.throughput_tok_s:.1f} tok/s | "
                f"peak in-flight = {cell.peak_in_flight}/{conc} | errors = {cell.errors}",
                flush=True,
            )

    # Fail closed: if any cell has errors, zero completions, or missing
    # usage info, the fragment is not trustworthy as a perf gate. Operators
    # can opt in to lenient mode with --allow-errors when debugging.
    invalid: list[tuple[CellStats, list[str]]] = []
    for c in cells:
        reasons = validate_cell(c)
        if reasons:
            invalid.append((c, reasons))

    if invalid:
        print("", file=sys.stderr)
        print(f"ERROR: {len(invalid)} of {len(cells)} cells failed validation:", file=sys.stderr)
        for c, reasons in invalid:
            print(
                f"  - [{c.engine}] prompt_len={c.prompt_len} concurrency={c.concurrency}: {', '.join(reasons)}",
                file=sys.stderr,
            )
        if not args.allow_errors:
            print(
                "Refusing to write fragment. Pass --allow-errors to override (NOT for perf gates).",
                file=sys.stderr,
            )
            return 2
        print("WARNING: --allow-errors set; writing fragment anyway (do not trust as perf gate).", file=sys.stderr)

    fragment = {
        "engine": args.engine,
        "host": args.host,
        "model": args.served_name,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "max_tokens": args.max_tokens,
        "invalid_cells": len(invalid),
        "cells": [asdict(c) for c in cells],
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(fragment, fh, indent=2)
        print(f"\nWrote {args.out_json}", flush=True)
    else:
        print(json.dumps(fragment, indent=2))

    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    if not args.fragments:
        print("ERROR: --fragments is required in merge mode", file=sys.stderr)
        return 2

    all_cells: list[CellStats] = []
    meta: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "max_tokens": None,
    }
    for path in args.fragments:
        with open(path, "r", encoding="utf-8") as fh:
            frag = json.load(fh)
        host_key = f"{frag['engine'].lower()}_host"
        meta[host_key] = frag.get("host")
        meta["model"] = frag.get("model")
        meta["max_tokens"] = frag.get("max_tokens", meta["max_tokens"])
        for cell_dict in frag.get("cells", []):
            extras = cell_dict.pop("extras", {})
            all_cells.append(CellStats(extras=extras, **cell_dict))

    if args.gpu:
        meta["gpu"] = args.gpu
    if args.cuda_visible_devices:
        meta["cuda_visible_devices"] = args.cuda_visible_devices

    report = render_markdown(all_cells, meta)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Wrote {args.report}", flush=True)
    else:
        print(report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="forge vs vLLM benchmark client")
    parser.add_argument(
        "--mode", choices=("benchmark", "merge"), default="benchmark",
        help="benchmark a single engine, or merge JSON fragments into a report",
    )

    # benchmark mode
    parser.add_argument("--host", help="engine OpenAI-compatible base URL")
    parser.add_argument("--engine", choices=("forge", "vLLM"), help="engine label for output")
    parser.add_argument("--served-name", default="test")
    parser.add_argument("--prompt-lens", default="128,1024")
    parser.add_argument("--concurrencies", default="1,8,32")
    parser.add_argument("--requests-per-cell", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out-json", help="write JSON fragment here (otherwise stdout)")
    parser.add_argument(
        "--allow-errors", action="store_true",
        help="continue and write fragment even when cells fail validation "
             "(errors>0, zero completions, missing usage). NEVER set this when running a perf gate.",
    )

    # merge mode
    parser.add_argument("--fragments", nargs="*", default=[], help="JSON fragments to merge")
    parser.add_argument("--report", help="markdown report path (otherwise stdout)")
    parser.add_argument("--gpu", help="GPU model string for the report header")
    parser.add_argument("--cuda-visible-devices", help="CUDA_VISIBLE_DEVICES used during runs")

    args = parser.parse_args()
    if args.mode == "benchmark":
        return cmd_benchmark(args)
    return cmd_merge(args)


if __name__ == "__main__":
    sys.exit(main())
