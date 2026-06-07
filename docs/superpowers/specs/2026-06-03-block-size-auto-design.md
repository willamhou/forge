# block_size auto: backend-driven default for paged KV cache

**Date:** 2026-06-03 (rev 2, post Codex review round 2)
**Status:** spec (pre-implementation)
**Branch:** `feat/quantized-decode` (or follow-on)
**Related:** `docs/plans/2026-02-23-flash-attention-v2-{design,plan}.md`,
memory `forge-fa2-paged-decode.md`, `gb10-qwen3-3way-bench.md`

## Problem

Forge already has a FA2 paged-decode path
(`forge-backend/forge-backend-cuda/src/backend.rs:1498-1580`) that wins
**−9% (C=1) to −13.3% (C=8) TPOT** on GB10 / Qwen3-4B versus the
self-rolled split-KV fallback. It is gated by env var
`FORGE_FA2_PAGED=1` and a CLI flag `--block-size 256` because FA2's
paged inner loop hard-requires `page_block_size % 256 == 0`. The default
of `--block-size 16` keeps the fast path off, so the win never
materialises unless the operator knows the incantation.

We want FA2 paged decode to be the **default code path** whenever the
backend + model geometry permits, without regressing:

- CPU backend (no FA2 at all).
- Unsupported head_dim (FA2 templates exist only for
  `{32, 64, 96, 128, 192, 256}`).
- F32 / BF16 / quantised dtypes. FA2 supports F16 and BF16, but
  forge's `paged_attention_into` only dispatches the FA2 branch from the
  `DType::F16` arm — BF16 currently hits `ForgeError::UnsupportedDtype`
  (`forge-backend/forge-backend-cuda/src/backend.rs:1489, 1684`). The
  auto-preference must match dispatch reality (F16 only), not FA2's
  theoretical capability.
- Short-context debugging where the 16× KV-block fragmentation jump
  matters.

## How peers do it

- **pegainfer** (`pegainfer-qwen3-4b/src/weights.rs:309`): hardcoded
  `page_size = 16`, uses self-written
  `paged_attention_decode_split_kv_cuda`. Does not use FA2 paged. Their
  edge over forge is not block-size related.
- **vLLM** (`vllm/config/cache.py:48`,
  `vllm/v1/attention/backend.py:175`):
  `Backend.get_supported_kernel_block_sizes() -> [int | MultipleOf]`
  plus `get_preferred_block_size(default) -> int` (FA backend returns
  `max(default, 64)`). `_apply_block_size_default` runs a startup
  protocol: user-supplied → validated, missing → backend.preferred,
  unsupported → adjusted. `num_gpu_blocks` is determined by a profiling
  pass against `gpu_memory_utilization`.

Forge takes the vLLM **shape** (backend declares capability, startup
negotiates) without vLLM's VRAM-profiling complexity. Forge's stage does
not need per-deployment memory budgeting yet; preserving today's logical
KV-token capacity is the right invariant.

## Design

### 1. CLI surface — `forge-server/src/main.rs`

Introduce one shared parser used by both flags:

```rust
#[derive(Clone, Copy, Debug)]
enum AutoUsize {
    Auto,
    Fixed(usize),
}

fn parse_auto_usize(s: &str) -> Result<AutoUsize, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(AutoUsize::Auto);
    }
    s.parse::<usize>()
        .map(AutoUsize::Fixed)
        .map_err(|_| format!("expected 'auto' or a positive integer, got '{s}'"))
}
```

Flags:

```rust
#[arg(long, default_value = "auto", value_parser = parse_auto_usize)]
block_size: AutoUsize,

#[arg(long, default_value = "auto", value_parser = parse_auto_usize)]
num_blocks: AutoUsize,
```

**Resolution call depth.** Backend selection happens in `main()`
(`forge-server/src/main.rs:136-147`) — the concrete `CpuBackend` or
`CudaBackend` is then passed into `run_server<B: Backend + Clone>`
(`main.rs:154-159`). The paged cache is constructed inside `run_server`
(`main.rs:185-203`). Resolution therefore happens **inside
`run_server`**, after the backend is in scope and **before** the
`match cli.kv_cache` arm — not in `main()` and not at parse time.

```rust
const DEFAULT_KV_TOKEN_CAPACITY: usize = 32_768; // = current 2048 * 16

// Inside run_server<B: Backend + Clone>(backend, cli, model_config, ...)
let resolved_block_size = match cli.block_size {
    AutoUsize::Fixed(0) => anyhow::bail!("--block-size must be >= 1"),
    AutoUsize::Fixed(n) => n,
    AutoUsize::Auto => backend.preferred_block_size(
        model_config.head_dim,
        model_config.dtype,
    ),
};

let resolved_num_blocks = match cli.num_blocks {
    AutoUsize::Fixed(0) => anyhow::bail!("--num-blocks must be >= 1"),
    AutoUsize::Fixed(n) => n,
    AutoUsize::Auto => {
        DEFAULT_KV_TOKEN_CAPACITY.div_ceil(resolved_block_size).max(1)
    }
};

info!(
    "Paged KV cache: block_size={resolved_block_size} \
     ({}), num_blocks={resolved_num_blocks} ({}), capacity={} tokens",
    if matches!(cli.block_size, AutoUsize::Auto) { "auto" } else { "set" },
    if matches!(cli.num_blocks, AutoUsize::Auto) { "auto" } else { "set" },
    resolved_block_size * resolved_num_blocks,
);
```

The old `if cli.block_size == 0 { bail }` check at `main.rs:115-117` is
removed (it operates on the wrong type now and fires before resolution
anyway). The bail moves into the `Fixed(0)` arms above. Both `paged` and
`naive` match arms read `resolved_block_size` / `resolved_num_blocks`;
naive ignores both values but it is harmless to forward the resolved
ints uniformly.

`PagedKvCache::new` and its info log are updated to take the resolved
`usize` values.

### 2. Backend trait — `forge-core/src/backend.rs`

Add an instance method with a safe default:

```rust
/// Preferred KV-cache block size for this backend / model geometry.
///
/// Backends override this when a fast attention path imposes shape
/// requirements. Default is 16, which all backends accept.
fn preferred_block_size(&self, _head_dim: usize, _dtype: DType) -> usize {
    16
}
```

`&self` rather than an associated function because:

- `run_server<B: Backend + Clone>` already has the concrete backend by
  the time it needs the answer (`forge-server/src/main.rs:154-159`).
- The trait is not used as a `dyn Backend` object — all consumers are
  generic; object-safety is not a constraint.
- Future signals (SM count, free VRAM, feature-flag state) can live on
  `self` without re-shaping the trait.

**Existing `Backend` impls touched by adding a default method:**

| Impl | File | Behaviour with default |
| ---- | ---- | ---------------------- |
| `CudaBackend` | `forge-backend/forge-backend-cuda/src/backend.rs:1770` | Overrides — returns 256 when FA2-eligible |
| `CpuBackend` | `forge-backend/forge-backend-cpu/src/backend.rs:47` | Inherits default (16) |
| `TestBackend` | `forge-kvcache/src/paged_cache.rs:515,532` (test module) | Inherits default (16); no other code change required |

The defaulted method means no existing impl block must be edited just to
compile — only `CudaBackend` adds an explicit override.

**Eligibility helper (canonical source of truth).** A single function in
`forge-backend/forge-backend-cuda/src/backend.rs` answers "does this
shape qualify for FA2 paged decode?". Both `preferred_block_size` and
the dispatch site at `backend.rs:1510-1518` call it:

```rust
// In forge-backend-cuda crate, gated by the same #[cfg(feature = "flash-attn")]
// that already wraps the dispatch block.
#[cfg(feature = "flash-attn")]
const FA2_SUPPORTED_HEAD_DIMS: [usize; 6] = [32, 64, 96, 128, 192, 256];

#[cfg(feature = "flash-attn")]
fn fa2_paged_eligible(head_dim: usize, dtype: DType, block_size: usize) -> bool {
    // Mirrors forge-backend-cuda paged_attention_into dispatch reality:
    // only the DType::F16 arm enters the FA2 branch today; BF16 hits
    // UnsupportedDtype at backend.rs:1684.
    matches!(dtype, DType::F16)
        && FA2_SUPPORTED_HEAD_DIMS.contains(&head_dim)
        && block_size % 256 == 0
}
```

`preferred_block_size` on `CudaBackend` becomes:

```rust
#[cfg(feature = "flash-attn")]
fn preferred_block_size(&self, head_dim: usize, dtype: DType) -> usize {
    // Probe the eligibility helper with the candidate block_size 256.
    if fa2_paged_eligible(head_dim, dtype, 256) { 256 } else { 16 }
}
// No #[cfg(not(feature = "flash-attn"))] override needed —
// the trait default (16) applies.
```

When BF16 support is later added to `paged_attention_into`'s dispatch,
the helper's `matches!(dtype, DType::F16)` clause is the only place
that needs to widen. Spec contract: **dispatch and preference must agree
through the helper, not by parallel hand-edits.**

### 3. FA2 dispatch — `forge-backend/forge-backend-cuda/src/backend.rs:1510-1580`

Change the env semantics:

- **Was:** opt-in. The branch only runs when
  `FORGE_FA2_PAGED == "1"` or `==` (case-insensitive) `"true"`.
- **Now:** opt-out kill switch. The branch only **skips** when
  `FORGE_FA2_PAGED == "0"` or `==` (case-insensitive) `"false"`. Absent
  variable, empty string, or any other value → FA2 runs whenever shape
  conditions allow.

**This is a deliberate behaviour change for unrecognised values.** A
user who previously set `FORGE_FA2_PAGED=on` saw the fallback path; the
new behaviour routes them to FA2. The only way to keep the fallback now
is `=0`, `=false`, or pinning `--block-size 16`. Mentioned here so it
gets into release notes / RUNBOOK, not buried.

Rationale (from Codex review): the FA2 path at
`forge-backend/forge-backend-cuda/src/backend.rs:1555-1586`
`return Ok(())`s immediately after the kernel launch — there is no
runtime fallback. Memory note `gb10-qwen3-3way-bench.md` records FA2
unit tests hitting `CUDA_ERROR_MISALIGNED_ADDRESS` on small / odd
shapes on Blackwell sm_121. Outright removal of the env gate would
leave operators with no recovery path if a previously-untested geometry
trips one of these edge cases. The kill switch preserves zero-config
default behaviour while keeping a documented escape hatch.

Dispatch condition becomes (still inside the existing
`#[cfg(feature = "flash-attn")]` block at `backend.rs:1510`):

```rust
let fa2_killed = std::env::var("FORGE_FA2_PAGED")
    .ok()
    .filter(|v| v == "0" || v.eq_ignore_ascii_case("false"))
    .is_some();
if !fa2_killed && fa2_paged_eligible(head_dim, dtype, block_size) {
    // FA2 launch as today
}
```

`fa2_paged_eligible` is the same helper from §2 — no parallel
`matches!` lists, no risk of drift.

The `cfg` wrapping is unchanged: a build without `--features flash-attn`
continues to skip the FA2 branch entirely.

User levers to take the fallback path explicitly:

- `--block-size 16` (or anything not divisible by 256).
- `FORGE_FA2_PAGED=0` env var.

### 4. Tests

**FA2 path-taken assertion (new requirement).** The existing test at
`forge-backend/forge-backend-cuda/tests/test_paged_attention.rs:715-734`
sets `FORGE_FA2_PAGED=1`, runs `paged_attention`, then checks output vs
F32 reference at 1% F16 tolerance. Simply deleting the `set_var` dance
leaves the test passing **even if dispatch silently routes to the
fallback** — both branches answer within 1% F16. That is not a
correctness regression today but it removes the only assertion that
"FA2 actually ran" once the env gate flips.

The test layout becomes a **differential check**:

1. With env unset (FA2 enabled by default given the shape), run
   `paged_attention` → record `out_fa2`.
2. With `FORGE_FA2_PAGED=0`, re-run on identical inputs → record
   `out_fallback`.
3. Assert **both** are within 1% F16 tol of the F32 reference (existing
   correctness invariant).
4. Assert `out_fa2` ≠ bit-identical to `out_fallback` **OR** add a
   lightweight dispatch counter on `CudaBackend` (atomic incremented
   inside the FA2 branch only) and assert its value changes after step
   1 but not after step 2.

Recommended: add a `CudaBackend` test-only counter behind
`#[cfg(any(test, feature = "fa2-dispatch-counter"))]` and read it from
the test. Differential output check is fine as a fallback but
`paged_attention` numerics are deterministic enough that
`out_fa2 != out_fallback` is empirically true on the chosen shape; it
is not a contract.

**Unit tests added:**

- `forge-server` parser tests: `"auto"`, case-insensitive `"AUTO"`,
  `"256"` → `Fixed(256)`, `"0"` → `Fixed(0)` (parses; runtime bail is
  separate), invalid string. Pure-logic tests, no feature gating.
- `forge-backend-cuda` `preferred_block_size` tests, gated
  **only on `feature = "flash-attn"`** (not on a `"cuda"` feature —
  `forge-backend-cuda`'s Cargo.toml has only `flash-attn`; `cuda` lives
  on `forge-server`):
  - `(head_dim=128, dtype=F16) → 256`
  - `(head_dim=128, dtype=BF16) → 16` (matches dispatch reality
    today)
  - `(head_dim=128, dtype=F32) → 16`
  - `(head_dim=48, dtype=F16) → 16` (unsupported head_dim)
  - Without `flash-attn` feature, the trait default applies (covered by
    the standard compile target without explicit override).
- `forge-kvcache::paged_cache` scheduler-capacity sanity: existing
  scheduler tests use `block_size=16` literals on purpose (probe
  short-seq scheduling) and stay as-is. A new test exercises
  `block_size=256` with a 4096-token prompt and asserts the scheduler
  needs exactly 16 blocks
  (`ceil(prompt_len / block_size)`), guarding the math used in
  `forge-scheduler/src/continuous.rs:211-220, 275-283`.

**Unchanged tests:** scheduler short-seq tests, decode persistent test,
all unit tests that exercise the split-KV fallback explicitly via
`--block-size` choices not divisible by 256.

### 5. Documentation

- `README.md:70`: `--block-size auto (default; 256 when CUDA + FA2
  applicable, else 16)`.
- `docs/CONTRIB.md:79`: same blurb plus a row for `--num-blocks` =
  `auto` (~32k token capacity preserved).
- `docs/RUNBOOK.md`: add an entry "when to override block_size":
  - CPU backend debugging — pin `--block-size 16` to keep KV memory
    small.
  - Short-context benchmarking — pin 16 to avoid up-to-255-token
    internal fragmentation per sequence (vs 15 with the old default).
  - Reproducing FA2 vs split-KV comparisons — lock 16 / 256 explicitly.
  - Document `FORGE_FA2_PAGED=0` as the runtime kill switch if FA2
    misbehaves on a new geometry. Note the `=on`/`=enabled`-style env
    values no longer disable FA2.
- `docs/codemaps/backend.md`: add `preferred_block_size` to the
  `Backend` trait surface listing.

### 6. Scope of touch (intentional)

Out-of-scope (no design change here):

- VRAM-budget profiling (`gpu_memory_utilization` equivalent). Forge
  has no profiling pass; adding one is a separate, larger change. Token
  capacity is the right invariant for now.
- New CLI flag like `--kv-cache-mem-gb`. Can be added later without
  re-touching this design.
- Sweeping block_size at runtime per request, or maintaining multiple
  block pools. vLLM and pegainfer both use single-pool designs; the
  payoff doesn't justify the scheduler + allocator + dispatch rework.

**No structural change but mathematically dependent (worth flagging):**

- `forge-scheduler/src/continuous.rs:211-220, 275-283`: scheduling
  admission and decode-block reservation read `block_size` from
  `CacheUsage` (`forge-core/src/kvcache.rs:21-24`) and compute
  `ceil(tokens / block_size)`. The math is unchanged; what changes is
  the input value the scheduler sees by default. The new test in §4
  guards this.
- `forge-kvcache/src/paged_cache.rs:38-43, 466-479`: the pool tensor
  shape is `[total_blocks, block_size, kv_dim]` and `CacheUsage`
  reports `block_size`. No structural change; allocation sizing is
  parameterised. With auto-resolution the allocated bytes stay constant
  by construction (`num_blocks × block_size = 32_768`).
- `cuda_graph_buckets 1,2,4,8,16,32`: graph capture is gated on decode
  batch size, not block size
  (`forge-runtime/src/engine.rs:122-130, 520-521`); no change.

## Edge-case verification (from Codex review)

| Scenario                                  | Behaviour                                                                 | Status |
| ----------------------------------------- | ------------------------------------------------------------------------- | ------ |
| `--kv-cache naive`                        | Internal block_size=1, ignores CLI value                                  | unaffected (naive cache ignores both resolved values) |
| `--backend cpu --block-size 256`          | Legal, no FA2 path, wasteful KV but functional                            | unaffected |
| 4096-token prefill with `block_size=256`  | Exactly 16 blocks/seq, scheduler grants by block count                    | math depends on block_size — new test guards |
| `cuda_graph_buckets 1,2,4,8,16,32`        | Graph capture not gated on block_size                                     | unaffected |
| Short-seq generation, `block_size=256`    | Up to 255 tokens internal fragmentation per sequence (was 15)             | **documented** in RUNBOOK |
| Unsupported head_dim on CUDA + flash-attn | `preferred_block_size` returns 16, FA2 dispatch skipped via helper       | covered via shared helper |
| F32 on CUDA                               | `preferred_block_size` returns 16, FA2 dispatch skipped via helper       | covered via shared helper |
| BF16 on CUDA                              | `preferred_block_size` returns 16, dispatch hits UnsupportedDtype on this dtype today regardless | covered (dispatch is the underlying constraint) |
| `FORGE_FA2_PAGED=on` (old truthy value)   | New semantics: FA2 runs. Was: FA2 skipped                                 | **behaviour change documented** |
| FA2 dispatch failure on new geometry      | No automatic fallback. `FORGE_FA2_PAGED=0` documented as kill switch     | mitigated |

## Risks

- **No runtime FA2 fallback.** Mitigated by the kill switch and by
  conservative `preferred_block_size` gating (only the validated
  head_dims + F16). A user hitting a previously-untested geometry can
  set `FORGE_FA2_PAGED=0` and continue. Auto-fallback is deferred to a
  follow-up.
- **`DEFAULT_KV_TOKEN_CAPACITY = 32768` is a wired-in invariant.**
  Adequate for 4–8B class models on consumer Blackwell. Bigger models
  / longer contexts will want `--num-blocks` set explicitly. Documented
  in RUNBOOK.
- **Internal fragmentation jump 15 → 255 tokens/seq.** Real but bounded:
  Qwen3-4B FP16 KV is ~115 KB/token, so worst-case waste is ~28 MB per
  active sequence. At C=8, ≤ ~220 MB total. Acceptable on GB10's
  ~16 GB budget after weights.
- **Env-var semantics flip is user-visible.** Old `=on`/`=enabled`-style
  values now enable FA2. Release notes + RUNBOOK call this out.

## Acceptance

After implementation:

- `cargo build --workspace`, `cargo build --release`, `cargo test
  --workspace` pass on CUDA host.
- `cargo build --release --no-default-features` passes (CPU-only).
- `cargo test -p forge-server` covers `AutoUsize` parser cases.
- `cargo test -p forge-backend-cuda --features flash-attn` covers
  `preferred_block_size` truth table.
- `cargo test -p forge-backend-cuda --features flash-attn
  test_paged_attention` runs the differential FA2-vs-fallback check and
  asserts the FA2 branch was actually taken (counter or differential
  output, per §4).
- `cargo test -p forge-kvcache` covers a new `block_size=256` scheduler
  capacity sanity test (4096-token prompt → 16 blocks).
- `bash scripts/test_server.sh /path/to/qwen3-4b` works with no extra
  flags and the startup log reports
  `block_size=256 (auto), num_blocks=128 (auto), capacity=32768 tokens`.
  End-to-end `/v1/chat/completions` returns a coherent reply (the
  primary user-facing API path is exercised through the default auto
  resolution).
- `bash scripts/benchmark.sh /path/to/qwen3-4b` produces TPOT numbers
  consistent with the FA2 path (within noise of the memory-recorded
  C=8 61.15 ms baseline).
- Setting `FORGE_FA2_PAGED=0` falls back to split-KV without error;
  setting `--block-size 16` does the same; both yield numerically
  equivalent answers within F16 tolerance.

## Implementation order (handed to writing-plans)

1. Eligibility helper + Backend trait method + impls (CudaBackend
   override; CpuBackend / TestBackend take default).
2. CLI `AutoUsize` parser + resolution logic inside `run_server`.
3. Flip FA2 dispatch from opt-in to kill-switch; route through helper.
4. Tests: parser, `preferred_block_size` truth table, scheduler capacity
   guard, FA2 path-taken differential (counter or output diff).
5. Docs (README, CONTRIB, RUNBOOK, codemap) — call out env-var
   semantics flip explicitly.
6. Bench sanity-check on Qwen3-4B (record TPOT/TTFT, confirm no
   regression vs memory baseline; e2e smoke test of `/v1/chat/completions`
   with default flags).
