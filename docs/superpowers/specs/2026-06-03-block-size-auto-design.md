# block_size auto: backend-driven default for paged KV cache

**Date:** 2026-06-03
**Status:** spec (pre-implementation)
**Branch:** `feat/quantized-decode` (or follow-on)
**Related:** `docs/plans/2026-02-23-flash-attention-v2-{design,plan}.md`,
memory `forge-fa2-paged-decode.md`, `gb10-qwen3-3way-bench.md`

## Problem

Forge already has a FA2 paged-decode path
(`forge-backend-cuda/src/backend.rs:1498-1580`) that wins **−9% (C=1) to
−13.3% (C=8) TPOT** on GB10 / Qwen3-4B versus the self-rolled split-KV
fallback. It is gated by env var `FORGE_FA2_PAGED=1` and a CLI flag
`--block-size 256` because FA2's paged inner loop hard-requires
`page_block_size % 256 == 0`. The default of `--block-size 16` keeps the
fast path off, so the win never materialises unless the operator knows the
incantation.

We want FA2 paged decode to be the **default code path** whenever the
backend + model geometry permits, without regressing:

- CPU backend (no FA2 at all).
- Unsupported head_dim (FA2 templates exist only for
  `{32, 64, 96, 128, 192, 256}`).
- F32 / quantised dtypes (FA2 is F16 / BF16 only).
- Short-context debugging where the 16x KV-block fragmentation jump
  matters.

## How peers do it

- **pegainfer** (`pegainfer-qwen3-4b/src/weights.rs:309`): hardcoded
  `page_size = 16`, uses self-written
  `paged_attention_decode_split_kv_cuda`. Does not use FA2 paged. Their
  edge over forge is not block-size related.
- **vLLM** (`vllm/config/cache.py:48`, `vllm/v1/attention/backend.py:175`):
  `Backend.get_supported_kernel_block_sizes() -> [int | MultipleOf]` plus
  `get_preferred_block_size(default) -> int` (FA backend returns
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

Resolution happens **after** backend selection and **before** the paged
cache match arm (`forge-server/src/main.rs:185-203`):

```rust
const DEFAULT_KV_TOKEN_CAPACITY: usize = 32_768; // = current 2048 * 16

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
removed; bail now lives in the `Fixed(0)` arm above.

`PagedKvCache::new` and the existing info log are updated to take the
resolved `usize` values. `NaiveKvCache::new` is unchanged — naive cache
ignores block size entirely (`forge-kvcache/src/naive.rs:188-204`).

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

Implementations:

- `CudaBackend::preferred_block_size`: when the `flash-attn` feature is
  compiled in **and** `head_dim ∈ {32, 64, 96, 128, 192, 256}` **and**
  `dtype ∈ {F16, BF16}`, return `256`. Otherwise `16`.
- `CpuBackend`: inherits the trait default (`16`).

The condition lives in one place — the same gate used by the FA2
dispatch site, factored into a small helper so the two stay in lock-step.

### 3. FA2 dispatch — `forge-backend-cuda/src/backend.rs:1510-1580`

Change the env semantics:

- **Was:** opt-in. `FORGE_FA2_PAGED=1` required to take the FA2 branch.
- **Now:** opt-out kill-switch. `FORGE_FA2_PAGED=0` (or `false`) forces
  the split-KV fallback. Any other value or absent variable lets the FA2
  branch run when shape conditions allow.

Rationale (from Codex review): the FA2 path at
`forge-backend-cuda/src/backend.rs:1555-1586` `return Ok(())`s immediately
after the kernel launch — there is no runtime fallback. Memory note
`gb10-qwen3-3way-bench.md` records FA2 unit tests hitting
`CUDA_ERROR_MISALIGNED_ADDRESS` on small / odd shapes on Blackwell
sm_121. Outright removal of the env gate would leave operators with no
recovery path if a previously-untested geometry trips one of these
edge cases. The kill-switch preserves zero-config default behaviour while
keeping a documented escape hatch.

Dispatch condition becomes (still inside the existing
`#[cfg(feature = "flash-attn")]` block at `backend.rs:1510`):

```rust
let fa2_supported_hdim = matches!(head_dim, 32 | 64 | 96 | 128 | 192 | 256);
let block_size_aligned = block_size % 256 == 0;
let fa2_killed = std::env::var("FORGE_FA2_PAGED")
    .ok()
    .filter(|v| v == "0" || v.eq_ignore_ascii_case("false"))
    .is_some();
if !fa2_killed && fa2_supported_hdim && block_size_aligned { /* FA2 */ }
```

The `cfg` wrapping is unchanged: a build without `--features flash-attn`
continues to skip the FA2 branch entirely.

User levers to take the fallback path explicitly:

- `--block-size 16` (or anything not divisible by 256).
- `FORGE_FA2_PAGED=0` env var.

### 4. Tests

- `forge-backend/forge-backend-cuda/tests/test_paged_attention.rs:718,734`:
  delete the `unsafe { set_var("FORGE_FA2_PAGED", "1") }` /
  `remove_var` calls — the FA2 path is now the default for that test's
  shape. Test still passes if the dispatch logic is correct.
- Add a parser unit test in `forge-server`: covers `"auto"`,
  case-insensitive `"AUTO"`, valid integer `"256"`, `"0"` (parses
  successfully as `Fixed(0)` — the runtime bail is separate), and an
  invalid string.
- Add a CudaBackend unit test (gated on `feature = "cuda"` +
  `feature = "flash-attn"`): `preferred_block_size(128, DType::F16) ==
  256`, `(128, DType::F32) == 16`, `(48, DType::F16) == 16` (unsupported
  head_dim).
- `forge-scheduler/tests/test_scheduler.rs` and other tests using
  `block_size = 16` literals are intentional (they probe short-seq
  scheduling) and stay as-is.

### 5. Documentation

- `README.md:70`: `--block-size auto (default; 256 when CUDA + FA2
  applicable, else 16)`.
- `docs/CONTRIB.md:79`: same blurb plus a row for `--num-blocks` =
  `auto` (~32k token capacity preserved).
- `docs/RUNBOOK.md`: add an entry "when to override block_size":
  - CPU backend debugging — pin `--block-size 16` to keep KV memory
    small.
  - Short-context benchmarking — pin 16 to avoid up-to-255-token internal
    fragmentation per sequence (vs 15 with the old default).
  - Reproducing FA2 vs split-KV comparisons — lock 16 / 256 explicitly.
  - Document `FORGE_FA2_PAGED=0` as the runtime kill switch if FA2
    misbehaves on a new geometry.
- `docs/codemaps/backend.md`: add `preferred_block_size` to the Backend
  trait surface listing.

### 6. Out of scope (intentional)

- VRAM-budget profiling (`gpu_memory_utilization` equivalent). Forge has
  no profiling pass; adding one is a separate, larger change. Token
  capacity is the right invariant for now.
- New CLI flag like `--kv-cache-mem-gb`. Can be added later without
  re-touching this design.
- Sweeping block_size at runtime per request, or maintaining multiple
  block pools. vLLM and pegainfer both use single-pool designs; the
  payoff doesn't justify the scheduler + allocator + dispatch rework.
- Changes to `cuda_graph_buckets`, `--max-prefill-tokens`, or the
  scheduler. None of them gate on block_size; verified in
  `forge-runtime/src/engine.rs:122-130, 520-521` and
  `forge-scheduler/src/continuous.rs:254-289`.

## Edge-case verification (from Codex review)

| Scenario                                  | Behaviour                                                                 | Status |
| ----------------------------------------- | ------------------------------------------------------------------------- | ------ |
| `--kv-cache naive`                        | Internal block_size=1, ignores CLI value                                  | unaffected |
| `--backend cpu --block-size 256`          | Legal, no FA2 path, wasteful KV but functional                            | unaffected |
| 4096-token prefill with `block_size=256`  | Exactly 16 blocks/seq, scheduler grants by block count                    | unaffected |
| `cuda_graph_buckets 1,2,4,8,16,32`        | Graph capture not gated on block_size                                     | unaffected |
| Short-seq generation, `block_size=256`    | Up to 255 tokens internal fragmentation per sequence (was 15)             | **documented** in RUNBOOK |
| Unsupported head_dim on CUDA + flash-attn | `preferred_block_size` returns 16, FA2 path skipped via head_dim gate    | covered |
| F32 / quantised dtype on CUDA             | `preferred_block_size` returns 16, FA2 path skipped via dtype gate       | covered |
| FA2 dispatch failure on new geometry      | No automatic fallback. `FORGE_FA2_PAGED=0` documented as kill switch     | mitigated |

## Risks

- **No runtime FA2 fallback.** Mitigated by the kill switch and by
  conservative `preferred_block_size` gating (only the validated
  head_dims + dtypes). A user hitting a previously-untested geometry can
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

## Acceptance

After implementation:

- `cargo build --workspace`, `cargo build --release`, `cargo test
  --workspace` pass on CUDA host.
- `cargo build --release --no-default-features` passes (CPU-only).
- `bash scripts/test_server.sh /path/to/qwen3-4b` works with no extra
  flags and the startup log reports `block_size=256 (auto), num_blocks=128
  (auto), capacity=32768 tokens`.
- `bash scripts/benchmark.sh /path/to/qwen3-4b` produces TPOT numbers
  consistent with the FA2 path (within noise of the memory-recorded
  C=8 61.15 ms baseline).
- Setting `FORGE_FA2_PAGED=0` falls back to split-KV without error;
  setting `--block-size 16` does the same.

## Implementation order (handed to writing-plans)

1. Backend trait method + impls.
2. CLI `AutoUsize` parser + resolution logic in `main.rs`.
3. Flip FA2 dispatch from opt-in to kill-switch.
4. Tests (parser, backend preference, remove env-var dance from
   FA2 test).
5. Docs (README, CONTRIB, RUNBOOK, codemap).
6. Bench sanity-check on Qwen3-4B (record TPOT/TTFT, confirm no
   regression).
