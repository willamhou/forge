# block_size auto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `--block-size auto` / `--num-blocks auto` so FA2 paged-decode is the default path on supported geometries, and remove the `FORGE_FA2_PAGED=1` opt-in (kept as `=0` kill-switch).

**Architecture:** Backend trait gains `preferred_block_size(head_dim, dtype) -> usize` (default 16; CudaBackend overrides to 256 when FA2 conditions hold, routed through a canonical `fa2_paged_eligible` helper shared with the FA2 dispatch site). CLI uses an `AutoUsize` enum with a clap value parser; resolution happens inside `run_server` after backend selection, before paged-cache construction. Token capacity is preserved by `num_blocks_auto = ceil(32_768 / block_size)`.

**Tech Stack:** Rust 2024 stable, clap derive, anyhow, CUDA via cudarc, FA2 via the vendored `forge-flash` crate, cargo workspace.

**Spec:** `docs/superpowers/specs/2026-06-03-block-size-auto-design.md` (rev 3, head `bb3a68a`).

**Baseline numbers (lock):** `/tmp/bench-2026-06-09/baseline.jsonl` (GB10 / Qwen3-4B FP16 / prompt~1024 / out=256). After implementation, the new default must match `forge_B` (FA2 paged) within ±1 ms TPOT noise; `FORGE_FA2_PAGED=0` and `--block-size 16` must each fall back to `forge_A`.

---

## Task 1: Canonical FA2 eligibility helper + truth-table test

**Files:**
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` (add helper near top of file or near the dispatch site at line 1498)

- [ ] **Step 1.1: Add a failing unit test for `fa2_paged_eligible`**

Append at the very end of `forge-backend/forge-backend-cuda/src/backend.rs`:

```rust
#[cfg(all(test, feature = "flash-attn"))]
mod tests_block_size_auto {
    use super::fa2_paged_eligible;
    use forge_core::DType;

    #[test]
    fn fa2_eligible_qwen_class_shape() {
        assert!(fa2_paged_eligible(128, DType::F16, 256));
    }

    #[test]
    fn fa2_rejects_unaligned_block_size() {
        assert!(!fa2_paged_eligible(128, DType::F16, 128));
        assert!(!fa2_paged_eligible(128, DType::F16, 16));
    }

    #[test]
    fn fa2_rejects_unsupported_head_dim() {
        assert!(!fa2_paged_eligible(48, DType::F16, 256));
        assert!(!fa2_paged_eligible(160, DType::F16, 256));
    }

    #[test]
    fn fa2_rejects_non_f16_dtype() {
        // Today the FA2 dispatch arm is reachable only from DType::F16; BF16
        // hits UnsupportedDtype at backend.rs:1684. Keep preference aligned.
        assert!(!fa2_paged_eligible(128, DType::BF16, 256));
        assert!(!fa2_paged_eligible(128, DType::F32, 256));
    }
}
```

- [ ] **Step 1.2: Run the test to verify it fails**

```bash
cargo test -p forge-backend-cuda --features flash-attn tests_block_size_auto
```

Expected: compile error — `fa2_paged_eligible` not defined in the parent module.

- [ ] **Step 1.3: Implement the helper**

Add at the top of `forge-backend/forge-backend-cuda/src/backend.rs`, immediately after the existing `use forge_core::...` block:

```rust
/// Head dims for which FA2 templates are instantiated in
/// `forge-flash/csrc/flash_attn/src/flash_fwd_launch_template.h`.
#[cfg(feature = "flash-attn")]
const FA2_SUPPORTED_HEAD_DIMS: [usize; 6] = [32, 64, 96, 128, 192, 256];

/// Canonical FA2 paged-decode eligibility predicate.
///
/// Both `CudaBackend::preferred_block_size` (probes with candidate
/// `block_size = 256`) and the dispatch gate in `paged_attention_into_impl`
/// route through this function. Mirrors dispatch reality: only the
/// `DType::F16` arm currently reaches the FA2 branch — BF16 / F32 hit
/// `UnsupportedDtype` at the match-default arm. When BF16 support is added
/// to the dispatch later, widen the `dtype` clause here once and both call
/// sites pick it up.
#[cfg(feature = "flash-attn")]
pub(crate) fn fa2_paged_eligible(head_dim: usize, dtype: forge_core::DType, block_size: usize) -> bool {
    matches!(dtype, forge_core::DType::F16)
        && FA2_SUPPORTED_HEAD_DIMS.contains(&head_dim)
        && block_size % 256 == 0
}
```

- [ ] **Step 1.4: Run the test to verify it passes**

```bash
cargo test -p forge-backend-cuda --features flash-attn tests_block_size_auto
```

Expected: all four tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add forge-backend/forge-backend-cuda/src/backend.rs
git commit -m "feat(backend-cuda): canonical fa2_paged_eligible predicate + truth-table tests

Single helper for FA2 paged-decode eligibility shared by the dispatch
site and the backend's preferred_block_size override (next task). Mirrors
dispatch reality (DType::F16 only); widening dtype later is a one-line
change in one place.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 2: Backend trait gains `preferred_block_size` with safe default

**Files:**
- Modify: `forge-core/src/backend.rs` (add method to `pub trait Backend` near line 73)

- [ ] **Step 2.1: Write a failing test for CpuBackend inheriting the default**

Append to `forge-backend/forge-backend-cpu/tests/test_paged_attention.rs` (existing test file in that crate):

```rust
#[test]
fn cpu_preferred_block_size_is_16_for_any_shape() {
    use forge_core::{Backend, DType};
    use forge_backend_cpu::CpuBackend;
    let b = CpuBackend::new();
    assert_eq!(b.preferred_block_size(128, DType::F16), 16);
    assert_eq!(b.preferred_block_size(128, DType::F32), 16);
    assert_eq!(b.preferred_block_size(64, DType::BF16), 16);
}
```

If the file does not exist yet, create it with the standard test header:

```rust
// forge-backend/forge-backend-cpu/tests/test_paged_attention.rs
// (test for the preferred_block_size default; existing CPU paged-attention
// tests live in other files — this file is named for forward extension.)
```

- [ ] **Step 2.2: Run the test to verify it fails**

```bash
cargo test -p forge-backend-cpu cpu_preferred_block_size_is_16_for_any_shape
```

Expected: compile error — `preferred_block_size` not found on `Backend` trait.

- [ ] **Step 2.3: Add the trait method with a default**

Modify `forge-core/src/backend.rs`. Inside `pub trait Backend: Send + Sync + 'static { ... }` (at line 73), add the method below alongside the other allocation/metadata methods (place near `name()` / `device_count()`):

```rust
    /// Preferred KV-cache block size for this backend and model geometry.
    ///
    /// Backends override this when a fast attention path imposes shape
    /// requirements (e.g. CudaBackend returns 256 when FA2 paged decode
    /// is eligible). Default is 16, which all backends accept.
    fn preferred_block_size(&self, _head_dim: usize, _dtype: DType) -> usize {
        16
    }
```

- [ ] **Step 2.4: Run the test to verify it passes**

```bash
cargo test -p forge-backend-cpu cpu_preferred_block_size_is_16_for_any_shape
```

Expected: PASS. `CudaBackend` and `TestBackend` (in `forge-kvcache/src/paged_cache.rs:515,532`) inherit the default automatically; no changes required to either yet.

- [ ] **Step 2.5: Workspace compile check**

```bash
cargo check --workspace
```

Expected: clean. No existing impl block must be edited because the method is defaulted.

- [ ] **Step 2.6: Commit**

```bash
git add forge-core/src/backend.rs forge-backend/forge-backend-cpu/tests/test_paged_attention.rs
git commit -m "feat(core): Backend::preferred_block_size with safe default (16)

Defaulted instance method so existing impls (CpuBackend, TestBackend in
forge-kvcache, CudaBackend) compile unchanged. CudaBackend will override
in the next commit; CPU and test backends keep the safe default.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 3: CudaBackend overrides `preferred_block_size` (256 when FA2 eligible)

**Files:**
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` (inside `impl Backend for CudaBackend` block at line 1770)

- [ ] **Step 3.1: Extend the truth-table test with CudaBackend integration coverage**

Append to the existing `tests_block_size_auto` module added in Task 1:

```rust
    #[test]
    #[cfg(feature = "cuda")]  // marked but forge-backend-cuda has no cuda feature; see step 3.2
    fn cuda_preferred_matches_eligibility() {
        // Skipped: see step 3.2 — forge-backend-cuda's tests cannot construct
        // a CudaBackend in unit tests without a device. Integration coverage
        // happens through Task 4's paged_attention test that actually launches.
    }
```

Actually skip this test step — `CudaBackend::new()` requires a device handle and unit-testing the override directly is brittle on shared CI. Instead, add a logic-only test that proves the override calls the helper correctly via a constant probe:

Append to `tests_block_size_auto`:

```rust
    /// CudaBackend's override must defer to `fa2_paged_eligible` with the
    /// candidate `256`. We can't construct a CudaBackend in this unit test
    /// (needs a device), so we assert the predicate covers the same cases
    /// the override needs.
    #[test]
    fn override_helper_probe_matrix() {
        assert!(fa2_paged_eligible(128, forge_core::DType::F16, 256));
        assert!(fa2_paged_eligible(64, forge_core::DType::F16, 256));
        assert!(!fa2_paged_eligible(128, forge_core::DType::F32, 256));
        assert!(!fa2_paged_eligible(48, forge_core::DType::F16, 256));
    }
```

- [ ] **Step 3.2: Run the test to verify it passes (it already does — guards the override contract)**

```bash
cargo test -p forge-backend-cuda --features flash-attn tests_block_size_auto::override_helper_probe_matrix
```

Expected: PASS.

- [ ] **Step 3.3: Implement the override**

In `forge-backend/forge-backend-cuda/src/backend.rs`, locate `impl Backend for CudaBackend { ... }` at line 1770. Add the override near the top of that block (after `fn name(...)` / `fn device_count(...)`):

```rust
    #[cfg(feature = "flash-attn")]
    fn preferred_block_size(&self, head_dim: usize, dtype: forge_core::DType) -> usize {
        // Probe the eligibility helper with the candidate block_size 256.
        if crate::fa2_paged_eligible(head_dim, dtype, 256) { 256 } else { 16 }
    }
```

When `flash-attn` is not compiled in, the trait default (16) applies — no explicit `#[cfg(not(feature = "flash-attn"))]` branch needed.

- [ ] **Step 3.4: Workspace compile**

```bash
cargo build --workspace
cargo build --release -p forge-server
```

Expected: clean.

- [ ] **Step 3.5: Commit**

```bash
git add forge-backend/forge-backend-cuda/src/backend.rs
git commit -m "feat(backend-cuda): preferred_block_size returns 256 when FA2 eligible

Routes through fa2_paged_eligible so dispatch and preference cannot drift.
Falls back to the trait default (16) for non-F16 dtypes or unsupported
head dims, and when the flash-attn feature is off.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 4: Flip FA2 dispatch from opt-in to kill-switch + route via helper

**Files:**
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs:1498-1518` (rewrite the env gate)
- Modify: `forge-backend/forge-backend-cuda/tests/test_paged_attention.rs:715-734` (replace the `set_var` dance with a differential check)

- [ ] **Step 4.1: Rewrite the integration test to drive the new behaviour first (TDD)**

In `forge-backend/forge-backend-cuda/tests/test_paged_attention.rs`, find the existing test that sets `FORGE_FA2_PAGED=1` (around line 715-734) and rewrite the env-var dance into a differential check. Replace the existing block:

```rust
// SAFETY: env-var mutation is not Send-safe in general, but cargo test
// for this crate runs tests in a single binary serially (no parallel
// backend access — see file header) so the temporary mutation is fine.
unsafe { std::env::set_var("FORGE_FA2_PAGED", "1") };
let f_out16 = backend
    .paged_attention(/* ... existing args ... */)
    .unwrap();
backend.synchronize().unwrap();
unsafe { std::env::remove_var("FORGE_FA2_PAGED") };
```

with:

```rust
// Default behaviour: env unset → FA2 branch runs because the shape
// (head_dim=128, dtype=F16, block_size=256) is FA2-eligible.
let f_out_fa2 = backend
    .paged_attention(/* ... existing args ... */)
    .unwrap();
backend.synchronize().unwrap();

// Kill switch: FORGE_FA2_PAGED=0 forces the split-KV fallback. Same
// inputs, must still be within 1% F16 tol of the F32 reference.
// SAFETY: cargo test for this crate runs serially (see file header).
unsafe { std::env::set_var("FORGE_FA2_PAGED", "0") };
let f_out_fallback = backend
    .paged_attention(/* ... same args ... */)
    .unwrap();
backend.synchronize().unwrap();
unsafe { std::env::remove_var("FORGE_FA2_PAGED") };
```

Then add the tolerance assertion for the fallback output by mirroring the existing FA2 check immediately after. The existing `f_diff` / `f_tol` / `assert!` pattern (around line 745-750) stays; add the same pattern keyed on `f_out_fallback`.

- [ ] **Step 4.2: Run the test to verify the FA2-default branch still produces correct numbers (it currently fails because dispatch needs `=1`)**

```bash
cargo test -p forge-backend-cuda --features flash-attn test_paged_attention
```

Expected: with env unset the dispatch goes into split-KV fallback today (opt-in semantics), so the output should still match the F32 reference — the test passes for the FA2 branch trivially. Confirm the test compiles, runs, and the new fallback branch also passes. If it does, move on; if the fallback assertion fires, fix the assertion before flipping dispatch.

- [ ] **Step 4.3: Flip the dispatch semantics**

In `forge-backend/forge-backend-cuda/src/backend.rs`, replace lines 1498-1518 (the comment block + the existing `fa2_enabled` check) with:

```rust
                // FA2 paged decode path — uses `flash_fwd_kvcache` against
                // forge's block pool. Same memory layout in both:
                // `[num_blocks, block_size, num_kv_heads * head_dim]` and
                // FA2's `[num_blocks, page_block_size, num_heads_k, head_dim]`
                // are byte-equivalent.
                //
                // Default: FA2 runs whenever the shape is eligible
                // (`fa2_paged_eligible`). Kill switch: `FORGE_FA2_PAGED=0`
                // (or `=false`, case-insensitive) forces the split-KV
                // fallback. Anything else, including absent variable, takes
                // FA2. NOTE: this is a behaviour change from the previous
                // opt-in semantics — users who had `FORGE_FA2_PAGED=on`
                // previously saw fallback; now they see FA2.
                //
                // We are inside the `DType::F16 => { ... }` arm of
                // `match q.dtype()` above, so pass `DType::F16` literally
                // (there is no `dtype` binding in scope).
                #[cfg(feature = "flash-attn")]
                {
                    let fa2_killed = std::env::var("FORGE_FA2_PAGED")
                        .ok()
                        .filter(|v| v == "0" || v.eq_ignore_ascii_case("false"))
                        .is_some();
                    if !fa2_killed
                        && crate::fa2_paged_eligible(head_dim, forge_core::DType::F16, block_size)
                    {
```

Leave the existing FA2 launch body (lines 1519-1580) untouched; only the gating prelude changes.

- [ ] **Step 4.4: Run the integration test against the new dispatch**

```bash
cargo test -p forge-backend-cuda --features flash-attn test_paged_attention -- --nocapture
```

Expected: both `f_out_fa2` and `f_out_fallback` are within 1% F16 tol of the F32 reference. The FA2 branch now runs by default; `=0` forces fallback.

- [ ] **Step 4.5: Workspace check**

```bash
cargo test -p forge-backend-cuda --features flash-attn
cargo build --release -p forge-server
```

Expected: clean.

- [ ] **Step 4.6: Commit**

```bash
git add forge-backend/forge-backend-cuda/src/backend.rs forge-backend/forge-backend-cuda/tests/test_paged_attention.rs
git commit -m "feat(backend-cuda): FA2 paged dispatch default-on, FORGE_FA2_PAGED=0 kill-switch

- Dispatch routes through canonical fa2_paged_eligible (Task 1).
- Env semantics flip: absent / any-non-zero value → FA2 runs; =0 / =false
  → split-KV fallback. Behaviour change for previously-truthy values like
  =on / =enabled is intentional; documented in RUNBOOK.
- Integration test replaces the unsafe set_var(\"1\") dance with a
  differential FA2-vs-fallback tolerance check.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 5: `AutoUsize` CLI parser

**Files:**
- Modify: `forge-server/src/main.rs` (add enum + parser; keep flag wiring for Task 6)

- [ ] **Step 5.1: Write the failing parser test**

Append to `forge-server/src/main.rs` (or to a new `forge-server/src/lib.rs` if you prefer; the existing crate is binary-only so the easiest spot is a `#[cfg(test)] mod` at the bottom of `main.rs`):

```rust
#[cfg(test)]
mod tests_auto_usize {
    use super::{parse_auto_usize, AutoUsize};

    #[test]
    fn parses_auto_lowercase() {
        assert!(matches!(parse_auto_usize("auto"), Ok(AutoUsize::Auto)));
    }

    #[test]
    fn parses_auto_uppercase() {
        assert!(matches!(parse_auto_usize("AUTO"), Ok(AutoUsize::Auto)));
    }

    #[test]
    fn parses_fixed_integer() {
        assert!(matches!(parse_auto_usize("256"), Ok(AutoUsize::Fixed(256))));
    }

    #[test]
    fn parses_fixed_zero() {
        // Parse succeeds; runtime bail lives in the resolve step (Task 6).
        assert!(matches!(parse_auto_usize("0"), Ok(AutoUsize::Fixed(0))));
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_auto_usize("abc").is_err());
        assert!(parse_auto_usize("16x").is_err());
    }
}
```

- [ ] **Step 5.2: Run the test to verify it fails**

```bash
cargo test -p forge-server tests_auto_usize
```

Expected: compile error — `parse_auto_usize` and `AutoUsize` not defined.

- [ ] **Step 5.3: Implement the parser**

In `forge-server/src/main.rs`, just below the existing `parse_buckets` function (around line 86-105), add:

```rust
/// CLI value that is either the literal `"auto"` (deferred to runtime
/// resolution against the backend / model geometry) or a fixed integer.
/// Used by `--block-size` and `--num-blocks`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum AutoUsize {
    Auto,
    Fixed(usize),
}

pub(crate) fn parse_auto_usize(s: &str) -> Result<AutoUsize, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(AutoUsize::Auto);
    }
    s.parse::<usize>()
        .map(AutoUsize::Fixed)
        .map_err(|_| format!("expected 'auto' or a positive integer, got '{s}'"))
}
```

- [ ] **Step 5.4: Run the test to verify it passes**

```bash
cargo test -p forge-server tests_auto_usize
```

Expected: all five tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add forge-server/src/main.rs
git commit -m "feat(server): AutoUsize enum + parse_auto_usize for clap value parser

Shared parser used by --block-size and --num-blocks in the next commit.
Parses 'auto' (case-insensitive) into AutoUsize::Auto and any positive
integer (including 0; runtime bail is separate) into AutoUsize::Fixed.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 6: CLI wiring + resolve inside `run_server`

**Files:**
- Modify: `forge-server/src/main.rs` (lines 60-66 for flag types, 115-117 for old bail removal, 185-203 for resolve + paged-cache construction)

- [ ] **Step 6.1: Write a failing unit test for the resolve helper**

Append to `forge-server/src/main.rs` `tests_auto_usize` module:

```rust
    use super::resolve_block_config;

    #[test]
    fn resolve_auto_both_uses_backend_preference_and_keeps_capacity() {
        let preferred = 256;
        let cfg = resolve_block_config(
            AutoUsize::Auto, AutoUsize::Auto, preferred,
        ).unwrap();
        assert_eq!(cfg.block_size, 256);
        assert_eq!(cfg.num_blocks, 128); // ceil(32_768 / 256) = 128
        assert_eq!(cfg.block_size * cfg.num_blocks, 32_768);
    }

    #[test]
    fn resolve_explicit_block_size_overrides_backend() {
        let cfg = resolve_block_config(
            AutoUsize::Fixed(16), AutoUsize::Auto, 256,
        ).unwrap();
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.num_blocks, 2_048); // 32_768 / 16
    }

    #[test]
    fn resolve_zero_block_size_bails() {
        let err = resolve_block_config(AutoUsize::Fixed(0), AutoUsize::Auto, 256);
        assert!(err.is_err());
    }

    #[test]
    fn resolve_zero_num_blocks_bails() {
        let err = resolve_block_config(AutoUsize::Auto, AutoUsize::Fixed(0), 256);
        assert!(err.is_err());
    }

    #[test]
    fn resolve_keeps_explicit_num_blocks() {
        let cfg = resolve_block_config(
            AutoUsize::Auto, AutoUsize::Fixed(4_096), 256,
        ).unwrap();
        assert_eq!(cfg.block_size, 256);
        assert_eq!(cfg.num_blocks, 4_096);
    }
```

- [ ] **Step 6.2: Run the test to verify it fails**

```bash
cargo test -p forge-server tests_auto_usize::resolve_
```

Expected: compile error — `resolve_block_config` and `BlockConfig` not defined.

- [ ] **Step 6.3: Implement the resolve helper**

Add to `forge-server/src/main.rs`, immediately after `parse_auto_usize` (from Task 5):

```rust
pub(crate) const DEFAULT_KV_TOKEN_CAPACITY: usize = 32_768;

pub(crate) struct BlockConfig {
    pub block_size: usize,
    pub num_blocks: usize,
}

/// Resolve the user's `--block-size` / `--num-blocks` choices against the
/// backend's preferred default and the fixed-token-capacity invariant.
///
/// - `Auto` block_size → `backend_preferred`.
/// - `Auto` num_blocks → `ceil(DEFAULT_KV_TOKEN_CAPACITY / block_size)`.
/// - `Fixed(0)` for either flag → bail with a helpful message.
pub(crate) fn resolve_block_config(
    block_size: AutoUsize,
    num_blocks: AutoUsize,
    backend_preferred: usize,
) -> anyhow::Result<BlockConfig> {
    let block_size = match block_size {
        AutoUsize::Fixed(0) => anyhow::bail!("--block-size must be >= 1"),
        AutoUsize::Fixed(n) => n,
        AutoUsize::Auto => backend_preferred,
    };
    let num_blocks = match num_blocks {
        AutoUsize::Fixed(0) => anyhow::bail!("--num-blocks must be >= 1"),
        AutoUsize::Fixed(n) => n,
        AutoUsize::Auto => DEFAULT_KV_TOKEN_CAPACITY
            .div_ceil(block_size)
            .max(1),
    };
    Ok(BlockConfig { block_size, num_blocks })
}
```

- [ ] **Step 6.4: Run the test to verify it passes**

```bash
cargo test -p forge-server tests_auto_usize::resolve_
```

Expected: all five tests PASS.

- [ ] **Step 6.5: Switch CLI flags to `AutoUsize` and drop the early bail**

In `forge-server/src/main.rs`:

1. Replace lines 60-66 (block_size / num_blocks flags):

```rust
    /// Block size for paged KV cache. `auto` picks `Backend::preferred_block_size`
    /// (CUDA + FA2-eligible → 256; else 16). Use an integer to override.
    #[arg(long, default_value = "auto", value_parser = parse_auto_usize)]
    block_size: AutoUsize,

    /// Total number of KV cache blocks. `auto` keeps the default token
    /// capacity (~32k) by computing `ceil(32_768 / block_size)`.
    #[arg(long, default_value = "auto", value_parser = parse_auto_usize)]
    num_blocks: AutoUsize,
```

2. Delete the early bail at lines 115-117 entirely (`if cli.block_size == 0 { anyhow::bail!(...) }`). The bail is now inside `resolve_block_config`.

- [ ] **Step 6.6: Wire `resolve_block_config` into `run_server` and `PagedKvCache::new`**

In `forge-server/src/main.rs`, find `async fn run_server<B: Backend + Clone>` (around line 154). Locate the `match cli.kv_cache.as_str()` block at line 185-203. Just **before** that match, insert:

```rust
    let block_cfg = resolve_block_config(
        cli.block_size,
        cli.num_blocks,
        backend.preferred_block_size(model_config.head_dim, model_config.dtype),
    )?;
    let block_size = block_cfg.block_size;
    let num_blocks = block_cfg.num_blocks;
    let block_label = match cli.block_size {
        AutoUsize::Auto => "auto",
        AutoUsize::Fixed(_) => "set",
    };
    let nb_label = match cli.num_blocks {
        AutoUsize::Auto => "auto",
        AutoUsize::Fixed(_) => "set",
    };
```

Inside the `"paged"` arm, replace `cli.num_blocks` / `cli.block_size` with the resolved `num_blocks` / `block_size` (two places — `PagedKvCache::new(...)` and the `info!` log). Update the `info!` log to:

```rust
                info!(
                    "Paged KV cache: block_size={block_size} ({block_label}), \
                     num_blocks={num_blocks} ({nb_label}), \
                     capacity={} tokens, kv_dim={}",
                    block_size * num_blocks,
                    model_config.num_key_value_heads * model_config.head_dim,
                );
```

Inside the `"naive"` arm, no change is needed — `NaiveKvCache::new` does not consume block size.

- [ ] **Step 6.7: Build + run smoke test**

```bash
cargo build --release -p forge-server
./target/release/forge-server --model-path /home/wilamhou/models/qwen3-4b --port 8951 &
SERVER_PID=$!
until curl -sf http://localhost:8951/v1/models > /dev/null 2>&1; do sleep 3; done
echo "READY"
curl -s -N -X POST http://localhost:8951/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"stream":false}' | head -c 400
echo
kill $SERVER_PID
```

Expected: startup log line `Paged KV cache: block_size=256 (auto), num_blocks=128 (auto), capacity=32768 tokens, kv_dim=...` and the chat response returns a coherent reply (not an HTTP error).

- [ ] **Step 6.8: Verify the kill switch and explicit override paths**

```bash
# Kill switch — should log block_size=256 still (auto) but skip FA2 in dispatch.
FORGE_FA2_PAGED=0 ./target/release/forge-server --model-path /home/wilamhou/models/qwen3-4b --port 8951 &
SERVER_PID=$!
until curl -sf http://localhost:8951/v1/models > /dev/null 2>&1; do sleep 3; done
# Hit a request, kill.
curl -s -X POST http://localhost:8951/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"stream":false}' | head -c 200
echo
kill $SERVER_PID

# Explicit override — block_size=16 should log "set" and route fallback by alignment.
./target/release/forge-server --model-path /home/wilamhou/models/qwen3-4b --block-size 16 --port 8951 &
SERVER_PID=$!
until curl -sf http://localhost:8951/v1/models > /dev/null 2>&1; do sleep 3; done
curl -s -X POST http://localhost:8951/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"stream":false}' | head -c 200
echo
kill $SERVER_PID
```

Expected: both responses coherent; the second startup log shows `block_size=16 (set), num_blocks=2048 (auto)`.

- [ ] **Step 6.9: Commit**

```bash
git add forge-server/src/main.rs
git commit -m "feat(server): --block-size / --num-blocks default to auto

Resolution lives in run_server (after backend selection, before cache
construction). Auto block_size queries Backend::preferred_block_size;
auto num_blocks keeps a fixed ~32k token capacity (DEFAULT_KV_TOKEN_CAPACITY).
Fixed(0) bails at resolve time. Startup log prints the resolved values
and whether each came from auto or an explicit flag.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 7: Scheduler capacity sanity test for `block_size=256`

**Files:**
- Modify: `forge-scheduler/tests/test_scheduler.rs` (append a new test alongside existing ones)

- [ ] **Step 7.1: Write the failing test**

Append to `forge-scheduler/tests/test_scheduler.rs`:

```rust
fn big_block_cache() -> CacheUsage {
    CacheUsage {
        total_blocks: 128, // = 32_768 / 256, matches DEFAULT_KV_TOKEN_CAPACITY
        used_blocks: 0,
        block_size: 256,
    }
}

#[test]
fn test_4096_token_prompt_reserves_16_blocks_at_block_size_256() {
    // Guards the ceil(prompt_len / block_size) math in
    // forge-scheduler/src/continuous.rs:211-220, 275-283 against the
    // auto-resolved block_size=256 default.
    let mut scheduler = ContinuousBatchingScheduler::new(SchedulerConfig {
        max_batch_size: 32,
        max_prefill_tokens: 8_192, // must admit the 4096-token prompt
        ..Default::default()
    });
    let cache = big_block_cache();

    let prompt: Vec<u32> = (0..4_096).collect();
    scheduler
        .enqueue(make_request("req-big", prompt.clone()))
        .unwrap();
    let batch = scheduler.schedule(&cache).unwrap();

    assert_eq!(batch.prefill_seqs.len(), 1);
    assert_eq!(batch.prefill_seqs[0].token_ids.len(), 4_096);

    // 4096 / 256 = 16 blocks reserved.
    let used_after = batch.prefill_seqs[0].blocks_needed.unwrap_or(0);
    assert_eq!(
        used_after, 16,
        "expected ceil(4096/256)=16 blocks, scheduler reported {used_after}"
    );
}
```

If `ScheduleBatch.prefill_seqs[0].blocks_needed` is not the actual field name in this codebase, replace the assertion with whatever the scheduler exposes that means "blocks reserved for this prefill". Check `forge-scheduler/src/continuous.rs` and `forge-core/src/scheduler.rs` for the correct field. If the scheduler does not expose this directly, assert via `cache.used_blocks` after a follow-up `cache.allocate(...)` call — adapt to the existing API. Goal: prove ceil math sees 16, not the wrong fraction.

- [ ] **Step 7.2: Run the test**

```bash
cargo test -p forge-scheduler test_4096_token_prompt_reserves_16_blocks_at_block_size_256
```

Expected: PASS (the math already works; this test only guards against a regression).

- [ ] **Step 7.3: Workspace check**

```bash
cargo test -p forge-scheduler
```

Expected: all scheduler tests PASS.

- [ ] **Step 7.4: Commit**

```bash
git add forge-scheduler/tests/test_scheduler.rs
git commit -m "test(scheduler): guard ceil math at block_size=256 (auto default)

A 4096-token prompt under block_size=256 must reserve exactly 16 blocks.
Test sits in forge-scheduler because the integration math lives in
continuous.rs:211-220, 275-283 (forge-kvcache has no dependency on the
scheduler crate, so it cannot host this test).

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 8: Docs — README, CONTRIB, RUNBOOK, codemap

**Files:**
- Modify: `README.md` (the flags table, around line 70)
- Modify: `docs/CONTRIB.md` (the flags table, around line 79)
- Modify: `docs/RUNBOOK.md`
- Modify: `docs/codemaps/backend.md`

- [ ] **Step 8.1: README flag descriptions**

Find the `--block-size` row in `README.md` (line ~70). Replace:

```
| `--block-size` | 16 | Paged cache block size (tokens) |
```

with:

```
| `--block-size` | `auto` | Paged cache block size (tokens). `auto` picks 256 when the CUDA backend + FA2 + F16 model conditions hold, else 16. |
| `--num-blocks` | `auto` | Total KV cache blocks. `auto` keeps ~32k token capacity by computing `ceil(32_768 / block_size)`. |
```

(If `--num-blocks` already has a row, update it instead of duplicating.)

- [ ] **Step 8.2: CONTRIB flag table**

Apply the same update to `docs/CONTRIB.md` near line 79.

- [ ] **Step 8.3: RUNBOOK — when to override block_size**

Append a new section to `docs/RUNBOOK.md` (place near the existing "Common knobs" / "Troubleshooting" sections; or create one if none exists):

```markdown
### When to override `--block-size`

The default is `auto`, which resolves to 256 on CUDA + FA2-eligible
models (head_dim ∈ {32,64,96,128,192,256}, F16) and to 16 otherwise.
Set the flag explicitly when:

- **CPU backend debugging.** `--block-size 16` keeps the per-sequence KV
  allocation small; 256 is wasteful when there is no FA2 to consume it.
- **Short-context benchmarking.** At `block_size=256` an active sequence
  can carry up to 255 tokens of internal fragmentation in its tail block
  (vs 15 at the old default). For very short generations (< 1k tokens)
  the working-set waste can matter; pin `--block-size 16`.
- **Reproducing FA2 vs split-KV comparisons.** Lock 16 and 256 explicitly
  rather than relying on the auto default.

### `FORGE_FA2_PAGED=0` kill switch

The FA2 dispatch path has no runtime fallback. If a new model geometry
triggers a CUDA error from inside `flash_fwd_kvcache`, set
`FORGE_FA2_PAGED=0` (or `=false`, case-insensitive) to force the
split-KV fallback without changing the block size. Note: the previous
`=1` opt-in semantics are gone — any non-zero value (including `=on`,
`=enabled`) now leaves FA2 enabled. The only disabling values are `=0`
and `=false`.
```

- [ ] **Step 8.4: Codemap — Backend trait surface**

In `docs/codemaps/backend.md`, find the bulleted list of `Backend` trait methods (search for `name(&self)` or `device_count`). Add:

```markdown
- `preferred_block_size(&self, head_dim: usize, dtype: DType) -> usize`
  — backends declare their preferred KV-cache block size for a given
  model geometry. CudaBackend returns 256 when FA2 paged-decode is
  eligible (`fa2_paged_eligible`), else 16; CpuBackend / TestBackend
  inherit the trait default (16).
```

- [ ] **Step 8.5: Verify docs render**

```bash
git diff README.md docs/CONTRIB.md docs/RUNBOOK.md docs/codemaps/backend.md
```

Expected: the changes show only the intended additions / row updates.

- [ ] **Step 8.6: Commit**

```bash
git add README.md docs/CONTRIB.md docs/RUNBOOK.md docs/codemaps/backend.md
git commit -m "docs: --block-size auto + FORGE_FA2_PAGED=0 kill-switch semantics

- README + CONTRIB flag tables updated for auto defaults.
- RUNBOOK gains 'when to override --block-size' and a note on the
  env-var semantics flip (=on / =enabled now enables FA2).
- Backend codemap lists preferred_block_size.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 9: Bench sanity-check + acceptance gate

**Files:** none (acceptance step, produces an artifact under `/tmp/`).

- [ ] **Step 9.1: Workspace full-test pass**

```bash
cargo test --workspace
cargo build --release --no-default-features
cargo build --release -p forge-server
```

Expected: all tests pass on CUDA host; CPU-only release build succeeds.

- [ ] **Step 9.2: Default-flag bench (matches baseline `forge_B`)**

```bash
mkdir -p /tmp/bench-$(date +%F)
./target/release/forge-server --model-path /home/wilamhou/models/qwen3-4b --port 8951 \
  > /tmp/bench-$(date +%F)/post_default.log 2>&1 &
SERVER_PID=$!
until curl -sf http://localhost:8951/v1/models > /dev/null 2>&1; do sleep 3; done
for c in 1 2 4 8; do
  echo "=== C=$c ==="
  python3 scripts/bench_concurrent.py http://localhost:8951 qwen3-4b 256 1024 $c 3
done | tee /tmp/bench-$(date +%F)/post_default.txt
kill $SERVER_PID
```

Expected: per-stream TPOT within ±1 ms of the lock baseline (`/tmp/bench-2026-06-09/baseline.jsonl` row `forge_B`): C=1 ~40.5 ms, C=2 ~40.8 ms, C=4 ~42.4 ms, C=8 ~46.5 ms.

- [ ] **Step 9.3: Kill-switch regression bench (matches baseline `forge_A`)**

```bash
FORGE_FA2_PAGED=0 ./target/release/forge-server --model-path /home/wilamhou/models/qwen3-4b --port 8951 \
  > /tmp/bench-$(date +%F)/post_killswitch.log 2>&1 &
SERVER_PID=$!
until curl -sf http://localhost:8951/v1/models > /dev/null 2>&1; do sleep 3; done
for c in 1 2 4 8; do
  echo "=== C=$c ==="
  python3 scripts/bench_concurrent.py http://localhost:8951 qwen3-4b 256 1024 $c 3
done | tee /tmp/bench-$(date +%F)/post_killswitch.txt
kill $SERVER_PID
```

Expected: per-stream TPOT within ±1 ms of baseline row `forge_A`: C=1 ~42.3 ms, C=2 ~42.5 ms, C=4 ~43.4 ms, C=8 ~48.7 ms. Proves the kill switch routes to the split-KV fallback.

- [ ] **Step 9.4: Final commit (only if no code drift) + push**

If steps 9.1-9.3 all match expectation, no code change is needed. Otherwise: investigate the deviation (timer noise, GPU clocks, regression in a tangent fix), patch, and re-run from 9.1.

```bash
git push origin feat/quantized-decode
```

Expected: branch up-to-date with origin.

---

## Self-Review

**Spec coverage:** every section in `docs/superpowers/specs/2026-06-03-block-size-auto-design.md` rev 3 is mapped:

| Spec section | Plan task |
|---|---|
| §1 CLI surface (AutoUsize, resolve in run_server) | Tasks 5, 6 |
| §2 Backend trait + fa2_paged_eligible helper + CudaBackend override | Tasks 1, 2, 3 |
| §3 FA2 dispatch flip (kill-switch + helper routing) | Task 4 |
| §4 Tests (parser, truth-table, scheduler capacity, FA2 differential) | Tasks 1, 5, 6, 7 (predicate test); Task 4 (differential) |
| §5 Documentation | Task 8 |
| §6 Scope of touch (no scheduler / cache structural change) | Confirmed in Task 7 (test only guards math); no other tasks touch those crates |
| Acceptance gates | Task 9 |

**Placeholder scan:** no TBD / TODO / "implement later"; every step has executable commands or full code. The only "may need to adapt" note is in Task 7 step 1 (`blocks_needed` field name) — annotated with where to look and what the test is proving, so the implementer can substitute the correct field without losing the intent.

**Type consistency:** `AutoUsize`, `BlockConfig`, `parse_auto_usize`, `resolve_block_config`, `DEFAULT_KV_TOKEN_CAPACITY`, `fa2_paged_eligible`, `FA2_SUPPORTED_HEAD_DIMS`, `preferred_block_size`, `FORGE_FA2_PAGED` — used consistently across tasks. `DType::F16` literal passed to the helper at the dispatch site matches the F16 arm invariant called out in Task 4.

**Cross-task dependency check:** Task 1 publishes `pub(crate) fa2_paged_eligible`; Task 3 calls it via `crate::fa2_paged_eligible`; Task 4 calls it via the same path. All three are in `forge-backend-cuda`, so `pub(crate)` is sufficient — no API leakage.
