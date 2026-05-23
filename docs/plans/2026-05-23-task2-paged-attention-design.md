# Task 2 — Paged Attention Kernel: Design

**Date:** 2026-05-23
**Status:** Design / not yet implemented
**Companion:** `docs/plans/2026-05-22-cuda-graphs-plan.md` (Task 2)
**Prereq for:** Task 2.5 (capture-safe ops) and Task 3 (CudaGraphCache)

## Why this task exists

Captured CUDA Graphs replay against the exact device pointers and shapes baked at capture time. The current decode attention path reconstructs full K/V tensors from `PagedKvCache::get_kv()` on every step — those tensors live at fresh device addresses each call, so a graph captured from this path can't replay correctly when the cache state changes.

**Replacement contract:** a `Backend::paged_attention` op that takes `(query, kv_pool, block_tables, kv_lens, ...)` directly, where `kv_pool` is a single persistent device allocation and `block_tables` is a device-resident i32 array. Same pool address every step → captured graph stays valid.

## Two things the parent plan did not foresee

Read of the codebase before writing this doc surfaced two facts that change Task 2's shape:

1. **`PagedKvCache` is currently CPU-resident.** `forge-kvcache/src/paged_cache.rs:20,22,74,321` — `key: Vec<f32>`, `value: Vec<f32>`, host vectors. Block tables are `Vec<usize>` on the host. There is no GPU pool today; `get_kv()` reconstructs full sequences on the host and the existing CUDA attention path consumes the H2D-copied result. **A paged_attention CUDA kernel cannot exist until the cache is migrated to device-resident storage.**
2. **`forge-flash` exposes only fixed-shape `flash_fwd`.** No `flash_fwd_varlen` is wired through `csrc/flash_api_forge.cu`. The parent plan's Option A ("wrap FA2 varlen with a gather") requires adding a `forge_flash_attn_varlen_fwd` extern + Rust wrapper before the gather kernel can be wired up.

Both must be addressed before any `Backend::paged_attention` implementation lands.

## Decision: Option B (from-scratch decode kernel), not Option A

Parent plan defaulted to Option A. After the findings above, **Option B is the lower-risk path for decode**:

| Aspect | Option A: FA2 varlen + gather | Option B: from-scratch paged kernel |
|---|---|---|
| Forge-flash changes | Add varlen FFI (~40 LOC C + Rust wrapper) | None |
| Extra device memory | Persistent packed `[total_tokens, num_heads, head_dim]` buffer per dispatch | None — kernel reads blocks in place |
| Kernel-side work | Gather kernel (~150 LOC CUDA) + FA2 varlen call | One paged-attention kernel (~250 LOC CUDA) |
| Maps to existing vLLM/SGLang kernel | No (gather is forge-specific) | Yes (vLLM's `single_query_cached_kv_attention` reference) |
| Capture-friendly | Gather + FA2 launch — depends on FA2 capture story we have NOT validated | Single kernel launch — fully cuLaunchKernel-compatible |
| Decode-only restriction | Works for prefill too | Decode-only (q_len=1, GQA-aware) — prefill stays on current `multi_head_attention` path |
| Total LOC | ~150 CUDA + ~40 C + ~30 Rust + ~80 wiring | ~250 CUDA + ~80 wiring |

The deciding factors: (a) no forge-flash extension needed, (b) no intermediate buffer to allocate-in-capture, (c) Task 1's spike never validated FA2-in-capture (the plan flags this as an open question), so betting decode on a graph-incompatible FA2 path is the wrong direction. Option B keeps prefill on the existing FA2 fixed-shape path (which we have already validated in isolation outside capture) and gives decode a self-contained capture-safe kernel.

If a later phase wants paged **prefill**, revisit Option A then.

## Sub-step ordering

The plan's original sub-steps were sized assuming PagedKvCache was already device-resident. Re-ordered for reality:

### 2.0 — Migrate `PagedKvCache` to device-resident pool (new prereq, ~2–3 days)

- Replace `BlockData { key: Vec<f32>, value: Vec<f32> }` with a single device-resident `kv_pool: CudaTensor` of shape `[num_blocks, block_size, 2 * num_kv_heads * head_dim]` (K and V interleaved per token) or two separate pools — pick during this task; vLLM uses two pools, SGLang fuses K+V into one. **Recommend two separate pools** for cleaner indexing and easier swizzling later.
- Replace `Vec<usize>` block tables with a per-seq `CudaTensor<i32>` mirror on device. Keep the host-side `Vec` as the source of truth for allocator bookkeeping; sync to device on `append` / `allocate`.
- `append(layer, K, V)`: write directly into the device pool via `memcpy_dtod` or a small `scatter_kv` kernel, no host detour.
- `get_kv()` stays for the non-paged-attention call sites (prefill, eviction debug) but issues a `gather` kernel device-to-device — host data flow is gone.
- CPU backend keeps a parallel `Vec<f32>` impl for portability + unit tests.
- Tests: `forge-kvcache/tests/test_paged_device_cache.rs` — round-trip append/read, multi-seq interleave, free + realloc, KV pool size accounting.
- Commit: `feat(kvcache): device-resident paged KV pool + i32 block tables`.

This is the riskiest sub-step because it touches every consumer of `PagedKvCache::get_kv()`. Run full E2E (`scripts/test_server.sh`) before commit.

### 2.1 — Add `Backend::paged_attention` trait method (~1h)

```rust
fn paged_attention(
    &self,
    q: &Self::Tensor,              // [batch, num_heads, head_dim]  (decode: q_len = 1)
    k_pool: &Self::Tensor,         // [num_blocks, block_size, num_kv_heads, head_dim]
    v_pool: &Self::Tensor,         // same shape as k_pool
    block_tables: &Self::Tensor,   // [batch, max_blocks_per_seq] i32, padding = -1
    kv_lens: &Self::Tensor,        // [batch] i32 — current KV length per seq
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Self::Tensor>;         // [batch, num_heads, head_dim]
```

CPU default impl: gather to contiguous K/V via the existing `Vec<f32>` shadow + naive attention loop. Slow but the existing CPU correctness tests already exercise this code path with one fewer copy.

Commit: `feat(core): Backend::paged_attention trait + CPU naive impl`.

### 2.2 — CUDA paged attention kernel (~3 days)

`forge-kernels/src/paged_attention.rs` exposes the kernel source string. Skeleton:

```c
// Each block = one (batch, head_group) pair.
// Block dim = head_dim (one thread per element of the output vector).
// Shared mem: warp-reduced softmax scratch + per-block KV cache lines.
extern "C" __global__ void paged_attention_f16(
    const half* __restrict__ q,             // [B, H_q, D]
    const half* __restrict__ k_pool,        // [N_blk, S_blk, H_kv, D]
    const half* __restrict__ v_pool,        // [N_blk, S_blk, H_kv, D]
    const int*  __restrict__ block_tables,  // [B, max_blk_per_seq]
    const int*  __restrict__ kv_lens,       // [B]
    half* __restrict__ out,                 // [B, H_q, D]
    int num_heads, int num_kv_heads, int head_dim,
    int max_blk_per_seq, int block_size,
    float scale);
```

Algorithm (decode q_len=1, GQA):

1. `head_idx = blockIdx.y`; `batch_idx = blockIdx.x`; `kv_head_idx = head_idx / heads_per_group`.
2. Load `q[batch_idx, head_idx, :]` into registers.
3. Two-pass softmax: (a) loop over blocks, compute logits per token, track running max + sum (online softmax); (b) loop again, accumulate `softmax(logits) * v`.
4. Write `out[batch_idx, head_idx, :]`.

Reference: vLLM `csrc/attention/attention_kernels.cu :: single_query_cached_kv_attention_kernel` for the structure; we re-implement to match forge's existing kernel style (no PyTorch deps, no tensor abstractions). F32 variant first, F16 + BF16 variants once F32 is correct.

Wire `Backend::paged_attention` for `CudaBackend` to launch this kernel.

Tests: `forge-backend/forge-backend-cuda/tests/test_paged_attention.rs` — compare paged-attention output against the existing `multi_head_attention` path on randomly populated KV cache states for 8 representative shapes (combinations of num_heads ∈ {32, 64}, num_kv_heads ∈ {4, 8, 32}, head_dim ∈ {128}, batch ∈ {1, 4, 16}, kv_len ∈ {32, 1024, 8000}). Tolerance `1e-3` (FP16/BF16).

Commit: `feat(cuda): paged attention kernel reading block tables directly`.

### 2.3 — Wire `LlamaAttention` decode path through `paged_attention` (~3h)

`forge-models/forge-model-llama/src/layers.rs:119-137` currently always calls `multi_head_attention()` after reconstructing K/V from the cache. Change:

- If `seq_len == 1` (decode), call `backend.paged_attention(q, k_pool, v_pool, block_tables, kv_lens, ...)`.
- Else (prefill / chunked prefill) keep the existing `multi_head_attention()` path.
- Both paths still call `cache.append(layer, K, V)` — the append happens before the attention call.

Commit: `feat(llama): route decode attention through paged_attention`.

### 2.4 — E2E + benchmark (~half day)

- `scripts/test_server.sh /path/to/model` — quality regression check.
- `scripts/benchmark.sh /path/to/model 100 256 8080` — perf check; expect decode TPOT to match or improve (less data movement → faster).
- If TPOT regresses, profile with `nsys` before merging. Decode is the critical path.

Commit: `perf: enable paged attention path for decode (E2E validated)`.

## Test plan summary

| Layer | Test file | What it asserts |
|---|---|---|
| KvCache | `forge-kvcache/tests/test_paged_device_cache.rs` | Device pool round-trip; multi-seq; block alloc / free |
| Backend trait | unit tests on CPU impl | Reference output matches naive gather + attend |
| CUDA kernel | `forge-backend/forge-backend-cuda/tests/test_paged_attention.rs` | Output ≈ `multi_head_attention` on 8 random shapes, tol 1e-3 |
| Model | `forge-models/forge-model-llama/tests/test_decode_paged.rs` | Logits identical between decode path on/off paged route |
| E2E | `scripts/test_server.sh` | Greedy decode produces same token stream as previous build |

## Open questions

1. **Block size choice.** vLLM defaults to 16, SGLang to 1–16, FlashInfer to 64. Larger = less block-table traffic, smaller = less internal fragmentation. Recommend matching forge's current `PagedKvCache` block_size (need to verify) and revisit in Task 7 (bucket promotion).
2. **K/V pool layout: separate vs interleaved.** Recommend separate, but the cost is one extra Tensor handle per layer. Decide in 2.0 PR.
3. **Quantized KV (FP8) compatibility.** Out of scope for Task 2 but the kernel signature should accept a generic dtype so future FP8 KV doesn't require a re-signature. The first-pass F32 / F16 / BF16 variants share one kernel template.
4. **Multi-GPU / TP impact.** When the two DGX-Spark boxes are linked (per `claude memory`, expected this week), TP would shard `num_kv_heads` across ranks. The kernel signature already takes `num_kv_heads` per-rank; no change needed for the kernel. PP doesn't touch this kernel.
5. **`disable_event_tracking` interaction.** Task 1 found that disabling cudarc's event tracking is a prereq for capture. The new paged_attention kernel must therefore not rely on cudarc auto-recording read/write events between block_tables and kv_lens tensors. Single-stream FIFO ordering covers this today; document in the kernel wiring code.

## Estimate

| Sub-step | Wall-clock |
|---|---|
| 2.0 device-resident PagedKvCache | 2–3 days |
| 2.1 trait method + CPU impl | 1h |
| 2.2 CUDA kernel (F32 → F16 → BF16) | 3 days |
| 2.3 LlamaAttention wiring | 3h |
| 2.4 E2E + bench | half day |
| **Total** | **~6–7 days** (was 4–5 in parent plan; +2 days for device-pool migration) |

Parent plan's Summary table needs updating; reflected in a separate `docs/plans/2026-05-22-cuda-graphs-plan.md` patch when this design is approved.

## Decision request

Two open choices before kicking off 2.0:

1. **K/V pool layout**: two separate `[N_blk, S_blk, H_kv, D]` pools, vs one interleaved `[N_blk, S_blk, 2, H_kv, D]` pool. Recommend two separate.
2. **Where to keep host-side block-table bookkeeping**: in `PagedKvCache` alongside the device mirror (recommend), or move entirely to device. Host-side is simpler for allocator logic; device mirror is small (max_seqs × max_blocks_per_seq × 4 bytes ≈ kilobytes).
