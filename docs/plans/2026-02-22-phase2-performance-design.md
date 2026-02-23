# Phase 2: Performance Pass Design

**Date:** 2026-02-22
**Branch:** feat/phase1-mvp (builds on completed batch decode work)
**Target hardware:** Ampere (SM80+) and Hopper (SM90+)

## Goal

Full performance pass across the inference pipeline: fused NVRTC kernels, batched decode attention, FlashAttention v2 FFI, and GPU-side sampling. Bottom-up approach — each step is independently shippable and testable.

## Approach

Hybrid kernel strategy:
- **NVRTC** for fused elementwise/norm kernels (existing pipeline, low risk)
- **FFI to FlashAttention v2** for prefill attention (battle-tested, maximum perf)
- **Custom NVRTC** for batched decode attention (FA2 is overkill for single-token queries)

Naive attention remains as a correctness fallback for all paths.

---

## Step 1: Fused NVRTC Kernels

Three kernel fusions targeting kernel-launch overhead in the decoder layer loop.

### 1a. `fused_residual_rms_norm`

**Current:** `backend.add(x, attn_output)` + `backend.rms_norm(residual, w)` = 2 launches per call, called twice per layer (pre-attention norm + pre-MLP norm).

**Fused:** Single kernel that:
1. Adds residual: `tmp = x + residual_in`
2. Writes `tmp` to `residual_out` (for next residual connection)
3. Computes RMS via shared-memory reduction
4. Normalizes: `out = (tmp / rms) * weight`

Grid: `(rows, 1, 1)`, Block: `(block_dim, 1, 1)`, shared mem for reduction.

Saves 2N kernel launches + 2N intermediate tensor allocations for N layers.

### 1b. `fused_qkv_projection`

**Current:** 3 separate cuBLAS GEMM calls for Q, K, V projections.

**Fused:** Concatenate `wq`, `wk`, `wv` into `wqkv` at model load time. Single GEMM + lightweight split kernel.

- `wqkv = cat([wq, wk, wv], dim=0)` — done once during loading
- `qkv = matmul(x, wqkv)` — 1 GEMM (better GPU utilization from larger tile)
- `(q, k, v) = split_qkv(qkv, q_size, kv_size)` — 1 lightweight kernel

Saves ~10-20us per layer from eliminated GEMM launch overhead. Larger single GEMM has better utilization.

### 1c. `fused_silu_mul`

**Current:** `backend.silu(gate_proj)` + `backend.mul(gate, up_proj)` = 2 launches.

**Fused:** Single elementwise kernel: `out[i] = (g / (1 + exp(-g))) * up[i]`.

Saves 1 kernel launch + 1 intermediate tensor per layer.

### Backend trait additions

```rust
fn fused_residual_rms_norm(&self, x: &T, residual: &T, weight: &T, eps: f32)
    -> Result<(T, T)>;  // (normalized, updated_residual)
fn fused_silu_mul(&self, gate: &T, up: &T) -> Result<T>;
fn split_qkv(&self, qkv: &T, q_size: usize, kv_size: usize)
    -> Result<(T, T, T)>;
```

Default impls fall back to unfused operations (CPU backend unchanged).

### Expected gain: 15-20% end-to-end

---

## Step 2: Batched Decode Attention Kernel

Replace the per-sequence attention loop in `forward_batch` with a single kernel launch.

### Problem

Current `forward_batch` loops N times (once per sequence), each iteration doing per-head attention with 7 kernel launches. For batch_size=16, num_heads=32: 3,584 kernel launches per layer.

### Solution

Single `batched_decode_attention` kernel:
- Grid: `(num_seqs, num_heads, 1)`
- Each thread block handles one (sequence, head) pair
- Loads 1 query token, streams through that sequence's full KV cache
- Online softmax (never materializes score vector)
- GQA-native (head-to-kv-head mapping in kernel)
- No causal mask needed (single query token sees all KV)

```cuda
__global__ void batched_decode_attention_f32(
    float* out,              // [num_seqs, num_heads, head_dim]
    const float* q,          // [num_seqs, num_heads, head_dim]
    const float** k_ptrs,    // per-seq K cache pointers [num_seqs]
    const float** v_ptrs,    // per-seq V cache pointers [num_seqs]
    const int* kv_lens,      // per-seq KV cache length [num_seqs]
    float scale,
    int num_heads, int num_kv_heads, int head_dim, int max_kv_len
);
```

### Pointer table construction

Small host-side helper builds arrays of device pointers from KV cache:
- For each `seq_id`, call `kv_cache.get_kv()` to get raw device pointers
- Upload pointer arrays + `kv_lens` to GPU (one small H2D transfer)
- Amortized over entire layer

### Integration

```rust
// In LlamaAttention::forward_batch, replace per-sequence loop:
// 1. Batch QKV projection + RoPE (unchanged)
// 2. Per-seq cache append (N × 1 token, cheap)
// 3. Single batched_decode_attention kernel launch
// 4. Output projection matmul
```

From ~3,584 launches → ~5 per layer.

### Backend trait addition

```rust
fn batched_decode_attention(
    &self, q: &Self::Tensor,
    kv_cache: &dyn KvCache<T = Self::Tensor>,
    seq_ids: &[u64], layer_idx: usize,
    num_heads: usize, num_kv_heads: usize, head_dim: usize, scale: f32,
) -> Result<Self::Tensor>;
```

Default impl: falls back to per-sequence loop.

### Expected gain: 1.5-2x decode throughput for typical batch sizes

---

## Step 3: FlashAttention v2 FFI

Replace naive attention for prefill (and long-context decode) with FlashAttention v2.

### Crate: `forge-flash`

Separate crate isolating C++ build complexity:

```
forge-flash/
├── Cargo.toml       # build-deps: cc, bindgen
├── build.rs         # Compiles FA2 C++ sources for SM80+SM90
├── csrc/            # Vendored FA2 sources + thin C wrapper
└── src/lib.rs       # Safe Rust wrapper
```

### C wrapper API

Two entry points:
- `flash_attn_fwd` — uniform sequence lengths (single-seq prefill, long decode)
- `flash_attn_varlen_fwd` — variable-length batched (batched prefill via `cu_seqlens`)

Both take raw device pointers, dimensions, scale, causal flag, CUDA stream.

### Build system

`build.rs` uses `cc::Build` with:
- `-gencode=arch=compute_80,code=sm_80` (Ampere)
- `-gencode=arch=compute_90,code=sm_90` (Hopper)
- Vendored CUTLASS headers
- Auto-detection of `$CUDA_HOME`

Build time: ~2-5 minutes (one-time, cargo-cached).

### Feature gating

```toml
# forge-backend-cuda/Cargo.toml
[features]
default = []
flash-attn = ["forge-flash"]
```

Opt-in with `cargo build --features flash-attn`. Without it, naive attention + batched decode kernel are used.

### Runtime dispatch

```
if seq_len_q == 1 && num_seqs > 1:
    → batched_decode_attention (Step 2 kernel)
elif feature "flash-attn" enabled:
    → FlashAttention v2
else:
    → naive attention (fallback)
```

FA2 requires F16/BF16 input — auto-cast from F32 if needed.

### Backend trait change

Refactor `compute_attention` from per-head to full multi-head:

```rust
fn compute_attention(
    &self, q: &T, k: &T, v: &T,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    scale: f32, is_causal: bool,
) -> Result<T>;
```

This replaces the current per-head loop in model layer code with a single backend call.

### Expected gain: 2-4x attention speedup for prefill and long-context decode

---

## Step 4: GPU-Side Sampling

Move sampling from CPU to GPU, eliminating the `[vocab_size]` device-to-host transfer per token.

### Split at constraint boundary

- **Unconstrained requests:** Full GPU pipeline, only 4 bytes (token ID) cross the bus
- **Constrained requests (FSM):** GPU top-k extracts K candidates (~400 bytes), CPU does FSM masking + final sample

### GPU kernels

**`fused_temperature_softmax`:** Single kernel applies temperature scaling + softmax via shared-memory reduction. Grid: `(num_seqs, 1, 1)`.

**`top_k_filter`:** Two-pass radix select. Pass 1: find k-th largest value via histogram in shared memory. Pass 2: zero out below threshold, renormalize.

**`top_p_multinomial`:** After top-k, prefix-sum probabilities, find top-p cutoff, sample with cuRAND.

**`argmax`:** Shared-memory reduction for greedy decoding (temperature=0). No softmax needed.

### Backend trait additions

```rust
fn gpu_sample(
    &self, logits: &T, temperature: f32, top_k: usize, top_p: f32,
    seed: Option<u64>,
) -> Result<Vec<u32>>;

fn gpu_top_k_candidates(
    &self, logits: &T, temperature: f32, k: usize,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<f32>>)>;
```

Default impls: fall back to `copy_to_host_f32` + CPU sampler.

### Engine integration

```rust
if seq.has_fsm_constraint() {
    // GPU top-k → CPU FSM filter → CPU sample
    let (candidates, probs) = backend.gpu_top_k_candidates(logits, temp, 50)?;
    let masked = seq.fsm.mask_candidates(&candidates, &probs);
    cpu_sample(&masked)
} else {
    // Full GPU pipeline → token ID
    backend.gpu_sample(logits, temp, top_k, top_p, seed)?
}
```

### Expected gain: 10-15% e2e latency, larger for big vocabs and large batches

---

## Summary

| Step | What | Expected Gain | Risk | Dependencies |
|------|------|--------------|------|-------------|
| 1 | Fused NVRTC kernels | 15-20% | Low | None |
| 2 | Batched decode attention | 1.5-2x decode | Medium | Step 1 (fused QKV) |
| 3 | FlashAttention v2 FFI | 2-4x attention | High (C++ build) | None (parallel with 1-2) |
| 4 | GPU sampling | 10-15% e2e | Low-Medium | None |

Each step is independently shippable. Naive attention remains as correctness fallback throughout.
