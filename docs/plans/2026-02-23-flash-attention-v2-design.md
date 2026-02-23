# FlashAttention v2 FFI Integration

## Context

After Phase 2 Tasks 1-4 (fused kernels + batched decode attention), the prefill path still uses naive per-head attention with O(N^2) memory. FlashAttention v2 provides a tiled, memory-efficient algorithm that is 2-4x faster for prefill and long-context decode.

**Goal:** Integrate FA2 via C++ FFI behind a feature gate. Zero impact when disabled.

**Decisions:**
- Vendor FA2 C++ sources (self-contained, reproducible)
- Target SM80 (Ampere) + SM90 (Hopper)
- Use for prefill + long single-sequence decode; batched decode keeps existing kernel

---

## Crate Structure

```
forge-flash/
├── Cargo.toml          # build-deps: cc; no runtime deps
├── build.rs            # cc::Build compiling FA2 CUDA sources
├── csrc/
│   ├── flash_api.cu    # Thin C wrapper exposing 2 entry points
│   ├── flash_fwd_*.cu  # Vendored FA2 forward kernels
│   ├── cutlass/        # Vendored CUTLASS headers (SM80+SM90)
│   └── ...
└── src/
    └── lib.rs          # extern "C" bindings + safe Rust wrapper
```

Pure FFI crate — no Rust logic beyond bindings. `forge-backend-cuda` depends on it behind `flash-attn` feature gate.

---

## FFI API

Two C entry points matching FA2's native interface:

```c
// Uniform sequence length (single-seq prefill, long decode)
void flash_attn_fwd(
    void* q, void* k, void* v, void* out,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    float scale, bool is_causal, int dtype, // 0=F16, 1=BF16
    cudaStream_t stream
);

// Variable-length batched (future: batched prefill via cu_seqlens)
void flash_attn_varlen_fwd(
    void* q, void* k, void* v, void* out,
    int* cu_seqlens_q, int* cu_seqlens_k,
    int max_seqlen_q, int max_seqlen_k,
    int total_q, int total_k,
    int num_heads, int num_heads_k, int head_dim,
    float scale, bool is_causal, int dtype,
    cudaStream_t stream
);
```

Rust wrapper in `forge-flash/src/lib.rs`:

```rust
pub unsafe fn flash_attn_fwd(
    q: u64, k: u64, v: u64, out: u64,  // CUdeviceptr
    batch_size: i32, seqlen_q: i32, seqlen_k: i32,
    num_heads: i32, num_heads_k: i32, head_dim: i32,
    scale: f32, is_causal: bool, dtype: FlashDType,
    stream: u64,  // CUstream
) { ... }
```

Only `flash_attn_fwd` needed initially. `varlen_fwd` is for future batched prefill.

---

## Runtime Dispatch

In `forge-backend-cuda/src/flash_attention.rs`:

```rust
pub fn attention_fwd(
    backend: &CudaBackend,
    q: &CudaTensor, k: &CudaTensor, v: &CudaTensor,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    scale: f32, is_causal: bool,
) -> Result<CudaTensor> {
    #[cfg(feature = "flash-attn")]
    {
        let (q, k, v) = cast_to_half_if_needed(backend, q, k, v)?;
        let out = forge_flash::flash_attn_fwd(...);
        return cast_back_if_needed(backend, &out, original_dtype);
    }

    naive_attention_causal(backend, q, k, v, scale, is_causal)
}
```

Dispatch logic in model layers:

```
decode (seq_len=1) + batch (N>1) → batched_decode_attention  [existing]
decode (seq_len=1) + single     → multi_head_attention       [FA2 or naive]
prefill (seq_len>1)             → multi_head_attention       [FA2 or naive]
```

### Backend Trait Addition

```rust
fn multi_head_attention(
    &self, q: &Self::Tensor, k: &Self::Tensor, v: &Self::Tensor,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    scale: f32, is_causal: bool,
) -> Result<Self::Tensor>;
```

Default impl: existing per-head loop from `compute_attention`. CUDA override: routes through `flash_attention::attention_fwd`.

This replaces the per-head loop in model code with a single backend call.

---

## Build System

### `forge-flash/build.rs`

```rust
fn main() {
    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    cc::Build::new()
        .cuda(true)
        .files(&["csrc/flash_api.cu", "csrc/flash_fwd_hdim64.cu", ...])
        .include("csrc/cutlass/include")
        .include(format!("{cuda_home}/include"))
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .flag("-O3")
        .flag("--use_fast_math")
        .compile("flash_attn");
}
```

### Feature Gating

```toml
# forge-backend-cuda/Cargo.toml
[features]
default = []
flash-attn = ["dep:forge-flash"]

[dependencies]
forge-flash = { path = "../forge-flash", optional = true }
```

Users opt in with `cargo build --features flash-attn`. Without it, everything compiles and works using naive attention + batched decode kernel.

Build time: ~2-5 minutes first build, cached thereafter.

---

## Testing

### Correctness

Compare FA2 output against naive attention:

```rust
#[test]
#[cfg(feature = "flash-attn")]
fn test_flash_matches_naive() {
    // Random Q/K/V in F16
    // Run both naive and FA2
    // Assert match within 1e-2 (F16 tolerance)
}
```

Test matrix:
- `seq_len`: 1, 32, 128, 2048
- `head_dim`: 64, 128
- GQA: `num_heads=32, num_kv_heads=8`
- Causal vs non-causal
- F16 and BF16

### Integration

```rust
#[test]
#[cfg(feature = "flash-attn")]
fn test_prefill_with_flash_attention() {
    // Build tiny model, run prefill with FA2
    // Compare logits against CPU reference
}
```

### CI

FA2 tests only run with `--features flash-attn`. Default `cargo test` skips them.

---

## Summary

| Component | Description |
|-----------|-------------|
| `forge-flash` crate | Vendored FA2 C++ + CUTLASS, `cc::Build` for SM80/SM90 |
| FFI API | `flash_attn_fwd` (uniform), `flash_attn_varlen_fwd` (future) |
| Backend trait | New `multi_head_attention`, replaces per-head loop |
| Dispatch | FA2 for prefill + long decode, batched kernel for multi-seq decode |
| Feature gate | `flash-attn`, opt-in, zero impact when disabled |
| Auto-cast | F32 → F16 before FA2, cast back after |

Expected gain: 2-4x attention speedup for prefill and long-context decode.
