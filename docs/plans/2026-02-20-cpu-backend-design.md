# CPU Backend Design

**Date:** 2026-02-20
**Status:** Approved
**Goal:** Add `forge-backend-cpu` crate so Forge can load and run TinyLlama (1.1B) without a GPU.

## Architecture

New crate `forge-backend/forge-backend-cpu/` implementing the `Backend` trait (23 methods).

```
forge-backend/
├── forge-backend-cuda/    # existing
└── forge-backend-cpu/     # new
    ├── Cargo.toml
    └── src/
        ├── lib.rs         # pub mod + re-exports
        ├── backend.rs     # CpuBackend impl Backend
        └── tensor.rs      # CpuTensor impl Tensor
```

## Tensor Representation

```rust
#[derive(Clone, Debug)]
pub struct CpuTensor {
    data: Arc<Vec<f32>>,   // all dtypes converted to f32 on ingest
    shape: Vec<usize>,
    dtype: DType,          // tracks original dtype for metadata
}
```

- `Arc<Vec<f32>>` — cheap Clone, zero-copy reshape
- f16/bf16 inputs converted to f32 in `copy_from_host_*`
- `copy_to_host_f32` returns `data.to_vec()`

## Backend

```rust
#[derive(Clone)]
pub struct CpuBackend;  // stateless, zero-cost Clone
```

- `synchronize()` → no-op
- `name()` → `"cpu"`
- `device_count()` → `1`

## Operation Implementation

| Op | Strategy |
|---|---|
| `matmul` | `cblas::sgemm` via OpenBLAS |
| `add`, `mul`, `mul_scalar` | `Vec<f32>` element-wise loops |
| `silu` | `x * (1.0 / (1.0 + (-x).exp()))` |
| `rms_norm` | `sqrt(mean(x²) + eps)`, scale by weight |
| `softmax` | `exp(x - max) / sum(exp)` (numerically stable) |
| `rope` | Complex rotation per pair |
| `embedding` | Gather rows by index |
| `reshape` | Change shape metadata, share `Arc` data |
| `transpose` | Copy with index remapping (2D only) |
| `cat` | `extend_from_slice` concatenation (dim 0 only) |
| `allocate` | `vec![0.0; numel]` |
| `allocate_zeros` | `vec![0.0; numel]` |

## Dependencies

```toml
[dependencies]
forge-core = { path = "../../forge-core" }
half.workspace = true
cblas = "0.4"
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

Using `openblas-src` with `system` feature — requires `libopenblas-dev` installed.

## Server Integration

Add `--backend cpu|cuda` CLI flag to `forge-server/src/main.rs`. Default: `cuda`.

```rust
#[arg(long, default_value = "cuda")]
backend: String,  // "cpu" or "cuda"
```

The main function branches on this to create the appropriate backend and monomorphized engine.

## Testing

- Port relevant tests from `forge-backend-cuda/tests/test_kernels.rs`
- Numerical accuracy: results should match CUDA within f32 epsilon
- Integration: load TinyLlama with CPU backend, run single forward pass

## Non-Goals

- Multi-threaded inference (single-threaded is fine)
- Quantized dtypes (I8, I4, Q4K, Q8K) — f32 only for now
- Performance parity with CUDA
