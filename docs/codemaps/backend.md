# Backend Codemap

> Freshness: 2026-02-21 | Branch: feat/phase1-mvp

## forge-core — Traits & Abstractions

### `forge-core/src/backend.rs`
```rust
pub trait Backend: Send + Sync + 'static {
    type Tensor: Tensor;
    fn matmul(a, b, trans_a, trans_b) -> Result<Tensor>
    fn matmul_f16(a, b, trans_a, trans_b) -> Result<Tensor>
    fn add(a, b) -> Result<Tensor>
    fn mul(a, b) -> Result<Tensor>
    fn rms_norm(x, weight, eps) -> Result<Tensor>
    fn rms_norm_f16(x, weight, eps) -> Result<Tensor>
    fn softmax(x) -> Result<Tensor>
    fn silu_elementwise_mul(gate, up) -> Result<Tensor>
    fn silu_elementwise_mul_f16(gate, up) -> Result<Tensor>
    fn rope(x, freqs_cos, freqs_sin) -> Result<Tensor>
    fn rope_f16(x, freqs_cos, freqs_sin) -> Result<Tensor>
    fn copy_to_host_f32(tensor) -> Result<Vec<f32>>
    fn copy_from_host_f32(data, shape) -> Result<Tensor>
    fn cast_f32_to_f16(tensor) -> Result<Tensor>
    fn cast_f16_to_f32(tensor) -> Result<Tensor>
    fn synchronize() -> Result<()>
}
```

### `forge-core/src/tensor.rs`
```rust
pub trait Tensor: Clone + Send + Sync + 'static {
    fn shape(&self) -> &[usize]
    fn dtype(&self) -> DType
    fn reshape(self, shape) -> Result<Self>
    fn slice(self, dim, start, end) -> Result<Self>
    fn contiguous(self) -> Result<Self>
}

pub enum DType { F32, F16, BF16, I32, U32 }
```

## forge-backend-cuda — CUDA Backend

### Files
| File | Purpose |
|------|---------|
| `src/lib.rs` | `CudaBackend` struct, `Backend` impl |
| `src/tensor.rs` | `CudaTensor` struct, `Tensor` impl, device memory management |
| `src/attention.rs` | Naive attention with causal masking |
| `src/flash_attention.rs` | Attention dispatch (delegates to naive; Phase 2: PagedAttention) |
| `build.rs` | No-op (reserved for future build steps) |
| `tests/test_kernels.rs` | Integration tests for CUDA ops |

### `CudaBackend`
- Wraps `cudarc::CudaDevice` + `CudaStream` + `CudaBlas`
- Constructor: `CudaBackend::new(device_ordinal) -> Result<Self>`
- All ops dispatch to cuBLAS (matmul) or custom CUDA kernels (element-wise)
- FP16 path: `matmul_f16` uses `cublasGemmEx` with `CUDA_R_16F`
- Attention: delegates to naive attention (Phase 2 will add PagedAttention)

### `CudaTensor`
- `data: CudaSlice<u8>` (type-erased GPU buffer)
- `shape: Vec<usize>`, `strides: Vec<usize>`, `dtype: DType`
- Methods: `f32_slice()`, `f16_slice()`, `f16_slice_mut()`, `bf16_slice_mut()`

## forge-backend-cpu — CPU Backend

### Files
| File | Purpose |
|------|---------|
| `src/lib.rs` | `CpuBackend` struct, `Backend` impl |
| `src/tensor.rs` | `CpuTensor` struct, `Tensor` impl |
| `src/ops.rs` | CPU compute: matmul, rms_norm, softmax, silu, rope |
| `tests/` | Unit tests for CPU ops |

### `CpuBackend`
- Pure Rust, no GPU dependency
- Matmul: uses `half::f16` for FP16, standard f32 loops
- All ops implemented in `ops.rs` with explicit loops

## forge-kernels — CUDA C++ Kernels

### Files
| File | Purpose |
|------|---------|
| `src/lib.rs` | Module declarations |
| `src/elementwise.rs` | Add, mul, mul_scalar, silu kernel sources (F32 + F16) |
| `src/norm.rs` | RMS norm, softmax kernel sources (F32 + F16) |
| `src/positional.rs` | RoPE, embedding kernel sources (F32 + F16) |
| `src/memory.rs` | Transpose, cast kernel sources (F32 + F16) |
| `src/attention.rs` | Extract head, causal mask, interleave heads kernel sources (F32 + F16) |

### Usage
Kernel source strings are concatenated at runtime in `CudaBackend::new()` and compiled to PTX via NVRTC.

## forge-kvcache — KV Cache

### Files
| File | Purpose |
|------|---------|
| `src/lib.rs` | Module exports |
| `src/naive.rs` | `NaiveKvCache<B>` — CPU-side f32 vectors |
| `src/paged_cache.rs` | `PagedKvCache<B>` — GPU block manager |
| `src/block_manager.rs` | `BlockManager` — block allocation/freeing |

### `KvCache` trait (`forge-core`)
```rust
pub trait KvCache: Send + Sync {
    type T: Tensor;
    fn allocate(seq_id, initial_len) -> Result<()>
    fn append(seq_id, layer, key, value) -> Result<()>
    fn get_kv(seq_id, layer) -> Result<(T, T)>
    fn get_block_table(seq_id) -> Result<Vec<usize>>
    fn get_seq_len(seq_id) -> Result<usize>
    fn free(seq_id) -> Result<()>
    fn usage() -> CacheUsage
    fn can_allocate(num_tokens) -> bool
}
```

### NaiveKvCache
- Stores K/V as CPU `Vec<f32>`, reconstructs device tensors on retrieval
- `max_total_tokens` (default 128K), `max_sequences` limits
- `usage()` sums actual tokens across sequences

### PagedKvCache
- Uses `BlockManager` for block-level allocation
- GPU-side storage in pre-allocated block pool
- Block table per sequence for paged access
