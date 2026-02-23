# Phase 2: Performance Pass Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Full performance pass — fused NVRTC kernels, batched decode attention, FlashAttention v2 FFI, GPU-side sampling.

**Architecture:** Bottom-up approach. Each task is independently shippable. Naive attention remains as correctness fallback. FlashAttention gated behind `flash-attn` feature flag. All new CUDA kernels added to forge-kernels NVRTC pipeline.

**Tech Stack:** Rust, CUDA (NVRTC runtime compilation), cuBLAS, FlashAttention v2 (C++ FFI), cudarc

---

## Task 1: Fused SiLU-Mul Kernel

Simplest fusion — merge `silu(gate)` + `mul(gate, up)` into one elementwise kernel. No shared memory, no reduction.

**Files:**
- Modify: `forge-kernels/src/elementwise.rs` — add `fused_silu_mul_f32` and `fused_silu_mul_f16` kernels
- Modify: `forge-core/src/backend.rs:28` — add `fused_silu_mul` trait method with default impl
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs:13-42` — add kernel function handles to `KernelFunctions`
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs:238` — add `fused_silu_mul` CUDA impl
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs` — add CPU override
- Modify: `forge-models/forge-model-llama/src/layers.rs:37-48` — update `LlamaMLP::forward` to use fused op
- Test: `forge-backend/forge-backend-cpu/tests/test_ops.rs` — add `test_fused_silu_mul`

**Step 1: Write the CUDA kernel source**

Add to `forge-kernels/src/elementwise.rs` F32_SRC (after `silu_f32`):

```cuda
extern "C" __global__ void fused_silu_mul_f32(
    float* out, const float* gate, const float* up, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
```

And F16_SRC:

```cuda
extern "C" __global__ void fused_silu_mul_f16(
    __half* out, const __half* gate, const __half* up, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        out[i] = __float2half((g / (1.0f + expf(-g))) * __half2float(up[i]));
    }
}
```

**Step 2: Add Backend trait method with default impl**

In `forge-core/src/backend.rs`, after `silu` (line 28):

```rust
/// Fused SiLU activation and element-wise multiply: out = silu(gate) * up
fn fused_silu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor) -> Result<Self::Tensor> {
    let activated = self.silu(gate)?;
    self.mul(&activated, up)
}
```

**Step 3: Add kernel handles and CUDA impl**

In `forge-backend/forge-backend-cuda/src/backend.rs`:
- Add `fused_silu_mul_f32: CudaFunction` and `fused_silu_mul_f16: CudaFunction` to `KernelFunctions` struct
- Load them in `CudaBackend::new()` alongside existing kernels
- Implement `fused_silu_mul` override using `LaunchConfig::for_num_elems(n)`

**Step 4: Add CPU override**

In `forge-backend/forge-backend-cpu/src/backend.rs`:

```rust
fn fused_silu_mul(&self, gate: &CpuTensor, up: &CpuTensor) -> Result<CpuTensor> {
    let g = gate.data();
    let u = up.data();
    let data: Vec<f32> = g.iter().zip(u.iter())
        .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui)
        .collect();
    Ok(CpuTensor::new(data, gate.shape().to_vec()))
}
```

**Step 5: Write test**

In `forge-backend/forge-backend-cpu/tests/test_ops.rs`:

```rust
#[test]
fn test_fused_silu_mul() {
    let backend = CpuBackend::new();
    let gate = backend.copy_from_host_f32(&[1.0, -1.0, 2.0, 0.0], &[4]).unwrap();
    let up = backend.copy_from_host_f32(&[2.0, 3.0, 1.0, 5.0], &[4]).unwrap();

    // Reference: silu then mul
    let ref_silu = backend.silu(&gate).unwrap();
    let ref_result = backend.mul(&ref_silu, &up).unwrap();

    // Fused
    let fused = backend.fused_silu_mul(&gate, &up).unwrap();

    let ref_data = ref_result.data();
    let fused_data = fused.data();
    for (a, b) in ref_data.iter().zip(fused_data.iter()) {
        assert!((a - b).abs() < 1e-6, "mismatch: {} vs {}", a, b);
    }
}
```

**Step 6: Run tests**

Run: `cargo test --workspace`
Expected: All pass including new `test_fused_silu_mul`

**Step 7: Update LlamaMLP to use fused op**

In `forge-models/forge-model-llama/src/layers.rs:37-48`, replace:

```rust
let gate = backend.silu(&gate)?;
let fused = backend.mul(&gate, &up)?;
```

With:

```rust
let fused = backend.fused_silu_mul(&gate, &up)?;
```

**Step 8: Run tests again**

Run: `cargo test --workspace`
Expected: All 209+ tests pass

**Step 9: Commit**

```bash
git add forge-kernels/src/elementwise.rs forge-core/src/backend.rs \
       forge-backend/forge-backend-cuda/src/backend.rs \
       forge-backend/forge-backend-cpu/src/backend.rs \
       forge-models/forge-model-llama/src/layers.rs \
       forge-backend/forge-backend-cpu/tests/test_ops.rs
git commit -m "perf: add fused SiLU-mul kernel"
```

---

## Task 2: Fused Residual + RMSNorm Kernel

Merge `add(x, residual)` + `rms_norm(sum, weight, eps)` into a single kernel with one shared-memory reduction pass. Outputs both normalized result and updated residual.

**Files:**
- Modify: `forge-kernels/src/norm.rs` — add `fused_residual_rms_norm_f32/f16` kernels
- Modify: `forge-core/src/backend.rs` — add `fused_residual_rms_norm` trait method
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — add kernel handles + CUDA impl
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs` — add CPU override
- Modify: `forge-models/forge-model-llama/src/layers.rs:292-352` — update `LlamaDecoderLayer::forward` and `forward_batch`
- Test: `forge-backend/forge-backend-cpu/tests/test_ops.rs`

**Step 1: Write the CUDA kernel source**

Add to `forge-kernels/src/norm.rs` F32_SRC (after `softmax_f32`):

```cuda
extern "C" __global__ void fused_residual_rms_norm_f32(
    float* norm_out,
    float* residual_out,
    const float* x,
    const float* residual_in,
    const float* weight,
    float eps,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const float* xi = x + row * cols;
    const float* ri = residual_in + row * cols;
    float* ro = residual_out + row * cols;
    float* no = norm_out + row * cols;

    extern __shared__ float shared[];

    // Phase 1: add residual, accumulate sum of squares
    float local_ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float tmp = xi[i] + ri[i];
        ro[i] = tmp;
        local_ss += tmp * tmp;
    }

    shared[threadIdx.x] = local_ss;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)cols + eps);

    // Phase 2: normalize
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        no[i] = ro[i] * rms * weight[i];
    }
}
```

And the F16 variant (same structure, with `__half2float`/`__float2half` conversions).

**Step 2: Add Backend trait method**

In `forge-core/src/backend.rs`:

```rust
/// Fused residual addition + RMSNorm.
/// Returns (normalized, updated_residual) where:
///   updated_residual = x + residual_in
///   normalized = rms_norm(updated_residual, weight, eps)
fn fused_residual_rms_norm(
    &self,
    x: &Self::Tensor,
    residual: &Self::Tensor,
    weight: &Self::Tensor,
    eps: f32,
) -> Result<(Self::Tensor, Self::Tensor)> {
    let sum = self.add(x, residual)?;
    let normed = self.rms_norm(&sum, weight, eps)?;
    Ok((normed, sum))
}
```

**Step 3: CUDA impl**

In `forge-backend/forge-backend-cuda/src/backend.rs`:
- Add `fused_residual_rms_norm_f32/f16` to `KernelFunctions`
- Load in `CudaBackend::new()`
- Override `fused_residual_rms_norm`:
  - Grid: `(rows, 1, 1)`, Block: `(block_dim, 1, 1)`, shared_mem: `block_dim * 4`
  - Allocate two output tensors (norm_out, residual_out)
  - Returns `(norm_out, residual_out)`

**Step 4: CPU override**

```rust
fn fused_residual_rms_norm(
    &self, x: &CpuTensor, residual: &CpuTensor, weight: &CpuTensor, eps: f32,
) -> Result<(CpuTensor, CpuTensor)> {
    let sum = self.add(x, residual)?;
    let normed = self.rms_norm(&sum, weight, eps)?;
    Ok((normed, sum))
}
```

**Step 5: Write test**

```rust
#[test]
fn test_fused_residual_rms_norm() {
    let backend = CpuBackend::new();
    let x = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let res = backend.copy_from_host_f32(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]).unwrap();
    let w = backend.copy_from_host_f32(&[1.0, 1.0, 1.0], &[3]).unwrap();

    // Reference
    let ref_sum = backend.add(&x, &res).unwrap();
    let ref_norm = backend.rms_norm(&ref_sum, &w, 1e-5).unwrap();

    // Fused
    let (fused_norm, fused_res) = backend.fused_residual_rms_norm(&x, &res, &w, 1e-5).unwrap();

    // Compare
    for (a, b) in ref_norm.data().iter().zip(fused_norm.data().iter()) {
        assert!((a - b).abs() < 1e-5);
    }
    for (a, b) in ref_sum.data().iter().zip(fused_res.data().iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}
```

**Step 6: Run tests**

Run: `cargo test --workspace`

**Step 7: Update decoder layer forward**

In `forge-models/forge-model-llama/src/layers.rs`, update `LlamaDecoderLayer::forward` (lines 292-319):

Before:
```rust
let normed = self.input_layernorm.forward(x, backend)?;
let attn_out = self.self_attn.forward(&normed, ...)?;
let x = backend.add(x, &attn_out)?;
let normed = self.post_attention_layernorm.forward(&x, backend)?;
let mlp_out = self.mlp.forward(&normed, backend)?;
backend.add(&x, &mlp_out)
```

After:
```rust
let normed = self.input_layernorm.forward(x, backend)?;
let attn_out = self.self_attn.forward(&normed, ...)?;
let (normed, x) = backend.fused_residual_rms_norm(
    &attn_out, x, &self.post_attention_layernorm.weight, self.post_attention_layernorm.eps,
)?;
let mlp_out = self.mlp.forward(&normed, backend)?;
backend.add(&x, &mlp_out)
```

This requires making `RMSNorm.weight` and `RMSNorm.eps` `pub(crate)`. Update the same pattern in `forward_batch`.

**Step 8: Run tests**

Run: `cargo test --workspace`
Expected: All pass

**Step 9: Commit**

```bash
git commit -m "perf: add fused residual+RMSNorm kernel"
```

---

## Task 3: Fused QKV Projection

Concatenate wq/wk/wv at load time, run a single GEMM, then split the result.

**Files:**
- Modify: `forge-core/src/backend.rs` — add `split_qkv` trait method
- Modify: `forge-kernels/src/memory.rs` — add `split_qkv_f32/f16` CUDA kernels
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — CUDA `split_qkv` impl
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs` — CPU `split_qkv` impl
- Modify: `forge-models/forge-model-llama/src/layers.rs:56-83` — change `LlamaAttention` to hold `wqkv`
- Modify: `forge-models/forge-model-llama/src/loader.rs:67-72` — concatenate at load time
- Test: `forge-backend/forge-backend-cpu/tests/test_ops.rs`
- Modify: `forge-models/forge-model-llama/tests/test_batch_forward.rs` — update test model construction

**Step 1: Add `split_qkv` kernel**

In `forge-kernels/src/memory.rs` — a lightweight kernel that extracts contiguous column ranges from a row-major matrix:

```cuda
extern "C" __global__ void split_qkv_f32(
    float* q_out, float* k_out, float* v_out,
    const float* qkv, unsigned int rows,
    unsigned int q_cols, unsigned int kv_cols
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;
    if (row >= rows) return;
    unsigned int total_cols = q_cols + kv_cols + kv_cols;
    const float* src = qkv + row * total_cols;
    // Q columns
    for (unsigned int c = col; c < q_cols; c += blockDim.x)
        q_out[row * q_cols + c] = src[c];
    // K columns
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        k_out[row * kv_cols + c] = src[q_cols + c];
    // V columns
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        v_out[row * kv_cols + c] = src[q_cols + kv_cols + c];
}
```

**Step 2: Add Backend trait method**

```rust
/// Split a concatenated QKV tensor [rows, q_size + kv_size + kv_size] into
/// (Q [rows, q_size], K [rows, kv_size], V [rows, kv_size]).
fn split_qkv(
    &self,
    qkv: &Self::Tensor,
    q_size: usize,
    kv_size: usize,
) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor)> {
    let rows = qkv.shape()[0];
    let q = self.slice_rows(&self.reshape(qkv, &[rows * (q_size + 2 * kv_size) / (q_size + 2 * kv_size), q_size + 2 * kv_size])?, 0, rows)?;
    // Default: use slice_rows for each portion (less efficient but correct)
    // Override with CUDA kernel for single-launch split
    todo!("default split_qkv impl")
}
```

Actually, a cleaner default impl using `copy_to_host_f32`:

```rust
fn split_qkv(
    &self,
    qkv: &Self::Tensor,
    q_size: usize,
    kv_size: usize,
) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor)> {
    let rows = qkv.shape()[0];
    let total_cols = q_size + 2 * kv_size;
    let data = self.copy_to_host_f32(qkv)?;
    let mut q_data = Vec::with_capacity(rows * q_size);
    let mut k_data = Vec::with_capacity(rows * kv_size);
    let mut v_data = Vec::with_capacity(rows * kv_size);
    for r in 0..rows {
        let row = &data[r * total_cols..(r + 1) * total_cols];
        q_data.extend_from_slice(&row[..q_size]);
        k_data.extend_from_slice(&row[q_size..q_size + kv_size]);
        v_data.extend_from_slice(&row[q_size + kv_size..]);
    }
    Ok((
        self.copy_from_host_f32(&q_data, &[rows, q_size])?,
        self.copy_from_host_f32(&k_data, &[rows, kv_size])?,
        self.copy_from_host_f32(&v_data, &[rows, kv_size])?,
    ))
}
```

**Step 3: CUDA and CPU impls**

CUDA: Use the `split_qkv_f32/f16` kernel.
CPU: Direct slice-based extraction (no host round-trip needed):

```rust
fn split_qkv(&self, qkv: &CpuTensor, q_size: usize, kv_size: usize) -> Result<(CpuTensor, CpuTensor, CpuTensor)> {
    let rows = qkv.shape()[0];
    let total_cols = q_size + 2 * kv_size;
    let data = qkv.data();
    let mut q = Vec::with_capacity(rows * q_size);
    let mut k = Vec::with_capacity(rows * kv_size);
    let mut v = Vec::with_capacity(rows * kv_size);
    for r in 0..rows {
        let row = &data[r * total_cols..(r + 1) * total_cols];
        q.extend_from_slice(&row[..q_size]);
        k.extend_from_slice(&row[q_size..q_size + kv_size]);
        v.extend_from_slice(&row[q_size + kv_size..]);
    }
    Ok((
        CpuTensor::new(q, vec![rows, q_size]),
        CpuTensor::new(k, vec![rows, kv_size]),
        CpuTensor::new(v, vec![rows, kv_size]),
    ))
}
```

**Step 4: Write test**

```rust
#[test]
fn test_split_qkv() {
    let backend = CpuBackend::new();
    // 2 rows, q_size=3, kv_size=2 → total_cols=7
    let qkv_data: Vec<f32> = (1..=14).map(|i| i as f32).collect();
    let qkv = backend.copy_from_host_f32(&qkv_data, &[2, 7]).unwrap();
    let (q, k, v) = backend.split_qkv(&qkv, 3, 2).unwrap();
    assert_eq!(q.shape(), &[2, 3]);
    assert_eq!(k.shape(), &[2, 2]);
    assert_eq!(v.shape(), &[2, 2]);
    assert_eq!(q.data(), &[1.0, 2.0, 3.0, 8.0, 9.0, 10.0]);
    assert_eq!(k.data(), &[4.0, 5.0, 11.0, 12.0]);
    assert_eq!(v.data(), &[6.0, 7.0, 13.0, 14.0]);
}
```

**Step 5: Run tests**

Run: `cargo test --workspace`

**Step 6: Update LlamaAttention struct**

In `forge-models/forge-model-llama/src/layers.rs`, change `LlamaAttention` to hold `wqkv` instead of `wq, wk, wv`:

```rust
pub struct LlamaAttention<B: Backend> {
    wqkv: B::Tensor,    // [hidden_size, q_proj_size + 2 * kv_proj_size]
    wo: B::Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_proj_size: usize,
    kv_proj_size: usize,
}
```

Update `new()` to accept `wqkv` (concatenated at load time).

Update `forward()` and `forward_batch()`:
```rust
// Before: 3 GEMM calls
// let q = backend.matmul(x, &self.wq)?;
// let k = backend.matmul(x, &self.wk)?;
// let v = backend.matmul(x, &self.wv)?;

// After: 1 GEMM + 1 split
let qkv = backend.matmul(x, &self.wqkv)?;
let (q, k, v) = backend.split_qkv(&qkv, self.q_proj_size, self.kv_proj_size)?;
```

**Step 7: Update loader**

In `forge-models/forge-model-llama/src/loader.rs:67-72`, concatenate weights at load time:

```rust
let wq = load_linear(loader, &format!("{prefix}.self_attn.q_proj.weight"), backend)?;
let wk = load_linear(loader, &format!("{prefix}.self_attn.k_proj.weight"), backend)?;
let wv = load_linear(loader, &format!("{prefix}.self_attn.v_proj.weight"), backend)?;
// Concatenate: wqkv = cat([wq, wk, wv], dim=1)
// wq: [hidden, q_proj], wk: [hidden, kv_proj], wv: [hidden, kv_proj]
// After transpose to column-first, we cat along columns (dim=1)
let wqkv = concat_weights(backend, &wq, &wk, &wv)?;
```

We need a `concat_weights` helper that transposes then cats along dim=1. Since our `cat` only supports dim=0, we'll transpose, cat dim=0, transpose back — or simpler: download to host, interleave columns, upload. Or: just use the fact that weights are already `[hidden_size, proj_size]` after transpose. We can:

```rust
fn concat_weights<B: Backend>(backend: &B, wq: &B::Tensor, wk: &B::Tensor, wv: &B::Tensor) -> Result<B::Tensor> {
    // wq: [hidden, q_proj], wk: [hidden, kv_proj], wv: [hidden, kv_proj]
    // We want wqkv: [hidden, q_proj + 2*kv_proj]
    // Since cat only works on dim=0, transpose → cat dim=0 → transpose back
    let wq_t = backend.transpose(wq, 0, 1)?;  // [q_proj, hidden]
    let wk_t = backend.transpose(wk, 0, 1)?;  // [kv_proj, hidden]
    let wv_t = backend.transpose(wv, 0, 1)?;  // [kv_proj, hidden]
    let cat_t = backend.cat(&[&wq_t, &wk_t, &wv_t], 0)?;  // [q+2*kv, hidden]
    backend.transpose(&cat_t, 0, 1)  // [hidden, q+2*kv]
}
```

**Step 8: Update test model construction**

In `forge-models/forge-model-llama/tests/test_batch_forward.rs`, update the model construction to create `wqkv` instead of separate `wq, wk, wv`.

**Step 9: Run tests**

Run: `cargo test --workspace`
Expected: All pass

**Step 10: Commit**

```bash
git commit -m "perf: fuse QKV into single GEMM with split kernel"
```

---

## Task 4: Batched Decode Attention Kernel

Replace the per-sequence attention loop with a single CUDA kernel that processes all decode sequences in one launch.

**Files:**
- Create: `forge-kernels/src/decode_attention.rs` — new kernel module
- Modify: `forge-kernels/src/lib.rs` — add `pub mod decode_attention`
- Modify: `forge-core/src/backend.rs` — add `batched_decode_attention` trait method
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — add CUDA impl with pointer table
- Modify: `forge-models/forge-model-llama/src/layers.rs:152-209` — replace per-seq loop in `forward_batch`
- Test: `forge-models/forge-model-llama/tests/test_batch_forward.rs` — update existing batch test

**Step 1: Write the decode attention kernel**

Create `forge-kernels/src/decode_attention.rs`:

```cuda
// Grid: (num_seqs, num_heads, 1)
// Block: (THREADS_PER_BLOCK, 1, 1)  — e.g. 128
// Shared mem: THREADS_PER_BLOCK * sizeof(float) * 3 (for max, sum, partial_v)
//
// Each thread block handles one (seq_idx, head_idx) pair.
// Streams through KV cache computing online softmax.
extern "C" __global__ void batched_decode_attention_f32(
    float* out,              // [num_seqs, num_heads * head_dim]
    const float* q,          // [num_seqs, num_heads * head_dim]
    const float* const* k_ptrs,  // [num_seqs] device pointers to K caches
    const float* const* v_ptrs,  // [num_seqs] device pointers to V caches
    const int* kv_lens,      // [num_seqs]
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int kv_len = kv_lens[seq_idx];
    int kv_head = head_idx / (num_heads / num_kv_heads);

    // Load query vector for this (seq, head) into registers
    const float* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const float* k_cache = k_ptrs[seq_idx];  // [kv_len, num_kv_heads * head_dim]
    const float* v_cache = v_ptrs[seq_idx];  // [kv_len, num_kv_heads * head_dim]

    extern __shared__ float smem[];
    float* s_max = smem;                                    // [blockDim.x]
    float* s_sum = smem + blockDim.x;                       // [blockDim.x]
    float* s_out = smem + 2 * blockDim.x;                   // [head_dim] (accumulator)

    // Initialize output accumulator in shared memory
    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;

    float thread_max = -1e30f;
    float thread_sum = 0.0f;

    // Stream through KV cache positions
    for (int t = tid; t < kv_len; t += blockDim.x) {
        // Compute q · k[t] for this head
        const float* k_t = k_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += q_ptr[d] * k_t[d];
        score *= scale;

        // Online softmax update
        float old_max = thread_max;
        if (score > thread_max) thread_max = score;
        float exp_old = expf(old_max - thread_max);
        float exp_score = expf(score - thread_max);
        thread_sum = thread_sum * exp_old + exp_score;

        // Accumulate weighted V
        const float* v_t = v_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&s_out[d], exp_score * v_t[d]);
        // NOTE: This naive approach with atomicAdd on shared mem is a starting point.
        // Production version would use warp-level reduction.
    }

    // Reduce max and sum across threads (shared memory reduction)
    s_max[tid] = thread_max;
    s_sum[tid] = thread_sum;
    __syncthreads();

    // ... (standard parallel reduction for max, then rescale sums, then reduce sums)
    // Final normalization: s_out[d] /= total_sum

    // Write to output
    float* out_ptr = out + seq_idx * num_heads * head_dim + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x)
        out_ptr[d] = s_out[d] / s_sum[0];
}
```

Note: The actual kernel will need careful implementation of the online softmax reduction across threads. The above is the algorithmic sketch — the implementation step will flesh out the warp-level reduction properly.

**Step 2: Add module to forge-kernels**

In `forge-kernels/src/lib.rs`, add: `pub mod decode_attention;`

**Step 3: Add Backend trait method**

```rust
/// Batched decode attention: N sequences × 1 query token each.
/// Fuses Q@K^T → softmax → @V for all sequences in a single kernel.
fn batched_decode_attention(
    &self,
    q: &Self::Tensor,              // [num_seqs, num_heads * head_dim]
    kv_cache: &dyn KvCache<T = Self::Tensor>,
    seq_ids: &[u64],
    layer_idx: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Self::Tensor>;
```

Default impl: falls back to per-sequence loop (existing `compute_attention` call per sequence).

**Step 4: CUDA impl — pointer table + kernel launch**

```rust
fn batched_decode_attention(&self, q: &CudaTensor, kv_cache: ...) -> Result<CudaTensor> {
    let num_seqs = seq_ids.len();
    // 1. Build pointer table: for each seq, get raw device ptrs from kv_cache.get_kv()
    let mut k_ptrs_host = Vec::with_capacity(num_seqs);
    let mut v_ptrs_host = Vec::with_capacity(num_seqs);
    let mut kv_lens_host = Vec::with_capacity(num_seqs);
    for &sid in seq_ids {
        let (k, v) = kv_cache.get_kv(sid, layer_idx)?;
        k_ptrs_host.push(k.device_ptr());  // raw *const f32
        v_ptrs_host.push(v.device_ptr());
        kv_lens_host.push(k.shape()[0] as i32);
    }
    // 2. Upload pointer arrays + kv_lens to GPU
    // 3. Launch kernel: grid=(num_seqs, num_heads), block=(128)
    //    shared_mem = 128 * 4 * 2 + head_dim * 4
    // 4. Return output tensor [num_seqs, num_heads * head_dim]
}
```

Note: `device_ptr()` needs to be added to `CudaTensor` if not already available.

**Step 5: Write test**

Update existing `test_batch_decode_matches_sequential` to verify the new kernel produces matching output.

**Step 6: Update forward_batch**

In `forge-models/forge-model-llama/src/layers.rs`, replace lines 182-208 (the per-sequence loop) with:

```rust
// Append to cache (still per-seq, 1 token each)
for i in 0..n {
    let k_row = backend.slice_rows(&k, i, 1)?;
    let v_row = backend.slice_rows(&v, i, 1)?;
    kv_cache.append(seq_ids[i], layer_idx, &k_row, &v_row)?;
}

// Single batched attention kernel
let attn_out = backend.batched_decode_attention(
    &q, &*kv_cache, seq_ids, layer_idx,
    self.num_heads, self.num_kv_heads, self.head_dim,
    1.0 / (self.head_dim as f32).sqrt(),
)?;
```

**Step 7: Run tests**

Run: `cargo test --workspace`

**Step 8: Commit**

```bash
git commit -m "perf: batched decode attention kernel (single launch)"
```

---

## Task 5: FlashAttention v2 FFI Crate

Create `forge-flash` crate wrapping FlashAttention v2 C++ library for prefill attention.

**Files:**
- Create: `forge-flash/Cargo.toml`
- Create: `forge-flash/build.rs`
- Create: `forge-flash/src/lib.rs`
- Create: `forge-flash/csrc/flash_api.h`
- Create: `forge-flash/csrc/flash_api.cpp`
- Vendor: `forge-flash/csrc/flash_attn/` — FlashAttention v2 source
- Modify: `Cargo.toml` (workspace) — add `forge-flash` member
- Modify: `forge-backend/forge-backend-cuda/Cargo.toml` — add optional `forge-flash` dep
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — dispatch to FA2 for prefill
- Modify: `forge-models/forge-model-llama/src/layers.rs` — refactor `compute_attention` to use new Backend method

**Step 1: Create forge-flash crate skeleton**

```toml
# forge-flash/Cargo.toml
[package]
name = "forge-flash"
version = "0.1.0"
edition = "2021"

[build-dependencies]
cc = "1"
```

**Step 2: Write C wrapper header and implementation**

`forge-flash/csrc/flash_api.h`:
```c
#pragma once
#include <stdint.h>

typedef struct {
    void* q;
    void* k;
    void* v;
    void* out;
    int batch_size;
    int seqlen_q;
    int seqlen_k;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    float softmax_scale;
    int is_causal;
    int dtype;  // 0=F16, 1=BF16
    void* stream;
} FlashAttnParams;

int flash_attn_fwd(FlashAttnParams* params);

typedef struct {
    void* q;
    void* k;
    void* v;
    void* out;
    int* cu_seqlens_q;
    int* cu_seqlens_k;
    int max_seqlen_q;
    int max_seqlen_k;
    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    float softmax_scale;
    int is_causal;
    int dtype;
    void* stream;
} FlashAttnVarlenParams;

int flash_attn_varlen_fwd(FlashAttnVarlenParams* params);
```

**Step 3: Write build.rs**

```rust
fn main() {
    let cuda_home = std::env::var("CUDA_HOME")
        .unwrap_or_else(|_| "/usr/local/cuda".into());

    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .file("csrc/flash_api.cpp")
        .include(format!("{}/include", cuda_home))
        .include("csrc/cutlass/include")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .compile("forge_flash");

    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    println!("cargo:rustc-link-lib=dylib=cudart");
}
```

**Step 4: Write Rust safe wrapper**

```rust
// forge-flash/src/lib.rs
#[repr(C)]
pub struct FlashAttnParams { /* mirrors C struct */ }

extern "C" {
    fn flash_attn_fwd(params: *mut FlashAttnParams) -> i32;
    fn flash_attn_varlen_fwd(params: *mut FlashAttnVarlenParams) -> i32;
}

pub fn flash_attention_forward(params: &mut FlashAttnParams) -> Result<(), String> {
    let ret = unsafe { flash_attn_fwd(params) };
    if ret != 0 { Err(format!("flash_attn_fwd failed: {}", ret)) }
    else { Ok(()) }
}
```

**Step 5: Feature-gate in cuda backend**

```toml
# forge-backend/forge-backend-cuda/Cargo.toml
[features]
default = []
flash-attn = ["forge-flash"]

[dependencies.forge-flash]
path = "../forge-flash"
optional = true
```

**Step 6: Add runtime dispatch**

In `forge-backend/forge-backend-cuda/src/backend.rs`, add a new `compute_attention` method that dispatches:

```rust
fn compute_attention_dispatch(
    &self, q: &CudaTensor, k: &CudaTensor, v: &CudaTensor,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    scale: f32, is_causal: bool,
) -> Result<CudaTensor> {
    #[cfg(feature = "flash-attn")]
    {
        // Cast to F16 if needed, call FA2
        return self.flash_attention_forward(q, k, v, ...);
    }
    // Fallback: naive per-head attention
    self.naive_attention(q, k, v, ...)
}
```

**Step 7: Vendor FlashAttention sources**

```bash
git submodule add https://github.com/Dao-AILab/flash-attention.git forge-flash/csrc/flash_attn
```

Or vendor a specific release tag.

**Step 8: Test**

Build with feature flag:
```bash
cargo build --features flash-attn
cargo test --workspace --features flash-attn
```

Verify that existing batch forward tests still pass (they exercise the attention path).

**Step 9: Commit**

```bash
git commit -m "feat: add FlashAttention v2 FFI crate (forge-flash)"
```

---

## Task 6: GPU-Side Sampling

Move temperature + top-k + softmax + sampling to GPU, avoiding full logit D→H transfer.

**Files:**
- Create: `forge-kernels/src/sampling.rs` — sampling CUDA kernels
- Modify: `forge-kernels/src/lib.rs` — add `pub mod sampling`
- Modify: `forge-core/src/backend.rs` — add `gpu_sample` and `gpu_top_k_candidates` trait methods
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — CUDA impls
- Modify: `forge-runtime/src/engine.rs:309-356,363-420` — use GPU sampling for unconstrained
- Test: `forge-backend/forge-backend-cuda/tests/test_ops.rs`

**Step 1: Write GPU sampling kernels**

Create `forge-kernels/src/sampling.rs`:

```cuda
// argmax: one block per row, shared-memory reduction
extern "C" __global__ void argmax_f32(
    unsigned int* out, const float* logits, unsigned int vocab_size
) { /* shared-mem reduction for max index per row */ }

// fused_temperature_softmax: applies temperature + softmax in one pass
extern "C" __global__ void fused_temperature_softmax_f32(
    float* probs, const float* logits, float temperature,
    unsigned int rows, unsigned int cols
) { /* find max, subtract+exp+div by temp, normalize */ }

// top_k_mask: zeros out all but top-k probabilities
extern "C" __global__ void top_k_mask_f32(
    float* probs, unsigned int vocab_size, unsigned int k
) { /* histogram-based k-th value finder + threshold application */ }

// multinomial_sample: samples from probability distribution using curand
extern "C" __global__ void multinomial_sample_f32(
    unsigned int* out, const float* probs,
    unsigned int vocab_size, unsigned long long seed, unsigned long long offset
) { /* per-row: generate random float, walk cumsum */ }
```

**Step 2: Add Backend trait methods**

```rust
/// GPU-side greedy sampling (argmax). Returns token IDs on host.
fn gpu_argmax(&self, logits: &Self::Tensor) -> Result<Vec<u32>> {
    // Default: copy to host, do argmax on CPU
    let data = self.copy_to_host_f32(logits)?;
    let vocab = logits.shape().last().copied().unwrap_or(0);
    let rows = data.len() / vocab;
    let mut ids = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &data[r * vocab..(r + 1) * vocab];
        let (idx, _) = row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        ids.push(idx as u32);
    }
    Ok(ids)
}

/// GPU-side multinomial sampling with temperature + top-k + top-p.
/// Returns token IDs on host.
fn gpu_sample(
    &self,
    logits: &Self::Tensor,
    temperature: f32,
    top_k: Option<usize>,
    top_p: f32,
    seed: Option<u64>,
    step: usize,
) -> Result<Vec<u32>> {
    // Default: copy to host, use CPU sampling
    // CUDA override: fused_temperature_softmax → top_k_mask → multinomial_sample
    todo!()
}

/// GPU-side top-k candidate extraction for constrained decoding.
/// Returns (token_ids, probabilities) on host for FSM filtering.
fn gpu_top_k_candidates(
    &self,
    logits: &Self::Tensor,
    temperature: f32,
    k: usize,
) -> Result<Vec<(Vec<u32>, Vec<f32>)>> {
    // Default: copy to host, sort, take top k
    todo!()
}
```

**Step 3: CUDA impls**

Wire up the kernels using the same pattern as existing ops (load functions in `CudaBackend::new`, launch with `LaunchConfig`).

**Step 4: Engine integration**

In `forge-runtime/src/engine.rs`, update `process_sequence` and `process_decode_batch`:

```rust
// For unconstrained sequences with temperature == 0:
//   let token_ids = self.backend.gpu_argmax(&output.logits)?;
// For unconstrained with temperature > 0:
//   let token_ids = self.backend.gpu_sample(&output.logits, temp, top_k, top_p, seed, step)?;
// For constrained:
//   let candidates = self.backend.gpu_top_k_candidates(&output.logits, temp, 50)?;
//   // CPU-side FSM masking + sample
```

**Step 5: Test**

Verify that GPU argmax matches CPU argmax for deterministic inputs. Verify that seeded GPU sampling produces reproducible results.

**Step 6: Commit**

```bash
git commit -m "perf: GPU-side sampling (argmax, top-k, multinomial)"
```

---

## Task 7: Integration Tests & Benchmarks

End-to-end validation that all performance changes produce correct output, plus benchmarks to measure gains.

**Files:**
- Modify: `forge-models/forge-model-llama/tests/test_batch_forward.rs` — verify batch output still matches sequential
- Create: `scripts/benchmark_decode.sh` — benchmark decode throughput before/after
- Modify: existing E2E tests to exercise batched + fused paths

**Step 1: Run full test suite**

```bash
cargo test --workspace
```

Verify all tests pass.

**Step 2: Run with FlashAttention (if available)**

```bash
cargo test --workspace --features flash-attn
```

**Step 3: Benchmark**

```bash
# Before changes (baseline — measure on master):
cargo bench --bench decode_throughput

# After changes:
cargo bench --bench decode_throughput
```

**Step 4: Commit final cleanup**

```bash
git commit -m "test: add Phase 2 integration tests and benchmarks"
```

---

## Summary: Execution Order

| Task | Depends On | Est. Kernel Launches Saved / Layer |
|------|-----------|-----------------------------------|
| 1. Fused SiLU-Mul | None | 1 |
| 2. Fused Residual+RMSNorm | None | 2 |
| 3. Fused QKV | None | 2 GEMM calls |
| 4. Batched Decode Attention | Tasks 1-3 (for full layer fusion) | N × num_heads × 7 → 1 |
| 5. FlashAttention FFI | None (parallel with 1-4) | Replaces per-head loop for prefill |
| 6. GPU Sampling | None (parallel with 1-5) | Eliminates D→H logit transfer |
| 7. Integration Tests | All above | — |

Tasks 1-3 can be done in any order. Task 4 builds on the refactored attention code. Tasks 5 and 6 are independent. Task 7 is final validation.
