# FlashAttention v2 FFI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate FlashAttention v2 via C++ FFI to replace naive per-head attention for prefill and long-context decode, achieving 2-4x speedup.

**Architecture:** New `forge-flash` crate vendors FA2 C++ sources + CUTLASS headers, compiled via `cc::Build` for SM80/SM90. Feature-gated behind `flash-attn` in `forge-backend-cuda`. A new `multi_head_attention` Backend trait method replaces the per-head loop in model code. Runtime dispatch: batched decode → existing kernel, everything else → FA2 (or naive fallback).

**Tech Stack:** Rust FFI (`extern "C"`), cc crate (build-time NVCC compilation), CUTLASS, cudarc 0.17

**Design doc:** `docs/plans/2026-02-23-flash-attention-v2-design.md`

---

## Task 1: Scaffold `forge-flash` Crate

**Files:**
- Create: `forge-flash/Cargo.toml`
- Create: `forge-flash/build.rs` (stub — compiles nothing yet)
- Create: `forge-flash/src/lib.rs` (stub — types + dummy functions)
- Modify: `Cargo.toml` (root workspace) — add member

**Step 1: Create `forge-flash/Cargo.toml`**

```toml
[package]
name = "forge-flash"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "FlashAttention v2 C++ FFI bindings for forge"

[build-dependencies]
cc = { version = "1", features = ["parallel"] }
```

**Step 2: Create stub `forge-flash/build.rs`**

```rust
/// Build script for forge-flash.
///
/// Compiles vendored FlashAttention v2 CUDA sources.
/// Requires CUDA toolkit with nvcc for SM80+SM90.
fn main() {
    // TODO: Task 3 will add cc::Build compilation here
    println!("cargo:rerun-if-changed=csrc/");
}
```

**Step 3: Create stub `forge-flash/src/lib.rs`**

```rust
//! FlashAttention v2 FFI bindings.
//!
//! Provides safe Rust wrappers around vendored FA2 C++ CUDA kernels.
//! Supports SM80 (Ampere) and SM90 (Hopper).

/// Data type for FlashAttention inputs.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum FlashDType {
    F16 = 0,
    BF16 = 1,
}
```

**Step 4: Add to workspace**

In root `Cargo.toml`, add `"forge-flash"` to `members` list (after `"forge-kernels"`).

Also add to `[workspace.dependencies]`:
```toml
forge-flash = { path = "forge-flash" }
```

**Step 5: Verify workspace compiles**

Run: `cargo check -p forge-flash`
Expected: SUCCESS (empty crate compiles)

**Step 6: Commit**

```bash
git add forge-flash/ Cargo.toml
git commit -m "chore: scaffold forge-flash crate"
```

---

## Task 2: Vendor FA2 Sources + CUTLASS Headers

**Files:**
- Create: `forge-flash/csrc/` directory tree with vendored sources

**Step 1: Clone FlashAttention v2 and CUTLASS**

```bash
cd /tmp
git clone --depth 1 --branch v2.7.4 https://github.com/Dao-AILab/flash-attention fa2-vendor
git clone --depth 1 --branch v3.6.0 https://github.com/NVIDIA/cutlass cutlass-vendor
```

Note: Use the latest stable tags. If these exact tags don't exist, use the latest available.

**Step 2: Copy FA2 forward-pass sources**

```bash
mkdir -p forge-flash/csrc

# Copy FA2 kernel sources (forward only — we don't need backward)
cp -r /tmp/fa2-vendor/csrc/flash_attn forge-flash/csrc/flash_attn

# Copy CUTLASS headers (only include/ directory needed)
mkdir -p forge-flash/csrc/cutlass
cp -r /tmp/cutlass-vendor/include forge-flash/csrc/cutlass/include
```

**Step 3: Create thin C wrapper**

Create `forge-flash/csrc/flash_api_forge.cu`:

```cuda
// Thin C wrapper for FlashAttention v2 forward pass.
// Exposes C-linkage entry points callable from Rust FFI.

#include "flash_attn/flash_api.h"

extern "C" {

void forge_flash_attn_fwd(
    void* q, void* k, void* v, void* out,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    float scale, bool is_causal, int dtype,
    void* stream
) {
    // Construct FA2 params and call mha_fwd
    // dtype: 0 = F16, 1 = BF16
    auto q_dtype = dtype == 0
        ? at::ScalarType::Half
        : at::ScalarType::BFloat16;

    // ... FA2 parameter setup and kernel launch
    // Implementation depends on exact FA2 API version
}

} // extern "C"
```

Note: The exact wrapper implementation depends on FA2's internal API. The FA2 repo's `csrc/flash_attn/flash_api.cpp` shows the pattern. We need to adapt it to work without PyTorch tensors — using raw device pointers + dimensions instead. This may require writing a custom entry point that constructs FA2's internal `Flash_fwd_params` struct directly.

**Key insight:** FA2 v2's internal API (`flash_fwd_kernel.h`) takes a `Flash_fwd_params` struct with raw pointers. We should bypass the PyTorch wrapper entirely and fill `Flash_fwd_params` directly. Reference: `csrc/flash_attn/src/flash.h` defines the struct.

**Step 4: Verify directory structure**

```
forge-flash/csrc/
├── flash_api_forge.cu     # Our thin C wrapper
├── flash_attn/            # Vendored FA2 sources
│   ├── flash_api.h
│   ├── src/
│   │   ├── flash.h
│   │   ├── flash_fwd_kernel.h
│   │   ├── flash_fwd_hdim64.cu
│   │   ├── flash_fwd_hdim128.cu
│   │   └── ...
│   └── ...
└── cutlass/
    └── include/
        └── cutlass/
            └── ...
```

**Step 5: Add `.gitignore` for vendored code tracking**

We DO want to track vendored code in git (self-contained builds), but add a note file:

Create `forge-flash/csrc/VENDORED.md`:
```markdown
# Vendored Sources

- FlashAttention v2.7.4: https://github.com/Dao-AILab/flash-attention
- CUTLASS v3.6.0: https://github.com/NVIDIA/cutlass

To update: re-run the vendoring steps from the implementation plan.
```

**Step 6: Commit**

```bash
git add forge-flash/csrc/
git commit -m "chore: vendor FlashAttention v2 + CUTLASS sources"
```

---

## Task 3: Write `build.rs` + FFI Bindings

**Files:**
- Modify: `forge-flash/build.rs` — full cc::Build compilation
- Modify: `forge-flash/src/lib.rs` — extern "C" bindings + safe wrapper

**Step 1: Write `forge-flash/build.rs`**

```rust
fn main() {
    println!("cargo:rerun-if-changed=csrc/");

    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Collect all FA2 forward kernel .cu files
    let cu_files: Vec<_> = std::fs::read_dir("csrc/flash_attn/src")
        .expect("csrc/flash_attn/src must exist")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "cu")
                && p.file_name()
                    .unwrap()
                    .to_str()
                    .is_some_and(|n| n.starts_with("flash_fwd"))
        })
        .collect();

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file("csrc/flash_api_forge.cu")
        .include("csrc/flash_attn/src")
        .include("csrc/flash_attn")
        .include("csrc/cutlass/include")
        .include(format!("{cuda_home}/include"))
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("-diag-suppress=177"); // suppress unused variable warnings in vendored code

    for f in &cu_files {
        build.file(f);
    }

    build.compile("flash_attn");

    println!("cargo:rustc-link-lib=static=flash_attn");
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
```

**Step 2: Write `forge-flash/src/lib.rs`**

```rust
//! FlashAttention v2 FFI bindings.
//!
//! Provides safe Rust wrappers around vendored FA2 C++ CUDA kernels.
//! Supports SM80 (Ampere) and SM90 (Hopper).

/// Data type for FlashAttention inputs.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum FlashDType {
    F16 = 0,
    BF16 = 1,
}

extern "C" {
    fn forge_flash_attn_fwd(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        scale: f32,
        is_causal: bool,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    );
}

/// Run FlashAttention v2 forward pass.
///
/// # Safety
/// - All device pointers must be valid CUDA allocations with correct sizes
/// - `stream` must be a valid `CUstream`
/// - Q shape: `[batch_size, seqlen_q, num_heads, head_dim]` (contiguous)
/// - K shape: `[batch_size, seqlen_k, num_heads_k, head_dim]` (contiguous)
/// - V shape: `[batch_size, seqlen_k, num_heads_k, head_dim]` (contiguous)
/// - out shape: `[batch_size, seqlen_q, num_heads, head_dim]` (pre-allocated)
/// - Data must be F16 or BF16 (not F32)
pub unsafe fn flash_fwd(
    q: u64,
    k: u64,
    v: u64,
    out: u64,
    batch_size: i32,
    seqlen_q: i32,
    seqlen_k: i32,
    num_heads: i32,
    num_heads_k: i32,
    head_dim: i32,
    scale: f32,
    is_causal: bool,
    dtype: FlashDType,
    stream: u64,
) {
    forge_flash_attn_fwd(
        q as *const core::ffi::c_void,
        k as *const core::ffi::c_void,
        v as *const core::ffi::c_void,
        out as *mut core::ffi::c_void,
        batch_size,
        seqlen_q,
        seqlen_k,
        num_heads,
        num_heads_k,
        head_dim,
        scale,
        is_causal,
        dtype as i32,
        stream as *mut core::ffi::c_void,
    );
}
```

**Step 3: Verify it compiles**

Run: `cargo build -p forge-flash`
Expected: SUCCESS (if CUDA toolkit is available; may fail without nvcc — that's expected on CI without GPU)

**Step 4: Commit**

```bash
git add forge-flash/build.rs forge-flash/src/lib.rs
git commit -m "feat: add FA2 build.rs + Rust FFI bindings"
```

---

## Task 4: Add `multi_head_attention` to Backend Trait

**Files:**
- Modify: `forge-core/src/backend.rs` — add trait method with default impl
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs` — (uses default impl, no changes needed)

**Step 1: Write the failing test**

Create test in `forge-backend/forge-backend-cpu/tests/test_ops.rs`:

```rust
#[test]
fn test_multi_head_attention_basic() {
    let backend = CpuBackend::new();
    // Q: [1, 2, 2, 4] (batch=1, seq_len=2, num_heads=2, head_dim=4)
    let q_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let q = backend.copy_from_host_f32(&q_data, &[1, 2, 2, 4]).unwrap();

    // K, V: [1, 3, 2, 4] (batch=1, kv_len=3, num_kv_heads=2, head_dim=4)
    let k_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.05).collect();
    let k = backend.copy_from_host_f32(&k_data, &[1, 3, 2, 4]).unwrap();
    let v_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.02 + 0.1).collect();
    let v = backend.copy_from_host_f32(&v_data, &[1, 3, 2, 4]).unwrap();

    let scale = 1.0 / (4.0_f32).sqrt();
    let result = backend
        .multi_head_attention(&q, &k, &v, 2, 2, 4, scale, true)
        .unwrap();

    // Output should be [1, 2, 2, 4] → flattened to [2, 8]
    assert_eq!(result.shape(), &[2, 8]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p forge-backend-cpu test_multi_head_attention_basic`
Expected: FAIL — `multi_head_attention` method does not exist

**Step 3: Add `multi_head_attention` to Backend trait**

In `forge-core/src/backend.rs`, add after `batched_decode_attention`:

```rust
    /// Multi-head scaled dot-product attention.
    ///
    /// Q: [1, seq_len, num_heads, head_dim]
    /// K: [1, kv_len, num_kv_heads, head_dim]
    /// V: [1, kv_len, num_kv_heads, head_dim]
    ///
    /// Returns: [seq_len, num_heads * head_dim]
    ///
    /// Default impl: per-head loop (extract_head → matmul → softmax → interleave).
    /// CUDA override: FlashAttention v2 when available.
    fn multi_head_attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<Self::Tensor> {
        let q_shape = q.shape();
        let seq_len = q_shape[1];
        let kv_len = k.shape()[1];
        let heads_per_group = num_heads / num_kv_heads;

        // Reshape to 3D for extract_head: strip batch dim (always 1)
        let q = self.reshape(q, &[seq_len, num_heads, head_dim])?;
        let k = self.reshape(k, &[kv_len, num_kv_heads, head_dim])?;
        let v = self.reshape(v, &[kv_len, num_kv_heads, head_dim])?;

        let mut head_outputs = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let kv_h = h / heads_per_group;

            let q_head = self.extract_head(&q, seq_len, num_heads, head_dim, h)?;
            let k_head = self.extract_head(&k, kv_len, num_kv_heads, head_dim, kv_h)?;
            let v_head = self.extract_head(&v, kv_len, num_kv_heads, head_dim, kv_h)?;

            let k_t = self.transpose(&k_head, 0, 1)?;
            let scores = self.matmul(&q_head, &k_t)?;
            let scores = self.mul_scalar(&scores, scale)?;

            let scores = if is_causal && seq_len > 1 {
                self.apply_causal_mask(&scores, seq_len, kv_len)?
            } else {
                scores
            };

            let attn = self.softmax(&scores, -1)?;
            head_outputs.push(self.matmul(&attn, &v_head)?);
        }

        let refs: Vec<&Self::Tensor> = head_outputs.iter().collect();
        self.interleave_heads(&refs, seq_len, head_dim)
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p forge-backend-cpu test_multi_head_attention_basic`
Expected: PASS

**Step 5: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests PASS (new method has default impl, nothing breaks)

**Step 6: Commit**

```bash
git add forge-core/src/backend.rs forge-backend/forge-backend-cpu/tests/test_ops.rs
git commit -m "feat: add multi_head_attention to Backend trait"
```

---

## Task 5: CUDA Backend — Override with FA2 Dispatch

**Files:**
- Modify: `forge-backend/forge-backend-cuda/src/flash_attention.rs` — FA2 dispatch
- Modify: `forge-backend/forge-backend-cuda/src/backend.rs` — override `multi_head_attention`
- Modify: `forge-backend/forge-backend-cuda/Cargo.toml` — add optional forge-flash dep

**Step 1: Add `forge-flash` as optional dependency**

In `forge-backend/forge-backend-cuda/Cargo.toml`:

```toml
[features]
default = []
flash-attn = ["dep:forge-flash"]

[dependencies]
forge-core.workspace = true
forge-kernels.workspace = true
cudarc = { version = "0.17", features = ["cublas", "nvrtc", "f16", "cuda-version-from-build-system"] }
half.workspace = true
forge-flash = { workspace = true, optional = true }
```

**Step 2: Update `flash_attention.rs` dispatch**

Replace the contents of `forge-backend/forge-backend-cuda/src/flash_attention.rs`:

```rust
//! Attention dispatch entry point.
//!
//! Routes to FlashAttention v2 (when `flash-attn` feature is enabled)
//! or falls back to naive per-head GPU attention.

use forge_core::{DType, Result};

use crate::attention::naive_attention_causal;
use crate::backend::CudaBackend;
use crate::tensor::CudaTensor;

/// Run multi-head scaled dot-product attention.
///
/// Q: [batch, seq_len, num_heads, head_dim]
/// K: [batch, kv_len, num_kv_heads, head_dim]
/// V: [batch, kv_len, num_kv_heads, head_dim]
///
/// Returns: [batch, seq_len, num_heads, head_dim]
pub fn attention_fwd(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
    is_causal: bool,
) -> Result<CudaTensor> {
    #[cfg(feature = "flash-attn")]
    {
        return flash_attn_dispatch(backend, q, k, v, scale, is_causal);
    }

    #[cfg(not(feature = "flash-attn"))]
    {
        naive_attention_causal(backend, q, k, v, scale, is_causal)
    }
}

#[cfg(feature = "flash-attn")]
fn flash_attn_dispatch(
    backend: &CudaBackend,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: f32,
    is_causal: bool,
) -> Result<CudaTensor> {
    use cudarc::driver::DevicePtr;
    use forge_core::{Backend, ForgeError, Tensor};

    let q_shape = q.shape();
    let batch_size = q_shape[0] as i32;
    let seqlen_q = q_shape[1] as i32;
    let num_heads = q_shape[2] as i32;
    let head_dim = q_shape[3] as i32;
    let seqlen_k = k.shape()[1] as i32;
    let num_heads_k = k.shape()[2] as i32;

    // FA2 requires F16 or BF16 — auto-cast from F32 if needed
    let original_dtype = q.dtype();
    let (q, k, v, fa_dtype) = match original_dtype {
        DType::F16 => (q.clone(), k.clone(), v.clone(), forge_flash::FlashDType::F16),
        DType::BF16 => (q.clone(), k.clone(), v.clone(), forge_flash::FlashDType::BF16),
        DType::F32 => {
            let q = backend.cast(q, DType::F16)?;
            let k = backend.cast(k, DType::F16)?;
            let v = backend.cast(v, DType::F16)?;
            (q, k, v, forge_flash::FlashDType::F16)
        }
    };

    // Allocate output tensor (same shape + dtype as Q after cast)
    let out = backend.allocate(q.shape(), q.dtype())?;

    // Get raw device pointers
    let (q_ptr, _q_guard) = match &q.data {
        crate::tensor::TensorData::F16(s) => s.device_ptr(&backend.stream),
        crate::tensor::TensorData::BF16(s) => s.device_ptr(&backend.stream),
        _ => return Err(ForgeError::InvalidArgument("FA2 requires F16/BF16".into())),
    };
    let (k_ptr, _k_guard) = match &k.data {
        crate::tensor::TensorData::F16(s) => s.device_ptr(&backend.stream),
        crate::tensor::TensorData::BF16(s) => s.device_ptr(&backend.stream),
        _ => return Err(ForgeError::InvalidArgument("FA2 requires F16/BF16".into())),
    };
    let (v_ptr, _v_guard) = match &v.data {
        crate::tensor::TensorData::F16(s) => s.device_ptr(&backend.stream),
        crate::tensor::TensorData::BF16(s) => s.device_ptr(&backend.stream),
        _ => return Err(ForgeError::InvalidArgument("FA2 requires F16/BF16".into())),
    };
    let (out_ptr, _out_guard) = match &out.data {
        crate::tensor::TensorData::F16(s) => s.device_ptr(&backend.stream),
        crate::tensor::TensorData::BF16(s) => s.device_ptr(&backend.stream),
        _ => return Err(ForgeError::InvalidArgument("FA2 requires F16/BF16".into())),
    };

    // Get raw stream pointer
    let stream_ptr: u64 = 0; // default stream; TODO: extract from backend.stream

    unsafe {
        forge_flash::flash_fwd(
            q_ptr, k_ptr, v_ptr, out_ptr,
            batch_size, seqlen_q, seqlen_k,
            num_heads, num_heads_k, head_dim,
            scale, is_causal, fa_dtype, stream_ptr,
        );
    }

    // Cast back to original dtype if needed
    if original_dtype == DType::F32 {
        backend.cast(&out, DType::F32)
    } else {
        Ok(out)
    }
}
```

**Step 3: Override `multi_head_attention` in CudaBackend**

In `forge-backend/forge-backend-cuda/src/backend.rs`, add the override in the `impl Backend for CudaBackend` block:

```rust
fn multi_head_attention(
    &self,
    q: &Self::Tensor,
    k: &Self::Tensor,
    v: &Self::Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    is_causal: bool,
) -> Result<Self::Tensor> {
    let q_shape = q.shape();
    let seq_len = q_shape[1];

    // Route through attention_fwd (FA2 when available, naive otherwise)
    let result_4d = crate::flash_attention::attention_fwd(self, q, k, v, scale, is_causal)?;

    // Flatten from [1, seq_len, num_heads, head_dim] → [seq_len, num_heads * head_dim]
    self.reshape(&result_4d, &[seq_len, num_heads * head_dim])
}
```

**Step 4: Verify existing attention tests still pass**

Run: `cargo test -p forge-backend-cuda` (if on GPU machine)
Expected: All existing `test_attention_fwd_*` tests PASS

**Step 5: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add forge-backend/forge-backend-cuda/Cargo.toml \
        forge-backend/forge-backend-cuda/src/flash_attention.rs \
        forge-backend/forge-backend-cuda/src/backend.rs
git commit -m "feat: CUDA multi_head_attention with FA2 dispatch"
```

---

## Task 6: Refactor Model Layers to Use `multi_head_attention`

**Files:**
- Modify: `forge-models/forge-model-llama/src/layers.rs` — replace `compute_attention` with `multi_head_attention`

**Step 1: Run existing tests to establish baseline**

Run: `cargo test -p forge-model-llama`
Expected: All tests PASS

**Step 2: Replace `compute_attention` call in `LlamaAttention::forward`**

In `forge-models/forge-model-llama/src/layers.rs`, the `forward` method currently calls `self.compute_attention(...)` at line ~131. Replace that call:

**Before (line ~131):**
```rust
let attn_out = self.compute_attention(&q, &k_4d, &v_4d, seq_len, kv_len, backend)?;
```

**After:**
```rust
let scale = 1.0 / (self.head_dim as f32).sqrt();
let attn_out = backend.multi_head_attention(
    &q, &k_4d, &v_4d,
    self.num_heads, self.num_kv_heads, self.head_dim,
    scale, true, // causal for prefill
)?;
```

**Step 3: Remove `compute_attention` method**

Delete the entire `compute_attention` private method (lines ~215-263 in `layers.rs`). It is no longer called — the logic now lives in the Backend trait's default impl.

**Step 4: Run tests to verify refactor**

Run: `cargo test -p forge-model-llama`
Expected: All tests PASS (behavior identical — default impl has same per-head loop)

Run: `cargo test --workspace`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add forge-models/forge-model-llama/src/layers.rs
git commit -m "refactor: use multi_head_attention backend method in LlamaAttention"
```

---

## Task 7: Correctness Tests

**Files:**
- Modify: `forge-backend/forge-backend-cuda/tests/test_kernels.rs` — update existing tests
- Modify: `forge-backend/forge-backend-cpu/tests/test_ops.rs` — add GQA test

**Step 1: Add GQA test for multi_head_attention (CPU)**

In `forge-backend/forge-backend-cpu/tests/test_ops.rs`:

```rust
#[test]
fn test_multi_head_attention_gqa() {
    let backend = CpuBackend::new();
    // GQA: num_heads=4, num_kv_heads=2 (heads_per_group=2)
    // Q: [1, 1, 4, 4], K/V: [1, 3, 2, 4]
    let q_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let q = backend.copy_from_host_f32(&q_data, &[1, 1, 4, 4]).unwrap();

    let k_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.05).collect();
    let k = backend.copy_from_host_f32(&k_data, &[1, 3, 2, 4]).unwrap();
    let v_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.02 + 0.1).collect();
    let v = backend.copy_from_host_f32(&v_data, &[1, 3, 2, 4]).unwrap();

    let scale = 1.0 / (4.0_f32).sqrt();
    let result = backend
        .multi_head_attention(&q, &k, &v, 4, 2, 4, scale, false)
        .unwrap();

    // Output: [1, 4*4] = [1, 16]
    assert_eq!(result.shape(), &[1, 16]);
}

#[test]
fn test_multi_head_attention_causal_mask() {
    let backend = CpuBackend::new();
    // Q: [1, 3, 2, 4] (seq_len=3), K/V: [1, 3, 2, 4]
    let q_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let q = backend.copy_from_host_f32(&q_data, &[1, 3, 2, 4]).unwrap();
    let k = backend.copy_from_host_f32(&q_data, &[1, 3, 2, 4]).unwrap();
    let v_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.02).collect();
    let v = backend.copy_from_host_f32(&v_data, &[1, 3, 2, 4]).unwrap();

    let scale = 1.0 / (4.0_f32).sqrt();

    // Causal: first token can only attend to itself
    let causal = backend
        .multi_head_attention(&q, &k, &v, 2, 2, 4, scale, true)
        .unwrap();
    // Non-causal: first token attends to all
    let non_causal = backend
        .multi_head_attention(&q, &k, &v, 2, 2, 4, scale, false)
        .unwrap();

    let c = backend.copy_to_host_f32(&causal).unwrap();
    let nc = backend.copy_to_host_f32(&non_causal).unwrap();

    // Results should differ (causal mask changes first token's attention)
    assert_ne!(c, nc, "causal and non-causal should produce different results");
}
```

**Step 2: Update existing CUDA `attention_fwd` tests**

In `forge-backend/forge-backend-cuda/tests/test_kernels.rs`, update `test_attention_fwd_f32_uses_naive` and `test_attention_fwd_f16_fallback` to also test through the `multi_head_attention` Backend method:

```rust
#[test]
fn test_multi_head_attention_matches_naive() {
    let backend = CudaBackend::new(0).unwrap();
    // Same test data as test_attention_fwd_f32_uses_naive
    // but call backend.multi_head_attention(...) instead of attention_fwd(...)
    // and verify output matches
}
```

**Step 3: Run all tests**

Run: `cargo test --workspace`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add forge-backend/forge-backend-cpu/tests/test_ops.rs \
        forge-backend/forge-backend-cuda/tests/test_kernels.rs
git commit -m "test: add multi_head_attention correctness tests"
```

---

## Task 8: FA2 Feature-Gated Tests

**Files:**
- Modify: `forge-backend/forge-backend-cuda/tests/test_kernels.rs`

These tests only compile and run when `--features flash-attn` is passed.

**Step 1: Add FA2-vs-naive correctness test**

```rust
#[test]
#[cfg(feature = "flash-attn")]
fn test_flash_attn_matches_naive_f16() {
    use forge_core::{Backend, DType, Tensor};
    let backend = CudaBackend::new(0).unwrap();

    let seq_len = 32;
    let kv_len = 32;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Random-ish F16 data
    let q_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|i| ((i as f32 * 0.017) % 1.0) - 0.5)
        .collect();
    let k_data: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| ((i as f32 * 0.013) % 1.0) - 0.5)
        .collect();
    let v_data: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| ((i as f32 * 0.011) % 1.0) - 0.5)
        .collect();

    let q_f32 = backend.copy_from_host_f32(&q_data, &[1, seq_len, num_heads, head_dim]).unwrap();
    let k_f32 = backend.copy_from_host_f32(&k_data, &[1, kv_len, num_kv_heads, head_dim]).unwrap();
    let v_f32 = backend.copy_from_host_f32(&v_data, &[1, kv_len, num_kv_heads, head_dim]).unwrap();

    // Naive attention (always available)
    let naive_out = crate::attention::naive_attention_causal(
        &backend, &q_f32, &k_f32, &v_f32, scale, true,
    ).unwrap();

    // FA2 via multi_head_attention (feature-gated dispatch)
    let fa2_out = backend.multi_head_attention(
        &q_f32, &k_f32, &v_f32,
        num_heads, num_kv_heads, head_dim, scale, true,
    ).unwrap();

    let naive_host = backend.copy_to_host_f32(&naive_out).unwrap();
    // FA2 output is [seq_len, num_heads * head_dim], naive is [1, seq_len, num_heads, head_dim]
    let naive_flat = backend.reshape(&naive_out, &[seq_len, num_heads * head_dim]).unwrap();
    let naive_host = backend.copy_to_host_f32(&naive_flat).unwrap();
    let fa2_host = backend.copy_to_host_f32(&fa2_out).unwrap();

    for i in 0..fa2_host.len() {
        assert!(
            (fa2_host[i] - naive_host[i]).abs() < 1e-2,
            "FA2 vs naive mismatch at {i}: fa2={} naive={} diff={}",
            fa2_host[i], naive_host[i], (fa2_host[i] - naive_host[i]).abs()
        );
    }
}
```

**Step 2: Verify test is skipped without feature**

Run: `cargo test -p forge-backend-cuda test_flash_attn`
Expected: 0 tests run (cfg-gated out)

**Step 3: Verify test runs with feature (on GPU machine)**

Run: `cargo test -p forge-backend-cuda --features flash-attn test_flash_attn`
Expected: PASS (if FA2 builds and GPU supports SM80+)

**Step 4: Commit**

```bash
git add forge-backend/forge-backend-cuda/tests/test_kernels.rs
git commit -m "test: add FA2-vs-naive correctness test (feature-gated)"
```

---

## Task 9: End-to-End Verification

**Step 1: Verify default build (no flash-attn feature)**

```bash
cargo test --workspace
cargo build --release
```

Expected: All pass, no regressions. FA2 code is completely compiled out.

**Step 2: Verify flash-attn build (on GPU machine)**

```bash
cargo build --release --features flash-attn
cargo test --workspace --features flash-attn
```

Expected: FA2 compiles (~2-5 min first time), all tests pass including FA2 correctness tests.

**Step 3: Verify model produces valid output with FA2**

Run existing `test_batch_forward.rs` tests with FA2 feature:

```bash
cargo test -p forge-model-llama --features forge-backend-cuda/flash-attn
```

Expected: All model tests PASS with FA2 dispatch active.

**Step 4: Final commit (if any fixups needed)**

```bash
git commit -m "fix: address FA2 integration issues from e2e testing"
```

---

## Execution Order

```
Task 1 (scaffold crate)       ← no deps
  ↓
Task 2 (vendor sources)       ← needs Task 1
  ↓
Task 3 (build.rs + FFI)       ← needs Task 2
  ↓
Task 4 (Backend trait)         ← independent of Tasks 1-3
  ↓
Task 5 (CUDA override)        ← needs Tasks 3 + 4
  ↓
Task 6 (model refactor)       ← needs Task 4
  ↓
Task 7 (correctness tests)    ← needs Tasks 4 + 6
  ↓
Task 8 (FA2 feature tests)    ← needs Task 5
  ↓
Task 9 (e2e verification)     ← needs all above
```

Tasks 1-3 (FFI crate) and Task 4 (Backend trait) can proceed in parallel.

---

## Verification Checklist

```bash
cargo test --workspace                                    # default: all pass
cargo build --release                                     # default: builds ok
cargo build --release --features flash-attn               # FA2: builds ok (GPU)
cargo test --workspace --features flash-attn              # FA2: all pass (GPU)
```
