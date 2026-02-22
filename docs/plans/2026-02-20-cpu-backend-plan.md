# CPU Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `forge-backend-cpu` crate implementing the `Backend` trait with OpenBLAS matmul, so Forge can load and run TinyLlama without a GPU.

**Architecture:** New crate `forge-backend/forge-backend-cpu/` with `CpuTensor` (Arc<Vec<f32>> + shape) and `CpuBackend` (stateless struct). Uses `cblas::sgemm` for matmul, hand-written loops for everything else. Server gets `--backend cpu|cuda` CLI flag.

**Tech Stack:** Rust, cblas 0.4, openblas-src 0.10 (system), half 2

---

### Task 1: Create crate skeleton + CpuTensor

**Files:**
- Create: `forge-backend/forge-backend-cpu/Cargo.toml`
- Create: `forge-backend/forge-backend-cpu/src/lib.rs`
- Create: `forge-backend/forge-backend-cpu/src/tensor.rs`
- Modify: `Cargo.toml` (workspace root)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "forge-backend-cpu"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
forge-core.workspace = true
half.workspace = true
cblas = "0.4"
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

**Step 2: Create tensor.rs**

```rust
use std::sync::Arc;

use forge_core::{DType, Tensor};

#[derive(Clone, Debug)]
pub struct CpuTensor {
    pub(crate) data: Arc<Vec<f32>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            dtype: DType::F32,
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Tensor for CpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}
```

**Step 3: Create lib.rs**

```rust
//! CPU backend for Forge inference engine.

pub mod tensor;
mod backend;

pub use backend::CpuBackend;
pub use tensor::CpuTensor;
```

**Step 4: Add to workspace Cargo.toml**

Add `"forge-backend/forge-backend-cpu"` to `members` list and add `forge-backend-cpu = { path = "forge-backend/forge-backend-cpu" }` to `[workspace.dependencies]`.

**Step 5: Verify it compiles**

Run: `cargo check -p forge-backend-cpu`
Expected: Compiles (backend.rs can be an empty placeholder for now)

**Step 6: Commit**

```
feat: add forge-backend-cpu crate skeleton with CpuTensor
```

---

### Task 2: Implement CpuBackend — device, allocation, data transfer

**Files:**
- Create: `forge-backend/forge-backend-cpu/src/backend.rs`

**Step 1: Write backend.rs with device/allocation/transfer ops**

```rust
use forge_core::{Backend, DType, ForgeError, Result, Tensor};

use crate::tensor::CpuTensor;

fn validate_shape(data_len: usize, shape: &[usize]) -> Result<()> {
    let expected: usize = shape.iter().product();
    if data_len != expected {
        return Err(ForgeError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![data_len],
        });
    }
    Ok(())
}

fn validate_same_len(a: &CpuTensor, b: &CpuTensor) -> Result<()> {
    if a.len() != b.len() {
        return Err(ForgeError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

#[derive(Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;

    fn name(&self) -> &str {
        "cpu"
    }

    fn device_count(&self) -> usize {
        1
    }

    fn allocate(&self, shape: &[usize], _dtype: DType) -> Result<CpuTensor> {
        let numel: usize = shape.iter().product();
        Ok(CpuTensor::new(vec![0.0; numel], shape.to_vec()))
    }

    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<CpuTensor> {
        self.allocate(shape, dtype)
    }

    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        Ok(CpuTensor::new(data.to_vec(), shape.to_vec()))
    }

    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        Ok(CpuTensor::new(f32_data, shape.to_vec()))
    }

    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        Ok(CpuTensor::new(f32_data, shape.to_vec()))
    }

    fn copy_to_host_f32(&self, tensor: &CpuTensor) -> Result<Vec<f32>> {
        Ok(tensor.data().to_vec())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    // Compute ops — stub for now, implemented in Tasks 3-5
    fn matmul(&self, _a: &CpuTensor, _b: &CpuTensor) -> Result<CpuTensor> {
        todo!("Task 3")
    }
    fn add(&self, _a: &CpuTensor, _b: &CpuTensor) -> Result<CpuTensor> {
        todo!("Task 4")
    }
    fn mul(&self, _a: &CpuTensor, _b: &CpuTensor) -> Result<CpuTensor> {
        todo!("Task 4")
    }
    fn mul_scalar(&self, _a: &CpuTensor, _scalar: f32) -> Result<CpuTensor> {
        todo!("Task 4")
    }
    fn silu(&self, _a: &CpuTensor) -> Result<CpuTensor> {
        todo!("Task 4")
    }
    fn rms_norm(&self, _x: &CpuTensor, _w: &CpuTensor, _eps: f32) -> Result<CpuTensor> {
        todo!("Task 5")
    }
    fn rope(&self, _x: &CpuTensor, _c: &CpuTensor, _s: &CpuTensor) -> Result<CpuTensor> {
        todo!("Task 5")
    }
    fn softmax(&self, _x: &CpuTensor, _dim: i32) -> Result<CpuTensor> {
        todo!("Task 5")
    }
    fn embedding(&self, _w: &CpuTensor, _i: &[u32]) -> Result<CpuTensor> {
        todo!("Task 4")
    }
    fn reshape(&self, _x: &CpuTensor, _shape: &[usize]) -> Result<CpuTensor> {
        todo!("Task 4")
    }
    fn transpose(&self, _x: &CpuTensor, _d0: usize, _d1: usize) -> Result<CpuTensor> {
        todo!("Task 4")
    }
    fn cat(&self, _t: &[&CpuTensor], _dim: usize) -> Result<CpuTensor> {
        todo!("Task 4")
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo check -p forge-backend-cpu`
Expected: Compiles with warnings about `todo!()` macros

**Step 3: Commit**

```
feat: implement CpuBackend device, allocation, and data transfer ops
```

---

### Task 3: Implement matmul with OpenBLAS

**Files:**
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs`
- Create: `forge-backend/forge-backend-cpu/tests/test_ops.rs`

**Step 1: Write the failing test**

Create `forge-backend/forge-backend-cpu/tests/test_ops.rs`:

```rust
use forge_backend_cpu::CpuBackend;
use forge_core::Backend;

#[test]
fn test_matmul_2x3_times_3x2() {
    let backend = CpuBackend::new();
    // A: [2,3], B: [3,2] -> C: [2,2]
    let a = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = backend.copy_from_host_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
    let c = backend.matmul(&a, &b).unwrap();
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_shape_mismatch() {
    let backend = CpuBackend::new();
    let a = backend.copy_from_host_f32(&[1.0, 2.0], &[1, 2]).unwrap();
    let b = backend.copy_from_host_f32(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
    assert!(backend.matmul(&a, &b).is_err());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p forge-backend-cpu test_matmul`
Expected: FAIL (panics on `todo!()`)

**Step 3: Implement matmul**

Replace the `matmul` method in `backend.rs`:

```rust
fn matmul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(ForgeError::InvalidArgument("matmul requires 2D tensors".into()));
    }
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    if b_shape[0] != k {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![k, n],
            got: b_shape.to_vec(),
        });
    }

    let mut c = vec![0.0f32; m * n];

    // cblas sgemm: C = alpha * A * B + beta * C
    // Row-major: use CblasRowMajor, CblasNoTrans, CblasNoTrans
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::None,
            m as i32,       // M
            n as i32,       // N
            k as i32,       // K
            1.0,            // alpha
            a.data(),       // A
            k as i32,       // lda
            b.data(),       // B
            n as i32,       // ldb
            0.0,            // beta
            &mut c,         // C
            n as i32,       // ldc
        );
    }

    Ok(CpuTensor::new(c, vec![m, n]))
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p forge-backend-cpu test_matmul`
Expected: 2 tests PASS

**Step 5: Commit**

```
feat: implement CPU matmul via cblas::sgemm (OpenBLAS)
```

---

### Task 4: Implement element-wise ops + shape ops

**Files:**
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs`
- Modify: `forge-backend/forge-backend-cpu/tests/test_ops.rs`

**Step 1: Write failing tests**

Append to `test_ops.rs`:

```rust
#[test]
fn test_add() {
    let backend = CpuBackend::new();
    let a = backend.copy_from_host_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let b = backend.copy_from_host_f32(&[4.0, 5.0, 6.0], &[3]).unwrap();
    let c = backend.add(&a, &b).unwrap();
    assert_eq!(backend.copy_to_host_f32(&c).unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_mul() {
    let backend = CpuBackend::new();
    let a = backend.copy_from_host_f32(&[2.0, 3.0, 4.0], &[3]).unwrap();
    let b = backend.copy_from_host_f32(&[5.0, 6.0, 7.0], &[3]).unwrap();
    let c = backend.mul(&a, &b).unwrap();
    assert_eq!(backend.copy_to_host_f32(&c).unwrap(), vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_mul_scalar() {
    let backend = CpuBackend::new();
    let a = backend.copy_from_host_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let c = backend.mul_scalar(&a, 2.5).unwrap();
    assert_eq!(backend.copy_to_host_f32(&c).unwrap(), vec![2.5, 5.0, 7.5]);
}

#[test]
fn test_silu() {
    let backend = CpuBackend::new();
    let x = backend.copy_from_host_f32(&[0.0, 1.0, -1.0], &[3]).unwrap();
    let out = backend.silu(&x).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    assert!((result[0] - 0.0).abs() < 1e-4);
    assert!((result[1] - 0.7311).abs() < 1e-3);
    assert!((result[2] - (-0.2689)).abs() < 1e-3);
}

#[test]
fn test_embedding() {
    let backend = CpuBackend::new();
    let weight = backend.copy_from_host_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[4, 3],
    ).unwrap();
    let out = backend.embedding(&weight, &[2, 0, 3]).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    assert_eq!(result, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
}

#[test]
fn test_reshape() {
    let backend = CpuBackend::new();
    let x = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let y = backend.reshape(&x, &[3, 2]).unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    // Data unchanged
    assert_eq!(backend.copy_to_host_f32(&y).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_transpose() {
    let backend = CpuBackend::new();
    // [2,3] -> [3,2]
    let x = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let y = backend.transpose(&x, 0, 1).unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    assert_eq!(backend.copy_to_host_f32(&y).unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_cat() {
    let backend = CpuBackend::new();
    let a = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = backend.copy_from_host_f32(&[5.0, 6.0], &[1, 2]).unwrap();
    let c = backend.cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(backend.copy_to_host_f32(&c).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p forge-backend-cpu`
Expected: New tests FAIL (panics on `todo!()`)

**Step 3: Implement all element-wise + shape ops**

Replace the `todo!()` stubs for `add`, `mul`, `mul_scalar`, `silu`, `embedding`, `reshape`, `transpose`, `cat`:

```rust
fn add(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
    validate_same_len(a, b)?;
    let data: Vec<f32> = a.data().iter().zip(b.data().iter()).map(|(x, y)| x + y).collect();
    Ok(CpuTensor::new(data, a.shape().to_vec()))
}

fn mul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
    validate_same_len(a, b)?;
    let data: Vec<f32> = a.data().iter().zip(b.data().iter()).map(|(x, y)| x * y).collect();
    Ok(CpuTensor::new(data, a.shape().to_vec()))
}

fn mul_scalar(&self, a: &CpuTensor, scalar: f32) -> Result<CpuTensor> {
    let data: Vec<f32> = a.data().iter().map(|x| x * scalar).collect();
    Ok(CpuTensor::new(data, a.shape().to_vec()))
}

fn silu(&self, a: &CpuTensor) -> Result<CpuTensor> {
    let data: Vec<f32> = a.data().iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    Ok(CpuTensor::new(data, a.shape().to_vec()))
}

fn embedding(&self, weight: &CpuTensor, indices: &[u32]) -> Result<CpuTensor> {
    let w_shape = weight.shape();
    if w_shape.len() != 2 {
        return Err(ForgeError::InvalidArgument(
            "embedding weight must be 2D [vocab_size, embedding_dim]".into(),
        ));
    }
    let vocab_size = w_shape[0];
    let dim = w_shape[1];
    let mut out = Vec::with_capacity(indices.len() * dim);
    for &idx in indices {
        if idx as usize >= vocab_size {
            return Err(ForgeError::InvalidArgument(format!(
                "embedding index {idx} out of range (vocab_size={vocab_size})"
            )));
        }
        let start = idx as usize * dim;
        out.extend_from_slice(&weight.data()[start..start + dim]);
    }
    Ok(CpuTensor::new(out, vec![indices.len(), dim]))
}

fn reshape(&self, x: &CpuTensor, shape: &[usize]) -> Result<CpuTensor> {
    let numel: usize = shape.iter().product();
    if numel != x.len() {
        return Err(ForgeError::ShapeMismatch {
            expected: shape.to_vec(),
            got: x.shape().to_vec(),
        });
    }
    // Zero-copy: share Arc data, just change shape
    Ok(CpuTensor {
        data: x.data.clone(),
        shape: shape.to_vec(),
        dtype: x.dtype,
    })
}

fn transpose(&self, x: &CpuTensor, dim0: usize, dim1: usize) -> Result<CpuTensor> {
    let shape = x.shape();
    if shape.len() != 2 || !((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0)) {
        return Err(ForgeError::InvalidArgument(
            "transpose currently only supports 2D tensors with dims (0,1)".into(),
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    let src = x.data();
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    Ok(CpuTensor::new(out, vec![cols, rows]))
}

fn cat(&self, tensors: &[&CpuTensor], dim: usize) -> Result<CpuTensor> {
    if tensors.is_empty() {
        return Err(ForgeError::InvalidArgument("empty tensor list".into()));
    }
    if dim != 0 {
        return Err(ForgeError::InvalidArgument(
            "cat currently only supports dim=0".into(),
        ));
    }
    let ndim = tensors[0].shape().len();
    for t in tensors.iter().skip(1) {
        if t.shape().len() != ndim {
            return Err(ForgeError::ShapeMismatch {
                expected: tensors[0].shape().to_vec(),
                got: t.shape().to_vec(),
            });
        }
        for d in 1..ndim {
            if t.shape()[d] != tensors[0].shape()[d] {
                return Err(ForgeError::ShapeMismatch {
                    expected: tensors[0].shape().to_vec(),
                    got: t.shape().to_vec(),
                });
            }
        }
    }
    let mut total_first_dim = 0;
    let mut all_data = Vec::new();
    for t in tensors {
        total_first_dim += t.shape()[0];
        all_data.extend_from_slice(t.data());
    }
    let mut out_shape = tensors[0].shape().to_vec();
    out_shape[0] = total_first_dim;
    Ok(CpuTensor::new(all_data, out_shape))
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p forge-backend-cpu`
Expected: All tests PASS

**Step 5: Commit**

```
feat: implement CPU element-wise ops (add, mul, silu, embedding, reshape, transpose, cat)
```

---

### Task 5: Implement rms_norm, softmax, rope

**Files:**
- Modify: `forge-backend/forge-backend-cpu/src/backend.rs`
- Modify: `forge-backend/forge-backend-cpu/tests/test_ops.rs`

**Step 1: Write failing tests**

Append to `test_ops.rs`:

```rust
#[test]
fn test_rms_norm() {
    let backend = CpuBackend::new();
    let x = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let w = backend.copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
    let out = backend.rms_norm(&x, &w, 1e-5).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let rms = (7.5f32).sqrt();
    let expected: Vec<f32> = vec![1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "got {a}, expected {b}");
    }
}

#[test]
fn test_rms_norm_multi_row() {
    let backend = CpuBackend::new();
    let x = backend.copy_from_host_f32(
        &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], &[2, 4],
    ).unwrap();
    let w = backend.copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
    let out = backend.rms_norm(&x, &w, 1e-5).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    for &v in &result[..4] {
        assert!((v - 1.0).abs() < 1e-4, "row 0: got {v}");
    }
    for &v in &result[4..] {
        assert!((v - 1.0).abs() < 1e-4, "row 1: got {v}");
    }
}

#[test]
fn test_softmax() {
    let backend = CpuBackend::new();
    let x = backend.copy_from_host_f32(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let out = backend.softmax(&x, -1).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    assert!(result[2] > result[1] && result[1] > result[0]);
}

#[test]
fn test_softmax_multi_row() {
    let backend = CpuBackend::new();
    let x = backend.copy_from_host_f32(
        &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3],
    ).unwrap();
    let out = backend.softmax(&x, -1).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let sum0: f32 = result[..3].iter().sum();
    let sum1: f32 = result[3..].iter().sum();
    assert!((sum0 - 1.0).abs() < 1e-5, "row 0 sum={sum0}");
    assert!((sum1 - 1.0).abs() < 1e-5, "row 1 sum={sum1}");
}

#[test]
fn test_rope() {
    let backend = CpuBackend::new();
    // x: [1, 1, 1, 4] — batch=1, seq_len=1, heads=1, head_dim=4
    let x = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
    // cos/sin for pos=0, half_dim=2: [1, 2]
    let cos = backend.copy_from_host_f32(&[1.0, 1.0], &[1, 2]).unwrap();
    let sin = backend.copy_from_host_f32(&[0.0, 0.0], &[1, 2]).unwrap();
    // With cos=1, sin=0, output should equal input
    let out = backend.rope(&x, &cos, &sin).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    for (i, (&got, &exp)) in result.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
        assert!((got - exp).abs() < 1e-4, "index {i}: got {got}, expected {exp}");
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p forge-backend-cpu test_rms_norm test_softmax test_rope`
Expected: FAIL (panics on `todo!()`)

**Step 3: Implement rms_norm, softmax, rope**

Replace the `todo!()` stubs:

```rust
fn rms_norm(&self, x: &CpuTensor, weight: &CpuTensor, eps: f32) -> Result<CpuTensor> {
    let shape = x.shape();
    let cols = *shape.last().unwrap();
    if weight.len() != cols {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![cols],
            got: weight.shape().to_vec(),
        });
    }
    let rows = x.len() / cols;
    let src = x.data();
    let w = weight.data();
    let mut out = vec![0.0f32; rows * cols];

    for row in 0..rows {
        let row_data = &src[row * cols..(row + 1) * cols];
        let ss: f32 = row_data.iter().map(|v| v * v).sum();
        let rms = (ss / cols as f32 + eps).sqrt().recip();
        for col in 0..cols {
            out[row * cols + col] = row_data[col] * rms * w[col];
        }
    }

    Ok(CpuTensor::new(out, shape.to_vec()))
}

fn softmax(&self, x: &CpuTensor, dim: i32) -> Result<CpuTensor> {
    let shape = x.shape();
    let ndim = shape.len() as i32;
    let normalized_dim = if dim < 0 { ndim + dim } else { dim };
    if normalized_dim != ndim - 1 {
        return Err(ForgeError::InvalidArgument(format!(
            "softmax only supports last dimension (got dim={dim}, ndim={ndim})"
        )));
    }
    let cols = *shape.last().unwrap();
    let rows = x.len() / cols;
    let src = x.data();
    let mut out = vec![0.0f32; rows * cols];

    for row in 0..rows {
        let row_data = &src[row * cols..(row + 1) * cols];
        let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for col in 0..cols {
            let v = (row_data[col] - max_val).exp();
            out[row * cols + col] = v;
            sum += v;
        }
        for col in 0..cols {
            out[row * cols + col] /= sum;
        }
    }

    Ok(CpuTensor::new(out, shape.to_vec()))
}

fn rope(
    &self,
    x: &CpuTensor,
    freqs_cos: &CpuTensor,
    freqs_sin: &CpuTensor,
) -> Result<CpuTensor> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(ForgeError::InvalidArgument(
            "rope expects 4D tensor [batch, seq_len, heads, head_dim]".into(),
        ));
    }
    let batch = shape[0];
    let seq_len = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];
    if head_dim % 2 != 0 {
        return Err(ForgeError::InvalidArgument("rope requires even head_dim".into()));
    }
    let half_dim = head_dim / 2;
    let expected_freq_len = seq_len * half_dim;
    if freqs_cos.len() < expected_freq_len || freqs_sin.len() < expected_freq_len {
        return Err(ForgeError::InvalidArgument(format!(
            "freq tensors need at least {} elements, got cos={} sin={}",
            expected_freq_len, freqs_cos.len(), freqs_sin.len()
        )));
    }

    let src = x.data();
    let cos = freqs_cos.data();
    let sin = freqs_sin.data();
    let mut out = vec![0.0f32; src.len()];

    for b in 0..batch {
        for pos in 0..seq_len {
            for head in 0..num_heads {
                let base = b * seq_len * num_heads * head_dim
                    + pos * num_heads * head_dim
                    + head * head_dim;
                for h in 0..half_dim {
                    let x0 = src[base + h];
                    let x1 = src[base + h + half_dim];
                    let cos_val = cos[pos * half_dim + h];
                    let sin_val = sin[pos * half_dim + h];
                    out[base + h] = x0 * cos_val - x1 * sin_val;
                    out[base + h + half_dim] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    Ok(CpuTensor::new(out, shape.to_vec()))
}
```

**Step 4: Run all tests**

Run: `cargo test -p forge-backend-cpu`
Expected: All tests PASS

**Step 5: Commit**

```
feat: implement CPU rms_norm, softmax, and rope
```

---

### Task 6: Wire CpuBackend into forge-server via --backend flag

**Files:**
- Modify: `Cargo.toml` (workspace root — add forge-backend-cpu to deps)
- Modify: `forge-server/Cargo.toml` (add forge-backend-cpu dep)
- Modify: `forge-server/src/main.rs`

**Step 1: Add forge-backend-cpu dependency**

In workspace `Cargo.toml`, the member is already added (Task 1). Now add to `forge-server/Cargo.toml`:

```toml
forge-backend-cpu.workspace = true
```

**Step 2: Add --backend CLI arg and branch in main.rs**

The key change: the server needs to branch on the backend type. Since `Engine`, `LlamaModel`, and `NaiveKvCache` are all generic over `B: Backend`, the main function needs to monomorphize for each backend. Extract the post-backend code into a generic helper:

```rust
use forge_backend_cpu::CpuBackend;
// ... existing imports ...

#[derive(Parser)]
#[command(name = "forge-server", about = "Forge LLM Inference Server")]
struct Cli {
    // ... existing fields ...

    /// Backend to use: "cuda" or "cpu"
    #[arg(long, default_value = "cuda")]
    backend: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ... tracing + cli + config loading (unchanged) ...

    match cli.backend.as_str() {
        "cpu" => {
            let backend = CpuBackend::new();
            info!("CPU backend initialized");
            run_server(backend, &cli, model_config).await
        }
        "cuda" => {
            let backend = CudaBackend::new(cli.device)?;
            info!("CUDA backend initialized (device {})", cli.device);
            run_server(backend, &cli, model_config).await
        }
        other => anyhow::bail!("Unknown backend: {other}. Use 'cpu' or 'cuda'."),
    }
}

async fn run_server<B: Backend + Clone>(
    backend: B,
    cli: &Cli,
    model_config: ModelConfig,
) -> anyhow::Result<()> {
    // Load model weights
    info!("Loading model from {}...", cli.model_path.display());
    let loader = SafeTensorsLoader::new(&cli.model_path)?;
    let model = load_llama_model(&loader, model_config.clone(), &backend)?;
    info!("Model loaded successfully");

    // Load tokenizer
    let tokenizer_path = cli.model_path.join("tokenizer.json");
    let tokenizer = ForgeTokenizer::from_file(&tokenizer_path)?;
    info!("Tokenizer loaded (eos_token_id={})", tokenizer.eos_token_id());

    // Load chat template
    let chat_template = load_chat_template(&cli.model_path)?;
    info!("Chat template loaded");

    // Create engine components
    let kv_cache = NaiveKvCache::new(backend.clone(), model_config.num_hidden_layers, cli.max_batch_size);
    let scheduler_config = SchedulerConfig {
        max_batch_size: cli.max_batch_size,
        max_prefill_tokens: cli.max_prefill_tokens,
    };
    let scheduler = ContinuousBatchingScheduler::new(scheduler_config);
    let (request_tx, request_rx) = mpsc::channel(1024);

    // Spawn engine
    let mut engine = Engine::new(model, backend, Box::new(scheduler), Box::new(kv_cache), request_rx);
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        if let Err(e) = engine.run().await {
            error!("Engine error: {e}");
        }
        let _ = shutdown_tx.send(true);
    });
    info!("Engine spawned");

    // Model name + HTTP server
    let model_name = cli.model_path.file_name()
        .and_then(|n| n.to_str()).unwrap_or("unknown").to_string();

    let state = Arc::new(AppState {
        model_name: model_name.clone(), tokenizer, chat_template, request_tx,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
        .route("/forge/v1/health", get(openai::health))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Forge serving '{model_name}' on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    info!("Engine exited, shutting down HTTP server");
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("Received Ctrl-C, shutting down");
                }
            }
        })
        .await?;

    Ok(())
}
```

**Step 3: Verify it compiles**

Run: `cargo check --workspace`
Expected: 0 errors, 0 warnings

**Step 4: Run all tests**

Run: `cargo test --workspace`
Expected: All existing tests still pass + new CPU backend tests pass

**Step 5: Commit**

```
feat: add --backend cpu|cuda CLI flag, wire CpuBackend into server
```

---

### Task 7: Run full test suite, verify workspace integrity

**Files:** (none modified — verification only)

**Step 1: Full workspace check**

Run: `cargo check --workspace`
Expected: 0 errors, 0 warnings

**Step 2: Full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass (57 existing + new CPU backend tests)

**Step 3: Verify CPU backend tests specifically**

Run: `cargo test -p forge-backend-cpu -- --nocapture`
Expected: All ops tests pass with output

**Step 4: Verify data transfer with f16/bf16**

Manually verify the f16/bf16 → f32 conversion in the test output (Task 2 covers this).

**Step 5: Commit if any fixups needed, otherwise done**

If any issues found, fix and commit. Otherwise, the CPU backend is complete.
