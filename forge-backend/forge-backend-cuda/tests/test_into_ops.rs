//! `_into` variant tests for hot-path Backend ops.
//!
//! For each op: verify the `_into` variant produces the same output as the
//! allocating variant, that the same caller-provided buffer can be reused
//! across multiple calls (the property CUDA Graph capture exploits), and
//! that shape/dtype mismatches are rejected upfront.
//!
//! Single test fn — GB10 cuBLAS fixture leak workaround (task #4).

use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, DType};

#[test]
fn into_variants_match_alloc_variants() {
    let backend = CudaBackend::new(0).expect("CUDA backend");

    // ── matmul_into vs matmul (F32) ───────────────────────────────
    let m = 4;
    let k = 8;
    let n = 6;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.3).collect();
    let a = backend.copy_from_host_f32(&a_data, &[m, k]).unwrap();
    let b = backend.copy_from_host_f32(&b_data, &[k, n]).unwrap();

    // Alloc variant.
    let c_alloc = backend.matmul(&a, &b).unwrap();
    backend.synchronize().unwrap();
    let c_alloc_host = backend.copy_to_host_f32(&c_alloc).unwrap();

    // Into variant: pre-allocate, then call repeatedly into same buffer.
    let mut c_into = backend.allocate_zeros(&[m, n], DType::F32).unwrap();
    for _ in 0..3 {
        backend.matmul_into(&mut c_into, &a, &b).unwrap();
        backend.synchronize().unwrap();
        let h = backend.copy_to_host_f32(&c_into).unwrap();
        for (i, (got, want)) in h.iter().zip(&c_alloc_host).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "matmul_into F32 diverges at [{i}]: got {got} want {want}"
            );
        }
    }

    // ── matmul_into F16 ───────────────────────────────────────────
    let a_f16 = backend.cast(&a, DType::F16).unwrap();
    let b_f16 = backend.cast(&b, DType::F16).unwrap();
    let c_alloc_f16 = backend.matmul(&a_f16, &b_f16).unwrap();
    backend.synchronize().unwrap();
    let c_alloc_f16_f32 = backend.cast(&c_alloc_f16, DType::F32).unwrap();
    let c_alloc_f16_host = backend.copy_to_host_f32(&c_alloc_f16_f32).unwrap();

    let mut c_into_f16 = backend.allocate_zeros(&[m, n], DType::F16).unwrap();
    backend.matmul_into(&mut c_into_f16, &a_f16, &b_f16).unwrap();
    backend.synchronize().unwrap();
    let c_into_f16_f32 = backend.cast(&c_into_f16, DType::F32).unwrap();
    let c_into_f16_host = backend.copy_to_host_f32(&c_into_f16_f32).unwrap();
    for (i, (got, want)) in c_into_f16_host.iter().zip(&c_alloc_f16_host).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "matmul_into F16 diverges at [{i}]: got {got} want {want}"
        );
    }

    // ── matmul_into validation ────────────────────────────────────
    // Wrong out shape.
    let mut wrong_shape = backend.allocate_zeros(&[m, n + 1], DType::F32).unwrap();
    let _ = backend
        .matmul_into(&mut wrong_shape, &a, &b)
        .expect_err("wrong out shape rejected");

    // Wrong out dtype.
    let mut wrong_dtype = backend.allocate_zeros(&[m, n], DType::F16).unwrap();
    let _ = backend
        .matmul_into(&mut wrong_dtype, &a, &b)
        .expect_err("wrong out dtype rejected");

    // a/b dtype mismatch (cast a → F16 but leave b as F32).
    let _ = backend
        .matmul_into(&mut c_into_f16, &a_f16, &b)
        .expect_err("a/b dtype mismatch rejected");

    // ── add_into vs add (F32) ─────────────────────────────────────
    let shape_v = [12];
    let v1_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.3).collect();
    let v2_data: Vec<f32> = (0..12).map(|i| (i as f32) - 5.0).collect();
    let v1 = backend.copy_from_host_f32(&v1_data, &shape_v).unwrap();
    let v2 = backend.copy_from_host_f32(&v2_data, &shape_v).unwrap();

    let sum_alloc = backend.add(&v1, &v2).unwrap();
    backend.synchronize().unwrap();
    let sum_alloc_host = backend.copy_to_host_f32(&sum_alloc).unwrap();

    let mut sum_into = backend.allocate_zeros(&shape_v, DType::F32).unwrap();
    for _ in 0..3 {
        backend.add_into(&mut sum_into, &v1, &v2).unwrap();
        backend.synchronize().unwrap();
        let h = backend.copy_to_host_f32(&sum_into).unwrap();
        for (i, (got, want)) in h.iter().zip(&sum_alloc_host).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "add_into F32 diverges at [{i}]: got {got} want {want}"
            );
        }
    }

    // ── add_into F16 ──────────────────────────────────────────────
    let v1_f16 = backend.cast(&v1, DType::F16).unwrap();
    let v2_f16 = backend.cast(&v2, DType::F16).unwrap();
    let sum_alloc_f16 = backend.add(&v1_f16, &v2_f16).unwrap();
    backend.synchronize().unwrap();
    let sum_alloc_f16_f32 = backend.cast(&sum_alloc_f16, DType::F32).unwrap();
    let sum_alloc_f16_host = backend.copy_to_host_f32(&sum_alloc_f16_f32).unwrap();

    let mut sum_into_f16 = backend.allocate_zeros(&shape_v, DType::F16).unwrap();
    backend.add_into(&mut sum_into_f16, &v1_f16, &v2_f16).unwrap();
    backend.synchronize().unwrap();
    let sum_into_f16_f32 = backend.cast(&sum_into_f16, DType::F32).unwrap();
    let sum_into_f16_host = backend.copy_to_host_f32(&sum_into_f16_f32).unwrap();
    for (i, (got, want)) in sum_into_f16_host.iter().zip(&sum_alloc_f16_host).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "add_into F16 diverges at [{i}]: got {got} want {want}"
        );
    }

    // ── add_into validation ───────────────────────────────────────
    let mut wrong_v = backend.allocate_zeros(&[13], DType::F32).unwrap();
    let _ = backend
        .add_into(&mut wrong_v, &v1, &v2)
        .expect_err("wrong out shape rejected");
    let mut wrong_dtype_v = backend.allocate_zeros(&shape_v, DType::F16).unwrap();
    let _ = backend
        .add_into(&mut wrong_dtype_v, &v1, &v2)
        .expect_err("wrong out dtype rejected");
}
