use forge_backend_cpu::CpuBackend;
use forge_core::{Backend, Tensor};

#[test]
fn test_matmul_2x3_times_3x2() {
    let backend = CpuBackend::new();
    let a = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let b = backend
        .copy_from_host_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])
        .unwrap();
    let c = backend.matmul(&a, &b).unwrap();
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_shape_mismatch() {
    let backend = CpuBackend::new();
    let a = backend.copy_from_host_f32(&[1.0, 2.0], &[1, 2]).unwrap();
    let b = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0], &[1, 3])
        .unwrap();
    assert!(backend.matmul(&a, &b).is_err());
}

#[test]
fn test_add() {
    let backend = CpuBackend::new();
    let a = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0], &[3])
        .unwrap();
    let b = backend
        .copy_from_host_f32(&[4.0, 5.0, 6.0], &[3])
        .unwrap();
    let c = backend.add(&a, &b).unwrap();
    assert_eq!(backend.copy_to_host_f32(&c).unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_mul() {
    let backend = CpuBackend::new();
    let a = backend
        .copy_from_host_f32(&[2.0, 3.0, 4.0], &[3])
        .unwrap();
    let b = backend
        .copy_from_host_f32(&[5.0, 6.0, 7.0], &[3])
        .unwrap();
    let c = backend.mul(&a, &b).unwrap();
    assert_eq!(
        backend.copy_to_host_f32(&c).unwrap(),
        vec![10.0, 18.0, 28.0]
    );
}

#[test]
fn test_mul_scalar() {
    let backend = CpuBackend::new();
    let a = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0], &[3])
        .unwrap();
    let c = backend.mul_scalar(&a, 2.5).unwrap();
    assert_eq!(backend.copy_to_host_f32(&c).unwrap(), vec![2.5, 5.0, 7.5]);
}

#[test]
fn test_silu() {
    let backend = CpuBackend::new();
    let x = backend
        .copy_from_host_f32(&[0.0, 1.0, -1.0], &[3])
        .unwrap();
    let out = backend.silu(&x).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    assert!((result[0] - 0.0).abs() < 1e-4);
    assert!((result[1] - 0.7311).abs() < 1e-3);
    assert!((result[2] - (-0.2689)).abs() < 1e-3);
}

#[test]
fn test_embedding() {
    let backend = CpuBackend::new();
    let weight = backend
        .copy_from_host_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[4, 3],
        )
        .unwrap();
    let out = backend.embedding(&weight, &[2, 0, 3]).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    assert_eq!(
        result,
        vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn test_reshape() {
    let backend = CpuBackend::new();
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let y = backend.reshape(&x, &[3, 2]).unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    assert_eq!(
        backend.copy_to_host_f32(&y).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_transpose() {
    let backend = CpuBackend::new();
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let y = backend.transpose(&x, 0, 1).unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    assert_eq!(
        backend.copy_to_host_f32(&y).unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn test_cat() {
    let backend = CpuBackend::new();
    let a = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap();
    let b = backend.copy_from_host_f32(&[5.0, 6.0], &[1, 2]).unwrap();
    let c = backend.cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(
        backend.copy_to_host_f32(&c).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_rms_norm() {
    let backend = CpuBackend::new();
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4])
        .unwrap();
    let w = backend
        .copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0], &[4])
        .unwrap();
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
    let x = backend
        .copy_from_host_f32(
            &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            &[2, 4],
        )
        .unwrap();
    let w = backend
        .copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0], &[4])
        .unwrap();
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
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0], &[1, 3])
        .unwrap();
    let out = backend.softmax(&x, -1).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    assert!(result[2] > result[1] && result[1] > result[0]);
}

#[test]
fn test_softmax_multi_row() {
    let backend = CpuBackend::new();
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3])
        .unwrap();
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
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4])
        .unwrap();
    let cos = backend
        .copy_from_host_f32(&[1.0, 1.0], &[1, 2])
        .unwrap();
    let sin = backend
        .copy_from_host_f32(&[0.0, 0.0], &[1, 2])
        .unwrap();
    let out = backend.rope(&x, &cos, &sin).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    for (i, (&got, &exp)) in result.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "index {i}: got {got}, expected {exp}"
        );
    }
}
