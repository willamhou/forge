use forge_backend_cuda::CudaBackend;
use forge_core::Backend;

#[test]
fn test_rms_norm() {
    let backend = CudaBackend::new(0).unwrap();
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
        assert!(
            (a - b).abs() < 1e-4,
            "got {a}, expected {b}, diff={}",
            (a - b).abs()
        );
    }
}

#[test]
fn test_rms_norm_multi_row() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend
        .copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], &[2, 4])
        .unwrap();
    let w = backend
        .copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0], &[4])
        .unwrap();
    let out = backend.rms_norm(&x, &w, 1e-5).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    // Row 0: all 1s, RMS = 1.0, so output ≈ 1.0
    // Row 1: all 2s, RMS = 2.0, so output ≈ 1.0
    for &v in &result[..4] {
        assert!((v - 1.0).abs() < 1e-4, "row 0: got {v}");
    }
    for &v in &result[4..] {
        assert!((v - 1.0).abs() < 1e-4, "row 1: got {v}");
    }
}

#[test]
fn test_silu() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend
        .copy_from_host_f32(&[0.0, 1.0, -1.0], &[3])
        .unwrap();
    let out = backend.silu(&x).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    // silu(x) = x * sigmoid(x)
    assert!((result[0] - 0.0).abs() < 1e-4);
    assert!((result[1] - 0.7311).abs() < 1e-3);
    assert!((result[2] - (-0.2689)).abs() < 1e-3);
}

#[test]
fn test_softmax() {
    let backend = CudaBackend::new(0).unwrap();
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
    let backend = CudaBackend::new(0).unwrap();
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
fn test_embedding() {
    let backend = CudaBackend::new(0).unwrap();
    // Embedding table: 4 tokens x 3 dims
    let weight = backend
        .copy_from_host_f32(
            &[
                1.0, 2.0, 3.0, // token 0
                4.0, 5.0, 6.0, // token 1
                7.0, 8.0, 9.0, // token 2
                10.0, 11.0, 12.0, // token 3
            ],
            &[4, 3],
        )
        .unwrap();
    let out = backend.embedding(&weight, &[2, 0, 3]).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    assert_eq!(result, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
}
