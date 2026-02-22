use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, Tensor};

#[test]
fn test_copy_roundtrip_f32() {
    let backend = CudaBackend::new(0).unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = backend.copy_from_host_f32(&data, &[2, 2]).unwrap();
    let result = backend.copy_to_host_f32(&tensor).unwrap();
    assert_eq!(data, result);
}

#[test]
fn test_tensor_shape() {
    let backend = CudaBackend::new(0).unwrap();
    let data = vec![0.0f32; 12];
    let tensor = backend.copy_from_host_f32(&data, &[3, 4]).unwrap();
    use forge_core::Tensor;
    assert_eq!(tensor.shape(), &[3, 4]);
}

#[test]
fn test_add() {
    let backend = CudaBackend::new(0).unwrap();
    let a = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[4])
        .unwrap();
    let b = backend
        .copy_from_host_f32(&[10.0, 20.0, 30.0, 40.0], &[4])
        .unwrap();
    let c = backend.add(&a, &b).unwrap();
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_mul() {
    let backend = CudaBackend::new(0).unwrap();
    let a = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[4])
        .unwrap();
    let b = backend
        .copy_from_host_f32(&[2.0, 3.0, 4.0, 5.0], &[4])
        .unwrap();
    let c = backend.mul(&a, &b).unwrap();
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_mul_scalar() {
    let backend = CudaBackend::new(0).unwrap();
    let a = backend.copy_from_host_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let c = backend.mul_scalar(&a, 2.0).unwrap();
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(result, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_matmul_2x3_times_3x2() {
    let backend = CudaBackend::new(0).unwrap();
    // A = [[1, 2, 3], [4, 5, 6]] (2x3, row-major)
    let a = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    // B = [[7, 8], [9, 10], [11, 12]] (3x2, row-major)
    let b = backend
        .copy_from_host_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])
        .unwrap();
    let c = backend.matmul(&a, &b).unwrap();
    let result = backend.copy_to_host_f32(&c).unwrap();
    // C = [[58, 64], [139, 154]]
    assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_allocate_zeros() {
    let backend = CudaBackend::new(0).unwrap();
    let t = backend
        .allocate_zeros(&[2, 3], forge_core::DType::F32)
        .unwrap();
    let result = backend.copy_to_host_f32(&t).unwrap();
    assert_eq!(result, vec![0.0; 6]);
}

#[test]
fn test_reshape() {
    let backend = CudaBackend::new(0).unwrap();
    let t = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let reshaped = backend.reshape(&t, &[3, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);
    let result = backend.copy_to_host_f32(&reshaped).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_reshape_mismatch() {
    let backend = CudaBackend::new(0).unwrap();
    let t = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap();
    assert!(backend.reshape(&t, &[3, 2]).is_err());
}

#[test]
fn test_transpose_2d() {
    let backend = CudaBackend::new(0).unwrap();
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
    let t = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let tr = backend.transpose(&t, 0, 1).unwrap();
    assert_eq!(tr.shape(), &[3, 2]);
    let result = backend.copy_to_host_f32(&tr).unwrap();
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_cat_dim0() {
    let backend = CudaBackend::new(0).unwrap();
    let a = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap();
    let b = backend
        .copy_from_host_f32(&[5.0, 6.0], &[1, 2])
        .unwrap();
    let c = backend.cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.shape(), &[3, 2]);
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_cat_1d() {
    let backend = CudaBackend::new(0).unwrap();
    let a = backend.copy_from_host_f32(&[1.0, 2.0], &[2]).unwrap();
    let b = backend.copy_from_host_f32(&[3.0, 4.0, 5.0], &[3]).unwrap();
    let c = backend.cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.shape(), &[5]);
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_rope() {
    let backend = CudaBackend::new(0).unwrap();
    // Shape: [1, 1, 1, 4] => batch=1, seq_len=1, heads=1, head_dim=4
    // Input: [1.0, 2.0, 3.0, 4.0]
    // half_dim = 2, x0=[1,2], x1=[3,4]
    // cos_freqs and sin_freqs shape: [1, 2] (seq_len=1, half_dim=2)
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4])
        .unwrap();
    // cos = [1.0, 0.0], sin = [0.0, 1.0]
    let cos_freqs = backend
        .copy_from_host_f32(&[1.0, 0.0], &[1, 2])
        .unwrap();
    let sin_freqs = backend
        .copy_from_host_f32(&[0.0, 1.0], &[1, 2])
        .unwrap();
    let out = backend.rope(&x, &cos_freqs, &sin_freqs).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    // out[0] = x0[0]*cos[0] - x1[0]*sin[0] = 1*1 - 3*0 = 1.0
    // out[1] = x0[1]*cos[1] - x1[1]*sin[1] = 2*0 - 4*1 = -4.0
    // out[2] = x0[0]*sin[0] + x1[0]*cos[0] = 1*0 + 3*1 = 3.0
    // out[3] = x0[1]*sin[1] + x1[1]*cos[1] = 2*1 + 4*0 = 2.0
    let expected = vec![1.0, -4.0, 3.0, 2.0];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "got {a}, expected {b}, diff={}",
            (a - b).abs()
        );
    }
}

#[test]
fn test_copy_shape_validation() {
    let backend = CudaBackend::new(0).unwrap();
    // 4 elements but shape says 6
    assert!(backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 3]).is_err());
}

#[test]
fn test_add_length_mismatch() {
    let backend = CudaBackend::new(0).unwrap();
    let a = backend.copy_from_host_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let b = backend.copy_from_host_f32(&[1.0, 2.0], &[2]).unwrap();
    assert!(backend.add(&a, &b).is_err());
}
