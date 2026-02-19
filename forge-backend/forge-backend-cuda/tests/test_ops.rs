use forge_backend_cuda::CudaBackend;
use forge_core::Backend;

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
