use forge_backend_cuda::{CudaBackend, GpuPagedKvCache};
use forge_core::{Backend, DType, KvCache, Tensor};

fn make_cache(
    backend: &CudaBackend,
    total_blocks: usize,
    block_size: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> GpuPagedKvCache {
    GpuPagedKvCache::new(
        backend.clone(),
        total_blocks,
        block_size,
        num_layers,
        num_kv_heads,
        head_dim,
        DType::F32,
    )
    .unwrap()
}

/// Test: allocate then free — block accounting is correct.
#[test]
fn test_allocate_and_free() {
    let backend = CudaBackend::new(0).unwrap();
    let mut cache = make_cache(&backend, 64, 16, 2, 4, 8);

    let usage0 = cache.usage();
    assert_eq!(usage0.used_blocks, 0);
    assert_eq!(usage0.total_blocks, 64);

    cache.allocate(1, 10).unwrap();
    let usage1 = cache.usage();
    // 10 tokens / 16 block_size = 1 block (ceil)
    assert_eq!(usage1.used_blocks, 1);

    cache.allocate(2, 32).unwrap();
    let usage2 = cache.usage();
    // 32 / 16 = 2 blocks
    assert_eq!(usage2.used_blocks, 3);

    cache.free(1).unwrap();
    let usage3 = cache.usage();
    assert_eq!(usage3.used_blocks, 2);

    cache.free(2).unwrap();
    let usage4 = cache.usage();
    assert_eq!(usage4.used_blocks, 0);
}

/// Test: append + get_kv round-trip correctness.
#[test]
fn test_append_and_get_kv() {
    let backend = CudaBackend::new(0).unwrap();
    let num_layers = 2;
    let num_kv_heads = 2;
    let head_dim = 4;
    let kv_dim = num_kv_heads * head_dim;
    let mut cache = make_cache(&backend, 64, 16, num_layers, num_kv_heads, head_dim);

    cache.allocate(1, 4).unwrap();

    // Append 3 tokens to both layers.
    let k_data: Vec<f32> = (0..3 * kv_dim).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..3 * kv_dim).map(|i| 100.0 + i as f32).collect();
    let k = backend.copy_from_host_f32(&k_data, &[3, kv_dim]).unwrap();
    let v = backend.copy_from_host_f32(&v_data, &[3, kv_dim]).unwrap();

    for layer in 0..num_layers {
        cache.append(1, layer, &k, &v).unwrap();
    }

    // Verify round-trip for each layer.
    for layer in 0..num_layers {
        let (k_out, v_out) = cache.get_kv(1, layer).unwrap();
        assert_eq!(k_out.shape(), &[3, kv_dim]);
        assert_eq!(v_out.shape(), &[3, kv_dim]);

        let k_host = backend.copy_to_host_f32(&k_out).unwrap();
        let v_host = backend.copy_to_host_f32(&v_out).unwrap();

        for i in 0..k_data.len() {
            assert!(
                (k_host[i] - k_data[i]).abs() < 1e-6,
                "layer {layer} k[{i}]: got {} expected {}",
                k_host[i],
                k_data[i]
            );
            assert!(
                (v_host[i] - v_data[i]).abs() < 1e-6,
                "layer {layer} v[{i}]: got {} expected {}",
                v_host[i],
                v_data[i]
            );
        }
    }
}

/// Test: append across block boundary (tokens span multiple blocks).
#[test]
fn test_block_boundary_crossing() {
    let backend = CudaBackend::new(0).unwrap();
    let block_size = 4;
    let num_kv_heads = 1;
    let head_dim = 2;
    let kv_dim = num_kv_heads * head_dim;
    let mut cache = make_cache(&backend, 64, block_size, 1, num_kv_heads, head_dim);

    cache.allocate(1, 8).unwrap();

    // Append 6 tokens — spans blocks 0 and 1 with block_size=4.
    let k_data: Vec<f32> = (0..6 * kv_dim).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..6 * kv_dim).map(|i| 50.0 + i as f32).collect();
    let k = backend.copy_from_host_f32(&k_data, &[6, kv_dim]).unwrap();
    let v = backend.copy_from_host_f32(&v_data, &[6, kv_dim]).unwrap();

    cache.append(1, 0, &k, &v).unwrap();

    let (k_out, v_out) = cache.get_kv(1, 0).unwrap();
    let k_host = backend.copy_to_host_f32(&k_out).unwrap();
    let v_host = backend.copy_to_host_f32(&v_out).unwrap();

    assert_eq!(k_out.shape(), &[6, kv_dim]);
    for i in 0..k_data.len() {
        assert!(
            (k_host[i] - k_data[i]).abs() < 1e-6,
            "k[{i}]: got {} expected {}",
            k_host[i],
            k_data[i]
        );
        assert!(
            (v_host[i] - v_data[i]).abs() < 1e-6,
            "v[{i}]: got {} expected {}",
            v_host[i],
            v_data[i]
        );
    }
}

/// Test: multiple sequences with independent data.
#[test]
fn test_multiple_sequences() {
    let backend = CudaBackend::new(0).unwrap();
    let kv_dim = 4;
    let mut cache = make_cache(&backend, 64, 16, 1, 2, 2);

    cache.allocate(10, 4).unwrap();
    cache.allocate(20, 4).unwrap();

    // Seq 10: 2 tokens with values starting at 0
    let k10: Vec<f32> = (0..2 * kv_dim).map(|i| i as f32).collect();
    let v10: Vec<f32> = (0..2 * kv_dim).map(|i| 100.0 + i as f32).collect();
    let k10_t = backend.copy_from_host_f32(&k10, &[2, kv_dim]).unwrap();
    let v10_t = backend.copy_from_host_f32(&v10, &[2, kv_dim]).unwrap();
    cache.append(10, 0, &k10_t, &v10_t).unwrap();

    // Seq 20: 3 tokens with values starting at 200
    let k20: Vec<f32> = (0..3 * kv_dim).map(|i| 200.0 + i as f32).collect();
    let v20: Vec<f32> = (0..3 * kv_dim).map(|i| 300.0 + i as f32).collect();
    let k20_t = backend.copy_from_host_f32(&k20, &[3, kv_dim]).unwrap();
    let v20_t = backend.copy_from_host_f32(&v20, &[3, kv_dim]).unwrap();
    cache.append(20, 0, &k20_t, &v20_t).unwrap();

    // Verify seq 10
    let (k_out, _) = cache.get_kv(10, 0).unwrap();
    let k_host = backend.copy_to_host_f32(&k_out).unwrap();
    assert_eq!(k_out.shape(), &[2, kv_dim]);
    for i in 0..k10.len() {
        assert!(
            (k_host[i] - k10[i]).abs() < 1e-6,
            "seq10 k[{i}]: got {} expected {}",
            k_host[i],
            k10[i]
        );
    }

    // Verify seq 20
    let (k_out, _) = cache.get_kv(20, 0).unwrap();
    let k_host = backend.copy_to_host_f32(&k_out).unwrap();
    assert_eq!(k_out.shape(), &[3, kv_dim]);
    for i in 0..k20.len() {
        assert!(
            (k_host[i] - k20[i]).abs() < 1e-6,
            "seq20 k[{i}]: got {} expected {}",
            k_host[i],
            k20[i]
        );
    }

    // Verify isolation: freeing one doesn't affect the other
    cache.free(10).unwrap();
    let (k_out, _) = cache.get_kv(20, 0).unwrap();
    let k_host = backend.copy_to_host_f32(&k_out).unwrap();
    for i in 0..k20.len() {
        assert!((k_host[i] - k20[i]).abs() < 1e-6);
    }
}

/// Test: incremental append (append 1 token at a time, simulating decode).
#[test]
fn test_incremental_append() {
    let backend = CudaBackend::new(0).unwrap();
    let kv_dim = 4;
    let num_layers = 2;
    let mut cache = make_cache(&backend, 64, 4, num_layers, 2, 2);

    cache.allocate(1, 8).unwrap();

    let mut expected_k: Vec<f32> = Vec::new();
    let mut expected_v: Vec<f32> = Vec::new();

    // Append 5 tokens one at a time (crosses block boundary at token 4 with block_size=4).
    for t in 0..5 {
        let k_data: Vec<f32> = (0..kv_dim).map(|d| (t * kv_dim + d) as f32).collect();
        let v_data: Vec<f32> = (0..kv_dim).map(|d| 100.0 + (t * kv_dim + d) as f32).collect();
        expected_k.extend_from_slice(&k_data);
        expected_v.extend_from_slice(&v_data);

        let k = backend.copy_from_host_f32(&k_data, &[1, kv_dim]).unwrap();
        let v = backend.copy_from_host_f32(&v_data, &[1, kv_dim]).unwrap();

        for layer in 0..num_layers {
            cache.append(1, layer, &k, &v).unwrap();
        }
    }

    // Verify all 5 tokens are correct in each layer.
    for layer in 0..num_layers {
        let (k_out, v_out) = cache.get_kv(1, layer).unwrap();
        assert_eq!(k_out.shape(), &[5, kv_dim]);

        let k_host = backend.copy_to_host_f32(&k_out).unwrap();
        let v_host = backend.copy_to_host_f32(&v_out).unwrap();

        for i in 0..expected_k.len() {
            assert!(
                (k_host[i] - expected_k[i]).abs() < 1e-6,
                "layer {layer} k[{i}]: got {} expected {}",
                k_host[i],
                expected_k[i]
            );
            assert!(
                (v_host[i] - expected_v[i]).abs() < 1e-6,
                "layer {layer} v[{i}]: got {} expected {}",
                v_host[i],
                expected_v[i]
            );
        }
    }
}

/// Test: can_allocate reports correctly.
#[test]
fn test_can_allocate() {
    let backend = CudaBackend::new(0).unwrap();
    let mut cache = make_cache(&backend, 4, 16, 1, 1, 4);

    assert!(cache.can_allocate(64)); // 64 tokens / 16 = 4 blocks = exactly all
    assert!(!cache.can_allocate(65)); // 65 tokens needs 5 blocks > 4

    cache.allocate(1, 32).unwrap(); // uses 2 blocks
    assert!(cache.can_allocate(32)); // 2 remaining blocks = 32 tokens
    assert!(!cache.can_allocate(33));
}

/// Test: get_seq_len reports correctly after appends.
#[test]
fn test_get_seq_len() {
    let backend = CudaBackend::new(0).unwrap();
    let kv_dim = 4;
    let mut cache = make_cache(&backend, 64, 16, 1, 2, 2);

    cache.allocate(1, 8).unwrap();
    assert_eq!(cache.get_seq_len(1).unwrap(), 0);

    let k = backend.copy_from_host_f32(&vec![0.0; 3 * kv_dim], &[3, kv_dim]).unwrap();
    let v = backend.copy_from_host_f32(&vec![0.0; 3 * kv_dim], &[3, kv_dim]).unwrap();
    cache.append(1, 0, &k, &v).unwrap();
    assert_eq!(cache.get_seq_len(1).unwrap(), 3);

    let k1 = backend.copy_from_host_f32(&vec![0.0; kv_dim], &[1, kv_dim]).unwrap();
    let v1 = backend.copy_from_host_f32(&vec![0.0; kv_dim], &[1, kv_dim]).unwrap();
    cache.append(1, 0, &k1, &v1).unwrap();
    assert_eq!(cache.get_seq_len(1).unwrap(), 4);
}

/// Test: pool_dtype returns the configured dtype.
#[test]
fn test_pool_dtype() {
    let backend = CudaBackend::new(0).unwrap();
    let cache = make_cache(&backend, 4, 4, 1, 1, 2);
    assert_eq!(cache.pool_dtype(), DType::F32);
}

/// Test: F16 cache round-trip.
#[test]
fn test_f16_round_trip() {
    let backend = CudaBackend::new(0).unwrap();
    let num_kv_heads = 2;
    let head_dim = 4;
    let kv_dim = num_kv_heads * head_dim;

    let mut cache = GpuPagedKvCache::new(
        backend.clone(),
        64,
        16,
        1,
        num_kv_heads,
        head_dim,
        DType::F16,
    )
    .unwrap();

    assert_eq!(cache.pool_dtype(), DType::F16);

    cache.allocate(1, 4).unwrap();

    // Create F16 tensors
    let k_f32: Vec<f32> = (0..3 * kv_dim).map(|i| i as f32 * 0.1).collect();
    let v_f32: Vec<f32> = (0..3 * kv_dim).map(|i| 1.0 + i as f32 * 0.1).collect();
    let k_f16: Vec<half::f16> = k_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_f16: Vec<half::f16> = v_f32.iter().map(|&x| half::f16::from_f32(x)).collect();

    let k = backend.copy_from_host_f16(&k_f16, &[3, kv_dim]).unwrap();
    let v = backend.copy_from_host_f16(&v_f16, &[3, kv_dim]).unwrap();

    cache.append(1, 0, &k, &v).unwrap();

    let (k_out, _) = cache.get_kv(1, 0).unwrap();
    assert_eq!(k_out.shape(), &[3, kv_dim]);
    assert_eq!(k_out.dtype(), DType::F16);

    // Cast back to F32 to compare
    let k_out_f32 = backend.cast(&k_out, DType::F32).unwrap();
    let k_host = backend.copy_to_host_f32(&k_out_f32).unwrap();

    for i in 0..k_f32.len() {
        // F16 has less precision
        assert!(
            (k_host[i] - k_f32[i]).abs() < 0.01,
            "f16 k[{i}]: got {} expected {}",
            k_host[i],
            k_f32[i]
        );
    }
}

/// Test: paged_attention_meta returns valid metadata.
#[test]
fn test_paged_attention_meta() {
    let backend = CudaBackend::new(0).unwrap();
    let kv_dim = 4;
    let mut cache = make_cache(&backend, 64, 16, 2, 2, 2);

    cache.allocate(1, 4).unwrap();
    cache.allocate(2, 4).unwrap();

    // Append some tokens
    let k = backend.copy_from_host_f32(&vec![1.0; 3 * kv_dim], &[3, kv_dim]).unwrap();
    let v = backend.copy_from_host_f32(&vec![2.0; 3 * kv_dim], &[3, kv_dim]).unwrap();
    for layer in 0..2 {
        cache.append(1, layer, &k, &v).unwrap();
    }

    let k2 = backend.copy_from_host_f32(&vec![3.0; 5 * kv_dim], &[5, kv_dim]).unwrap();
    let v2 = backend.copy_from_host_f32(&vec![4.0; 5 * kv_dim], &[5, kv_dim]).unwrap();
    for layer in 0..2 {
        cache.append(2, layer, &k2, &v2).unwrap();
    }

    let seq_ids = [1, 2];
    let meta = cache.paged_attention_meta(&seq_ids, 0).unwrap();

    assert_eq!(meta.num_seqs, 2);
    assert_eq!(meta.block_size, 16);
    assert_eq!(meta.kv_dim, kv_dim);
    assert!(meta.block_tables_ptr != 0);
    assert!(meta.kv_lens_ptr != 0);
    assert!(meta.k_pool_ptr != 0);
    assert!(meta.v_pool_ptr != 0);

    // Meta for layer 1 should reuse cached block tables
    let meta1 = cache.paged_attention_meta(&seq_ids, 1).unwrap();
    assert_eq!(meta1.block_tables_ptr, meta.block_tables_ptr);
    assert_eq!(meta1.kv_lens_ptr, meta.kv_lens_ptr);
    // But pool pointers differ per layer
    assert_ne!(meta1.k_pool_ptr, meta.k_pool_ptr);
}

/// Test: supports_paged_attention returns true.
#[test]
fn test_supports_paged_attention() {
    let backend = CudaBackend::new(0).unwrap();
    let cache = make_cache(&backend, 4, 4, 1, 1, 2);
    assert!(cache.supports_paged_attention());
}
