use forge_backend_cuda::{CudaBackend, GpuPagedKvCache};
use forge_core::{Backend, DType, KvCache, Tensor};

/// Test: paged decode attention matches batched decode attention output.
///
/// Sets up identical KV data in both a GpuPagedKvCache (for paged path) and
/// contiguous tensors (for batched path), then compares outputs.
#[test]
fn test_paged_decode_matches_batched_decode() {
    let backend = CudaBackend::new(0).unwrap();
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Two sequences with different KV lengths
    let kv_len_1 = 5;
    let kv_len_2 = 3;

    // Generate deterministic KV data
    let k1_data: Vec<f32> = (0..kv_len_1 * kv_dim)
        .map(|i| ((i as f32 * 0.017) % 1.0) - 0.5)
        .collect();
    let v1_data: Vec<f32> = (0..kv_len_1 * kv_dim)
        .map(|i| ((i as f32 * 0.013) % 1.0) - 0.5)
        .collect();
    let k2_data: Vec<f32> = (0..kv_len_2 * kv_dim)
        .map(|i| ((i as f32 * 0.023) % 1.0) - 0.5)
        .collect();
    let v2_data: Vec<f32> = (0..kv_len_2 * kv_dim)
        .map(|i| ((i as f32 * 0.019) % 1.0) - 0.5)
        .collect();

    // Q for decode: 1 token per sequence = [2, num_heads * head_dim]
    let q_data: Vec<f32> = (0..2 * num_heads * head_dim)
        .map(|i| ((i as f32 * 0.031) % 1.0) - 0.5)
        .collect();
    let q = backend
        .copy_from_host_f32(&q_data, &[2, num_heads * head_dim])
        .unwrap();

    // === Paged path ===
    let mut cache = GpuPagedKvCache::new(
        backend.clone(),
        64,
        4, // small block_size to test multi-block
        1, // single layer
        num_kv_heads,
        head_dim,
        DType::F32,
    )
    .unwrap();

    cache.allocate(1, kv_len_1).unwrap();
    cache.allocate(2, kv_len_2).unwrap();

    let k1 = backend
        .copy_from_host_f32(&k1_data, &[kv_len_1, kv_dim])
        .unwrap();
    let v1 = backend
        .copy_from_host_f32(&v1_data, &[kv_len_1, kv_dim])
        .unwrap();
    let k2 = backend
        .copy_from_host_f32(&k2_data, &[kv_len_2, kv_dim])
        .unwrap();
    let v2 = backend
        .copy_from_host_f32(&v2_data, &[kv_len_2, kv_dim])
        .unwrap();

    cache.append(1, 0, &k1, &v1).unwrap();
    cache.append(2, 0, &k2, &v2).unwrap();

    let meta = cache.paged_attention_meta(&[1, 2], 0).unwrap();
    let paged_out = backend
        .paged_decode_attention(&q, &meta, num_heads, num_kv_heads, head_dim, scale)
        .unwrap();
    let paged_host = backend.copy_to_host_f32(&paged_out).unwrap();

    // === Batched path (reference) ===
    // Gather contiguous KV per sequence
    let (k1_full, v1_full) = cache.get_kv(1, 0).unwrap();
    let (k2_full, v2_full) = cache.get_kv(2, 0).unwrap();

    let batched_out = backend
        .batched_decode_attention(
            &q,
            &[k1_full, k2_full],
            &[v1_full, v2_full],
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();
    let batched_host = backend.copy_to_host_f32(&batched_out).unwrap();

    // Compare outputs
    assert_eq!(paged_out.shape(), batched_out.shape());
    for i in 0..paged_host.len() {
        assert!(
            (paged_host[i] - batched_host[i]).abs() < 1e-3,
            "paged vs batched mismatch at {i}: paged={} batched={} diff={}",
            paged_host[i],
            batched_host[i],
            (paged_host[i] - batched_host[i]).abs()
        );
    }
}

/// Test: paged decode with single sequence.
#[test]
fn test_paged_decode_single_seq() {
    let backend = CudaBackend::new(0).unwrap();
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 4;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_len = 4;

    let mut cache = GpuPagedKvCache::new(
        backend.clone(),
        16,
        4,
        1,
        num_kv_heads,
        head_dim,
        DType::F32,
    )
    .unwrap();

    cache.allocate(1, kv_len).unwrap();

    let k_data: Vec<f32> = (0..kv_len * kv_dim)
        .map(|i| ((i as f32 * 0.1) % 1.0) - 0.5)
        .collect();
    let v_data: Vec<f32> = (0..kv_len * kv_dim)
        .map(|i| ((i as f32 * 0.07) % 1.0) - 0.5)
        .collect();
    let k = backend
        .copy_from_host_f32(&k_data, &[kv_len, kv_dim])
        .unwrap();
    let v = backend
        .copy_from_host_f32(&v_data, &[kv_len, kv_dim])
        .unwrap();
    cache.append(1, 0, &k, &v).unwrap();

    let q_data: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| ((i as f32 * 0.03) % 1.0) - 0.5)
        .collect();
    let q = backend
        .copy_from_host_f32(&q_data, &[1, num_heads * head_dim])
        .unwrap();

    let meta = cache.paged_attention_meta(&[1], 0).unwrap();
    let paged_out = backend
        .paged_decode_attention(&q, &meta, num_heads, num_kv_heads, head_dim, scale)
        .unwrap();

    // Compare with batched
    let (k_full, v_full) = cache.get_kv(1, 0).unwrap();
    let batched_out = backend
        .batched_decode_attention(
            &q,
            &[k_full],
            &[v_full],
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

    let paged_host = backend.copy_to_host_f32(&paged_out).unwrap();
    let batched_host = backend.copy_to_host_f32(&batched_out).unwrap();

    for i in 0..paged_host.len() {
        assert!(
            (paged_host[i] - batched_host[i]).abs() < 1e-3,
            "single-seq paged vs batched at {i}: {} vs {}",
            paged_host[i],
            batched_host[i]
        );
    }
}

/// Test: GQA (grouped query attention) with paged path.
#[test]
fn test_paged_decode_gqa() {
    let backend = CudaBackend::new(0).unwrap();
    // 4 Q heads, 2 KV heads (GQA ratio = 2)
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_len = 6;

    let mut cache = GpuPagedKvCache::new(
        backend.clone(),
        32,
        4,
        1,
        num_kv_heads,
        head_dim,
        DType::F32,
    )
    .unwrap();

    cache.allocate(1, kv_len).unwrap();

    let k_data: Vec<f32> = (0..kv_len * kv_dim)
        .map(|i| ((i as f32 * 0.017) % 1.0) - 0.5)
        .collect();
    let v_data: Vec<f32> = (0..kv_len * kv_dim)
        .map(|i| ((i as f32 * 0.013) % 1.0) - 0.5)
        .collect();
    let k = backend
        .copy_from_host_f32(&k_data, &[kv_len, kv_dim])
        .unwrap();
    let v = backend
        .copy_from_host_f32(&v_data, &[kv_len, kv_dim])
        .unwrap();
    cache.append(1, 0, &k, &v).unwrap();

    let q_data: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| ((i as f32 * 0.031) % 1.0) - 0.5)
        .collect();
    let q = backend
        .copy_from_host_f32(&q_data, &[1, num_heads * head_dim])
        .unwrap();

    let meta = cache.paged_attention_meta(&[1], 0).unwrap();
    let paged_out = backend
        .paged_decode_attention(&q, &meta, num_heads, num_kv_heads, head_dim, scale)
        .unwrap();

    let (k_full, v_full) = cache.get_kv(1, 0).unwrap();
    let batched_out = backend
        .batched_decode_attention(
            &q,
            &[k_full],
            &[v_full],
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

    let paged_host = backend.copy_to_host_f32(&paged_out).unwrap();
    let batched_host = backend.copy_to_host_f32(&batched_out).unwrap();

    assert_eq!(paged_out.shape(), &[1, num_heads * head_dim]);
    for i in 0..paged_host.len() {
        assert!(
            (paged_host[i] - batched_host[i]).abs() < 1e-3,
            "GQA paged vs batched at {i}: {} vs {} diff={}",
            paged_host[i],
            batched_host[i],
            (paged_host[i] - batched_host[i]).abs()
        );
    }
}

/// Test: block table cache is reused across layers.
#[test]
fn test_block_table_cache_reuse() {
    let backend = CudaBackend::new(0).unwrap();
    let kv_dim = 4;
    let num_layers = 4;
    let mut cache = GpuPagedKvCache::new(
        backend.clone(),
        64,
        16,
        num_layers,
        2,
        2,
        DType::F32,
    )
    .unwrap();

    cache.allocate(1, 4).unwrap();

    let k = backend.copy_from_host_f32(&vec![1.0; 2 * kv_dim], &[2, kv_dim]).unwrap();
    let v = backend.copy_from_host_f32(&vec![2.0; 2 * kv_dim], &[2, kv_dim]).unwrap();
    for layer in 0..num_layers {
        cache.append(1, layer, &k, &v).unwrap();
    }

    // Get meta for all layers — block table pointer should be the same
    let meta0 = cache.paged_attention_meta(&[1], 0).unwrap();
    let bt_ptr = meta0.block_tables_ptr;
    let kl_ptr = meta0.kv_lens_ptr;

    for layer in 1..num_layers {
        let meta = cache.paged_attention_meta(&[1], layer).unwrap();
        assert_eq!(
            meta.block_tables_ptr, bt_ptr,
            "layer {layer}: block_tables_ptr should be cached"
        );
        assert_eq!(
            meta.kv_lens_ptr, kl_ptr,
            "layer {layer}: kv_lens_ptr should be cached"
        );
    }
}
