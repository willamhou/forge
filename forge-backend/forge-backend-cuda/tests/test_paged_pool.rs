//! Integration test for CUDA paged_write_kv / paged_gather_kv overrides.
//!
//! Single test fn on purpose: GB10 has a known cuBLAS handle leak across
//! multiple CudaBackend::new() calls in the same test binary (task #4).
//! Keeping all assertions in one fn means one process → one backend → one
//! cuBLAS context.

use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, DType, Tensor};

#[test]
fn paged_pool_device_roundtrip() {
    let backend = CudaBackend::new(0).expect("CUDA backend init");

    // Pool shape: [4 blocks, 2 tokens/block, 3 dims/token] → 12 slots total.
    let mut pool = backend
        .allocate_zeros(&[4, 2, 3], DType::F32)
        .expect("pool alloc");

    // ── Test 1: contiguous prefill into one block ─────────────────────
    //
    // Write tokens [a,b,c,d] (each kv_dim=3) into slots [0,1,2,3]
    // which is block 0 (slots 0,1) and block 1 (slots 0,1).
    let src_a: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let src_a_t = backend
        .copy_from_host_f32(&src_a, &[4, 3])
        .expect("src_a upload");
    backend
        .paged_write_kv(&mut pool, &src_a_t, &[0, 1, 2, 3])
        .expect("paged_write_kv contiguous");
    backend.synchronize().unwrap();

    let pool_after_a = backend.copy_to_host_f32(&pool).expect("pool readback");
    assert_eq!(pool_after_a.len(), 24, "pool element count");
    assert_eq!(
        &pool_after_a[0..12],
        &src_a[..],
        "blocks 0+1 should hold src_a"
    );
    assert_eq!(&pool_after_a[12..24], &[0.0_f32; 12], "blocks 2+3 untouched");

    // ── Test 2: scattered write into non-contiguous slots ─────────────
    //
    // Write 3 tokens into slots [4, 9, 5] → block 2 slot 0, block 4? no, slot 9 is
    // block 4 which doesn't exist. Use [4, 7, 5] instead:
    //   slot 4 = block 2, position 0
    //   slot 7 = block 3, position 1
    //   slot 5 = block 2, position 1
    // Mapping is NOT monotonic → coalescing must NOT merge runs.
    let src_b: Vec<f32> = vec![100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 300.0, 301.0, 302.0];
    let src_b_t = backend
        .copy_from_host_f32(&src_b, &[3, 3])
        .expect("src_b upload");
    backend
        .paged_write_kv(&mut pool, &src_b_t, &[4, 7, 5])
        .expect("paged_write_kv scattered");
    backend.synchronize().unwrap();

    let pool_after_b = backend.copy_to_host_f32(&pool).expect("pool readback 2");
    // block 0+1 unchanged
    assert_eq!(&pool_after_b[0..12], &src_a[..]);
    // block 2 slot 0 = src_b token 0
    assert_eq!(&pool_after_b[12..15], &[100.0, 101.0, 102.0]);
    // block 2 slot 1 = src_b token 2
    assert_eq!(&pool_after_b[15..18], &[300.0, 301.0, 302.0]);
    // block 3 slot 0 untouched
    assert_eq!(&pool_after_b[18..21], &[0.0, 0.0, 0.0]);
    // block 3 slot 1 = src_b token 1
    assert_eq!(&pool_after_b[21..24], &[200.0, 201.0, 202.0]);

    // ── Test 3: gather a full sequence spanning multiple blocks ───────
    //
    // Read 4 tokens from blocks [0, 1] → reconstructs src_a.
    let gathered_full = backend
        .paged_gather_kv(&pool, &[0, 1], 4)
        .expect("paged_gather_kv full");
    assert_eq!(gathered_full.shape(), &[4, 3]);
    let g_full_host = backend.copy_to_host_f32(&gathered_full).unwrap();
    assert_eq!(g_full_host, src_a);

    // ── Test 4: gather with partial last block ────────────────────────
    //
    // Read 3 tokens from blocks [0, 1] → should give tokens 0,1,2 of src_a.
    let gathered_partial = backend
        .paged_gather_kv(&pool, &[0, 1], 3)
        .expect("paged_gather_kv partial");
    assert_eq!(gathered_partial.shape(), &[3, 3]);
    let g_partial_host = backend.copy_to_host_f32(&gathered_partial).unwrap();
    assert_eq!(g_partial_host, src_a[..9].to_vec());

    // ── Test 5: gather from non-adjacent blocks (block table indirection) ─
    //
    // Read 4 tokens from blocks [2, 3] → uses our scattered writes from test 2.
    let gathered_scatter = backend
        .paged_gather_kv(&pool, &[2, 3], 4)
        .expect("paged_gather_kv scatter");
    let g_scatter_host = backend.copy_to_host_f32(&gathered_scatter).unwrap();
    assert_eq!(
        g_scatter_host,
        vec![
            100.0, 101.0, 102.0, // block 2 slot 0 = src_b token 0
            300.0, 301.0, 302.0, // block 2 slot 1 = src_b token 2
            0.0, 0.0, 0.0, // block 3 slot 0 (untouched)
            200.0, 201.0, 202.0, // block 3 slot 1 = src_b token 1
        ]
    );

    // ── Test 6: empty slot_mapping is a no-op ─────────────────────────
    let src_empty = backend.copy_from_host_f32(&[], &[0, 3]).unwrap();
    backend
        .paged_write_kv(&mut pool, &src_empty, &[])
        .expect("paged_write_kv empty");
    let pool_after_empty = backend.copy_to_host_f32(&pool).unwrap();
    assert_eq!(pool_after_empty, pool_after_b);

    // ── Test 7: out-of-bounds slot is rejected ────────────────────────
    let bad = backend
        .paged_write_kv(&mut pool, &src_a_t, &[100]) // capacity is 8 slots
        .expect_err("oob slot should fail");
    let _ = bad; // any error variant is fine
}
