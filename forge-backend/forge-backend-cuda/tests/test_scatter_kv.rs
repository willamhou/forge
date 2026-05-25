//! Parity test: the capture-safe `scatter_kv` (device slot_mapping, kernel)
//! must produce byte-identical pool contents to the host-loop `paged_write_kv`.
//!
//! Single test fn on purpose: GB10 has a known cuBLAS handle leak across
//! multiple CudaBackend::new() calls in one binary (see test_paged_pool.rs).

use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, DType};

#[test]
fn scatter_kv_matches_host_loop() {
    let backend = CudaBackend::new(0).expect("CUDA backend init");

    // Pool shape: [4 blocks, 2 tokens/block, 3 dims/token] → 8 slots total.
    let shape = [4usize, 2, 3];

    // Helper: fresh zeroed F32 pool.
    let fresh_pool = || backend.allocate_zeros(&shape, DType::F32).expect("pool alloc");

    // ── Case 1: scattered (non-monotonic) slots [4, 7, 5] ────────────
    let src: Vec<f32> = vec![
        100.0, 101.0, 102.0, // token 0
        200.0, 201.0, 202.0, // token 1
        300.0, 301.0, 302.0, // token 2
    ];
    let src_t = backend.copy_from_host_f32(&src, &[3, 3]).expect("src upload");
    let slots = [4i32, 7, 5];

    let mut pool_ref = fresh_pool();
    backend
        .paged_write_kv(&mut pool_ref, &src_t, &slots)
        .expect("host-loop write");
    backend.synchronize().unwrap();
    let expected = backend.copy_to_host_f32(&pool_ref).unwrap();

    let mut pool_scatter = fresh_pool();
    let n = backend.stage_slot_mapping(&slots).expect("stage");
    assert_eq!(n, 3);
    backend
        .scatter_kv(&mut pool_scatter, &src_t, n)
        .expect("scatter_kv");
    backend.synchronize().unwrap();
    let got = backend.copy_to_host_f32(&pool_scatter).unwrap();

    assert_eq!(got, expected, "scattered scatter_kv must match host loop");

    // ── Case 2: contiguous slots [0,1,2,3] (4 tokens) ────────────────
    let src4: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let src4_t = backend.copy_from_host_f32(&src4, &[4, 3]).expect("src4 upload");
    let slots4 = [0i32, 1, 2, 3];

    let mut pool_ref2 = fresh_pool();
    backend.paged_write_kv(&mut pool_ref2, &src4_t, &slots4).unwrap();
    backend.synchronize().unwrap();
    let expected2 = backend.copy_to_host_f32(&pool_ref2).unwrap();

    let mut pool_scatter2 = fresh_pool();
    let n2 = backend.stage_slot_mapping(&slots4).unwrap();
    backend.scatter_kv(&mut pool_scatter2, &src4_t, n2).unwrap();
    backend.synchronize().unwrap();
    let got2 = backend.copy_to_host_f32(&pool_scatter2).unwrap();
    assert_eq!(got2, expected2, "contiguous scatter_kv must match host loop");

    // ── Case 3: scratch reuse after a larger stage (grow), then small ─
    // Stage a big slot_mapping to force the scratch to grow, then a small
    // one — scatter_kv must read only the first n_rows and ignore stale tail.
    let big: Vec<i32> = (0..64).map(|i| i % 8).collect();
    backend.stage_slot_mapping(&big).expect("stage big (grow)");
    let n3 = backend.stage_slot_mapping(&slots).expect("stage small");
    let mut pool_scatter3 = fresh_pool();
    backend.scatter_kv(&mut pool_scatter3, &src_t, n3).unwrap();
    backend.synchronize().unwrap();
    let got3 = backend.copy_to_host_f32(&pool_scatter3).unwrap();
    assert_eq!(got3, expected, "scatter_kv after grow must still match");

    // ── Case 4: n_rows = 0 is a no-op ────────────────────────────────
    let mut pool_zero = fresh_pool();
    backend.scatter_kv(&mut pool_zero, &src_t, 0).unwrap();
    backend.synchronize().unwrap();
    let got_zero = backend.copy_to_host_f32(&pool_zero).unwrap();
    assert_eq!(got_zero, vec![0.0_f32; 24], "n_rows=0 leaves pool untouched");

    // ── Case 5: F16 parity ───────────────────────────────────────────
    let src_f16_data: Vec<half::f16> =
        (0..9).map(|i| half::f16::from_f32(i as f32 * 0.5)).collect();
    let src_f16 = backend.copy_from_host_f16(&src_f16_data, &[3, 3]).unwrap();

    let mut pool_f16_ref = backend.allocate_zeros(&shape, DType::F16).unwrap();
    backend.paged_write_kv(&mut pool_f16_ref, &src_f16, &slots).unwrap();
    backend.synchronize().unwrap();
    let exp_f16 = backend.copy_to_host_f32(&backend.cast(&pool_f16_ref, DType::F32).unwrap()).unwrap();

    let mut pool_f16_scatter = backend.allocate_zeros(&shape, DType::F16).unwrap();
    let nf = backend.stage_slot_mapping(&slots).unwrap();
    backend.scatter_kv(&mut pool_f16_scatter, &src_f16, nf).unwrap();
    backend.synchronize().unwrap();
    let got_f16 = backend.copy_to_host_f32(&backend.cast(&pool_f16_scatter, DType::F32).unwrap()).unwrap();
    assert_eq!(got_f16, exp_f16, "F16 scatter_kv must match host loop");

    // ── Case 6: dtype mismatch rejected ──────────────────────────────
    let mut pool_f16_bad = backend.allocate_zeros(&shape, DType::F16).unwrap();
    backend.stage_slot_mapping(&slots).unwrap();
    let _ = backend
        .scatter_kv(&mut pool_f16_bad, &src_t, 3)
        .expect_err("F32 src into F16 pool must fail");
}
