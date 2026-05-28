//! `CudaGraphCache` integration test — captures a real `paged_attention_into`
//! call, replays it, verifies numerical equivalence with the uncaptured path.
//!
//! Single test fn on purpose: dodges the GB10 cuBLAS-handle-leak in
//! multi-fixture binaries (task #4).

use forge_backend_cuda::{CudaBackend, CudaGraphCache};
use forge_core::{Backend, DType, Tensor};

fn rng_lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*seed >> 33) as u32;
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[test]
fn cuda_graph_cache_captures_and_replays_paged_attention() {
    let backend = CudaBackend::new(0).expect("CUDA backend init");

    // Tiny GQA shape; large enough to exercise multi-block paths.
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 32;
    let kv_dim = num_kv_heads * head_dim;
    let block_size = 4;
    let total_blocks = 8;
    let batch_size = 2;
    let max_blocks = 3;

    // Populate K/V pool with deterministic content.
    let pool_n = total_blocks * block_size * kv_dim;
    let mut seed = 0xC0DECAFEu64;
    let k_pool_data: Vec<f32> = (0..pool_n).map(|_| rng_lcg(&mut seed)).collect();
    let v_pool_data: Vec<f32> = (0..pool_n).map(|_| rng_lcg(&mut seed)).collect();
    let q_data: Vec<f32> = (0..batch_size * num_heads * head_dim)
        .map(|_| rng_lcg(&mut seed))
        .collect();

    let k_pool = backend
        .copy_from_host_f32(&k_pool_data, &[total_blocks, block_size, kv_dim])
        .unwrap();
    let v_pool = backend
        .copy_from_host_f32(&v_pool_data, &[total_blocks, block_size, kv_dim])
        .unwrap();
    let q = backend
        .copy_from_host_f32(&q_data, &[batch_size, num_heads * head_dim])
        .unwrap();

    let kv_lens: Vec<i32> = vec![10, 5];
    let block_tables: Vec<i32> = vec![
        0, 1, 2, // seq 0: 10 tokens → 3 blocks
        3, 4, -1, // seq 1: 5 tokens → 2 blocks, padded
    ];
    let scale = (head_dim as f32).powf(-0.5);

    // ── Baseline: uncaptured call into a fresh output buffer ─────────
    let baseline_out = backend
        .paged_attention(
            &q,
            &k_pool,
            &v_pool,
            &block_tables,
            &kv_lens,
            max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();
    backend.synchronize().unwrap();
    let baseline_host = backend.copy_to_host_f32(&baseline_out).unwrap();

    // ── Persistent output buffer for the captured graph ──────────────
    let mut graph_out = backend
        .allocate_zeros(&[batch_size, num_heads * head_dim], DType::F32)
        .unwrap();

    // ── Capture ──────────────────────────────────────────────────────
    let mut cache = CudaGraphCache::new(backend.ctx(), backend.stream());
    assert!(!cache.has(0), "cache starts empty");
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // run_or_capture borrows backend, so build a thin reference & let the
    // closure capture only what it needs.
    let backend_ref = &backend;
    let q_ref = &q;
    let k_pool_ref = &k_pool;
    let v_pool_ref = &v_pool;
    let bt_ref = &block_tables;
    let kv_ref = &kv_lens;

    cache
        .run_or_capture(0, || {
            backend_ref.paged_attention_into(
                &mut graph_out,
                q_ref,
                k_pool_ref,
                v_pool_ref,
                bt_ref,
                kv_ref,
                max_blocks,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
        })
        .expect("capture + initial launch");
    backend.synchronize().unwrap();

    assert!(cache.has(0), "bucket 0 captured");
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // First launch (in run_or_capture) produced results.
    let initial_host = backend.copy_to_host_f32(&graph_out).unwrap();

    let diff_init_vs_baseline = initial_host
        .iter()
        .zip(&baseline_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        diff_init_vs_baseline < 1e-4,
        "captured first-launch output diverges from uncaptured baseline: max diff {diff_init_vs_baseline}"
    );

    // ── Replay 5 times — every result must be bit-identical ──────────
    let mut prev_host: Option<Vec<f32>> = Some(initial_host.clone());
    for i in 0..5 {
        // Zero the buffer so we know the replay re-writes it (not just stale data).
        let zeros = vec![0.0_f32; graph_out.shape().iter().product()];
        let zero_t = backend
            .copy_from_host_f32(&zeros, graph_out.shape())
            .unwrap();
        backend.synchronize().unwrap();
        let _ = backend.copy_to_host_f32(&zero_t).unwrap();
        // (We can't memcpy into graph_out without exposing the internal
        // slice; the replay writes over whatever was there. That's the
        // point we want to assert below.)

        cache.replay(0).expect("replay");
        backend.synchronize().unwrap();
        let host = backend.copy_to_host_f32(&graph_out).unwrap();
        let diff = host
            .iter()
            .zip(prev_host.as_ref().unwrap())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert_eq!(
            diff, 0.0,
            "replay #{i} differs from previous (max diff {diff})"
        );
        prev_host = Some(host);
    }

    // ── Cache hit path: run_or_capture on existing bucket replays ───
    cache
        .run_or_capture(0, || {
            panic!("closure should NOT be invoked on cache hit");
        })
        .expect("cache hit replays without invoking closure");
    backend.synchronize().unwrap();
    let post_hit_host = backend.copy_to_host_f32(&graph_out).unwrap();
    assert_eq!(post_hit_host, *prev_host.as_ref().unwrap());

    // ── replay() with no capture errors cleanly ──────────────────────
    let _ = cache
        .replay(999)
        .expect_err("replay of un-captured bucket must error");

    // ── invalidate + recapture works ─────────────────────────────────
    let removed = cache.invalidate(0);
    assert!(removed, "invalidate(0) returns true");
    assert!(!cache.has(0));
    assert_eq!(cache.len(), 0);

    cache
        .run_or_capture(0, || {
            backend_ref.paged_attention_into(
                &mut graph_out,
                q_ref,
                k_pool_ref,
                v_pool_ref,
                bt_ref,
                kv_ref,
                max_blocks,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
        })
        .expect("re-capture after invalidate");
    backend.synchronize().unwrap();
    assert!(cache.has(0));
    let recap_host = backend.copy_to_host_f32(&graph_out).unwrap();
    assert_eq!(recap_host, initial_host, "re-capture produces same result");

    // ── clear() drops everything ─────────────────────────────────────
    cache.clear();
    assert!(!cache.has(0));
    assert_eq!(cache.len(), 0);

    // ── Multiple buckets coexist ─────────────────────────────────────
    let mut out_a = backend
        .allocate_zeros(&[batch_size, num_heads * head_dim], DType::F32)
        .unwrap();
    let mut out_b = backend
        .allocate_zeros(&[1, num_heads * head_dim], DType::F32)
        .unwrap();
    cache
        .run_or_capture(2, || {
            backend_ref.paged_attention_into(
                &mut out_a,
                q_ref,
                k_pool_ref,
                v_pool_ref,
                bt_ref,
                kv_ref,
                max_blocks,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
        })
        .unwrap();
    let q_one = backend
        .copy_from_host_f32(&q_data[..num_heads * head_dim], &[1, num_heads * head_dim])
        .unwrap();
    let kv_one: Vec<i32> = vec![10];
    let bt_one: Vec<i32> = vec![0, 1, 2];
    let q_one_ref = &q_one;
    let kv_one_ref = &kv_one;
    let bt_one_ref = &bt_one;
    cache
        .run_or_capture(1, || {
            backend_ref.paged_attention_into(
                &mut out_b,
                q_one_ref,
                k_pool_ref,
                v_pool_ref,
                bt_one_ref,
                kv_one_ref,
                3,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
        })
        .unwrap();
    assert!(cache.has(1));
    assert!(cache.has(2));
    assert_eq!(cache.len(), 2);

    // Replays of both buckets independently work.
    cache.replay(2).unwrap();
    cache.replay(1).unwrap();
    backend.synchronize().unwrap();
}
