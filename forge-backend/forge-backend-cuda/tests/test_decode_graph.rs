//! `DecodeGraphRunner` integration test.
//!
//! Exercises the dispatch logic (bucket match → capture/replay; non-bucket
//! → fallback) using `paged_attention_into` as the capturable unit (the only
//! kernel currently capture-safe end-to-end; Task 5 will extend this to the
//! full decode forward pass).
//!
//! Single test fn — GB10 cuBLAS-handle-leak workaround (task #4).

use forge_backend_cuda::{CudaBackend, DecodeGraphRunner};
use forge_core::{Backend, DType};

fn rng(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (*seed >> 33) as u32;
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[test]
fn decode_graph_runner_dispatches_buckets() {
    let backend = CudaBackend::new(0).expect("CUDA backend");

    // GQA tiny shape; KV pool populated deterministically.
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 32;
    let kv_dim = num_kv_heads * head_dim;
    let block_size = 4;
    let total_blocks = 8;
    let max_blocks = 3;

    let mut seed = 0xBEEFu64;
    let pool_n = total_blocks * block_size * kv_dim;
    let k_data: Vec<f32> = (0..pool_n).map(|_| rng(&mut seed)).collect();
    let v_data: Vec<f32> = (0..pool_n).map(|_| rng(&mut seed)).collect();

    let k_pool = backend
        .copy_from_host_f32(&k_data, &[total_blocks, block_size, kv_dim])
        .unwrap();
    let v_pool = backend
        .copy_from_host_f32(&v_data, &[total_blocks, block_size, kv_dim])
        .unwrap();

    let scale = (head_dim as f32).powf(-0.5);

    // ── Runner with buckets [1, 2, 4] (de-dup'd + sorted from user input) ──
    let mut runner = DecodeGraphRunner::with_buckets(
        backend.ctx(),
        backend.stream(),
        vec![4, 2, 1, 4], // intentional duplicates / unsorted
    );
    assert_eq!(runner.buckets(), &[1, 2, 4]);
    assert!(runner.is_bucket(2));
    assert!(!runner.is_bucket(3));
    assert!(!runner.is_bucket(5));
    assert_eq!(runner.captured_count(), 0);

    // Build batch=2 inputs.
    let q_data_b2: Vec<f32> = (0..2 * num_heads * head_dim).map(|_| rng(&mut seed)).collect();
    let q_b2 = backend
        .copy_from_host_f32(&q_data_b2, &[2, num_heads * head_dim])
        .unwrap();
    let kv_lens_b2: Vec<i32> = vec![10, 5];
    let bt_b2: Vec<i32> = vec![0, 1, 2, 3, 4, -1];
    let mut out_b2 = backend
        .allocate_zeros(&[2, num_heads * head_dim], DType::F32)
        .unwrap();

    // ── Bucket exact match → captures + launches ─────────────────────
    runner
        .dispatch(2, || {
            backend.paged_attention_into(
                &mut out_b2,
                &q_b2,
                &k_pool,
                &v_pool,
                &bt_b2,
                &kv_lens_b2,
                max_blocks,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
        })
        .expect("dispatch batch=2 (bucket hit)");
    backend.synchronize().unwrap();
    assert_eq!(runner.captured_count(), 1, "batch=2 captured");

    let captured_host = backend.copy_to_host_f32(&out_b2).unwrap();

    // ── Bucket replay path: second dispatch must NOT invoke the closure ──
    runner
        .dispatch(2, || {
            panic!("closure should NOT run on bucket cache hit");
        })
        .expect("dispatch batch=2 (replay)");
    backend.synchronize().unwrap();
    let replayed_host = backend.copy_to_host_f32(&out_b2).unwrap();
    assert_eq!(replayed_host, captured_host, "replay reproduces capture");
    assert_eq!(runner.captured_count(), 1, "still one bucket");

    // ── Non-bucket batch size → fallback (closure runs, no capture) ─
    let q_data_b3: Vec<f32> = (0..3 * num_heads * head_dim).map(|_| rng(&mut seed)).collect();
    let q_b3 = backend
        .copy_from_host_f32(&q_data_b3, &[3, num_heads * head_dim])
        .unwrap();
    let kv_lens_b3: Vec<i32> = vec![10, 5, 8];
    let bt_b3: Vec<i32> = vec![0, 1, 2, 3, 4, -1, 5, 6, -1];
    let mut out_b3 = backend
        .allocate_zeros(&[3, num_heads * head_dim], DType::F32)
        .unwrap();

    let mut closure_ran = false;
    {
        let closure_ran_ref = &mut closure_ran;
        runner
            .dispatch(3, || {
                *closure_ran_ref = true;
                backend.paged_attention_into(
                    &mut out_b3,
                    &q_b3,
                    &k_pool,
                    &v_pool,
                    &bt_b3,
                    &kv_lens_b3,
                    max_blocks,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                )
            })
            .expect("dispatch batch=3 (fallback)");
    }
    backend.synchronize().unwrap();
    assert!(closure_ran, "fallback path must invoke closure");
    assert_eq!(
        runner.captured_count(),
        1,
        "non-bucket dispatch must not capture"
    );

    // ── Two distinct buckets coexist ─────────────────────────────────
    let q_data_b1: Vec<f32> = (0..num_heads * head_dim).map(|_| rng(&mut seed)).collect();
    let q_b1 = backend
        .copy_from_host_f32(&q_data_b1, &[1, num_heads * head_dim])
        .unwrap();
    let kv_lens_b1: Vec<i32> = vec![10];
    let bt_b1: Vec<i32> = vec![0, 1, 2];
    let mut out_b1 = backend
        .allocate_zeros(&[1, num_heads * head_dim], DType::F32)
        .unwrap();

    runner
        .dispatch(1, || {
            backend.paged_attention_into(
                &mut out_b1,
                &q_b1,
                &k_pool,
                &v_pool,
                &bt_b1,
                &kv_lens_b1,
                max_blocks,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
        })
        .expect("dispatch batch=1 (bucket hit, fresh capture)");
    backend.synchronize().unwrap();
    assert_eq!(runner.captured_count(), 2);

    // Replays of both buckets independently.
    runner
        .dispatch(1, || panic!("should replay"))
        .unwrap();
    runner
        .dispatch(2, || panic!("should replay"))
        .unwrap();
    backend.synchronize().unwrap();

    // ── invalidate(bucket) drops just that bucket ───────────────────
    assert!(runner.invalidate(2));
    assert!(!runner.invalidate(2), "second invalidate returns false");
    assert_eq!(runner.captured_count(), 1, "only batch=1 remains");
    // batch=1 still replays.
    runner
        .dispatch(1, || panic!("batch=1 still cached"))
        .unwrap();

    // ── invalidate_all drops the rest ───────────────────────────────
    runner.invalidate_all();
    assert_eq!(runner.captured_count(), 0);

    // After full invalidation, next dispatch on a bucket recaptures.
    runner
        .dispatch(2, || {
            backend.paged_attention_into(
                &mut out_b2,
                &q_b2,
                &k_pool,
                &v_pool,
                &bt_b2,
                &kv_lens_b2,
                max_blocks,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
        })
        .expect("re-capture after invalidate_all");
    backend.synchronize().unwrap();
    assert_eq!(runner.captured_count(), 1);

    // Empty bucket list: every dispatch is fallback.
    let mut no_buckets = DecodeGraphRunner::with_buckets(backend.ctx(), backend.stream(), vec![]);
    assert!(no_buckets.buckets().is_empty());
    assert!(!no_buckets.is_bucket(0));
    assert!(!no_buckets.is_bucket(1));
    let mut closure_ran = false;
    {
        let closure_ran_ref = &mut closure_ran;
        no_buckets
            .dispatch(4, || {
                *closure_ran_ref = true;
                Ok(())
            })
            .unwrap();
    }
    assert!(closure_ran);
    assert_eq!(no_buckets.captured_count(), 0);
}
