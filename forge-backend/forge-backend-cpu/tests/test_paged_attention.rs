//! Reference test for `Backend::paged_attention` default impl.
//!
//! Verifies the host-fallback path produces the same logits as gathering the
//! K/V manually and running `multi_head_attention` on the result. This is the
//! reference CUDA Task 2.2's kernel will be compared against.

use forge_backend_cpu::CpuBackend;
use forge_core::{Backend, DType, Tensor};
use forge_kvcache::paged_cache::PagedKvCache;
use forge_core::KvCache;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol || ((x - y).abs() / x.abs().max(y.abs()).max(1e-6)) <= tol)
}

#[test]
fn paged_attention_matches_gather_then_attend_gqa() {
    // Small but non-trivial GQA shape:
    //   num_heads = 4, num_kv_heads = 2 (group size 2), head_dim = 4
    //   2 sequences in the batch
    //   block_size = 2, total_blocks = 8
    let backend = CpuBackend::new();
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let kv_dim = num_kv_heads * head_dim;
    let block_size = 2;
    let total_blocks = 8;
    let num_layers = 1;
    let layer = 0;

    let mut cache = PagedKvCache::new(
        backend.clone(),
        total_blocks,
        block_size,
        num_layers,
        num_kv_heads,
        head_dim,
        DType::F32,
    )
    .unwrap();

    // Seq 1: 5 tokens → 3 blocks; deterministic K/V values
    let seq1_len = 5;
    let k1: Vec<f32> = (0..seq1_len * kv_dim).map(|i| (i as f32) * 0.01).collect();
    let v1: Vec<f32> = (0..seq1_len * kv_dim)
        .map(|i| (i as f32) * 0.02 + 0.5)
        .collect();
    cache.allocate(1, seq1_len).unwrap();
    cache
        .append(
            1,
            layer,
            &backend.copy_from_host_f32(&k1, &[seq1_len, kv_dim]).unwrap(),
            &backend.copy_from_host_f32(&v1, &[seq1_len, kv_dim]).unwrap(),
        )
        .unwrap();

    // Seq 2: 3 tokens → 2 blocks
    let seq2_len = 3;
    let k2: Vec<f32> = (0..seq2_len * kv_dim)
        .map(|i| (i as f32) * 0.03 - 0.2)
        .collect();
    let v2: Vec<f32> = (0..seq2_len * kv_dim)
        .map(|i| (i as f32) * 0.04 + 0.1)
        .collect();
    cache.allocate(2, seq2_len).unwrap();
    cache
        .append(
            2,
            layer,
            &backend.copy_from_host_f32(&k2, &[seq2_len, kv_dim]).unwrap(),
            &backend.copy_from_host_f32(&v2, &[seq2_len, kv_dim]).unwrap(),
        )
        .unwrap();

    // Query: one row per seq, shape [batch, num_heads * head_dim]
    let q_data: Vec<f32> = (0..2 * num_heads * head_dim)
        .map(|i| (i as f32) * 0.05 - 0.3)
        .collect();
    let q = backend
        .copy_from_host_f32(&q_data, &[2, num_heads * head_dim])
        .unwrap();

    // Batch metadata
    let (block_tables, kv_lens, max_blocks_per_seq) =
        cache.batch_block_tables(&[1, 2]).unwrap();
    assert_eq!(kv_lens, vec![seq1_len as i32, seq2_len as i32]);

    // ── Run paged_attention ──────────────────────────────────────────
    let scale = (head_dim as f32).powf(-0.5);
    let out_paged = backend
        .paged_attention(
            &q,
            cache.k_pool(layer).unwrap(),
            cache.v_pool(layer).unwrap(),
            &block_tables,
            &kv_lens,
            max_blocks_per_seq,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

    // ── Reference: paged_gather_kv per seq, then batched_decode_attention ──
    let mut k_caches = Vec::new();
    let mut v_caches = Vec::new();
    for &id in &[1u64, 2] {
        let blocks = cache.get_block_table(id).unwrap();
        let len = cache.get_seq_len(id).unwrap();
        let k = backend
            .paged_gather_kv(cache.k_pool(layer).unwrap(), &blocks, len)
            .unwrap();
        let v = backend
            .paged_gather_kv(cache.v_pool(layer).unwrap(), &blocks, len)
            .unwrap();
        k_caches.push(k);
        v_caches.push(v);
    }
    let out_ref = backend
        .batched_decode_attention(
            &q,
            &k_caches,
            &v_caches,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

    let h_paged = backend.copy_to_host_f32(&out_paged).unwrap();
    let h_ref = backend.copy_to_host_f32(&out_ref).unwrap();

    assert_eq!(out_paged.shape(), out_ref.shape());
    assert_eq!(out_paged.shape(), &[2, num_heads * head_dim]);
    assert!(
        approx_eq(&h_paged, &h_ref, 1e-5),
        "paged_attention vs reference mismatch:\n  paged = {:?}\n  ref   = {:?}",
        h_paged,
        h_ref
    );
}

#[test]
fn paged_attention_validates_inputs() {
    let backend = CpuBackend::new();
    let pool = backend
        .allocate_zeros(&[4, 2, 8], DType::F32) // 4 blocks, 2 tokens/block, kv_dim=8
        .unwrap();
    let q = backend
        .copy_from_host_f32(&vec![0.0; 16], &[1, 16]) // batch=1, num_heads*head_dim=16
        .unwrap();

    // num_heads=4, num_kv_heads=2, head_dim=4 → kv_dim=8 (matches pool)
    let scale = 0.5;

    // 1. block_tables length must equal batch * max_blocks_per_seq
    assert!(backend
        .paged_attention(&q, &pool, &pool, &[0, -1, 0], &[1], 2, 4, 2, 4, scale)
        .is_err());

    // 2. pool kv_dim mismatch (claim head_dim=2 → expected kv_dim=4, pool is 8)
    assert!(backend
        .paged_attention(&q, &pool, &pool, &[0, -1], &[1], 2, 4, 2, 2, scale)
        .is_err());

    // 3. negative kv_len
    assert!(backend
        .paged_attention(&q, &pool, &pool, &[0, -1], &[-1], 2, 4, 2, 4, scale)
        .is_err());

    // 4. num_heads not divisible by num_kv_heads
    assert!(backend
        .paged_attention(&q, &pool, &pool, &[0, -1], &[1], 2, 5, 2, 4, scale)
        .is_err());

    // 5. num_kv_heads == 0 must error cleanly, NOT panic on the `num_heads %
    //    num_kv_heads` divide-by-zero (Codex review on PR #5).
    assert!(backend
        .paged_attention(&q, &pool, &pool, &[0, -1], &[1], 2, 4, 0, 4, scale)
        .is_err());

    // 6. Valid call succeeds (1 seq, 1 block, 1 token, all-zeros)
    let _ = backend
        .paged_attention(&q, &pool, &pool, &[0, -1], &[1], 2, 4, 2, 4, scale)
        .unwrap();
}
