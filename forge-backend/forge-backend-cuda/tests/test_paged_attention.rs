//! CUDA paged_attention kernel — F32 numerical equivalence test.
//!
//! Runs the CUDA kernel on a representative GQA shape and verifies the output
//! matches the host-fallback default impl (gather-per-seq + batched_decode_attention).
//!
//! Single test fn to dodge the GB10 cuBLAS fixture leak (task #4).

use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, DType, Tensor};

fn rng_lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (*seed >> 33) as u32;
    // Map to roughly (-1, 1)
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[test]
fn paged_attention_cuda_matches_default_impl() {
    let backend = CudaBackend::new(0).expect("CUDA backend");

    // Shape: GQA (4 query heads, 2 KV heads, group size 2), head_dim 32.
    // 2 sequences in the batch with different KV lengths spanning multiple blocks.
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 32;
    let kv_dim = num_kv_heads * head_dim;
    let block_size = 4;
    let total_blocks = 8;
    let batch_size = 2;

    // Sequence lengths chosen to (a) cross a block boundary and (b) not be a multiple of block_size.
    let kv_lens: Vec<i32> = vec![10, 5];
    // Block layout: seq 0 uses blocks [0, 1, 2] (10 tokens in 3 blocks of size 4),
    // seq 1 uses blocks [3, 4] (5 tokens in 2 blocks). max_blocks_per_seq = 3.
    let max_blocks = 3;
    let block_tables: Vec<i32> = vec![
        0, 1, 2, // seq 0
        3, 4, -1, // seq 1, padded
    ];

    // Deterministic K/V pool — populate every slot, even unused ones.
    let pool_n = total_blocks * block_size * kv_dim;
    let mut seed = 0xC0FFEEu64;
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

    let scale = (head_dim as f32).powf(-0.5);

    // ── CUDA kernel path ───────────────────────────────────────────
    let out_cuda = backend
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
    let out_cuda_host = backend.copy_to_host_f32(&out_cuda).unwrap();

    // ── Reference: gather per seq via paged_gather_kv + batched_decode_attention ──
    let mut k_caches = Vec::new();
    let mut v_caches = Vec::new();
    for b in 0..batch_size {
        let row = &block_tables[b * max_blocks..(b + 1) * max_blocks];
        let block_ids: Vec<usize> = row
            .iter()
            .take_while(|&&id| id >= 0)
            .map(|&id| id as usize)
            .collect();
        let kv_len = kv_lens[b] as usize;
        k_caches.push(backend.paged_gather_kv(&k_pool, &block_ids, kv_len).unwrap());
        v_caches.push(backend.paged_gather_kv(&v_pool, &block_ids, kv_len).unwrap());
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
    backend.synchronize().unwrap();
    let out_ref_host = backend.copy_to_host_f32(&out_ref).unwrap();

    assert_eq!(out_cuda.shape(), &[batch_size, num_heads * head_dim]);
    assert_eq!(out_cuda.shape(), out_ref.shape());
    assert_eq!(out_cuda_host.len(), out_ref_host.len());

    let max_abs_diff = out_cuda_host
        .iter()
        .zip(&out_ref_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_abs_val = out_ref_host.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let tol = 1e-4_f32.max(max_abs_val * 1e-4);
    assert!(
        max_abs_diff < tol,
        "max abs diff {max_abs_diff} > tol {tol} (max abs val in ref: {max_abs_val})\n  cuda[0..8] = {:?}\n  ref[0..8]  = {:?}",
        &out_cuda_host[..out_cuda_host.len().min(8)],
        &out_ref_host[..out_ref_host.len().min(8)],
    );

    // ── Single-batch sanity check ─────────────────────────────────
    // batch_size = 1 should reuse the kernel with grid_dim.x = 1.
    let q1 = backend
        .copy_from_host_f32(
            &q_data[..num_heads * head_dim],
            &[1, num_heads * head_dim],
        )
        .unwrap();
    let out1 = backend
        .paged_attention(
            &q1,
            &k_pool,
            &v_pool,
            &block_tables[..max_blocks],
            &kv_lens[..1],
            max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();
    assert_eq!(out1.shape(), &[1, num_heads * head_dim]);
    let out1_host = backend.copy_to_host_f32(&out1).unwrap();
    // First batch_size=1 row should equal the corresponding row of out_cuda
    let diff1 = out1_host
        .iter()
        .zip(out_cuda_host.iter().take(num_heads * head_dim))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        diff1 < 1e-5,
        "batch=1 result differs from batch=2 first row: max diff {diff1}"
    );

    // ── Rejection: non-F32 dtype unsupported ──────────────────────
    let _ = backend
        .paged_attention(
            &q,
            &k_pool,
            &v_pool,
            &block_tables,
            &kv_lens,
            max_blocks,
            num_heads + 1, // not divisible by num_kv_heads → rejected upfront
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect_err("invalid head config should fail");

    // ── Rejection: shape mismatch on q ────────────────────────────
    let q_wrong = backend
        .copy_from_host_f32(&q_data[..16], &[1, 16])
        .unwrap();
    let _ = backend
        .paged_attention(
            &q_wrong,
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
        .expect_err("q shape mismatch should fail");

    // Allocate F16 pool to check dtype rejection.
    let pool_f16 = backend
        .allocate_zeros(&[total_blocks, block_size, kv_dim], DType::F16)
        .unwrap();
    let q_f16 = backend.cast(&q, DType::F16).unwrap();
    let _ = backend
        .paged_attention(
            &q_f16,
            &pool_f16,
            &pool_f16,
            &block_tables,
            &kv_lens,
            max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect_err("F16 pool not yet supported");
}
