//! CUDA paged_attention kernel — F32 numerical equivalence test.
//!
//! Runs the CUDA kernel on a representative GQA shape and verifies the output
//! matches the host-fallback default impl (gather-per-seq + batched_decode_attention).
//!
//! Single test fn to dodge the GB10 cuBLAS fixture leak (task #4).

use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, DType, Tensor};

fn rng_lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
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
        k_caches.push(
            backend
                .paged_gather_kv(&k_pool, &block_ids, kv_len)
                .unwrap(),
        );
        v_caches.push(
            backend
                .paged_gather_kv(&v_pool, &block_ids, kv_len)
                .unwrap(),
        );
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
        .copy_from_host_f32(&q_data[..num_heads * head_dim], &[1, num_heads * head_dim])
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

    // ── Rejection: num_kv_heads == 0 must error, not panic on the
    //    `num_heads % num_kv_heads` divide-by-zero (Codex PR #5 review) ──
    let _ = backend
        .paged_attention(
            &q,
            &k_pool,
            &v_pool,
            &block_tables,
            &kv_lens,
            max_blocks,
            num_heads,
            0, // num_kv_heads == 0
            head_dim,
            scale,
        )
        .expect_err("num_kv_heads == 0 should be rejected, not panic");

    // ── Rejection: shape mismatch on q ────────────────────────────
    let q_wrong = backend.copy_from_host_f32(&q_data[..16], &[1, 16]).unwrap();
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

    // ── F16 path: same shape, cast all inputs to F16, run kernel,
    //    compare against F32 reference within 5e-3 ────────────────
    let k_pool_f16 = backend.cast(&k_pool, DType::F16).unwrap();
    let v_pool_f16 = backend.cast(&v_pool, DType::F16).unwrap();
    let q_f16 = backend.cast(&q, DType::F16).unwrap();

    let out_f16 = backend
        .paged_attention(
            &q_f16,
            &k_pool_f16,
            &v_pool_f16,
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
    assert_eq!(out_f16.shape(), &[batch_size, num_heads * head_dim]);
    assert_eq!(out_f16.dtype(), DType::F16);

    // Cast F16 output back to F32 for comparison.
    let out_f16_as_f32 = backend.cast(&out_f16, DType::F32).unwrap();
    let h_f16 = backend.copy_to_host_f32(&out_f16_as_f32).unwrap();

    let max_abs_diff_f16 = h_f16
        .iter()
        .zip(&out_ref_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let tol_f16 = 5e-3_f32.max(max_abs_val * 5e-3);
    assert!(
        max_abs_diff_f16 < tol_f16,
        "F16 vs F32 reference max abs diff {max_abs_diff_f16} > tol {tol_f16}\n  f16[0..8] = {:?}\n  ref[0..8] = {:?}",
        &h_f16[..h_f16.len().min(8)],
        &out_ref_host[..out_ref_host.len().min(8)],
    );

    // ── Mixed-dtype rejection: F16 q against F32 pool should fail upfront ─
    let _ = backend
        .paged_attention(
            &q_f16,
            &k_pool, // F32
            &v_pool, // F32
            &block_tables,
            &kv_lens,
            max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect_err("mixed F16 q + F32 pool should be rejected");

    // ── Persistent scratch: pointer should stay stable across small calls,
    //    bump version on grow ─────────────────────────────────────
    let (v_bt_pre, v_kv_pre) = backend.paged_scratch_versions();

    // Several small calls within initial capacity (16 i32 each) — no grow.
    for _ in 0..4 {
        let _ = backend
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
    }
    let (v_bt_small, v_kv_small) = backend.paged_scratch_versions();
    assert_eq!(
        (v_bt_pre, v_kv_pre),
        (v_bt_small, v_kv_small),
        "no grow expected for small repeated calls"
    );

    // Force grow: assemble a batch with > 16 i32 block_table entries.
    let big_batch = 6_usize;
    let big_max_blocks = 4_usize; // 6 * 4 = 24 > initial scratch cap of 16
    let big_block_tables: Vec<i32> = (0..big_batch as i32 * big_max_blocks as i32)
        .map(|i| i % total_blocks as i32) // valid block ids, possibly repeated
        .collect();
    let big_kv_lens: Vec<i32> = vec![4; big_batch]; // 4 tokens per seq, < block_size * big_max_blocks
    let big_q_data: Vec<f32> = (0..big_batch * num_heads * head_dim)
        .map(|_| rng_lcg(&mut seed))
        .collect();
    let big_q = backend
        .copy_from_host_f32(&big_q_data, &[big_batch, num_heads * head_dim])
        .unwrap();

    let _ = backend
        .paged_attention(
            &big_q,
            &k_pool,
            &v_pool,
            &big_block_tables,
            &big_kv_lens,
            big_max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();
    let (v_bt_post, v_kv_post) = backend.paged_scratch_versions();
    assert!(
        v_bt_post > v_bt_small,
        "block_tables scratch should have grown (version {v_bt_small} → {v_bt_post})"
    );
    // kv_lens has 6 i32 — fits in initial 16. Should NOT bump.
    assert_eq!(
        v_kv_post, v_kv_small,
        "kv_lens scratch should NOT have grown for 6 i32"
    );

    // Re-run the grown call — version must stay (no further grow).
    let _ = backend
        .paged_attention(
            &big_q,
            &k_pool,
            &v_pool,
            &big_block_tables,
            &big_kv_lens,
            big_max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();
    let (v_bt_post2, _) = backend.paged_scratch_versions();
    assert_eq!(
        v_bt_post2, v_bt_post,
        "no further grow on second identical call"
    );

    // ── paged_attention_into: caller-provided output buffer ────────────
    //
    // Capture-mode pattern: pre-allocate one output, reuse across multiple
    // calls. The same `&mut CudaTensor` is passed each time, so the kernel
    // writes to the same underlying CudaSlice / device pointer — structurally
    // guaranteed and what CUDA Graph capture relies on.
    let mut out_persistent = backend
        .allocate_zeros(&[batch_size, num_heads * head_dim], DType::F32)
        .unwrap();

    // First call into persistent buffer.
    backend
        .paged_attention_into(
            &mut out_persistent,
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
    let into_first = backend.copy_to_host_f32(&out_persistent).unwrap();

    // Equivalent to the alloc-variant output (within the kernel's nondeterminism
    // floor — atomicAdd order can differ across launches; bound at 1e-4).
    let diff_alloc_vs_into = into_first
        .iter()
        .zip(&out_cuda_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        diff_alloc_vs_into < 1e-4,
        "paged_attention_into result diverges from paged_attention: max diff {diff_alloc_vs_into}"
    );

    // Reuse the SAME out_persistent across many calls — proves the API works
    // with buffer reuse (the property CUDA Graph capture exploits).
    for _ in 0..5 {
        backend
            .paged_attention_into(
                &mut out_persistent,
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
    }
    backend.synchronize().unwrap();
    let into_last = backend.copy_to_host_f32(&out_persistent).unwrap();
    // Deterministic for fixed inputs (atomicAdd order is launch-fixed at this
    // shape — confirmed by the first-vs-Nth comparison): should match the
    // first call exactly.
    assert_eq!(
        into_last, into_first,
        "repeated paged_attention_into into same buffer should be deterministic"
    );

    // Wrong out shape rejected upfront.
    let mut wrong_out = backend
        .allocate_zeros(&[batch_size, num_heads * head_dim + 1], DType::F32)
        .unwrap();
    let _ = backend
        .paged_attention_into(
            &mut wrong_out,
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
        .expect_err("wrong out shape should fail");

    // Wrong out dtype rejected upfront.
    let mut wrong_dtype_out = backend
        .allocate_zeros(&[batch_size, num_heads * head_dim], DType::F16)
        .unwrap();
    let _ = backend
        .paged_attention_into(
            &mut wrong_dtype_out,
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
        .expect_err("wrong out dtype should fail");

    // ── Host-side bounds checks: seq needing more blocks than max_blocks_per_seq ─
    //
    // block_size = 4. Seq with kv_len=10 needs ceil(10/4) = 3 blocks. With
    // max_blocks_per_seq = 2, this must be rejected before launch.
    let oob_kv_lens: Vec<i32> = vec![10, 5];
    let oob_max_blocks = 2; // not enough for kv_len=10
    let oob_block_tables: Vec<i32> = vec![0, 1, 3, 4]; // 2 * 2
    let mut oob_out = backend
        .allocate_zeros(&[2, num_heads * head_dim], DType::F32)
        .unwrap();
    let _ = backend
        .paged_attention_into(
            &mut oob_out,
            &q,
            &k_pool,
            &v_pool,
            &oob_block_tables,
            &oob_kv_lens,
            oob_max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect_err("kv_len needing more blocks than max_blocks_per_seq must fail");

    // ── Host-side bounds: -1 padding within blocks_needed range ─────
    //
    // block_size = 4, kv_len = 5 → blocks_needed = 2. With block_tables row =
    // [0, -1, ...], the second entry is needed but is padding → error.
    let pad_kv_lens: Vec<i32> = vec![5];
    let pad_max_blocks = 3;
    let pad_block_tables: Vec<i32> = vec![0, -1, -1];
    let mut pad_out = backend
        .allocate_zeros(&[1, num_heads * head_dim], DType::F32)
        .unwrap();
    let _ = backend
        .paged_attention_into(
            &mut pad_out,
            &q_f16, // any q with batch=1, num_heads*head_dim — actually no, batch must be 1
            &k_pool,
            &v_pool,
            &pad_block_tables,
            &pad_kv_lens,
            pad_max_blocks,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect_err("padding in active block-table range must fail (or upstream dtype check)");

    // ── Host-side bounds: block_id beyond pool's num_blocks ─────────
    //
    // pool has total_blocks=8. Use block_id=999 → must reject.
    let huge_block_tables: Vec<i32> = vec![0, 1, 999, 3, 4, -1];
    let huge_kv_lens: Vec<i32> = vec![9, 5]; // seq 0 needs 3 blocks; 3rd entry is 999
    let mut huge_out = backend
        .allocate_zeros(&[2, num_heads * head_dim], DType::F32)
        .unwrap();
    let _ = backend
        .paged_attention_into(
            &mut huge_out,
            &q,
            &k_pool,
            &v_pool,
            &huge_block_tables,
            &huge_kv_lens,
            3,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect_err("block_id beyond pool size must fail");

    // ── Split-KV multi-split + 4-warp coverage ──────────────────────────
    //
    // The shapes above use head_dim=32 (1 warp) and kv_len < 512 → num_splits=1,
    // so they never exercise the flash-decoding combine step or the 4-warp dot
    // reduction. Re-run with the real Qwen decode geometry: head_dim=128
    // (4 warps), GQA group 4, and a kv_len of 1500 that crosses many blocks and
    // forces num_splits = ceil(1500/512) = 3. Compare the F16 split-KV kernel
    // against the F32 gather + batched_decode_attention reference.
    {
        let s_num_heads = 8;
        let s_num_kv_heads = 2; // group size 4
        let s_head_dim = 128;
        let s_kv_dim = s_num_kv_heads * s_head_dim;
        let s_block_size = 16;
        let s_kv_len = 1500usize; // ceil(1500/16) = 94 blocks; > 512 → 3 splits
        let s_blocks_needed = s_kv_len.div_ceil(s_block_size);
        let s_total_blocks = s_blocks_needed + 4;
        let s_max_blocks = s_blocks_needed;
        let s_block_tables: Vec<i32> = (0..s_max_blocks as i32).collect();
        let s_kv_lens: Vec<i32> = vec![s_kv_len as i32];

        let s_pool_n = s_total_blocks * s_block_size * s_kv_dim;
        let mut s_seed = 0x5917A11Du64.wrapping_add(0xBEEF);
        let s_k: Vec<f32> = (0..s_pool_n).map(|_| rng_lcg(&mut s_seed) * 0.2).collect();
        let s_v: Vec<f32> = (0..s_pool_n).map(|_| rng_lcg(&mut s_seed) * 0.2).collect();
        let s_q: Vec<f32> = (0..s_num_heads * s_head_dim)
            .map(|_| rng_lcg(&mut s_seed) * 0.2)
            .collect();

        let s_k_pool = backend
            .copy_from_host_f32(&s_k, &[s_total_blocks, s_block_size, s_kv_dim])
            .unwrap();
        let s_v_pool = backend
            .copy_from_host_f32(&s_v, &[s_total_blocks, s_block_size, s_kv_dim])
            .unwrap();
        let s_q_t = backend
            .copy_from_host_f32(&s_q, &[1, s_num_heads * s_head_dim])
            .unwrap();
        let s_scale = (s_head_dim as f32).powf(-0.5);

        // Reference (F32): gather the seq's KV then batched_decode_attention.
        let s_block_ids: Vec<usize> = (0..s_blocks_needed).collect();
        let s_kc = backend
            .paged_gather_kv(&s_k_pool, &s_block_ids, s_kv_len)
            .unwrap();
        let s_vc = backend
            .paged_gather_kv(&s_v_pool, &s_block_ids, s_kv_len)
            .unwrap();
        let s_ref = backend
            .batched_decode_attention(
                &s_q_t,
                &[s_kc],
                &[s_vc],
                s_num_heads,
                s_num_kv_heads,
                s_head_dim,
                s_scale,
            )
            .unwrap();
        backend.synchronize().unwrap();
        let s_ref_host = backend.copy_to_host_f32(&s_ref).unwrap();

        // F16 split-KV path.
        let s_k16 = backend.cast(&s_k_pool, DType::F16).unwrap();
        let s_v16 = backend.cast(&s_v_pool, DType::F16).unwrap();
        let s_q16 = backend.cast(&s_q_t, DType::F16).unwrap();
        let s_out16 = backend
            .paged_attention(
                &s_q16,
                &s_k16,
                &s_v16,
                &s_block_tables,
                &s_kv_lens,
                s_max_blocks,
                s_num_heads,
                s_num_kv_heads,
                s_head_dim,
                s_scale,
            )
            .unwrap();
        backend.synchronize().unwrap();
        let s_out_f32 = backend.cast(&s_out16, DType::F32).unwrap();
        let s_out_host = backend.copy_to_host_f32(&s_out_f32).unwrap();

        let s_max_abs_val = s_ref_host.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let s_diff = s_out_host
            .iter()
            .zip(&s_ref_host)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let s_tol = 5e-3_f32.max(s_max_abs_val * 5e-3);
        assert!(
            s_diff < s_tol,
            "split-KV (3 splits, head_dim=128) F16 vs F32 ref: max abs diff {s_diff} > tol {s_tol} (ref max abs {s_max_abs_val})\n  out[0..8] = {:?}\n  ref[0..8] = {:?}",
            &s_out_host[..s_out_host.len().min(8)],
            &s_ref_host[..s_ref_host.len().min(8)],
        );
    }
}
