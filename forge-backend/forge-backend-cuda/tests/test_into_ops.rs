//! `_into` variant tests for hot-path Backend ops.
//!
//! For each op: verify the `_into` variant produces the same output as the
//! allocating variant, that the same caller-provided buffer can be reused
//! across multiple calls (the property CUDA Graph capture exploits), and
//! that shape/dtype mismatches are rejected upfront.
//!
//! Single test fn — GB10 cuBLAS fixture leak workaround (task #4).

use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, DType};

#[test]
fn into_variants_match_alloc_variants() {
    let backend = CudaBackend::new(0).expect("CUDA backend");

    // ── matmul_into vs matmul (F32) ───────────────────────────────
    let m = 4;
    let k = 8;
    let n = 6;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.3).collect();
    let a = backend.copy_from_host_f32(&a_data, &[m, k]).unwrap();
    let b = backend.copy_from_host_f32(&b_data, &[k, n]).unwrap();

    // Alloc variant.
    let c_alloc = backend.matmul(&a, &b).unwrap();
    backend.synchronize().unwrap();
    let c_alloc_host = backend.copy_to_host_f32(&c_alloc).unwrap();

    // Into variant: pre-allocate, then call repeatedly into same buffer.
    let mut c_into = backend.allocate_zeros(&[m, n], DType::F32).unwrap();
    for _ in 0..3 {
        backend.matmul_into(&mut c_into, &a, &b).unwrap();
        backend.synchronize().unwrap();
        let h = backend.copy_to_host_f32(&c_into).unwrap();
        for (i, (got, want)) in h.iter().zip(&c_alloc_host).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "matmul_into F32 diverges at [{i}]: got {got} want {want}"
            );
        }
    }

    // ── matmul_into F16 ───────────────────────────────────────────
    let a_f16 = backend.cast(&a, DType::F16).unwrap();
    let b_f16 = backend.cast(&b, DType::F16).unwrap();
    let c_alloc_f16 = backend.matmul(&a_f16, &b_f16).unwrap();
    backend.synchronize().unwrap();
    let c_alloc_f16_f32 = backend.cast(&c_alloc_f16, DType::F32).unwrap();
    let c_alloc_f16_host = backend.copy_to_host_f32(&c_alloc_f16_f32).unwrap();

    let mut c_into_f16 = backend.allocate_zeros(&[m, n], DType::F16).unwrap();
    backend
        .matmul_into(&mut c_into_f16, &a_f16, &b_f16)
        .unwrap();
    backend.synchronize().unwrap();
    let c_into_f16_f32 = backend.cast(&c_into_f16, DType::F32).unwrap();
    let c_into_f16_host = backend.copy_to_host_f32(&c_into_f16_f32).unwrap();
    for (i, (got, want)) in c_into_f16_host.iter().zip(&c_alloc_f16_host).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "matmul_into F16 diverges at [{i}]: got {got} want {want}"
        );
    }

    // ── matmul_into validation ────────────────────────────────────
    // Wrong out shape.
    let mut wrong_shape = backend.allocate_zeros(&[m, n + 1], DType::F32).unwrap();
    let _ = backend
        .matmul_into(&mut wrong_shape, &a, &b)
        .expect_err("wrong out shape rejected");

    // Wrong out dtype.
    let mut wrong_dtype = backend.allocate_zeros(&[m, n], DType::F16).unwrap();
    let _ = backend
        .matmul_into(&mut wrong_dtype, &a, &b)
        .expect_err("wrong out dtype rejected");

    // a/b dtype mismatch (cast a → F16 but leave b as F32).
    let _ = backend
        .matmul_into(&mut c_into_f16, &a_f16, &b)
        .expect_err("a/b dtype mismatch rejected");

    // ── add_into vs add (F32) ─────────────────────────────────────
    let shape_v = [12];
    let v1_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.3).collect();
    let v2_data: Vec<f32> = (0..12).map(|i| (i as f32) - 5.0).collect();
    let v1 = backend.copy_from_host_f32(&v1_data, &shape_v).unwrap();
    let v2 = backend.copy_from_host_f32(&v2_data, &shape_v).unwrap();

    let sum_alloc = backend.add(&v1, &v2).unwrap();
    backend.synchronize().unwrap();
    let sum_alloc_host = backend.copy_to_host_f32(&sum_alloc).unwrap();

    let mut sum_into = backend.allocate_zeros(&shape_v, DType::F32).unwrap();
    for _ in 0..3 {
        backend.add_into(&mut sum_into, &v1, &v2).unwrap();
        backend.synchronize().unwrap();
        let h = backend.copy_to_host_f32(&sum_into).unwrap();
        for (i, (got, want)) in h.iter().zip(&sum_alloc_host).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "add_into F32 diverges at [{i}]: got {got} want {want}"
            );
        }
    }

    // ── add_into F16 ──────────────────────────────────────────────
    let v1_f16 = backend.cast(&v1, DType::F16).unwrap();
    let v2_f16 = backend.cast(&v2, DType::F16).unwrap();
    let sum_alloc_f16 = backend.add(&v1_f16, &v2_f16).unwrap();
    backend.synchronize().unwrap();
    let sum_alloc_f16_f32 = backend.cast(&sum_alloc_f16, DType::F32).unwrap();
    let sum_alloc_f16_host = backend.copy_to_host_f32(&sum_alloc_f16_f32).unwrap();

    let mut sum_into_f16 = backend.allocate_zeros(&shape_v, DType::F16).unwrap();
    backend
        .add_into(&mut sum_into_f16, &v1_f16, &v2_f16)
        .unwrap();
    backend.synchronize().unwrap();
    let sum_into_f16_f32 = backend.cast(&sum_into_f16, DType::F32).unwrap();
    let sum_into_f16_host = backend.copy_to_host_f32(&sum_into_f16_f32).unwrap();
    for (i, (got, want)) in sum_into_f16_host
        .iter()
        .zip(&sum_alloc_f16_host)
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-3,
            "add_into F16 diverges at [{i}]: got {got} want {want}"
        );
    }

    // ── add_into validation ───────────────────────────────────────
    let mut wrong_v = backend.allocate_zeros(&[13], DType::F32).unwrap();
    let _ = backend
        .add_into(&mut wrong_v, &v1, &v2)
        .expect_err("wrong out shape rejected");
    let mut wrong_dtype_v = backend.allocate_zeros(&shape_v, DType::F16).unwrap();
    let _ = backend
        .add_into(&mut wrong_dtype_v, &v1, &v2)
        .expect_err("wrong out dtype rejected");

    // ── rms_norm_into vs rms_norm (F32) ───────────────────────────
    let rows = 3;
    let cols = 8;
    let x_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let weight_data: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.05).collect();
    let x = backend.copy_from_host_f32(&x_data, &[rows, cols]).unwrap();
    let weight = backend.copy_from_host_f32(&weight_data, &[cols]).unwrap();
    let eps = 1e-5_f32;

    let rms_alloc = backend.rms_norm(&x, &weight, eps).unwrap();
    backend.synchronize().unwrap();
    let rms_alloc_host = backend.copy_to_host_f32(&rms_alloc).unwrap();

    let mut rms_into = backend.allocate_zeros(&[rows, cols], DType::F32).unwrap();
    for _ in 0..3 {
        backend
            .rms_norm_into(&mut rms_into, &x, &weight, eps)
            .unwrap();
    }
    backend.synchronize().unwrap();
    let rms_into_host = backend.copy_to_host_f32(&rms_into).unwrap();
    for (i, (got, want)) in rms_into_host.iter().zip(&rms_alloc_host).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "rms_norm_into F32 diverges at [{i}]: got {got} want {want}"
        );
    }

    // ── fused_residual_rms_norm_into ──────────────────────────────
    let residual_data: Vec<f32> = (0..rows * cols).map(|i| -0.2 + (i as f32) * 0.07).collect();
    let residual = backend
        .copy_from_host_f32(&residual_data, &[rows, cols])
        .unwrap();
    let (frrn_norm_alloc, frrn_res_alloc) = backend
        .fused_residual_rms_norm(&x, &residual, &weight, eps)
        .unwrap();
    backend.synchronize().unwrap();
    let frrn_norm_alloc_host = backend.copy_to_host_f32(&frrn_norm_alloc).unwrap();
    let frrn_res_alloc_host = backend.copy_to_host_f32(&frrn_res_alloc).unwrap();

    let mut frrn_norm = backend.allocate_zeros(&[rows, cols], DType::F32).unwrap();
    let mut frrn_res = backend.allocate_zeros(&[rows, cols], DType::F32).unwrap();
    backend
        .fused_residual_rms_norm_into(&mut frrn_norm, &mut frrn_res, &x, &residual, &weight, eps)
        .unwrap();
    backend.synchronize().unwrap();
    let frrn_norm_host = backend.copy_to_host_f32(&frrn_norm).unwrap();
    let frrn_res_host = backend.copy_to_host_f32(&frrn_res).unwrap();
    for (i, (got, want)) in frrn_norm_host.iter().zip(&frrn_norm_alloc_host).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "fused_residual_rms_norm_into normed diverges at [{i}]: got {got} want {want}"
        );
    }
    for (i, (got, want)) in frrn_res_host.iter().zip(&frrn_res_alloc_host).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "fused_residual_rms_norm_into residual diverges at [{i}]: got {got} want {want}"
        );
    }

    // ── fused_silu_mul_into ───────────────────────────────────────
    let gate_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.13 - 0.6).collect();
    let up_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.09 + 0.2).collect();
    let gate = backend
        .copy_from_host_f32(&gate_data, &[rows, cols])
        .unwrap();
    let up = backend.copy_from_host_f32(&up_data, &[rows, cols]).unwrap();
    let silu_alloc = backend.fused_silu_mul(&gate, &up).unwrap();
    backend.synchronize().unwrap();
    let silu_alloc_host = backend.copy_to_host_f32(&silu_alloc).unwrap();

    let mut silu_into = backend.allocate_zeros(&[rows, cols], DType::F32).unwrap();
    for _ in 0..3 {
        backend
            .fused_silu_mul_into(&mut silu_into, &gate, &up)
            .unwrap();
    }
    backend.synchronize().unwrap();
    let silu_into_host = backend.copy_to_host_f32(&silu_into).unwrap();
    for (i, (got, want)) in silu_into_host.iter().zip(&silu_alloc_host).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "fused_silu_mul_into F32 diverges at [{i}]: got {got} want {want}"
        );
    }

    // ── embedding_into ────────────────────────────────────────────
    let vocab = 16usize;
    let embed_dim = 6usize;
    let emb_weight_data: Vec<f32> = (0..vocab * embed_dim).map(|i| (i as f32) * 0.04).collect();
    let emb_weight = backend
        .copy_from_host_f32(&emb_weight_data, &[vocab, embed_dim])
        .unwrap();
    let indices: Vec<u32> = vec![0, 5, 12, 3];

    let emb_alloc = backend.embedding(&emb_weight, &indices).unwrap();
    backend.synchronize().unwrap();
    let emb_alloc_host = backend.copy_to_host_f32(&emb_alloc).unwrap();

    let mut emb_into = backend
        .allocate_zeros(&[indices.len(), embed_dim], DType::F32)
        .unwrap();
    for _ in 0..3 {
        backend
            .embedding_into(&mut emb_into, &emb_weight, &indices)
            .unwrap();
    }
    backend.synchronize().unwrap();
    let emb_into_host = backend.copy_to_host_f32(&emb_into).unwrap();
    assert_eq!(emb_into_host, emb_alloc_host);

    // ── cast_into (F32 → F16 → F32 round trip) ───────────────────
    let mut cast_to_f16 = backend.allocate_zeros(&[rows, cols], DType::F16).unwrap();
    backend.cast_into(&mut cast_to_f16, &x).unwrap();
    backend.synchronize().unwrap();
    // cast back to F32 for comparison
    let mut cast_back = backend.allocate_zeros(&[rows, cols], DType::F32).unwrap();
    backend.cast_into(&mut cast_back, &cast_to_f16).unwrap();
    backend.synchronize().unwrap();
    let cast_back_host = backend.copy_to_host_f32(&cast_back).unwrap();
    let x_host = backend.copy_to_host_f32(&x).unwrap();
    for (i, (got, want)) in cast_back_host.iter().zip(&x_host).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "cast F32→F16→F32 roundtrip diverges at [{i}]: got {got} want {want}"
        );
    }

    // Same-dtype cast_into is a d2d copy.
    let mut cast_same = backend.allocate_zeros(&[rows, cols], DType::F32).unwrap();
    backend.cast_into(&mut cast_same, &x).unwrap();
    backend.synchronize().unwrap();
    let cast_same_host = backend.copy_to_host_f32(&cast_same).unwrap();
    assert_eq!(cast_same_host, x_host);

    // ── Validation on the new ops ────────────────────────────────
    let mut wrong = backend
        .allocate_zeros(&[rows, cols + 1], DType::F32)
        .unwrap();
    let _ = backend
        .rms_norm_into(&mut wrong, &x, &weight, eps)
        .expect_err("rms_norm_into wrong shape rejected");
    let mut wrong_dt = backend.allocate_zeros(&[rows, cols], DType::F16).unwrap();
    let _ = backend
        .rms_norm_into(&mut wrong_dt, &x, &weight, eps)
        .expect_err("rms_norm_into wrong dtype rejected");
    let _ = backend
        .fused_silu_mul_into(&mut wrong_dt, &gate, &up)
        .expect_err("fused_silu_mul_into wrong dtype rejected");
    let mut emb_wrong = backend
        .allocate_zeros(&[indices.len() + 1, embed_dim], DType::F32)
        .unwrap();
    let _ = backend
        .embedding_into(&mut emb_wrong, &emb_weight, &indices)
        .expect_err("embedding_into wrong shape rejected");

    // ── split_qkv_into vs split_qkv ──────────────────────────────
    let split_rows = 4;
    let q_size = 8;
    let kv_size = 4;
    let total_cols = q_size + 2 * kv_size; // 16
    let qkv_data: Vec<f32> = (0..split_rows * total_cols)
        .map(|i| (i as f32) * 0.05 - 0.3)
        .collect();
    let qkv = backend
        .copy_from_host_f32(&qkv_data, &[split_rows, total_cols])
        .unwrap();

    let (q_a, k_a, v_a) = backend.split_qkv(&qkv, q_size, kv_size).unwrap();
    backend.synchronize().unwrap();
    let q_alloc = backend.copy_to_host_f32(&q_a).unwrap();
    let k_alloc = backend.copy_to_host_f32(&k_a).unwrap();
    let v_alloc = backend.copy_to_host_f32(&v_a).unwrap();

    let mut q_into = backend
        .allocate_zeros(&[split_rows, q_size], DType::F32)
        .unwrap();
    let mut k_into = backend
        .allocate_zeros(&[split_rows, kv_size], DType::F32)
        .unwrap();
    let mut v_into = backend
        .allocate_zeros(&[split_rows, kv_size], DType::F32)
        .unwrap();
    backend
        .split_qkv_into(&mut q_into, &mut k_into, &mut v_into, &qkv, q_size, kv_size)
        .unwrap();
    backend.synchronize().unwrap();
    assert_eq!(backend.copy_to_host_f32(&q_into).unwrap(), q_alloc);
    assert_eq!(backend.copy_to_host_f32(&k_into).unwrap(), k_alloc);
    assert_eq!(backend.copy_to_host_f32(&v_into).unwrap(), v_alloc);

    // Wrong q_out shape rejected
    let mut q_wrong = backend
        .allocate_zeros(&[split_rows, q_size + 1], DType::F32)
        .unwrap();
    let _ = backend
        .split_qkv_into(
            &mut q_wrong,
            &mut k_into,
            &mut v_into,
            &qkv,
            q_size,
            kv_size,
        )
        .expect_err("split_qkv_into wrong q shape rejected");

    // ── slice_rows_into vs slice_rows ────────────────────────────
    let row_src_data: Vec<f32> = (0..20 * 5).map(|i| i as f32 * 0.1).collect();
    let row_src = backend.copy_from_host_f32(&row_src_data, &[20, 5]).unwrap();
    let row_alloc = backend.slice_rows(&row_src, 5, 7).unwrap();
    backend.synchronize().unwrap();
    let row_alloc_host = backend.copy_to_host_f32(&row_alloc).unwrap();

    let mut row_into = backend.allocate_zeros(&[7, 5], DType::F32).unwrap();
    backend
        .slice_rows_into(&mut row_into, &row_src, 5, 7)
        .unwrap();
    backend.synchronize().unwrap();
    assert_eq!(backend.copy_to_host_f32(&row_into).unwrap(), row_alloc_host);

    // Reuse same buffer: subsequent slices overwrite (and second slice
    // should be different data, proving the buffer reuses cleanly).
    backend
        .slice_rows_into(&mut row_into, &row_src, 0, 7)
        .unwrap();
    backend.synchronize().unwrap();
    let row_into_host2 = backend.copy_to_host_f32(&row_into).unwrap();
    assert_eq!(row_into_host2, row_src_data[0..35]);

    // Out-of-bounds start_row + num_rows rejected
    let mut row_bad = backend.allocate_zeros(&[5, 5], DType::F32).unwrap();
    let _ = backend
        .slice_rows_into(&mut row_bad, &row_src, 18, 5)
        .expect_err("slice_rows_into oob rejected");

    // ── rope_into vs rope (F32) ──────────────────────────────────
    let batch = 1;
    let rseq = 3;
    let heads = 2;
    let dim = 4; // head_dim even
    let half = dim / 2;
    let rope_x: Vec<f32> = (0..batch * rseq * heads * dim)
        .map(|i| (i as f32) * 0.07)
        .collect();
    let rope_cos: Vec<f32> = (0..rseq * half).map(|i| ((i as f32) * 0.3).cos()).collect();
    let rope_sin: Vec<f32> = (0..rseq * half).map(|i| ((i as f32) * 0.3).sin()).collect();
    let x_rope = backend
        .copy_from_host_f32(&rope_x, &[batch, rseq, heads, dim])
        .unwrap();
    let cos_t = backend
        .copy_from_host_f32(&rope_cos, &[rseq, half])
        .unwrap();
    let sin_t = backend
        .copy_from_host_f32(&rope_sin, &[rseq, half])
        .unwrap();

    let rope_alloc = backend.rope(&x_rope, &cos_t, &sin_t).unwrap();
    backend.synchronize().unwrap();
    let rope_alloc_host = backend.copy_to_host_f32(&rope_alloc).unwrap();

    let mut rope_into = backend
        .allocate_zeros(&[batch, rseq, heads, dim], DType::F32)
        .unwrap();
    for _ in 0..3 {
        backend
            .rope_into(&mut rope_into, &x_rope, &cos_t, &sin_t)
            .unwrap();
    }
    backend.synchronize().unwrap();
    assert_eq!(
        backend.copy_to_host_f32(&rope_into).unwrap(),
        rope_alloc_host
    );

    // Wrong out shape rejected
    let mut rope_wrong = backend
        .allocate_zeros(&[batch, rseq, heads, dim + 2], DType::F32)
        .unwrap();
    let _ = backend
        .rope_into(&mut rope_wrong, &x_rope, &cos_t, &sin_t)
        .expect_err("rope_into wrong out shape rejected");

    // ── rope_with_host_freqs_into — persistent rope_cos/rope_sin scratch ─
    let mut rope_via_host = backend
        .allocate_zeros(&[batch, rseq, heads, dim], DType::F32)
        .unwrap();
    backend
        .rope_with_host_freqs_into(&mut rope_via_host, &x_rope, &rope_cos, &rope_sin)
        .unwrap();
    backend.synchronize().unwrap();
    assert_eq!(
        backend.copy_to_host_f32(&rope_via_host).unwrap(),
        rope_alloc_host,
        "rope_with_host_freqs_into result differs from rope_into"
    );

    // Versions baseline; small inputs should fit initial scratch cap (16 elems each).
    let (v_cos0, v_sin0) = backend.rope_scratch_versions();
    for _ in 0..3 {
        backend
            .rope_with_host_freqs_into(&mut rope_via_host, &x_rope, &rope_cos, &rope_sin)
            .unwrap();
    }
    backend.synchronize().unwrap();
    let (v_cos1, v_sin1) = backend.rope_scratch_versions();
    assert_eq!((v_cos0, v_sin0), (v_cos1, v_sin1), "no grow expected");

    // Force grow: 20-elem cos/sin → exceeds initial 16-element scratch.
    let big_seq = 5; // 5 * (4/2) = 10 — still under 16, hm
    let big_dim = 8; // 5 * (8/2) = 20 > 16 → grow
    let big_x: Vec<f32> = vec![0.0; batch * big_seq * heads * big_dim];
    let big_cos: Vec<f32> = vec![0.0; big_seq * (big_dim / 2)];
    let big_sin: Vec<f32> = vec![0.0; big_seq * (big_dim / 2)];
    let bx = backend
        .copy_from_host_f32(&big_x, &[batch, big_seq, heads, big_dim])
        .unwrap();
    let mut big_out = backend
        .allocate_zeros(&[batch, big_seq, heads, big_dim], DType::F32)
        .unwrap();
    backend
        .rope_with_host_freqs_into(&mut big_out, &bx, &big_cos, &big_sin)
        .unwrap();
    backend.synchronize().unwrap();
    let (v_cos2, v_sin2) = backend.rope_scratch_versions();
    assert!(
        v_cos2 > v_cos1 && v_sin2 > v_sin1,
        "rope scratch should have grown for big call (cos {v_cos1}→{v_cos2}, sin {v_sin1}→{v_sin2})"
    );

    // ── embedding indices scratch ─────────────────────────────────
    let emb_indices2: Vec<u32> = vec![0, 5, 12, 3];
    let v_idx0 = backend.embedding_scratch_version();

    // initial scratch cap = 16, 4 indices fits — no grow
    let mut emb_out2 = backend
        .allocate_zeros(&[emb_indices2.len(), embed_dim], DType::F32)
        .unwrap();
    backend
        .embedding_into(&mut emb_out2, &emb_weight, &emb_indices2)
        .unwrap();
    backend.synchronize().unwrap();
    let v_idx1 = backend.embedding_scratch_version();
    assert_eq!(v_idx0, v_idx1, "no embedding scratch grow for 4 indices");

    // Force grow: 20-index call.
    let big_indices: Vec<u32> = (0..20).map(|i| (i % (vocab as u32)) as u32).collect();
    let mut big_emb_out = backend
        .allocate_zeros(&[big_indices.len(), embed_dim], DType::F32)
        .unwrap();
    backend
        .embedding_into(&mut big_emb_out, &emb_weight, &big_indices)
        .unwrap();
    backend.synchronize().unwrap();
    let v_idx2 = backend.embedding_scratch_version();
    assert!(
        v_idx2 > v_idx1,
        "embedding scratch should have grown (was {v_idx1}, now {v_idx2})"
    );

    // Verify embedding_into still produces correct output after grow.
    let big_emb_host = backend.copy_to_host_f32(&big_emb_out).unwrap();
    for (i, &idx) in big_indices.iter().enumerate() {
        let start = (idx as usize) * embed_dim;
        for d in 0..embed_dim {
            let want = emb_weight_data[start + d];
            let got = big_emb_host[i * embed_dim + d];
            assert!(
                (got - want).abs() < 1e-6,
                "embedding output diverges at [{i}][{d}]: got {got} want {want}"
            );
        }
    }

    // ── add_bias: broadcast bias[c] over rows of x[r,c] (Qwen2 QKV bias) ──
    let bias_rows = 3;
    let bias_cols = 5;
    let bx_data: Vec<f32> = (0..bias_rows * bias_cols).map(|i| i as f32 * 0.5).collect();
    let bias_data: Vec<f32> = vec![1.0, -2.0, 0.25, 10.0, -0.5];
    let bx = backend
        .copy_from_host_f32(&bx_data, &[bias_rows, bias_cols])
        .unwrap();
    let bias = backend
        .copy_from_host_f32(&bias_data, &[bias_cols])
        .unwrap();

    // F32
    let biased = backend.add_bias(&bx, &bias).unwrap();
    backend.synchronize().unwrap();
    let biased_host = backend.copy_to_host_f32(&biased).unwrap();
    for r in 0..bias_rows {
        for c in 0..bias_cols {
            let want = bx_data[r * bias_cols + c] + bias_data[c];
            let got = biased_host[r * bias_cols + c];
            assert!(
                (got - want).abs() < 1e-5,
                "add_bias F32 diverges at [{r}][{c}]: got {got} want {want}"
            );
        }
    }

    // F16
    let bx_f16 = backend.cast(&bx, DType::F16).unwrap();
    let bias_f16 = backend.cast(&bias, DType::F16).unwrap();
    let biased_f16 = backend.add_bias(&bx_f16, &bias_f16).unwrap();
    backend.synchronize().unwrap();
    let biased_f16_f32 = backend.cast(&biased_f16, DType::F32).unwrap();
    let biased_f16_host = backend.copy_to_host_f32(&biased_f16_f32).unwrap();
    for r in 0..bias_rows {
        for c in 0..bias_cols {
            let want = bx_data[r * bias_cols + c] + bias_data[c];
            let got = biased_f16_host[r * bias_cols + c];
            assert!(
                (got - want).abs() < 1e-2,
                "add_bias F16 diverges at [{r}][{c}]: got {got} want {want}"
            );
        }
    }

    // Wrong bias length rejected.
    let wrong_bias = backend.copy_from_host_f32(&[0.0; 4], &[4]).unwrap();
    let _ = backend
        .add_bias(&bx, &wrong_bias)
        .expect_err("add_bias wrong bias length should fail");
}
