//! Numerical-consistency test for the Q8_0 quantized GEMV kernel.
//!
//! Both paths use the *same* quantized weights, so the only differences are
//! f32-accumulation order and f16 rounding — the test isolates kernel
//! correctness, not quantization error.
//!
//! - Path A (quantized): `quantize_q8_0(W)` → `copy_from_host_quant` → `wq`
//!   ([n, k] Q8_0); run `matmul_quant_into(A, wq)` = `A · Wᵀ` → [m, n].
//! - Path B (reference): dequant the *same* quantized bytes back to f16 →
//!   `W_deq` [n, k]; transpose to [k, n]; run the ordinary cuBLAS
//!   `matmul_into(A, W_kn)` = `A · W_kn` → [m, n].
//!
//! Single test fn (GB10 cuBLAS fixture leak workaround — see other CUDA tests).

use forge_backend_cuda::{CudaBackend, quantize_q8_0};
use forge_core::{Backend, DType, Tensor};
use half::f16;

/// Host dequant of Q8_0 bytes, bit-identical to the kernel's reconstruction
/// (`w = scale * q`) and to `forge-loader::dequantize_q8_0`.
fn dequant_q8_0_to_f16(bytes: &[u8], n_elements: usize) -> Vec<f16> {
    const BLOCK_BYTES: usize = 34;
    let mut out = Vec::with_capacity(n_elements);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &q in &block[2..] {
            out.push(f16::from_f32(scale * (q as i8) as f32));
        }
    }
    out.truncate(n_elements);
    out
}

/// Deterministic pseudo-random f32 in roughly [-amp, amp].
fn pseudo_random(i: usize, salt: u64, amp: f32) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    // Map top 24 bits to (-1, 1).
    let u = ((x >> 40) as f32) * (1.0 / 16_777_216.0); // [0, 1)
    (u * 2.0 - 1.0) * amp
}

fn run_case(backend: &CudaBackend, m: usize, n: usize, k: usize, salt: u64) {
    assert_eq!(k % 32, 0, "k must be a multiple of 32");

    // Random [n, k] f16 weights.
    let w_flat: Vec<f16> = (0..n * k)
        .map(|i| f16::from_f32(pseudo_random(i, salt, 2.0)))
        .collect();

    // Path A: quantize → upload as Q8_0 tensor [n, k].
    let w_bytes = quantize_q8_0(&w_flat);
    let wq = backend
        .copy_from_host_quant(&w_bytes, &[n, k], DType::Q8_0)
        .expect("copy_from_host_quant");

    // Path B reference weights: dequant the SAME quantized bytes back to f16,
    // upload as [n, k] f16, then transpose → [k, n] for ordinary matmul.
    let w_deq = dequant_q8_0_to_f16(&w_bytes, n * k);
    let w_nk = backend
        .copy_from_host_f16(&w_deq, &[n, k])
        .expect("upload dequant weights");
    let w_kn = backend.transpose(&w_nk, 0, 1).expect("transpose to [k, n]");
    assert_eq!(w_kn.shape(), &[k, n]);

    // Random [m, k] f16 activation.
    let a_flat: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(pseudo_random(i, salt ^ 0xABCD, 1.0)))
        .collect();
    let a = backend
        .copy_from_host_f16(&a_flat, &[m, k])
        .expect("upload activation");

    // Path A: quantized GEMV → A · Wᵀ = [m, n].
    let mut out_q = backend.allocate_zeros(&[m, n], DType::F16).unwrap();
    backend
        .matmul_quant_into(&mut out_q, &a, &wq)
        .expect("matmul_quant_into");
    backend.synchronize().unwrap();
    let out_q_host = backend.copy_to_host_f32(&out_q).unwrap();
    assert_eq!(out_q_host.len(), m * n);

    // ── Primary reference: host f32 matmul of the SAME dequantized weights.
    // This isolates kernel correctness from cuBLAS's f16-accumulation choice
    // (which itself rounds per step over k terms). a/w are read at the exact
    // f16 precision the GPU sees.
    let a_f32: Vec<f32> = a_flat.iter().map(|x| x.to_f32()).collect();
    let w_f32: Vec<f32> = w_deq.iter().map(|x| x.to_f32()).collect(); // [n, k]
    let mut out_host_ref = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a_f32[i * k + l] * w_f32[j * k + l];
            }
            out_host_ref[i * n + j] = acc;
        }
    }

    // Per-element pass criterion: abs error small OR relative error small.
    // An f16 output has ~10 mantissa bits, so a correct element's error is
    // bounded by ~a few f16 ULP — i.e. ~rel 1e-2, or, near zero where relative
    // error is meaningless, an absolute floor at the f16 ULP of the largest
    // output magnitude (catastrophic cancellation over k terms can leave a
    // near-zero output with an abs error of a few ULP of the inputs).
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut max_out_mag = 0.0f32;
    for v in &out_host_ref {
        max_out_mag = max_out_mag.max(v.abs());
    }
    // Absolute floor: a few f16 ULP at the peak output magnitude. f16 ULP at
    // magnitude X is ~X/1024; allow 8 ULP plus a small constant.
    let abs_floor = (max_out_mag * 8.0 / 1024.0).max(1.0e-2);
    let mut worst_i = 0usize;
    for (i, (got, want)) in out_q_host.iter().zip(&out_host_ref).enumerate() {
        let abs = (got - want).abs();
        let rel = abs / want.abs().max(1.0e-6);
        if abs > max_abs_err {
            max_abs_err = abs;
            worst_i = i;
        }
        // Element passes if EITHER bound holds.
        if abs <= abs_floor || rel <= 1.0e-2 {
            continue;
        }
        max_rel_err = max_rel_err.max(rel);
    }
    println!(
        "q8_0 gemv m={m} n={n} k={k} vs host-f32: max_abs_err={max_abs_err:.5} (idx {worst_i}) max_out_mag={max_out_mag:.3} abs_floor={abs_floor:.5}"
    );
    assert!(
        max_rel_err == 0.0,
        "q8_0 gemv vs host-f32: some element fails both abs ({abs_floor}) and rel (1e-2) bounds; worst rel {max_rel_err} (m={m} n={n} k={k})"
    );

    // ── Cross-check: cuBLAS f16 matmul of the dequantized weights (path B in
    // the spec). cuBLAS may accumulate in f16, so this is a looser bound — it
    // confirms the result also matches the existing f16 GEMM path within f16
    // accumulation noise over k terms.
    let mut out_ref = backend.allocate_zeros(&[m, n], DType::F16).unwrap();
    backend
        .matmul_into(&mut out_ref, &a, &w_kn)
        .expect("matmul_into");
    backend.synchronize().unwrap();
    let out_ref_host = backend.copy_to_host_f32(&out_ref).unwrap();
    let mut max_abs_err_blas = 0.0f32;
    for (got, want) in out_q_host.iter().zip(&out_ref_host) {
        max_abs_err_blas = max_abs_err_blas.max((got - want).abs());
    }
    println!("q8_0 gemv m={m} n={n} k={k} vs cuBLAS-f16: max_abs_err={max_abs_err_blas:.5}");
    // f16-accumulation over k terms drifts ~rel 1e-2; scale the bound to the
    // peak output magnitude.
    let blas_tol = (max_out_mag * 2.0e-2).max(5.0e-2);
    assert!(
        max_abs_err_blas <= blas_tol,
        "q8_0 gemv vs cuBLAS-f16 max_abs_err {max_abs_err_blas} > {blas_tol} (m={m} n={n} k={k})"
    );
}

#[test]
fn gemv_q8_0_matches_dequant_matmul() {
    let backend = CudaBackend::new(0).expect("CUDA backend");

    // m=1 (the real decode case) and m=4, with a small and a large weight.
    run_case(&backend, 1, 256, 512, 0x1111);
    run_case(&backend, 4, 256, 512, 0x2222);
    // Larger, closer to a real Qwen3 projection.
    run_case(&backend, 1, 2560, 2560, 0x3333);
    run_case(&backend, 4, 2560, 2560, 0x4444);

    // ── Validation: wrong dtypes / shapes are rejected. ───────────
    let a = backend
        .copy_from_host_f16(&[f16::from_f32(0.5); 32], &[1, 32])
        .unwrap();
    let w_bytes = quantize_q8_0(&[f16::from_f32(0.1); 32]);
    let wq = backend
        .copy_from_host_quant(&w_bytes, &[1, 32], DType::Q8_0)
        .unwrap();

    // Wrong out dtype (F32).
    let mut out_f32 = backend.allocate_zeros(&[1, 1], DType::F32).unwrap();
    backend
        .matmul_quant_into(&mut out_f32, &a, &wq)
        .expect_err("out must be F16");

    // Wrong out shape.
    let mut out_bad = backend.allocate_zeros(&[1, 2], DType::F16).unwrap();
    backend
        .matmul_quant_into(&mut out_bad, &a, &wq)
        .expect_err("wrong out shape rejected");

    // a not F16.
    let a_f32 = backend.copy_from_host_f32(&[0.5f32; 32], &[1, 32]).unwrap();
    let mut out_ok = backend.allocate_zeros(&[1, 1], DType::F16).unwrap();
    backend
        .matmul_quant_into(&mut out_ok, &a_f32, &wq)
        .expect_err("a must be F16");

    // w not quantized.
    let w_f16 = backend
        .copy_from_host_f16(&[f16::from_f32(0.1); 32], &[1, 32])
        .unwrap();
    backend
        .matmul_quant_into(&mut out_ok, &a, &w_f16)
        .expect_err("w must be Q8_0");
}
