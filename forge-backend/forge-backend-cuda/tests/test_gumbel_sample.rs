//! GPU Gumbel-max multinomial sampling (`Backend::sample_gumbel`):
//! determinism, low-temperature → argmax, peaked + uniform distributions,
//! and the F16 path. (Cross CPU/CUDA bit-parity is not asserted — `logf`
//! differs in the last bits across the two, which can flip near-ties; the
//! algorithm and RNG are identical by construction.)

use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, DType};

#[test]
fn gumbel_is_deterministic_per_seed_step() {
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[0.5, 1.0, 0.2, 0.8, 0.1, 0.9], &[1, 6])
        .unwrap();
    let a = backend.sample_gumbel(&logits, 1.0, 12345, 7).unwrap();
    let b = backend.sample_gumbel(&logits, 1.0, 12345, 7).unwrap();
    assert_eq!(a, b, "same (seed, step) must reproduce the same draw");
}

#[test]
fn gumbel_low_temperature_is_argmax() {
    // As T → 0 the logit term dominates the O(1) Gumbel noise, so the sample
    // collapses to the greedy argmax (index 2 here).
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 0.0], &[1, 4])
        .unwrap();
    let greedy = backend.argmax(&logits).unwrap();
    for step in 0..32u32 {
        let s = backend.sample_gumbel(&logits, 0.01, 999, step).unwrap();
        assert_eq!(s, greedy, "low-T sample must match argmax (step {step})");
    }
}

#[test]
fn gumbel_peaked_logits_pick_the_peak() {
    // One logit far above the rest dominates Gumbel noise at T=1 → always picked.
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[0.0, 0.0, 20.0, 0.0, 0.0], &[1, 5])
        .unwrap();
    for step in 0..64u32 {
        let s = backend.sample_gumbel(&logits, 1.0, 7, step).unwrap();
        assert_eq!(s, vec![2], "peaked dist must pick the peak (step {step})");
    }
}

#[test]
fn gumbel_uniform_logits_cover_all_tokens() {
    // Uniform logits at T=1 → uniform sampling: over many draws every token
    // appears and counts are roughly balanced.
    let backend = CudaBackend::new(0).unwrap();
    let vocab = 4usize;
    let logits = backend.copy_from_host_f32(&[0.0; 4], &[1, vocab]).unwrap();
    let draws = 2000usize;
    let mut counts = vec![0usize; vocab];
    for step in 0..draws as u32 {
        let s = backend.sample_gumbel(&logits, 1.0, 2024, step).unwrap();
        counts[s[0] as usize] += 1;
    }
    let expected = draws / vocab;
    for (tok, &c) in counts.iter().enumerate() {
        assert!(c > 0, "token {tok} never sampled");
        // Generous band (±40% of expected) — just a sanity check, not a GoF test.
        assert!(
            c > expected * 6 / 10 && c < expected * 14 / 10,
            "token {tok} count {c} far from expected ~{expected}"
        );
    }
}

#[test]
fn gumbel_batched_rows_independent() {
    // Two rows with different peaks → each row samples its own peak.
    let backend = CudaBackend::new(0).unwrap();
    let data = [
        0.0, 0.0, 20.0, 0.0, // row 0 → 2
        20.0, 0.0, 0.0, 0.0, // row 1 → 0
    ];
    let logits = backend.copy_from_host_f32(&data, &[2, 4]).unwrap();
    let s = backend.sample_gumbel(&logits, 1.0, 5, 3).unwrap();
    assert_eq!(s, vec![2, 0]);
}

#[test]
fn gumbel_f16_runs_and_picks_peak() {
    let backend = CudaBackend::new(0).unwrap();
    let f32_t = backend
        .copy_from_host_f32(&[0.0, 15.0, 0.0, 0.0], &[1, 4])
        .unwrap();
    let f16_t = backend.cast(&f32_t, DType::F16).unwrap();
    let s = backend.sample_gumbel(&f16_t, 1.0, 1, 1).unwrap();
    assert_eq!(s, vec![1]);
}

#[test]
fn gumbel_rejects_nonpositive_temperature() {
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend.copy_from_host_f32(&[1.0, 2.0], &[1, 2]).unwrap();
    assert!(backend.sample_gumbel(&logits, 0.0, 1, 1).is_err());
}

// ── Per-row `sample` (mixed greedy + Gumbel) ──────────────────────────

#[test]
fn sample_all_greedy_matches_argmax() {
    // temp <= 0 on every row → pure argmax, RNG-independent.
    let backend = CudaBackend::new(0).unwrap();
    let data = [1.0, 2.0, 9.0, 0.0, 5.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 4.0];
    let logits = backend.copy_from_host_f32(&data, &[3, 4]).unwrap();
    let temps = [0.0f32; 3];
    let min_ps = [0.0f32; 3];
    let top_ks = [0u32; 3];
    let top_ps = [1.0f32; 3];
    let seeds = [1u64; 3];
    let steps = [0u32; 3];
    assert_eq!(
        backend
            .sample(&logits, &temps, &min_ps, &top_ks, &top_ps, &seeds, &steps)
            .unwrap(),
        backend.argmax(&logits).unwrap()
    );
}

#[test]
fn sample_single_row_matches_scalar_gumbel() {
    // One row: per-row sample with temp>0 must equal the scalar sample_gumbel
    // (both use row index 0 and the same (seed, step) RNG key).
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[0.5, 1.0, 0.2, 0.8, 0.1, 0.9], &[1, 6])
        .unwrap();
    for (seed, step) in [(1u64, 0u32), (42, 7), (12345, 99)] {
        let via_perrow = backend
            .sample(&logits, &[0.7], &[0.0], &[0], &[1.0], &[seed], &[step])
            .unwrap();
        let via_scalar = backend.sample_gumbel(&logits, 0.7, seed, step).unwrap();
        assert_eq!(via_perrow, via_scalar, "(seed {seed}, step {step})");
    }
}

#[test]
fn sample_mixed_batch_greedy_and_sampled() {
    // row 0: greedy (temp 0) over peaked logits → index 2.
    // row 1: sampled (temp 1) over peaked logits → peak still dominates → 0.
    let backend = CudaBackend::new(0).unwrap();
    let data = [
        0.0, 0.0, 20.0, 0.0, // row 0 greedy → 2
        20.0, 0.0, 0.0, 0.0, // row 1 sampled → 0 (peak dominates)
    ];
    let logits = backend.copy_from_host_f32(&data, &[2, 4]).unwrap();
    let out = backend
        .sample(
            &logits,
            &[0.0, 1.0],
            &[0.0, 0.0],
            &[0, 0],
            &[1.0, 1.0],
            &[7, 7],
            &[0, 3],
        )
        .unwrap();
    assert_eq!(out, vec![2, 0]);
}

#[test]
fn sample_rejects_param_length_mismatch() {
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    // 2 rows but only 1 temp.
    assert!(
        backend
            .sample(
                &logits,
                &[1.0],
                &[0.0, 0.0],
                &[0, 0],
                &[1.0, 1.0],
                &[1, 2],
                &[0, 0]
            )
            .is_err()
    );
}

#[test]
fn sample_min_p_filters_low_prob_tokens() {
    // softmax([1.1,0,0,0]) ≈ [0.50, 0.167, 0.167, 0.167]. With min_p=0.5 the
    // keep threshold is 0.5*0.50 = 0.25; only the peak (0.50) survives, so every
    // draw picks it. With min_p disabled the 0.167 tokens win ~half the time.
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[1.1, 0.0, 0.0, 0.0], &[1, 4])
        .unwrap();
    for step in 0..32u32 {
        let s = backend
            .sample(&logits, &[1.0], &[0.5], &[0], &[1.0], &[7], &[step])
            .unwrap();
        assert_eq!(s, vec![0], "min_p must restrict to the peak (step {step})");
    }
    // With min_p disabled (0.0) the same logits can occasionally pick others.
    let mut seen_other = false;
    for step in 0..200u32 {
        let s = backend
            .sample(&logits, &[1.0], &[0.0], &[0], &[1.0], &[7], &[step])
            .unwrap();
        if s != vec![0] {
            seen_other = true;
            break;
        }
    }
    assert!(seen_other, "without min_p, low-prob tokens should sometimes win");
}

#[test]
fn sample_top_k_restricts_to_k_highest() {
    // top_k=1 keeps only the highest-logit token → always picked.
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[0.5, 3.0, 1.0, 0.2, 2.0], &[1, 5])
        .unwrap();
    for step in 0..32u32 {
        let s = backend
            .sample(&logits, &[1.0], &[0.0], &[1], &[1.0], &[9], &[step])
            .unwrap();
        assert_eq!(s, vec![1], "top_k=1 must pick the argmax (step {step})");
    }
}

#[test]
fn sample_top_p_restricts_to_nucleus() {
    // softmax([5,0,0,0]) ≈ [0.985, 0.005×3]; top_p=0.9 keeps only the peak.
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[5.0, 0.0, 0.0, 0.0], &[1, 4])
        .unwrap();
    for step in 0..32u32 {
        let s = backend
            .sample(&logits, &[1.0], &[0.0], &[0], &[0.9], &[3], &[step])
            .unwrap();
        assert_eq!(s, vec![0], "top_p=0.9 must keep only the peak (step {step})");
    }
}

#[test]
fn sample_top_k_matches_cpu_reference() {
    // Distribution check: GPU top-k sampling covers exactly the top-k set and
    // nothing outside it, over many draws.
    let backend = CudaBackend::new(0).unwrap();
    let data = [3.0f32, 1.0, 2.5, 0.5, 2.0, 0.1, 2.8, 0.3];
    let logits = backend.copy_from_host_f32(&data, &[1, 8]).unwrap();
    // top-3 by logit are indices 0 (3.0), 6 (2.8), 2 (2.5).
    let allowed = [0u32, 6, 2];
    let mut seen = std::collections::HashSet::new();
    for step in 0..400u32 {
        let s = backend
            .sample(&logits, &[1.0], &[0.0], &[3], &[1.0], &[5], &[step])
            .unwrap();
        assert!(
            allowed.contains(&s[0]),
            "top_k=3 sampled {} outside the top-3 set (step {step})",
            s[0]
        );
        seen.insert(s[0]);
    }
    assert_eq!(seen.len(), 3, "all 3 top-k tokens should appear over 400 draws");
}
