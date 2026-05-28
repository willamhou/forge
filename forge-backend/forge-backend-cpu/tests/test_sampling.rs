//! CPU reference for `Backend::{argmax, sample_gumbel}` default impls (the
//! host-fallback sampling path). The CUDA kernels mirror this exact algorithm
//! and RNG, so these also pin down the shared contract.

use forge_backend_cpu::CpuBackend;
use forge_core::Backend;

#[test]
fn argmax_highest_index_tie_break() {
    let backend = CpuBackend::new();
    // Equal maxima at 1 and 3 → highest index (matches Iterator::max_by).
    let logits = backend
        .copy_from_host_f32(&[0.0, 5.0, 0.0, 5.0, 1.0], &[1, 5])
        .unwrap();
    assert_eq!(backend.argmax(&logits).unwrap(), vec![3]);

    let batched = backend
        .copy_from_host_f32(&[1.0, 9.0, 2.0, 5.0, 1.0, 2.0], &[2, 3])
        .unwrap();
    assert_eq!(backend.argmax(&batched).unwrap(), vec![1, 0]);
}

#[test]
fn gumbel_deterministic_and_low_temp_is_argmax() {
    let backend = CpuBackend::new();
    let logits = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 0.0], &[1, 4])
        .unwrap();
    // Deterministic per (seed, step).
    let a = backend.sample_gumbel(&logits, 1.0, 42, 3).unwrap();
    let b = backend.sample_gumbel(&logits, 1.0, 42, 3).unwrap();
    assert_eq!(a, b);
    // Low temperature collapses to argmax.
    let greedy = backend.argmax(&logits).unwrap();
    for step in 0..16u32 {
        assert_eq!(
            backend.sample_gumbel(&logits, 0.01, 1, step).unwrap(),
            greedy
        );
    }
}

#[test]
fn gumbel_uniform_logits_cover_all_tokens() {
    let backend = CpuBackend::new();
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
        assert!(
            c > expected * 6 / 10 && c < expected * 14 / 10,
            "token {tok} count {c} far from expected ~{expected}"
        );
    }
}

#[test]
fn sample_min_p_filters_low_prob_tokens() {
    // Mirrors the CUDA test: softmax([1.1,0,0,0]) ≈ [0.50, 0.167×3]; min_p=0.5
    // keeps only the peak (threshold 0.25), so every draw picks token 0.
    let backend = CpuBackend::new();
    let logits = backend
        .copy_from_host_f32(&[1.1, 0.0, 0.0, 0.0], &[1, 4])
        .unwrap();
    for step in 0..32u32 {
        let s = backend
            .sample(&logits, &[1.0], &[0.5], &[0], &[1.0], &[7], &[step])
            .unwrap();
        assert_eq!(s, vec![0], "min_p must restrict to the peak (step {step})");
    }
    // Without min_p the 0.167 tokens win sometimes.
    let mut seen_other = false;
    for step in 0..200u32 {
        if backend
            .sample(&logits, &[1.0], &[0.0], &[0], &[1.0], &[7], &[step])
            .unwrap()
            != vec![0]
        {
            seen_other = true;
            break;
        }
    }
    assert!(
        seen_other,
        "without min_p, low-prob tokens should sometimes win"
    );
}

#[test]
fn sample_top_k_restricts_to_k_highest() {
    // top_k=1 keeps only the single highest-logit token, so every draw picks
    // it regardless of temperature/noise.
    let backend = CpuBackend::new();
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
    // softmax([5,0,0,0]) ≈ [0.985, 0.005×3]. top_p=0.9 keeps only the peak
    // (its mass alone exceeds 0.9), so every draw picks it.
    let backend = CpuBackend::new();
    let logits = backend
        .copy_from_host_f32(&[5.0, 0.0, 0.0, 0.0], &[1, 4])
        .unwrap();
    for step in 0..32u32 {
        let s = backend
            .sample(&logits, &[1.0], &[0.0], &[0], &[0.9], &[3], &[step])
            .unwrap();
        assert_eq!(
            s,
            vec![0],
            "top_p=0.9 must keep only the peak (step {step})"
        );
    }
}

#[test]
fn gumbel_rejects_nonpositive_temperature() {
    let backend = CpuBackend::new();
    let logits = backend.copy_from_host_f32(&[1.0, 2.0], &[1, 2]).unwrap();
    assert!(backend.sample_gumbel(&logits, 0.0, 1, 1).is_err());
    assert!(backend.sample_gumbel(&logits, -1.0, 1, 1).is_err());
}
