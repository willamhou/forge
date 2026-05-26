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
        assert_eq!(backend.sample_gumbel(&logits, 0.01, 1, step).unwrap(), greedy);
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
fn gumbel_rejects_nonpositive_temperature() {
    let backend = CpuBackend::new();
    let logits = backend.copy_from_host_f32(&[1.0, 2.0], &[1, 2]).unwrap();
    assert!(backend.sample_gumbel(&logits, 0.0, 1, 1).is_err());
    assert!(backend.sample_gumbel(&logits, -1.0, 1, 1).is_err());
}
