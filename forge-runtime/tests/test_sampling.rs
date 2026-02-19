use std::collections::HashMap;

use forge_core::{SamplingContext, SamplingParams};
use forge_runtime::sampling::{CpuSampler, LogitProcessorPipeline};

#[test]
fn test_greedy_sampling() {
    let logits = vec![0.1, 0.3, 0.9, 0.2];
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let sampler = CpuSampler;
    let result = sampler.sample_single(&logits, &params, &[]).unwrap();
    assert_eq!(result.token_id, 2);
}

#[test]
fn test_temperature_scaling() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let params = SamplingParams {
        temperature: 0.5,
        ..Default::default()
    };
    let pipeline = LogitProcessorPipeline::from_params(&params);
    let ctx = SamplingContext {
        generated_tokens: &[],
        prompt_tokens: &[],
        token_counts: &HashMap::new(),
    };
    pipeline.apply(&mut logits, &ctx);
    // After temperature 0.5: logits should be [2.0, 4.0, 6.0]
    assert!((logits[0] - 2.0).abs() < 1e-5);
    assert!((logits[1] - 4.0).abs() < 1e-5);
    assert!((logits[2] - 6.0).abs() < 1e-5);
}

#[test]
fn test_repetition_penalty() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    let params = SamplingParams {
        repetition_penalty: 2.0,
        temperature: 1.0,
        ..Default::default()
    };
    let pipeline = LogitProcessorPipeline::from_params(&params);
    let generated = [1u32, 3];
    let mut counts = HashMap::new();
    counts.insert(1u32, 1);
    counts.insert(3u32, 1);
    let ctx = SamplingContext {
        generated_tokens: &generated,
        prompt_tokens: &[],
        token_counts: &counts,
    };
    pipeline.apply(&mut logits, &ctx);
    // Positive logits for generated tokens should be divided by penalty
    assert!((logits[1] - 1.0).abs() < 1e-5); // 2.0 / 2.0
    assert!((logits[3] - 2.0).abs() < 1e-5); // 4.0 / 2.0
    // Unpenalized tokens unchanged
    assert!((logits[0] - 1.0).abs() < 1e-5);
    assert!((logits[2] - 3.0).abs() < 1e-5);
}

#[test]
fn test_greedy_deterministic() {
    let logits = vec![0.5, 0.1, 0.8, 0.3];
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let sampler = CpuSampler;
    // Should always return the same result
    for _ in 0..10 {
        let result = sampler.sample_single(&logits, &params, &[]).unwrap();
        assert_eq!(result.token_id, 2);
    }
}

#[test]
fn test_seeded_sampling_reproducible() {
    let logits = vec![1.0, 1.0, 1.0, 1.0]; // uniform
    let params = SamplingParams {
        temperature: 1.0,
        seed: Some(42),
        ..Default::default()
    };
    let sampler = CpuSampler;

    let result1 = sampler.sample_single(&logits, &params, &[]).unwrap();
    let result2 = sampler.sample_single(&logits, &params, &[]).unwrap();
    assert_eq!(result1.token_id, result2.token_id);
}

#[test]
fn test_negative_logit_repetition_penalty() {
    let mut logits = vec![-1.0, -2.0, 3.0];
    let params = SamplingParams {
        repetition_penalty: 2.0,
        temperature: 1.0,
        ..Default::default()
    };
    let pipeline = LogitProcessorPipeline::from_params(&params);
    let generated = [0u32, 1];
    let mut counts = HashMap::new();
    counts.insert(0u32, 1);
    counts.insert(1u32, 1);
    let ctx = SamplingContext {
        generated_tokens: &generated,
        prompt_tokens: &[],
        token_counts: &counts,
    };
    pipeline.apply(&mut logits, &ctx);
    // Negative logits should be MULTIPLIED by penalty (making them more negative)
    assert!((logits[0] - (-2.0)).abs() < 1e-5); // -1.0 * 2.0
    assert!((logits[1] - (-4.0)).abs() < 1e-5); // -2.0 * 2.0
    // Unpenalized token unchanged
    assert!((logits[2] - 3.0).abs() < 1e-5);
}
