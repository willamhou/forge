use std::collections::HashMap;

use forge_core::{SampleResult, SamplingContext, SamplingParams};

use crate::constraints::fsm::FsmConstraint;

/// Logit processor pipeline: applies temperature and penalties in the logit domain.
/// Top-k, top-p, and min-p are applied in the probability domain by `CpuSampler`.
pub struct LogitProcessorPipeline {
    temperature: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
}

impl LogitProcessorPipeline {
    pub fn from_params(params: &SamplingParams) -> Self {
        Self {
            temperature: params.temperature,
            repetition_penalty: params.repetition_penalty,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
        }
    }

    /// Apply all logit processors in order. Modifies `logits` in place.
    pub fn apply(&self, logits: &mut [f32], ctx: &SamplingContext<'_>) {
        self.apply_repetition_penalty(logits, ctx);
        self.apply_presence_frequency_penalty(logits, ctx);
        self.apply_temperature(logits);
    }

    fn apply_repetition_penalty(&self, logits: &mut [f32], ctx: &SamplingContext<'_>) {
        if (self.repetition_penalty - 1.0).abs() < f32::EPSILON {
            return;
        }
        // Deduplicate: penalize each unique token exactly once regardless
        // of how many times it appears in generated_tokens.
        let mut seen = std::collections::HashSet::new();
        for &token_id in ctx.generated_tokens {
            if !seen.insert(token_id) {
                continue;
            }
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.repetition_penalty;
                } else {
                    logits[idx] *= self.repetition_penalty;
                }
            }
        }
    }

    fn apply_presence_frequency_penalty(&self, logits: &mut [f32], ctx: &SamplingContext<'_>) {
        if self.presence_penalty.abs() < f32::EPSILON
            && self.frequency_penalty.abs() < f32::EPSILON
        {
            return;
        }
        for (&token_id, &count) in ctx.token_counts {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] -=
                    self.presence_penalty + self.frequency_penalty * (count as f32);
            }
        }
    }

    fn apply_temperature(&self, logits: &mut [f32]) {
        if self.temperature <= 0.0 || (self.temperature - 1.0).abs() < f32::EPSILON {
            return;
        }
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }
    }
}

/// CPU-based sampler: greedy or multinomial sampling with top-k/top-p/min-p filtering.
pub struct CpuSampler;

impl CpuSampler {
    /// Sample a single token from logits.
    ///
    /// If `temperature == 0`, uses greedy (argmax).
    /// Otherwise, applies top-k, top-p, min-p filtering, then multinomial sampling.
    pub fn sample_single(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        generated_tokens: &[u32],
    ) -> forge_core::Result<SampleResult> {
        self.sample_with_constraint(logits, params, generated_tokens, None)
    }

    /// Sample a single token with an optional FSM constraint.
    ///
    /// The constraint mask is applied in the logit domain (before softmax),
    /// after penalties and temperature scaling. The generation step count is
    /// mixed into seeded RNG so each step produces a different draw.
    pub fn sample_with_constraint(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        generated_tokens: &[u32],
        constraint: Option<(&dyn FsmConstraint, u32)>,
    ) -> forge_core::Result<SampleResult> {
        let token_counts = count_tokens(generated_tokens);
        let ctx = SamplingContext {
            generated_tokens,
            prompt_tokens: &[],
            token_counts: &token_counts,
        };

        let mut processed = logits.to_vec();

        let pipeline = LogitProcessorPipeline::from_params(params);
        pipeline.apply(&mut processed, &ctx);

        // Apply FSM constraint mask (step 8 in the pipeline)
        if let Some((fsm, state)) = constraint {
            fsm.mask_logits(state, &mut processed);
        }

        if params.temperature <= 0.0 {
            return self.greedy(&processed);
        }

        // Convert to probabilities via softmax
        let mut probs = softmax(&processed);

        // Apply top-k
        if let Some(k) = params.top_k {
            apply_top_k(&mut probs, k);
        }

        // Apply top-p (nucleus)
        if params.top_p < 1.0 {
            apply_top_p(&mut probs, params.top_p);
        }

        // Apply min-p
        if let Some(min_p_val) = params.min_p {
            apply_min_p(&mut probs, min_p_val);
        }

        // Renormalize after filtering
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        } else {
            // All probability mass was zeroed (e.g. FSM constraint eliminated all
            // candidates after top-k/top-p). Fall back to greedy over the
            // post-penalty logits to avoid sampling from an all-zero distribution.
            return self.greedy(&processed);
        }

        // Multinomial sampling — pass step count so seeded RNG advances
        self.multinomial(&probs, params.seed, generated_tokens.len())
    }

    fn greedy(&self, logits: &[f32]) -> forge_core::Result<SampleResult> {
        let (token_id, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| {
                forge_core::ForgeError::InvalidArgument("empty logits".into())
            })?;

        // Compute logprob from normalized probability (softmax), not raw logit.
        // This is consistent with multinomial sampling which returns ln(p).
        let probs = softmax(logits);
        let p = probs[token_id];
        let logprob = if p > 0.0 { p.ln() } else { f32::NEG_INFINITY };

        Ok(SampleResult {
            token_id: token_id as u32,
            logprob,
        })
    }

    fn multinomial(
        &self,
        probs: &[f32],
        seed: Option<u64>,
        step: usize,
    ) -> forge_core::Result<SampleResult> {
        use rand::prelude::*;

        let mut rng: Box<dyn RngCore> = match seed {
            // Mix the step counter into the seed so each generation step
            // produces a different but deterministic random draw.
            Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(
                s.wrapping_add(step as u64),
            )),
            None => Box::new(rand::thread_rng()),
        };

        let r: f32 = rng.r#gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= r {
                let logprob = if p > 0.0 { p.ln() } else { f32::NEG_INFINITY };
                return Ok(SampleResult {
                    token_id: i as u32,
                    logprob,
                });
            }
        }

        // Fallback: return last token (rounding error)
        let last = probs.len() - 1;
        Ok(SampleResult {
            token_id: last as u32,
            logprob: if probs[last] > 0.0 {
                probs[last].ln()
            } else {
                f32::NEG_INFINITY
            },
        })
    }
}

fn count_tokens(tokens: &[u32]) -> HashMap<u32, usize> {
    let mut counts = HashMap::new();
    for &t in tokens {
        *counts.entry(t).or_insert(0) += 1;
    }
    counts
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Zero out all but the top-k highest probabilities.
/// `k == 0` means "disable top-k" (keep all tokens), matching common API conventions.
fn apply_top_k(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Collect the indices to keep (exactly the top k)
    let keep: std::collections::HashSet<usize> =
        indexed[..k].iter().map(|&(idx, _)| idx).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *p = 0.0;
        }
    }
}

/// Zero out tokens below the nucleus (cumulative probability) threshold.
fn apply_top_p(probs: &mut [f32], top_p: f32) {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative > top_p {
            cutoff_idx = i + 1; // include this token
            break;
        }
    }

    // Zero out tokens not in the top-p nucleus
    let keep: std::collections::HashSet<usize> =
        indexed[..cutoff_idx].iter().map(|&(idx, _)| idx).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *p = 0.0;
        }
    }
}

/// Zero out tokens whose probability is less than min_p * max_probability.
fn apply_min_p(probs: &mut [f32], min_p: f32) {
    let max_prob = probs
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}
