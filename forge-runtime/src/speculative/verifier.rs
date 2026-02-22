//! Speculative decoding verifier.
//!
//! Verifies draft tokens against the target model's logits using greedy
//! verification. In a single forward pass with all draft tokens, we check
//! whether the target model would have produced the same tokens.

/// Result of verifying a batch of draft tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationResult {
    /// Number of draft tokens that were accepted (matched target model output).
    pub accepted_count: usize,
    /// The token the target model would produce at the first rejected position
    /// (or after all accepted tokens). This is always a valid next token.
    pub correction_token: u32,
}

/// Greedy verification: accept draft tokens while they match the argmax
/// of the target model's logits at each position.
///
/// * `draft_tokens` — the candidate token sequence from the drafter
/// * `logits` — target model logits for positions [0..draft_len+1], flattened
///   as `[(draft_len + 1) * vocab_size]`. Position i contains the logits
///   *after* processing tokens [0..=i] from the draft.
/// * `vocab_size` — vocabulary size for slicing logits
///
/// The model is run on the full draft sequence in one forward pass, producing
/// logits for each position. We compare argmax(logits[i]) with draft_tokens[i].
pub fn verify_greedy(
    draft_tokens: &[u32],
    logits: &[f32],
    vocab_size: usize,
) -> VerificationResult {
    let num_positions = logits.len() / vocab_size;
    debug_assert!(
        num_positions >= 1,
        "need at least 1 position in logits"
    );

    let mut accepted = 0;

    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        if i >= num_positions {
            break;
        }
        let pos_logits = &logits[i * vocab_size..(i + 1) * vocab_size];
        let argmax = argmax_f32(pos_logits);

        if argmax == draft_token {
            accepted += 1;
        } else {
            // First mismatch — the correction is the target model's choice
            return VerificationResult {
                accepted_count: accepted,
                correction_token: argmax,
            };
        }
    }

    // All draft tokens accepted. The correction token comes from the bonus
    // position (index `draft_len`), which is the model's prediction after
    // consuming all draft tokens.
    debug_assert!(
        num_positions > draft_tokens.len(),
        "logits must contain draft_len + 1 positions for the bonus token; \
         got {} positions for {} draft tokens",
        num_positions,
        draft_tokens.len()
    );
    let bonus_pos = draft_tokens.len().min(num_positions - 1);
    let bonus_logits =
        &logits[bonus_pos * vocab_size..(bonus_pos + 1) * vocab_size];
    let correction_token = argmax_f32(bonus_logits);

    VerificationResult {
        accepted_count: accepted,
        correction_token,
    }
}

/// Compute the acceptance rate from a history of verification results.
/// Returns a value in [0.0, 1.0]. Returns 0.0 if no history is available.
pub fn acceptance_rate(accepted_counts: &[usize], draft_lengths: &[usize]) -> f64 {
    debug_assert_eq!(
        accepted_counts.len(),
        draft_lengths.len(),
        "accepted_counts and draft_lengths must have the same length"
    );
    if draft_lengths.is_empty() {
        return 0.0;
    }
    let total_accepted: usize = accepted_counts.iter().sum();
    let total_drafted: usize = draft_lengths.iter().sum();
    if total_drafted == 0 {
        return 0.0;
    }
    total_accepted as f64 / total_drafted as f64
}

/// Compute an adaptive draft length based on acceptance rate.
/// Uses the formula: `ceil(1 / (1 - acceptance_rate))`, clamped to [min_k, max_k].
pub fn adaptive_draft_len(rate: f64, min_k: usize, max_k: usize) -> usize {
    if rate >= 1.0 {
        return max_k;
    }
    if rate <= 0.0 {
        return min_k;
    }
    let k = (1.0 / (1.0 - rate)).ceil() as usize;
    k.clamp(min_k, max_k)
}

fn argmax_f32(slice: &[f32]) -> u32 {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(vocab_size: usize, argmax_per_pos: &[u32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; argmax_per_pos.len() * vocab_size];
        for (pos, &token) in argmax_per_pos.iter().enumerate() {
            logits[pos * vocab_size + token as usize] = 10.0;
        }
        logits
    }

    #[test]
    fn test_all_accepted() {
        let vocab_size = 100;
        let draft = vec![5, 10, 15];
        // Target model agrees: positions 0,1,2 produce 5,10,15
        // Position 3 (bonus) produces token 20
        let logits = make_logits(vocab_size, &[5, 10, 15, 20]);

        let result = verify_greedy(&draft, &logits, vocab_size);
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.correction_token, 20);
    }

    #[test]
    fn test_first_rejected() {
        let vocab_size = 100;
        let draft = vec![5, 10, 15];
        // Target disagrees at position 0: produces 99 instead of 5
        let logits = make_logits(vocab_size, &[99, 10, 15, 20]);

        let result = verify_greedy(&draft, &logits, vocab_size);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.correction_token, 99);
    }

    #[test]
    fn test_partial_accept() {
        let vocab_size = 100;
        let draft = vec![5, 10, 15];
        // Agrees on 5, 10 but disagrees at position 2: produces 42 instead of 15
        let logits = make_logits(vocab_size, &[5, 10, 42, 20]);

        let result = verify_greedy(&draft, &logits, vocab_size);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.correction_token, 42);
    }

    #[test]
    fn test_empty_draft() {
        let vocab_size = 100;
        let draft: Vec<u32> = vec![];
        // Single position logits for the "bonus" token
        let logits = make_logits(vocab_size, &[7]);

        let result = verify_greedy(&draft, &logits, vocab_size);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.correction_token, 7);
    }

    #[test]
    fn test_acceptance_rate_calculation() {
        let accepted = vec![3, 2, 0, 4];
        let drafted = vec![4, 4, 4, 4];
        let rate = acceptance_rate(&accepted, &drafted);
        assert!((rate - 9.0 / 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_acceptance_rate_empty() {
        assert_eq!(acceptance_rate(&[], &[]), 0.0);
    }

    #[test]
    fn test_adaptive_draft_len() {
        // 50% acceptance → ceil(1/0.5) = 2
        assert_eq!(adaptive_draft_len(0.5, 1, 8), 2);
        // 75% acceptance → ceil(1/0.25) = 4
        assert_eq!(adaptive_draft_len(0.75, 1, 8), 4);
        // 90% acceptance → ceil(1/0.1) = 10, clamped to 8
        assert_eq!(adaptive_draft_len(0.9, 1, 8), 8);
        // 0% acceptance → min_k = 1
        assert_eq!(adaptive_draft_len(0.0, 1, 8), 1);
        // 100% acceptance → max_k = 8
        assert_eq!(adaptive_draft_len(1.0, 1, 8), 8);
    }
}
