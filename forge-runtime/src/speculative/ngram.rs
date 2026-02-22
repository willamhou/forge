//! N-gram based speculative drafter.
//!
//! Builds an n-gram index from the prompt context and uses it to predict
//! the most likely next tokens. This is a simple, zero-cost draft method
//! (no draft model needed) that works well for repetitive or patterned text.

use std::collections::HashMap;

/// N-gram drafter that predicts candidates from prompt context patterns.
pub struct NgramDrafter {
    /// Order of the n-gram model (context window size).
    n: usize,
    /// Maps n-gram context (last n-1 tokens) → candidate token frequencies.
    index: HashMap<Vec<u32>, HashMap<u32, u32>>,
    /// Maximum number of draft tokens to generate per step.
    max_draft_len: usize,
}

impl NgramDrafter {
    /// Create a new N-gram drafter.
    ///
    /// * `n` — n-gram order (e.g., 3 for trigrams). Minimum 2.
    /// * `max_draft_len` — maximum draft tokens per speculation step.
    pub fn new(n: usize, max_draft_len: usize) -> Self {
        assert!(n >= 2, "n-gram order must be at least 2");
        Self {
            n,
            index: HashMap::new(),
            max_draft_len,
        }
    }

    /// Build (or rebuild) the n-gram index from a token sequence.
    pub fn build_index(&mut self, tokens: &[u32]) {
        self.index.clear();
        if tokens.len() < self.n {
            return;
        }
        for window in tokens.windows(self.n) {
            let context = &window[..self.n - 1];
            let next_token = window[self.n - 1];
            *self
                .index
                .entry(context.to_vec())
                .or_default()
                .entry(next_token)
                .or_insert(0) += 1;
        }
    }

    /// Incrementally add a single token to the index, updating n-grams
    /// that end with this token.
    pub fn add_token(&mut self, recent_tokens: &[u32], new_token: u32) {
        // We need at least n-1 recent tokens plus the new token to form an n-gram
        if recent_tokens.len() < self.n - 1 {
            return;
        }
        let start = recent_tokens.len() - (self.n - 1);
        let context = &recent_tokens[start..];
        *self
            .index
            .entry(context.to_vec())
            .or_default()
            .entry(new_token)
            .or_insert(0) += 1;
    }

    /// Draft up to `max_draft_len` candidate tokens given the current context.
    ///
    /// Returns a sequence of predicted token IDs. The draft stops early if
    /// no n-gram match is found for the current context.
    pub fn draft(&self, context_tokens: &[u32]) -> Vec<u32> {
        let mut draft = Vec::with_capacity(self.max_draft_len);
        let mut extended: Vec<u32> = context_tokens.to_vec();

        for _ in 0..self.max_draft_len {
            if extended.len() < self.n - 1 {
                break;
            }
            let start = extended.len() - (self.n - 1);
            let ctx = &extended[start..];

            match self.predict_next(ctx) {
                Some(token) => {
                    draft.push(token);
                    extended.push(token);
                }
                None => break,
            }
        }

        draft
    }

    /// Predict the most common next token for a given context.
    fn predict_next(&self, context: &[u32]) -> Option<u32> {
        self.index.get(context).and_then(|freq_map| {
            freq_map
                .iter()
                .max_by_key(|&(_, count)| *count)
                .map(|(&token, _)| token)
        })
    }

    /// Return the n-gram order.
    pub fn order(&self) -> usize {
        self.n
    }

    /// Return the max draft length.
    pub fn max_draft_len(&self) -> usize {
        self.max_draft_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_index_basic() {
        let mut drafter = NgramDrafter::new(3, 4);
        // tokens: [1,2,3,2,3,4]
        // trigrams: (1,2)->3, (2,3)->2, (3,2)->3, (2,3)->4
        // context (2,3) maps to {2:1, 4:1}
        drafter.build_index(&[1, 2, 3, 2, 3, 4]);
        assert!(drafter.index.contains_key(&vec![1, 2]));
        assert!(drafter.index.contains_key(&vec![2, 3]));
    }

    #[test]
    fn test_predict_most_common() {
        let mut drafter = NgramDrafter::new(3, 4);
        // (A,B)->C appears 3 times, (A,B)->D appears 1 time
        drafter.build_index(&[10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 40]);
        let draft = drafter.draft(&[10, 20]);
        assert!(!draft.is_empty());
        // Most common after (10,20) is 30 (3 occurrences vs 1 for 40)
        assert_eq!(draft[0], 30);
    }

    #[test]
    fn test_draft_chain() {
        let mut drafter = NgramDrafter::new(2, 4);
        // bigrams: 1->2, 2->3, 3->4, 4->5
        drafter.build_index(&[1, 2, 3, 4, 5]);
        let draft = drafter.draft(&[1]);
        // Should chain: 1->2->3->4
        assert_eq!(draft, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_draft_stops_on_unknown_context() {
        let mut drafter = NgramDrafter::new(3, 8);
        drafter.build_index(&[1, 2, 3]);
        // Only trigram: (1,2)->3
        let draft = drafter.draft(&[1, 2]);
        // Predicts 3 for context (1,2), then context becomes (2,3) which is unknown
        assert_eq!(draft, vec![3]);
    }

    #[test]
    fn test_draft_max_length() {
        let mut drafter = NgramDrafter::new(2, 2);
        // Repeating pattern: 1->1->1->...
        drafter.build_index(&[1, 1, 1, 1, 1]);
        let draft = drafter.draft(&[1]);
        assert_eq!(draft.len(), 2); // limited by max_draft_len
    }

    #[test]
    fn test_empty_context() {
        let mut drafter = NgramDrafter::new(3, 4);
        drafter.build_index(&[1, 2, 3]);
        let draft = drafter.draft(&[]);
        assert!(draft.is_empty());
    }

    #[test]
    fn test_too_short_tokens() {
        let mut drafter = NgramDrafter::new(3, 4);
        drafter.build_index(&[1]); // too short for trigrams
        assert!(drafter.index.is_empty());
    }

    #[test]
    fn test_add_token_incremental() {
        let mut drafter = NgramDrafter::new(3, 4);
        drafter.build_index(&[1, 2, 3]);
        // Index has: (1,2)->3
        // Now add token 4 with recent context [2,3]
        drafter.add_token(&[2, 3], 4);
        // Now (2,3)->4 should exist
        let draft = drafter.draft(&[2, 3]);
        assert_eq!(draft[0], 4);
    }

    #[test]
    #[should_panic(expected = "n-gram order must be at least 2")]
    fn test_invalid_order() {
        NgramDrafter::new(1, 4);
    }
}
