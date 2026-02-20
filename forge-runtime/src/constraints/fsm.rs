//! Core FSM trait for structured output constraints.

use std::collections::{HashMap, HashSet};

use forge_core::Result;

/// A vocabulary mapping: token string → token ID.
/// Built once per model, reused across requests.
pub struct TokenVocab {
    /// token_id → token string (decoded bytes)
    pub id_to_token: Vec<String>,
    /// Total vocabulary size
    pub vocab_size: usize,
}

impl TokenVocab {
    /// Build vocabulary from a decode function.
    /// `decode_fn` takes a token_id and returns the decoded string.
    pub fn from_decode_fn<F>(vocab_size: usize, decode_fn: F) -> Self
    where
        F: Fn(u32) -> Option<String>,
    {
        let mut id_to_token = Vec::with_capacity(vocab_size);
        for id in 0..vocab_size as u32 {
            let token = decode_fn(id).unwrap_or_default();
            id_to_token.push(token);
        }
        Self {
            id_to_token,
            vocab_size,
        }
    }
}

/// Trait for FSM-based generation constraints.
///
/// An FSM constraint tracks the current state and provides a set of allowed
/// token IDs at each generation step. After a token is selected, the FSM
/// advances to the next state.
pub trait FsmConstraint: Send {
    /// Get the initial state ID.
    fn initial_state(&self) -> u32;

    /// Get the set of allowed token IDs for the given state.
    /// Returns `None` if the state is invalid (should not happen in normal use).
    fn allowed_tokens(&self, state: u32) -> Option<&[u32]>;

    /// Advance the FSM: given current state and chosen token, return next state.
    /// Returns `None` if the transition is invalid.
    fn next_state(&self, state: u32, token_id: u32) -> Option<u32>;

    /// Check if the given state is a final (accepting) state.
    fn is_final_state(&self, state: u32) -> bool;

    /// Apply the constraint as a logit mask: set disallowed token logits to -inf.
    fn mask_logits(&self, state: u32, logits: &mut [f32]) {
        if let Some(allowed) = self.allowed_tokens(state) {
            // Use a boolean mask: O(V) init + O(allowed) mark + O(V) apply
            let mut mask = vec![false; logits.len()];
            for &id in allowed {
                if (id as usize) < mask.len() {
                    mask[id as usize] = true;
                }
            }
            for (i, logit) in logits.iter_mut().enumerate() {
                if !mask[i] {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Pre-computed token-level FSM index.
///
/// Maps each DFA state to the set of vocabulary tokens that can be
/// appended without leaving the language defined by the regex/schema.
pub struct TokenFsmIndex {
    /// state_id → list of allowed token_ids
    state_to_allowed: HashMap<u32, Vec<u32>>,
    /// (state_id, token_id) → next_state_id
    transitions: HashMap<(u32, u32), u32>,
    /// Set of accepting states (HashSet for O(1) lookup)
    final_states: HashSet<u32>,
    /// Initial state
    initial_state: u32,
}

impl TokenFsmIndex {
    pub fn new(
        state_to_allowed: HashMap<u32, Vec<u32>>,
        transitions: HashMap<(u32, u32), u32>,
        final_states: Vec<u32>,
        initial_state: u32,
    ) -> Self {
        Self {
            state_to_allowed,
            transitions,
            final_states: final_states.into_iter().collect(),
            initial_state,
        }
    }
}

impl FsmConstraint for TokenFsmIndex {
    fn initial_state(&self) -> u32 {
        self.initial_state
    }

    fn allowed_tokens(&self, state: u32) -> Option<&[u32]> {
        self.state_to_allowed.get(&state).map(|v| v.as_slice())
    }

    fn next_state(&self, state: u32, token_id: u32) -> Option<u32> {
        self.transitions.get(&(state, token_id)).copied()
    }

    fn is_final_state(&self, state: u32) -> bool {
        self.final_states.contains(&state)
    }
}

/// Build a token-level FSM index from a character-level DFA.
///
/// For each DFA state, we try feeding each vocabulary token (character by
/// character) through the DFA. If all characters lead to valid transitions,
/// the token is allowed from that state, and the resulting DFA state after
/// the last character is the next state.
pub fn build_token_index(
    dfa: &dyn CharDfa,
    vocab: &TokenVocab,
) -> Result<TokenFsmIndex> {
    let all_states = dfa.all_states();
    let initial_state = dfa.initial_state();
    let final_states = dfa.final_states();

    let mut state_to_allowed: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut transitions: HashMap<(u32, u32), u32> = HashMap::new();

    for &state in &all_states {
        let mut allowed = Vec::new();

        for token_id in 0..vocab.vocab_size as u32 {
            let token_str = &vocab.id_to_token[token_id as usize];
            if token_str.is_empty() {
                continue;
            }

            // Walk the DFA through each character of this token
            let mut current = state;
            let mut valid = true;
            for byte in token_str.bytes() {
                match dfa.next_state(current, byte) {
                    Some(next) => current = next,
                    None => {
                        valid = false;
                        break;
                    }
                }
            }

            if valid {
                allowed.push(token_id);
                transitions.insert((state, token_id), current);
            }
        }

        state_to_allowed.insert(state, allowed);
    }

    Ok(TokenFsmIndex::new(
        state_to_allowed,
        transitions,
        final_states,
        initial_state,
    ))
}

/// Trait abstracting a character-level DFA for use in token index building.
pub trait CharDfa {
    fn initial_state(&self) -> u32;
    fn next_state(&self, state: u32, byte: u8) -> Option<u32>;
    fn is_final_state(&self, state: u32) -> bool;
    fn all_states(&self) -> Vec<u32>;
    fn final_states(&self) -> Vec<u32>;
}
