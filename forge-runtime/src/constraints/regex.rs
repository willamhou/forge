//! Regex-based FSM constraint using `regex-automata` DFA.

use std::collections::HashSet;

use regex_automata::dfa::dense;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID;
use regex_automata::util::start;
use regex_automata::Anchored;

use forge_core::{ForgeError, Result};

use super::fsm::{CharDfa, TokenVocab, build_token_index, TokenFsmIndex};

/// Maximum allowed regex pattern length to prevent abuse.
const MAX_PATTERN_LEN: usize = 8192;
/// Maximum DFA size in bytes (10 MB).
const MAX_DFA_SIZE: usize = 10 * 1024 * 1024;

/// A character-level DFA built from a regex pattern.
pub struct RegexDfa {
    dfa: dense::DFA<Vec<u32>>,
    /// Cached mapping from our u32 state IDs to StateIDs
    state_map: Vec<StateID>,
    /// Reverse mapping: StateID → u32
    state_id_to_u32: std::collections::HashMap<StateID, u32>,
    initial: u32,
    /// EOI state ID (state after sending end-of-input)
    eoi_state_map: std::collections::HashMap<StateID, StateID>,
}

impl RegexDfa {
    /// Compile a regex pattern into a character-level DFA.
    pub fn new(pattern: &str) -> Result<Self> {
        if pattern.len() > MAX_PATTERN_LEN {
            return Err(ForgeError::InvalidArgument(format!(
                "regex pattern too long ({} bytes, max {})",
                pattern.len(),
                MAX_PATTERN_LEN
            )));
        }

        // Build DFA with size limits to prevent memory exhaustion
        let dfa = dense::Builder::new()
            .configure(
                dense::DFA::config()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .dfa_size_limit(Some(MAX_DFA_SIZE)),
            )
            .build(pattern)
            .map_err(|e| {
                ForgeError::Internal(format!("regex DFA compilation error: {e}"))
            })?;

        // Get start state with anchored config (matches start of input)
        let start_config = start::Config::new().anchored(Anchored::Yes);
        let start_id = dfa
            .start_state(&start_config)
            .map_err(|e| ForgeError::Internal(format!("no start state: {e}")))?;

        // Enumerate all reachable states via traversal
        let mut state_map = Vec::new();
        let mut state_id_to_u32 = std::collections::HashMap::new();
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        let mut eoi_state_map = std::collections::HashMap::new();

        stack.push(start_id);
        visited.insert(start_id);

        while let Some(sid) = stack.pop() {
            let u32_id = state_map.len() as u32;
            state_id_to_u32.insert(sid, u32_id);
            state_map.push(sid);

            // Explore transitions for all byte values
            for byte in 0..=255u8 {
                let next = dfa.next_state(sid, byte);
                if !visited.contains(&next) {
                    visited.insert(next);
                    stack.push(next);
                }
            }

            // Also explore the EOI transition for final state detection
            let eoi = dfa.next_eoi_state(sid);
            eoi_state_map.insert(sid, eoi);
            if !visited.contains(&eoi) {
                visited.insert(eoi);
                stack.push(eoi);
            }
        }

        let initial = state_id_to_u32[&start_id];

        Ok(Self {
            dfa,
            state_map,
            state_id_to_u32,
            initial,
            eoi_state_map,
        })
    }

    fn sid_from_u32(&self, state: u32) -> Option<StateID> {
        self.state_map.get(state as usize).copied()
    }
}

impl CharDfa for RegexDfa {
    fn initial_state(&self) -> u32 {
        self.initial
    }

    fn next_state(&self, state: u32, byte: u8) -> Option<u32> {
        let sid = self.sid_from_u32(state)?;
        let next_sid = self.dfa.next_state(sid, byte);
        // Dead states are not valid transitions
        if self.dfa.is_dead_state(next_sid) {
            return None;
        }
        self.state_id_to_u32.get(&next_sid).copied()
    }

    fn is_final_state(&self, state: u32) -> bool {
        // A state is final if, after sending EOI (end-of-input), the DFA
        // transitions to a match state. This correctly handles $ anchoring.
        let sid = match self.sid_from_u32(state) {
            Some(s) => s,
            None => return false,
        };
        // Check if this state is already a match state (no $ needed)
        if self.dfa.is_match_state(sid) {
            return true;
        }
        // Check if sending EOI from this state reaches a match state
        if let Some(&eoi_sid) = self.eoi_state_map.get(&sid) {
            return self.dfa.is_match_state(eoi_sid);
        }
        false
    }

    fn all_states(&self) -> Vec<u32> {
        (0..self.state_map.len() as u32).collect()
    }

    fn final_states(&self) -> Vec<u32> {
        (0..self.state_map.len() as u32)
            .filter(|i| self.is_final_state(*i))
            .collect()
    }
}

/// Build a token-level FSM index from a regex pattern and vocabulary.
pub fn build_regex_fsm(pattern: &str, vocab: &TokenVocab) -> Result<TokenFsmIndex> {
    let char_dfa = RegexDfa::new(pattern)?;
    build_token_index(&char_dfa, vocab)
}
