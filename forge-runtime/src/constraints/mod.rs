//! Structured output constraints via finite state machines (FSMs).
//!
//! Constraints guide token generation by masking disallowed tokens at each step.
//! Two constraint types are supported:
//! - **Regex**: compile a regex pattern into a character-level DFA, then build a
//!   token-level index mapping FSM states to allowed vocabulary tokens.
//! - **JSON Schema**: convert a JSON Schema to a regex pattern, then use the
//!   regex FSM machinery.

pub mod fsm;
pub mod json_schema;
pub mod regex;

pub use fsm::{FsmConstraint, TokenVocab};
