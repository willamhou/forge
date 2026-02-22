//! Tests for the FSM constraint system: regex and JSON schema.

use forge_runtime::constraints::fsm::{FsmConstraint, TokenVocab};
use forge_runtime::constraints::json_schema::schema_to_regex;
use forge_runtime::constraints::regex::{RegexDfa, build_regex_fsm};

/// Helper: create a simple vocabulary for testing.
/// Maps token_id to single characters or short strings.
fn test_vocab(tokens: &[&str]) -> TokenVocab {
    let id_to_token: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
    let vocab_size = id_to_token.len();
    TokenVocab {
        id_to_token,
        vocab_size,
    }
}

// ===== RegexDfa Tests =====

#[test]
fn test_regex_dfa_simple_literal() {
    use forge_runtime::constraints::fsm::CharDfa;
    let dfa = RegexDfa::new("abc").unwrap();
    let initial = dfa.initial_state();

    // Walk through 'a', 'b', 'c'
    let s1 = dfa.next_state(initial, b'a').expect("a should be valid");
    let s2 = dfa.next_state(s1, b'b').expect("b should be valid");
    let s3 = dfa.next_state(s2, b'c').expect("c should be valid");
    assert!(dfa.is_final_state(s3));
    assert!(!dfa.is_final_state(initial));
    assert!(!dfa.is_final_state(s1));
}

#[test]
fn test_regex_dfa_alternation() {
    use forge_runtime::constraints::fsm::CharDfa;
    let dfa = RegexDfa::new("(yes|no)").unwrap();
    let initial = dfa.initial_state();

    // "yes" path
    let s1 = dfa.next_state(initial, b'y').expect("y valid");
    let s2 = dfa.next_state(s1, b'e').expect("e valid");
    let s3 = dfa.next_state(s2, b's').expect("s valid");
    assert!(dfa.is_final_state(s3));

    // "no" path
    let n1 = dfa.next_state(initial, b'n').expect("n valid");
    let n2 = dfa.next_state(n1, b'o').expect("o valid");
    assert!(dfa.is_final_state(n2));

    // Invalid: 'x'
    assert!(dfa.next_state(initial, b'x').is_none());
}

#[test]
fn test_regex_dfa_digit_pattern() {
    use forge_runtime::constraints::fsm::CharDfa;
    let dfa = RegexDfa::new("[0-9]+").unwrap();
    let initial = dfa.initial_state();

    let s1 = dfa.next_state(initial, b'5').expect("digit valid");
    assert!(dfa.is_final_state(s1));

    let s2 = dfa.next_state(s1, b'3').expect("second digit valid");
    assert!(dfa.is_final_state(s2));

    // Letters invalid from initial
    assert!(dfa.next_state(initial, b'a').is_none());
}

// ===== Token-Level FSM Tests =====

#[test]
fn test_token_fsm_simple() {
    // Vocab: 0="a", 1="b", 2="c", 3="ab", 4="bc"
    let vocab = test_vocab(&["a", "b", "c", "ab", "bc"]);

    let fsm = build_regex_fsm("abc", &vocab).unwrap();

    let initial = fsm.initial_state();
    let allowed = fsm.allowed_tokens(initial).unwrap();

    // From initial, "a" (token 0) and "ab" (token 3) should be allowed
    assert!(allowed.contains(&0), "token 'a' should be allowed");
    assert!(allowed.contains(&3), "token 'ab' should be allowed");
    assert!(!allowed.contains(&1), "token 'b' should NOT be allowed from start");

    // After token "a" (0), we should be able to continue with "b" or "bc"
    let next = fsm.next_state(initial, 0).unwrap();
    let allowed2 = fsm.allowed_tokens(next).unwrap();
    assert!(allowed2.contains(&1), "token 'b' should be allowed after 'a'");
    assert!(allowed2.contains(&4), "token 'bc' should be allowed after 'a'");
}

#[test]
fn test_token_fsm_full_walk() {
    // Vocab: individual characters
    let vocab = test_vocab(&["h", "e", "l", "o", "he", "lo", "hello"]);

    let fsm = build_regex_fsm("hello", &vocab).unwrap();

    // Walk: "hello" as single token
    let initial = fsm.initial_state();
    let allowed = fsm.allowed_tokens(initial).unwrap();
    assert!(allowed.contains(&6), "token 'hello' should be allowed");

    // After "hello", we should be in a final state
    let final_state = fsm.next_state(initial, 6).unwrap();
    assert!(fsm.is_final_state(final_state));
}

#[test]
fn test_token_fsm_alternation() {
    let vocab = test_vocab(&["y", "e", "s", "n", "o", "yes", "no"]);

    let fsm = build_regex_fsm("(yes|no)", &vocab).unwrap();
    let initial = fsm.initial_state();
    let allowed = fsm.allowed_tokens(initial).unwrap();

    // "yes" (5) and "no" (6) and "y" (0) and "n" (3) should be allowed
    assert!(allowed.contains(&5), "token 'yes' allowed");
    assert!(allowed.contains(&6), "token 'no' allowed");
    assert!(allowed.contains(&0), "token 'y' allowed");
    assert!(allowed.contains(&3), "token 'n' allowed");
    // "e", "s", "o" should NOT be allowed from start
    assert!(!allowed.contains(&1), "token 'e' not allowed from start");
}

#[test]
fn test_fsm_mask_logits() {
    let vocab = test_vocab(&["a", "b", "c"]);
    let fsm = build_regex_fsm("a", &vocab).unwrap();

    let initial = fsm.initial_state();
    let mut logits = vec![1.0, 2.0, 3.0];
    fsm.mask_logits(initial, &mut logits);

    // Only "a" (token 0) should remain, others should be -inf
    assert!(logits[0].is_finite(), "token 'a' logit should remain");
    assert!(logits[1].is_infinite() && logits[1] < 0.0, "token 'b' should be -inf");
    assert!(logits[2].is_infinite() && logits[2] < 0.0, "token 'c' should be -inf");
}

// ===== Sampler Integration Test =====

#[test]
fn test_sampler_with_constraint() {
    use forge_core::SamplingParams;
    use forge_runtime::sampling::CpuSampler;

    // Vocab: 0="a", 1="b", 2="c"
    let vocab = test_vocab(&["a", "b", "c"]);
    let fsm = build_regex_fsm("a", &vocab).unwrap();

    // Logits favor "c" (token 2) heavily, but constraint only allows "a" (token 0)
    let logits = vec![0.1, 0.2, 10.0];
    let params = SamplingParams {
        temperature: 0.0, // greedy
        ..Default::default()
    };
    let sampler = CpuSampler;
    let result = sampler
        .sample_with_constraint(&logits, &params, &[], Some((&fsm, fsm.initial_state())))
        .unwrap();

    assert_eq!(result.token_id, 0, "should select 'a' despite 'c' having higher logit");
}

#[test]
fn test_sampler_without_constraint() {
    use forge_core::SamplingParams;
    use forge_runtime::sampling::CpuSampler;

    let logits = vec![0.1, 0.2, 10.0];
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let sampler = CpuSampler;
    let result = sampler
        .sample_with_constraint(&logits, &params, &[], None)
        .unwrap();

    assert_eq!(result.token_id, 2, "should select highest logit without constraint");
}

// ===== JSON Schema Tests =====

#[test]
fn test_schema_to_regex_string() {
    let schema = r#"{"type": "string"}"#;
    let regex = schema_to_regex(schema).unwrap();
    // Should produce a pattern that matches JSON strings
    assert!(regex.starts_with('"'));
    assert!(regex.ends_with('"'));
}

#[test]
fn test_schema_to_regex_integer() {
    let schema = r#"{"type": "integer"}"#;
    let regex = schema_to_regex(schema).unwrap();
    // Should match integers like 42, -1, 0
    let dfa = RegexDfa::new(&regex).unwrap();
    use forge_runtime::constraints::fsm::CharDfa;

    // Test "42"
    let mut state = dfa.initial_state();
    for byte in b"42" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));

    // Test "-1"
    let mut state = dfa.initial_state();
    for byte in b"-1" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));

    // Test "0"
    let mut state = dfa.initial_state();
    for byte in b"0" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));
}

#[test]
fn test_schema_to_regex_boolean() {
    let schema = r#"{"type": "boolean"}"#;
    let regex = schema_to_regex(schema).unwrap();
    let dfa = RegexDfa::new(&regex).unwrap();
    use forge_runtime::constraints::fsm::CharDfa;

    // "true"
    let mut state = dfa.initial_state();
    for byte in b"true" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));

    // "false"
    let mut state = dfa.initial_state();
    for byte in b"false" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));
}

#[test]
fn test_schema_to_regex_number() {
    let schema = r#"{"type": "number"}"#;
    let regex = schema_to_regex(schema).unwrap();
    let dfa = RegexDfa::new(&regex).unwrap();
    use forge_runtime::constraints::fsm::CharDfa;

    // "3.14"
    let mut state = dfa.initial_state();
    for byte in b"3.14" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));

    // "-1.5e10"
    let mut state = dfa.initial_state();
    for byte in b"-1.5e10" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));
}

#[test]
fn test_schema_to_regex_enum() {
    let schema = r#"{"enum": ["red", "green"]}"#;
    let regex = schema_to_regex(schema).unwrap();
    let dfa = RegexDfa::new(&regex).unwrap();
    use forge_runtime::constraints::fsm::CharDfa;

    // "red"  (note: enum values are JSON strings, so include quotes)
    let mut state = dfa.initial_state();
    for byte in br#""red""# {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));
}

#[test]
fn test_schema_to_regex_const() {
    let schema = r#"{"const": "hello"}"#;
    let regex = schema_to_regex(schema).unwrap();
    let dfa = RegexDfa::new(&regex).unwrap();
    use forge_runtime::constraints::fsm::CharDfa;

    let mut state = dfa.initial_state();
    for byte in br#""hello""# {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));
}

#[test]
fn test_schema_to_regex_null() {
    let schema = r#"{"type": "null"}"#;
    let regex = schema_to_regex(schema).unwrap();
    let dfa = RegexDfa::new(&regex).unwrap();
    use forge_runtime::constraints::fsm::CharDfa;

    let mut state = dfa.initial_state();
    for byte in b"null" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));
}

#[test]
fn test_schema_to_regex_anyof() {
    let schema = r#"{"anyOf": [{"type": "integer"}, {"type": "boolean"}]}"#;
    let regex = schema_to_regex(schema).unwrap();
    let dfa = RegexDfa::new(&regex).unwrap();
    use forge_runtime::constraints::fsm::CharDfa;

    // Integer "42"
    let mut state = dfa.initial_state();
    for byte in b"42" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));

    // Boolean "true"
    let mut state = dfa.initial_state();
    for byte in b"true" {
        state = dfa.next_state(state, *byte).expect("should be valid");
    }
    assert!(dfa.is_final_state(state));
}

// ===== Token-level JSON Schema FSM Test =====

#[test]
fn test_json_schema_token_fsm() {
    // Test that a boolean schema constrains token generation
    let schema = r#"{"type": "boolean"}"#;

    // Vocab: individual characters + common tokens
    let vocab = test_vocab(&[
        "t", "r", "u", "e", "f", "a", "l", "s",
        "true", "false", "null", "1", "0",
    ]);

    use forge_runtime::constraints::json_schema::build_json_schema_fsm;
    let fsm = build_json_schema_fsm(schema, &vocab).unwrap();
    let initial = fsm.initial_state();
    let allowed = fsm.allowed_tokens(initial).unwrap();

    // "true" (8) and "false" (9) should be allowed
    assert!(allowed.contains(&8), "token 'true' should be allowed");
    assert!(allowed.contains(&9), "token 'false' should be allowed");
    // "t" (0) and "f" (4) should be allowed (start of "true"/"false")
    assert!(allowed.contains(&0), "token 't' should be allowed");
    assert!(allowed.contains(&4), "token 'f' should be allowed");
    // "null" (10), "1" (11), "0" (12) should NOT be allowed
    assert!(!allowed.contains(&10), "token 'null' should not be allowed");
    assert!(!allowed.contains(&11), "token '1' should not be allowed");
    assert!(!allowed.contains(&12), "token '0' should not be allowed");
}

// ===== Edge Cases =====

#[test]
fn test_empty_token_skipped() {
    // Empty tokens should not be in the allowed set
    let vocab = test_vocab(&["", "a", "b"]);
    let fsm = build_regex_fsm("a", &vocab).unwrap();
    let initial = fsm.initial_state();
    let allowed = fsm.allowed_tokens(initial).unwrap();

    assert!(!allowed.contains(&0), "empty token should not be allowed");
    assert!(allowed.contains(&1), "token 'a' should be allowed");
}

#[test]
fn test_invalid_regex_returns_error() {
    let result = RegexDfa::new("[invalid");
    assert!(result.is_err());
}

#[test]
fn test_invalid_json_schema_returns_error() {
    let result = schema_to_regex("not valid json");
    assert!(result.is_err());
}

#[test]
fn test_unsupported_schema_ref() {
    let schema = r##"{"$ref": "#/definitions/Foo"}"##;
    let result = schema_to_regex(schema);
    assert!(result.is_err());
}
