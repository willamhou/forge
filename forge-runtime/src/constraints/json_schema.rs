//! JSON Schema → regex converter for structured output.
//!
//! Converts a JSON Schema into a regex pattern that matches any valid JSON
//! string conforming to the schema. The regex is then compiled into a DFA
//! for token-level constrained generation.

use forge_core::{ForgeError, Result};

use super::fsm::{TokenFsmIndex, TokenVocab};
use super::regex::build_regex_fsm;

// JSON literal patterns
const WS: &str = r"[ \t\n\r]*";
const STRING_INNER: &str = r#"([^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"#;
const INTEGER: &str = r"-?(0|[1-9][0-9]*)";
const NUMBER: &str = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?";
const BOOLEAN: &str = r"(true|false)";
const NULL: &str = r"null";

/// Convert a JSON Schema string to a regex pattern.
pub fn schema_to_regex(schema_str: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_str)
        .map_err(|e| ForgeError::InvalidArgument(format!("invalid JSON schema: {e}")))?;
    schema_node_to_regex(&schema)
}

/// Build a token-level FSM from a JSON Schema string and vocabulary.
pub fn build_json_schema_fsm(schema_str: &str, vocab: &TokenVocab) -> Result<TokenFsmIndex> {
    let pattern = schema_to_regex(schema_str)?;
    build_regex_fsm(&pattern, vocab)
}

/// Recursively convert a JSON Schema node to a regex pattern.
fn schema_node_to_regex(schema: &serde_json::Value) -> Result<String> {
    // Handle boolean schemas
    if let Some(b) = schema.as_bool() {
        return if b {
            // true schema: matches any JSON value
            Ok(any_json_value())
        } else {
            Err(ForgeError::InvalidArgument(
                "false schema rejects all values".into(),
            ))
        };
    }

    let obj = schema.as_object().ok_or_else(|| {
        ForgeError::InvalidArgument("schema must be an object or boolean".into())
    })?;

    // Handle enum
    if let Some(enum_values) = obj.get("enum") {
        return enum_to_regex(enum_values);
    }

    // Handle const
    if let Some(const_val) = obj.get("const") {
        return Ok(json_value_to_regex_literal(const_val));
    }

    // Handle anyOf / oneOf
    if let Some(any_of) = obj.get("anyOf").or_else(|| obj.get("oneOf")) {
        return any_of_to_regex(any_of);
    }

    // Handle allOf (just use the first schema — simplified)
    if let Some(all_of) = obj.get("allOf") {
        if let Some(arr) = all_of.as_array() {
            if let Some(first) = arr.first() {
                return schema_node_to_regex(first);
            }
        }
    }

    // Handle $ref — not supported in this simple implementation
    if obj.contains_key("$ref") {
        return Err(ForgeError::InvalidArgument(
            "$ref is not supported; inline all definitions".into(),
        ));
    }

    // Determine type — missing `type` means any JSON value is valid
    let type_str = match obj.get("type").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return Ok(any_json_value()),
    };

    match type_str {
        "string" => string_schema_to_regex(obj),
        "integer" => Ok(INTEGER.to_string()),
        "number" => Ok(NUMBER.to_string()),
        "boolean" => Ok(BOOLEAN.to_string()),
        "null" => Ok(NULL.to_string()),
        "array" => array_schema_to_regex(obj),
        "object" => object_schema_to_regex(obj),
        other => Err(ForgeError::InvalidArgument(format!(
            "unsupported JSON Schema type: {other}"
        ))),
    }
}

/// Convert a string schema to regex, respecting minLength, maxLength, pattern.
fn string_schema_to_regex(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<String> {
    if let Some(pattern) = obj.get("pattern").and_then(|v| v.as_str()) {
        // Validate the inner pattern compiles independently to prevent injection
        regex_automata::dfa::dense::DFA::new(pattern).map_err(|e| {
            ForgeError::InvalidArgument(format!("invalid string pattern: {e}"))
        })?;
        // Wrap in a non-capturing group to prevent breakout.
        // JSON Schema `pattern` means the string content must fully match the
        // regex, so wrap without repetition quantifier.
        return Ok(format!(r#""(?:{pattern})""#));
    }

    let min_len = obj
        .get("minLength")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let max_len = obj.get("maxLength").and_then(|v| v.as_u64());

    let char_pattern = if let Some(enumeration) = obj.get("enum") {
        return enum_to_regex(enumeration);
    } else {
        // Standard JSON string character
        format!(r#"([^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{{4}})"#)
    };

    let quantifier = match max_len {
        Some(max) => format!("{{{},{}}}", min_len, max),
        None if min_len > 0 => format!("{{{},}}", min_len),
        None => "*".to_string(),
    };

    Ok(format!(r#""{}{}""#, char_pattern, quantifier))
}

/// Convert an array schema to regex.
fn array_schema_to_regex(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<String> {
    let item_regex = if let Some(items) = obj.get("items") {
        schema_node_to_regex(items)?
    } else {
        any_json_value()
    };

    let min_items = obj
        .get("minItems")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let max_items = obj.get("maxItems").and_then(|v| v.as_u64());

    if min_items == 0 && max_items.is_none() {
        // Zero or more items
        Ok(format!(
            r"\[{WS}({item_regex}({WS},{WS}{item_regex})*)?{WS}\]"
        ))
    } else if let Some(max) = max_items {
        let max = max as usize;
        if max == 0 {
            // Empty array only
            Ok(format!(r"\[{WS}\]"))
        } else if min_items == 0 {
            // Up to max items
            let additional = if max > 1 {
                format!("({WS},{WS}{item_regex}){{0,{}}}", max - 1)
            } else {
                String::new()
            };
            Ok(format!(
                r"\[{WS}({item_regex}{additional})?{WS}\]"
            ))
        } else {
            // min..=max items
            let required_extra = if min_items > 1 {
                format!("({WS},{WS}{item_regex}){{{}}}", min_items - 1)
            } else {
                String::new()
            };
            let optional_extra = if max > min_items {
                format!("({WS},{WS}{item_regex}){{0,{}}}", max - min_items)
            } else {
                String::new()
            };
            Ok(format!(
                r"\[{WS}{item_regex}{required_extra}{optional_extra}{WS}\]"
            ))
        }
    } else {
        // min or more items (no max)
        let required_extra = if min_items > 1 {
            format!("({WS},{WS}{item_regex}){{{}}}", min_items - 1)
        } else {
            String::new()
        };
        Ok(format!(
            r"\[{WS}{item_regex}{required_extra}({WS},{WS}{item_regex})*{WS}\]"
        ))
    }
}

/// Maximum number of object properties that can use the full permutation
/// strategy. Beyond this limit we fall back to fixed-order to prevent
/// combinatorial explosion (n! permutations).
const MAX_PERMUTATION_PROPERTIES: usize = 6;

/// Convert an object schema to regex.
///
/// Properties can appear in any order (JSON semantics). Required properties
/// must always be present; optional properties may be omitted.
///
/// For small property counts (<= 6) we enumerate all valid permutations.
/// For larger schemas we fall back to fixed property order to avoid
/// factorial blowup in regex size.
fn object_schema_to_regex(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<String> {
    let properties = obj.get("properties").and_then(|v| v.as_object());
    let required: std::collections::HashSet<&str> = obj
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    match properties {
        Some(props) if !props.is_empty() => {
            let prop_entries: Vec<(&String, &serde_json::Value)> = props.iter().collect();

            // Build per-property regex patterns and required flags.
            let mut prop_patterns: Vec<String> = Vec::new();
            let mut is_required: Vec<bool> = Vec::new();
            for (key, value_schema) in &prop_entries {
                let value_regex = schema_node_to_regex(value_schema)?;
                let escaped_key = regex_escape(key);
                let prop_pattern = format!(
                    r#"{WS}"{escaped_key}"{WS}:{WS}{value_regex}"#
                );
                prop_patterns.push(prop_pattern);
                is_required.push(required.contains(key.as_str()));
            }

            let n = prop_patterns.len();

            // For large schemas, fall back to fixed-order output with
            // optional wrapping. This avoids n! regex blowup.
            if n > MAX_PERMUTATION_PROPERTIES {
                return object_schema_fixed_order(&prop_patterns, &is_required);
            }

            // Generate all valid orderings. Each ordering is a permutation of
            // the property indices where required properties are always present
            // and optional properties may be included or omitted.
            // n is guaranteed <= MAX_PERMUTATION_PROPERTIES (6) here,
            // so 1u64 << n and n! are both safe.
            let mut valid_combos: Vec<Vec<usize>> = Vec::new();
            for mask in 0..(1u64 << n) {
                let subset: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
                // Must include all required properties.
                let has_all_required = (0..n)
                    .filter(|&i| is_required[i])
                    .all(|i| subset.contains(&i));
                if has_all_required {
                    valid_combos.push(subset);
                }
            }

            // For each valid subset, generate all permutations.
            let mut alternatives: Vec<String> = Vec::new();

            // Empty object (only if no required properties)
            if !is_required.iter().any(|&r| r) {
                alternatives.push(String::new()); // matches \{ WS \}
            }

            for combo in &valid_combos {
                if combo.is_empty() {
                    continue; // handled by empty-object case above
                }
                let mut perms = Vec::new();
                permutations(combo, &mut perms);
                for perm in perms {
                    let joined = perm
                        .iter()
                        .map(|&i| prop_patterns[i].as_str())
                        .collect::<Vec<_>>()
                        .join(&format!("{WS},"));
                    alternatives.push(joined);
                }
            }

            if alternatives.is_empty() {
                // Should not happen, but fall back to empty object
                Ok(format!(r"\{{{WS}\}}"))
            } else if alternatives.len() == 1 && alternatives[0].is_empty() {
                Ok(format!(r"\{{{WS}\}}"))
            } else {
                let non_empty: Vec<&str> = alternatives
                    .iter()
                    .filter(|a| !a.is_empty())
                    .map(|a| a.as_str())
                    .collect();
                let has_empty = alternatives.iter().any(|a| a.is_empty());
                let inner = non_empty.join("|");
                if has_empty {
                    Ok(format!(r"\{{{WS}({inner})?{WS}\}}"))
                } else {
                    Ok(format!(r"\{{{WS}({inner}){WS}\}}"))
                }
            }
        }
        _ => {
            // Empty object or no properties specified
            Ok(format!(r"\{{{WS}\}}"))
        }
    }
}

/// Fixed-order fallback for schemas with more than MAX_PERMUTATION_PROPERTIES.
///
/// Produces a regex where properties appear in the iteration order provided
/// by the schema, with optional properties wrapped in `(...)?`.
///
/// Strategy: each non-first property carries a leading comma inside its
/// group. Optional properties at the start have no comma and are followed
/// by a conditional comma on the next entry.
fn object_schema_fixed_order(prop_patterns: &[String], is_required: &[bool]) -> Result<String> {
    if prop_patterns.is_empty() {
        return Ok(format!(r"\{{{WS}\}}"));
    }

    // Split into required and optional indices for cleaner generation.
    // In fixed-order mode we output all required properties separated by
    // commas, and each optional property is wrapped with its own leading
    // comma in a `(...)?` group.  This avoids double/trailing/leading
    // comma issues regardless of which optionals are present.
    //
    // Example with (opt, req1, opt2, req2):
    //   ({opt}{WS},)? {req1} ({WS},{opt2})? {WS},{req2}
    //
    // The trick: the first required property never has a leading comma.
    // Optional properties *before* the first required carry a trailing comma
    // inside their group (consumed by the following required property).
    // Optional properties *after* a required carry a leading comma inside
    // their group.

    let mut parts: Vec<String> = Vec::new();
    let mut seen_required = false;

    for (i, pattern) in prop_patterns.iter().enumerate() {
        if is_required[i] {
            if !seen_required {
                // First required property: no leading comma, but preceding
                // optional properties (if any) already end with a comma.
                parts.push(pattern.clone());
                seen_required = true;
            } else {
                // Subsequent required: always preceded by comma
                parts.push(format!("{WS},{pattern}"));
            }
        } else if !seen_required {
            // Optional before any required property: include trailing comma
            // so the following required property doesn't get a spurious comma
            // when this optional is absent.
            parts.push(format!("({pattern}{WS},)?"));
        } else {
            // Optional after a required property: leading comma inside group
            parts.push(format!("({WS},{pattern})?"));
        }
    }

    let inner = parts.join("");
    if !seen_required {
        // All optional — the entire body is optional
        // Strip trailing commas from the optional groups since there's
        // nothing following them. Rebuild without trailing commas.
        let mut opt_parts: Vec<String> = Vec::new();
        for pattern in prop_patterns {
            if opt_parts.is_empty() {
                opt_parts.push(pattern.clone());
            } else {
                opt_parts.push(format!("{WS},{pattern}"));
            }
        }
        let opt_inner = opt_parts.join("");
        Ok(format!(r"\{{{WS}({opt_inner})?{WS}\}}"))
    } else {
        Ok(format!(r"\{{{WS}{inner}{WS}\}}"))
    }
}

/// Generate all permutations of a slice.
fn permutations(items: &[usize], result: &mut Vec<Vec<usize>>) {
    if items.len() <= 1 {
        result.push(items.to_vec());
        return;
    }
    for (i, &item) in items.iter().enumerate() {
        let mut rest: Vec<usize> = items.to_vec();
        rest.remove(i);
        let mut sub_perms = Vec::new();
        permutations(&rest, &mut sub_perms);
        for mut perm in sub_perms {
            perm.insert(0, item);
            result.push(perm);
        }
    }
}

/// Convert an enum to regex (alternation of literal values).
fn enum_to_regex(values: &serde_json::Value) -> Result<String> {
    let arr = values.as_array().ok_or_else(|| {
        ForgeError::InvalidArgument("enum must be an array".into())
    })?;

    let alternatives: Vec<String> = arr
        .iter()
        .map(|v| json_value_to_regex_literal(v))
        .collect();

    Ok(format!("({})", alternatives.join("|")))
}

/// Convert anyOf/oneOf to regex (alternation).
fn any_of_to_regex(schemas: &serde_json::Value) -> Result<String> {
    let arr = schemas.as_array().ok_or_else(|| {
        ForgeError::InvalidArgument("anyOf/oneOf must be an array".into())
    })?;

    let alternatives: Result<Vec<String>> = arr
        .iter()
        .map(schema_node_to_regex)
        .collect();

    Ok(format!("({})", alternatives?.join("|")))
}

/// Convert a JSON value to its regex literal representation.
fn json_value_to_regex_literal(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => NULL.to_string(),
        serde_json::Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        serde_json::Value::Number(n) => regex_escape(&n.to_string()),
        serde_json::Value::String(s) => {
            // JSON-encode the string content first (handles quotes, backslashes,
            // control characters), then regex-escape the encoded interior.
            let json_encoded = serde_json::to_string(s).unwrap_or_default();
            // json_encoded includes surrounding quotes, so strip them and
            // regex-escape the interior to produce a literal regex match.
            let inner = &json_encoded[1..json_encoded.len() - 1];
            format!(r#""{}""#, regex_escape(inner))
        }
        serde_json::Value::Array(arr) => {
            let elements: Vec<String> = arr.iter().map(json_value_to_regex_literal).collect();
            format!(
                r"\[{WS}{}{WS}\]",
                elements.join(&format!("{WS},{WS}"))
            )
        }
        serde_json::Value::Object(obj) => {
            let entries: Vec<String> = obj
                .iter()
                .map(|(k, v)| {
                    let json_key = serde_json::to_string(k).unwrap_or_default();
                    let key_inner = &json_key[1..json_key.len() - 1];
                    format!(
                        r#"{WS}"{}"{WS}:{WS}{}"#,
                        regex_escape(key_inner),
                        json_value_to_regex_literal(v)
                    )
                })
                .collect();
            format!(
                r"\{{{WS}{}{WS}\}}",
                entries.join(&format!("{WS},{WS}"))
            )
        }
    }
}

/// Regex pattern matching any JSON value.
///
/// Uses self-referencing patterns for arrays and objects to ensure only
/// syntactically valid JSON is accepted.  The recursion is bounded to
/// two levels of nesting since deeper recursion would produce an overly
/// large DFA; for unconstrained schemas this is a reasonable trade-off.
fn any_json_value() -> String {
    // Leaf values: strings, numbers, booleans, null
    let leaf = format!(
        r#""{}"|{}|{}|{}"#,
        STRING_INNER, NUMBER, BOOLEAN, NULL
    );
    // Level-0: leaf only (used as elements inside level-1 containers)
    let atom = format!("({leaf})");
    // Level-1: leaf or one-level array/object
    let array_1 = format!(r"\[{WS}({atom}({WS},{WS}{atom})*)?{WS}\]");
    let obj_val = &atom;
    let obj_entry = format!(r#"{WS}"{}"{WS}:{WS}{obj_val}"#, STRING_INNER);
    let object_1 = format!(r"\{{{WS}({obj_entry}({WS},{WS}{obj_entry})*)?{WS}\}}");
    format!("({leaf}|{array_1}|{object_1})")
}

/// Escape special regex characters in a string.
fn regex_escape(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^'
            | '$' => {
                escaped.push('\\');
                escaped.push(c);
            }
            _ => escaped.push(c),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_schema() {
        let schema = r#"{"type": "string"}"#;
        let regex = schema_to_regex(schema).unwrap();
        assert!(regex.contains('"'));
    }

    #[test]
    fn test_integer_schema() {
        let schema = r#"{"type": "integer"}"#;
        let regex = schema_to_regex(schema).unwrap();
        assert_eq!(regex, INTEGER);
    }

    #[test]
    fn test_boolean_schema() {
        let schema = r#"{"type": "boolean"}"#;
        let regex = schema_to_regex(schema).unwrap();
        assert_eq!(regex, BOOLEAN);
    }

    #[test]
    fn test_null_schema() {
        let schema = r#"{"type": "null"}"#;
        let regex = schema_to_regex(schema).unwrap();
        assert_eq!(regex, NULL);
    }

    #[test]
    fn test_enum_schema() {
        let schema = r#"{"enum": ["red", "green", "blue"]}"#;
        let regex = schema_to_regex(schema).unwrap();
        assert!(regex.contains("red"));
        assert!(regex.contains("green"));
        assert!(regex.contains("blue"));
    }

    #[test]
    fn test_object_schema() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }"#;
        let regex = schema_to_regex(schema).unwrap();
        assert!(regex.contains("name"));
        assert!(regex.contains("age"));
    }

    #[test]
    fn test_regex_escape() {
        assert_eq!(regex_escape("hello"), "hello");
        assert_eq!(regex_escape("a.b"), r"a\.b");
        assert_eq!(regex_escape("a+b"), r"a\+b");
    }

    #[test]
    fn test_all_optional_properties_comma_handling() {
        // Schema: two optional properties, no required
        // Must match: {}, {"a":1}, {"a":1,"b":2}
        // Must NOT match: {"a":1"b":2} (missing comma)
        let schema = r#"{
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            }
        }"#;
        let regex = schema_to_regex(schema).unwrap();
        let dfa = regex_automata::dfa::dense::DFA::new(&regex).unwrap();
        use regex_automata::dfa::Automaton;
        use regex_automata::Anchored;
        use regex_automata::util::start;

        let matches = |input: &str| -> bool {
            let start_config = start::Config::new().anchored(Anchored::Yes);
            let Ok(start_state) = dfa.start_state(&start_config) else {
                return false;
            };
            let mut state = start_state;
            for &b in input.as_bytes() {
                state = dfa.next_state(state, b);
            }
            state = dfa.next_eoi_state(state);
            dfa.is_match_state(state)
        };

        assert!(matches(r#"{}"#), "should match empty object");
        assert!(matches(r#"{"a":1}"#), "should match single optional");
        assert!(matches(r#"{"a":1,"b":2}"#), "should match both optionals with comma");
        assert!(!matches(r#"{"a":1"b":2}"#), "should reject missing comma between optionals");
    }

    #[test]
    fn test_optional_before_required_no_leading_comma() {
        // Schema: optional "opt" then required "req"
        // Must match: {"req":1} and {"opt":"x","req":1}
        // Must NOT match: {,"req":1}
        let schema = r#"{
            "type": "object",
            "properties": {
                "opt": {"type": "string"},
                "req": {"type": "integer"}
            },
            "required": ["req"]
        }"#;
        let regex = schema_to_regex(schema).unwrap();
        let dfa = regex_automata::dfa::dense::DFA::new(&regex).unwrap();
        use regex_automata::dfa::Automaton;
        use regex_automata::Anchored;
        use regex_automata::util::start;

        // Helper to check if a string matches the regex
        let matches = |input: &str| -> bool {
            let start_config = start::Config::new().anchored(Anchored::Yes);
            let Ok(start_state) = dfa.start_state(&start_config) else {
                return false;
            };
            let mut state = start_state;
            for &b in input.as_bytes() {
                state = dfa.next_state(state, b);
            }
            state = dfa.next_eoi_state(state);
            dfa.is_match_state(state)
        };

        // Valid: required only
        assert!(matches(r#"{"req":1}"#), "should match required-only");
        // Invalid: leading comma
        assert!(!matches(r#"{,"req":1}"#), "should reject leading comma");
    }
}
