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

    // Determine type
    let type_str = obj
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("object");

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
        // Use `+` (one or more) rather than `*` so the pattern must match at least
        // once, preserving the schema's constraint semantics.
        return Ok(format!(r#""(?:{pattern})+""#));
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
        if min_items == 0 {
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

/// Convert an object schema to regex.
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

            // Build the object pattern.
            // Each property after the first needs a comma separator, but when
            // an optional property precedes a required one the comma must be
            // conditional on the optional property actually appearing.
            let mut result = format!(r"\{{{WS}");
            let mut prev_was_required = false;
            let mut any_emitted = false;

            for (key, value_schema) in &prop_entries {
                let value_regex = schema_node_to_regex(value_schema)?;
                let escaped_key = regex_escape(key);
                let prop_pattern = format!(
                    r#"{WS}"{escaped_key}"{WS}:{WS}{value_regex}"#
                );

                let is_required = required.contains(key.as_str());

                if !any_emitted {
                    // First property — no comma needed
                    if is_required {
                        result.push_str(&prop_pattern);
                        prev_was_required = true;
                    } else {
                        result.push_str(&format!("({prop_pattern})?"));
                        prev_was_required = false;
                    }
                } else if is_required {
                    if prev_was_required {
                        // Both this and previous are required: mandatory comma
                        result.push_str(&format!("{WS},{prop_pattern}"));
                    } else {
                        // Previous was optional: comma is conditional on it
                        result.push_str(&format!("({WS},)?{prop_pattern}"));
                    }
                    prev_was_required = true;
                } else {
                    // Optional property after something — include comma in the group
                    result.push_str(&format!("({WS},{prop_pattern})?"));
                    prev_was_required = false;
                }

                any_emitted = true;
            }

            result.push_str(&format!("{WS}\\}}"));
            Ok(result)
        }
        _ => {
            // Empty object or no properties specified
            Ok(format!(r"\{{{WS}\}}"))
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
            format!(r#""{}""#, regex_escape(s))
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
                    format!(
                        r#"{WS}"{}"{WS}:{WS}{}"#,
                        regex_escape(k),
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
fn any_json_value() -> String {
    // Simplified: matches strings, numbers, booleans, null, arrays, objects
    format!(
        r#"("{}"|{}|{}|{}|\[.*\]|\{{.*\}})"#,
        STRING_INNER, NUMBER, BOOLEAN, NULL
    )
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
}
