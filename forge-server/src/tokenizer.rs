use std::path::Path;

use forge_core::{ForgeError, Result};
use tokenizers::Tokenizer as HfTokenizer;

pub struct ForgeTokenizer {
    inner: HfTokenizer,
    eos_token_id: u32,
}

impl ForgeTokenizer {
    /// Create from an existing HuggingFace tokenizer instance.
    pub fn from_inner(inner: HfTokenizer, eos_token_id: u32) -> Self {
        Self {
            inner,
            eos_token_id,
        }
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let inner =
            HfTokenizer::from_file(path).map_err(|e| ForgeError::Tokenizer(e.to_string()))?;

        // Try to read eos_token_id from tokenizer_config.json (most reliable source).
        let config_eos = path.parent().and_then(|dir| {
            let config_path = dir.join("tokenizer_config.json");
            let text = std::fs::read_to_string(config_path).ok()?;
            let value: serde_json::Value = serde_json::from_str(&text).ok()?;
            // Try numeric eos_token_id first, then resolve eos_token string
            if let Some(id) = value.get("eos_token_id").and_then(|v| v.as_u64()) {
                return Some(id as u32);
            }
            // eos_token can be a string or {"content": "..."} object
            let eos_str = value.get("eos_token").and_then(|v| {
                v.as_str()
                    .map(String::from)
                    .or_else(|| v.get("content").and_then(|c| c.as_str()).map(String::from))
            })?;
            inner.token_to_id(&eos_str)
        });

        // Fall back to common token strings, then to id 2
        let eos_token_id = config_eos
            .or_else(|| inner.token_to_id("</s>"))
            .or_else(|| inner.token_to_id("<|endoftext|>"))
            .or_else(|| inner.token_to_id("<|im_end|>"))
            .or_else(|| inner.token_to_id("<|eot_id|>"))
            .unwrap_or(2);

        Ok(Self {
            inner,
            eos_token_id,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| ForgeError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| ForgeError::Tokenizer(e.to_string()))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

/// Incremental decoder for streaming token-by-token output.
#[derive(Default)]
pub struct IncrementalDecoder {
    pending_ids: Vec<u32>,
    prev_text_len: usize,
    /// Cached prefix text corresponding to `prev_text_len` bytes. Used to
    /// detect suffix rewrites where `decoded.len()` stays the same but the
    /// content changes (e.g. tokenizer whitespace normalization).
    prev_text: String,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_token(&mut self, token_id: u32, tokenizer: &ForgeTokenizer) -> Option<String> {
        self.pending_ids.push(token_id);
        let decoded = tokenizer.decode(&self.pending_ids).ok()?;

        if decoded.ends_with('\u{FFFD}') {
            return None; // incomplete UTF-8, keep accumulating
        }

        // Guard against tokenizers that rewrite earlier text on new tokens
        // (e.g. whitespace normalization). If the decoded prefix no longer
        // matches what we previously emitted, skip this token's output since
        // we cannot unsend already-streamed text.
        if decoded.len() < self.prev_text_len
            || !decoded.starts_with(&self.prev_text)
        {
            // Update tracking to the full decoded text so subsequent tokens
            // diff correctly against the current decode state.
            self.prev_text_len = decoded.len();
            self.prev_text = decoded;
            return None;
        }
        let new_text = decoded[self.prev_text_len..].to_string();

        if new_text.is_empty() {
            return None;
        }

        self.prev_text_len = decoded.len();
        self.prev_text = decoded;

        Some(new_text)
    }

    /// Flush any remaining pending bytes (e.g. on stream finish).
    pub fn flush(&mut self, tokenizer: &ForgeTokenizer) -> Option<String> {
        if self.pending_ids.is_empty() {
            return None;
        }
        let decoded = tokenizer.decode(&self.pending_ids).ok()?;
        self.pending_ids.clear();

        // On rewrite, skip re-emitting already-streamed text.
        let new_text = if decoded.len() < self.prev_text_len
            || !decoded.starts_with(&self.prev_text)
        {
            String::new()
        } else {
            decoded[self.prev_text_len..].to_string()
        };

        self.prev_text_len = 0;
        self.prev_text.clear();
        if new_text.is_empty() {
            None
        } else {
            Some(new_text)
        }
    }
}
