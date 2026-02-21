use std::path::Path;

use forge_core::{ForgeError, Result};
use tokenizers::Tokenizer as HfTokenizer;

pub struct ForgeTokenizer {
    inner: HfTokenizer,
    eos_token_id: u32,
}

impl ForgeTokenizer {
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
        // (e.g. whitespace normalization). If decoded is shorter than expected,
        // reset prev_text_len to avoid panic or dropped characters.
        if decoded.len() < self.prev_text_len {
            self.prev_text_len = 0;
        }
        let new_text = decoded[self.prev_text_len..].to_string();

        if new_text.is_empty() {
            return None;
        }

        // Reset to avoid O(N^2) re-decoding. Keep the last token as context
        // for multi-byte character boundaries.
        if self.pending_ids.len() > 4 {
            let keep = self.pending_ids.len().saturating_sub(2);
            self.pending_ids.drain(..keep);
            // Recompute prev_text_len relative to the trimmed buffer
            self.prev_text_len = tokenizer
                .decode(&self.pending_ids)
                .map(|s| s.len())
                .unwrap_or(0);
        } else {
            self.prev_text_len = decoded.len();
        }

        Some(new_text)
    }

    /// Flush any remaining pending bytes (e.g. on stream finish).
    pub fn flush(&mut self, tokenizer: &ForgeTokenizer) -> Option<String> {
        if self.pending_ids.is_empty() {
            return None;
        }
        let decoded = tokenizer.decode(&self.pending_ids).ok()?;
        if decoded.len() < self.prev_text_len {
            self.prev_text_len = 0;
        }
        let new_text = decoded[self.prev_text_len..].to_string();
        self.pending_ids.clear();
        self.prev_text_len = 0;
        if new_text.is_empty() {
            None
        } else {
            Some(new_text)
        }
    }
}
