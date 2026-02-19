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
        // Try to find EOS token id from common conventions
        let eos_token_id = inner
            .token_to_id("</s>")
            .or_else(|| inner.token_to_id("<|endoftext|>"))
            .or_else(|| inner.token_to_id("<|im_end|>"))
            .unwrap_or(2); // fallback
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
pub struct IncrementalDecoder {
    pending_ids: Vec<u32>,
    prev_text_len: usize,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self {
            pending_ids: Vec::new(),
            prev_text_len: 0,
        }
    }

    pub fn add_token(&mut self, token_id: u32, tokenizer: &ForgeTokenizer) -> Option<String> {
        self.pending_ids.push(token_id);
        let decoded = tokenizer.decode(&self.pending_ids).ok()?;

        if decoded.ends_with('\u{FFFD}') {
            return None; // incomplete UTF-8
        }

        let new_text = decoded[self.prev_text_len..].to_string();
        self.prev_text_len = decoded.len();

        if new_text.is_empty() {
            None
        } else {
            Some(new_text)
        }
    }
}
