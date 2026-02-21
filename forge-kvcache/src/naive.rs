//! Naive KV cache: stores K/V as CPU-side f32 vectors, reconstructing
//! device tensors on retrieval. Correct but slow — paired with naive attention.
//! Will be replaced by paged GPU-side cache with FlashAttention in Task 16.

use std::collections::HashMap;

use forge_core::{Backend, CacheUsage, ForgeError, KvCache, Result, Tensor};

/// Per-layer cached K and V for a single sequence.
struct LayerCache {
    /// Accumulated key data, [total_tokens, kv_dim] flattened
    key_data: Vec<f32>,
    /// Accumulated value data, [total_tokens, kv_dim] flattened
    value_data: Vec<f32>,
    /// Width of each row (num_kv_heads * head_dim)
    kv_dim: usize,
    /// Number of tokens cached
    num_tokens: usize,
}

/// Per-sequence cache across all layers.
struct SeqCache {
    layers: HashMap<usize, LayerCache>,
}

pub struct NaiveKvCache<B: Backend> {
    backend: B,
    sequences: HashMap<u64, SeqCache>,
    num_layers: usize,
    max_sequences: usize,
    /// Maximum total tokens across all sequences (for usage reporting).
    max_total_tokens: usize,
}

impl<B: Backend> NaiveKvCache<B> {
    pub fn new(backend: B, num_layers: usize, max_sequences: usize) -> Self {
        // Default to 128K tokens total capacity (reasonable for naive CPU-side cache).
        Self::with_max_tokens(backend, num_layers, max_sequences, 128 * 1024)
    }

    pub fn with_max_tokens(
        backend: B,
        num_layers: usize,
        max_sequences: usize,
        max_total_tokens: usize,
    ) -> Self {
        Self {
            backend,
            sequences: HashMap::new(),
            num_layers,
            max_sequences,
            max_total_tokens,
        }
    }
}

impl<B: Backend + Clone> KvCache for NaiveKvCache<B> {
    type T = B::Tensor;

    fn allocate(&mut self, seq_id: u64, _initial_len: usize) -> Result<()> {
        if self.sequences.len() >= self.max_sequences && !self.sequences.contains_key(&seq_id) {
            return Err(ForgeError::OutOfMemory(format!(
                "max_sequences ({}) reached",
                self.max_sequences
            )));
        }
        self.sequences.insert(
            seq_id,
            SeqCache {
                layers: HashMap::new(),
            },
        );
        Ok(())
    }

    fn append(
        &mut self,
        seq_id: u64,
        layer: usize,
        key: &B::Tensor,
        value: &B::Tensor,
    ) -> Result<()> {
        if layer >= self.num_layers {
            return Err(ForgeError::InvalidArgument(format!(
                "layer {layer} exceeds num_layers {}",
                self.num_layers
            )));
        }

        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;

        // key/value shape: [new_tokens, kv_dim]
        let key_shape = key.shape();
        let new_tokens = key_shape[0];
        let kv_dim = if key_shape.len() > 1 { key_shape[1] } else { 1 };

        let key_f32 = self.backend.copy_to_host_f32(key)?;
        let val_f32 = self.backend.copy_to_host_f32(value)?;

        let layer_cache = seq.layers.entry(layer).or_insert_with(|| LayerCache {
            key_data: Vec::new(),
            value_data: Vec::new(),
            kv_dim,
            num_tokens: 0,
        });

        layer_cache.key_data.extend_from_slice(&key_f32);
        layer_cache.value_data.extend_from_slice(&val_f32);
        layer_cache.num_tokens += new_tokens;

        Ok(())
    }

    fn get_kv(&self, seq_id: u64, layer: usize) -> Result<(B::Tensor, B::Tensor)> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;

        let layer_cache = seq.layers.get(&layer).ok_or_else(|| {
            ForgeError::InvalidArgument(format!(
                "No cached KV for seq {seq_id} layer {layer}"
            ))
        })?;

        let shape = &[layer_cache.num_tokens, layer_cache.kv_dim];
        let key = self.backend.copy_from_host_f32(&layer_cache.key_data, shape)?;
        let value = self.backend.copy_from_host_f32(&layer_cache.value_data, shape)?;
        Ok((key, value))
    }

    fn get_block_table(&self, _seq_id: u64) -> Result<Vec<usize>> {
        // Naive cache doesn't use block tables
        Ok(Vec::new())
    }

    fn get_seq_len(&self, seq_id: u64) -> Result<usize> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;

        // All layers have the same token count; use layer 0 if it exists
        Ok(seq
            .layers
            .values()
            .next()
            .map(|lc| lc.num_tokens)
            .unwrap_or(0))
    }

    fn free(&mut self, seq_id: u64) -> Result<()> {
        self.sequences
            .remove(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(())
    }

    fn usage(&self) -> CacheUsage {
        // Sum the token count per sequence (use max across layers per seq).
        let used_tokens: usize = self
            .sequences
            .values()
            .map(|seq| {
                seq.layers
                    .values()
                    .map(|lc| lc.num_tokens)
                    .max()
                    .unwrap_or(0)
            })
            .sum();

        CacheUsage {
            total_blocks: self.max_total_tokens,
            used_blocks: used_tokens,
            block_size: 1,
        }
    }

    fn can_allocate(&self, num_tokens: usize) -> bool {
        if self.sequences.len() >= self.max_sequences {
            return false;
        }
        let usage = self.usage();
        usage.free_blocks() >= num_tokens
    }
}
