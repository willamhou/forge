//! Naive KV cache: stores K/V as device tensors (GPU-resident on CUDA).
//!
//! Uses `backend.cat()` for appending (device-to-device, no CPU roundtrip).
//! `get_kv` returns cloned tensor references (cheap Arc bump on CudaBackend).
//! Phase 2 will replace with paged GPU-side cache + PagedAttention.

use std::collections::HashMap;

use forge_core::{Backend, CacheUsage, ForgeError, KvCache, Result, Tensor};

/// Per-layer cached K and V tensors for a single sequence.
struct LayerCache<T: Tensor> {
    /// Accumulated key tensor: [total_tokens, kv_dim] on device
    key: Option<T>,
    /// Accumulated value tensor: [total_tokens, kv_dim] on device
    value: Option<T>,
    /// Number of tokens cached
    num_tokens: usize,
}

/// Per-sequence cache across all layers.
struct SeqCache<T: Tensor> {
    layers: HashMap<usize, LayerCache<T>>,
}

pub struct NaiveKvCache<B: Backend> {
    backend: B,
    sequences: HashMap<u64, SeqCache<B::Tensor>>,
    num_layers: usize,
    max_sequences: usize,
    /// Maximum total tokens across all sequences (for usage reporting).
    max_total_tokens: usize,
}

impl<B: Backend> NaiveKvCache<B> {
    pub fn new(backend: B, num_layers: usize, max_sequences: usize) -> Self {
        // Default to 128K tokens total capacity.
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

        let new_tokens = key.shape()[0];

        // Take existing tensors from cache (releases &mut borrow on sequences
        // so we can call self.backend methods below).
        let (existing_key, existing_value, prev_tokens) = {
            let seq = self
                .sequences
                .get_mut(&seq_id)
                .ok_or(ForgeError::SeqNotFound(seq_id))?;

            let layer_cache = seq.layers.entry(layer).or_insert_with(|| LayerCache {
                key: None,
                value: None,
                num_tokens: 0,
            });

            (
                layer_cache.key.take(),
                layer_cache.value.take(),
                layer_cache.num_tokens,
            )
        };

        // Concatenate on device (no CPU roundtrip)
        let new_key = match existing_key {
            Some(existing) => self.backend.cat(&[&existing, key], 0)?,
            None => key.clone(),
        };
        let new_value = match existing_value {
            Some(existing) => self.backend.cat(&[&existing, value], 0)?,
            None => value.clone(),
        };

        // Store back
        let seq = self.sequences.get_mut(&seq_id).unwrap();
        let layer_cache = seq.layers.get_mut(&layer).unwrap();
        layer_cache.key = Some(new_key);
        layer_cache.value = Some(new_value);
        layer_cache.num_tokens = prev_tokens + new_tokens;

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

        let key = layer_cache
            .key
            .as_ref()
            .ok_or_else(|| {
                ForgeError::Internal(format!("seq {seq_id} layer {layer}: key tensor missing"))
            })?
            .clone();
        let value = layer_cache
            .value
            .as_ref()
            .ok_or_else(|| {
                ForgeError::Internal(format!("seq {seq_id} layer {layer}: value tensor missing"))
            })?
            .clone();

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
