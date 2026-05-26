use crate::Result;
use crate::tensor::Tensor;

/// Inputs for a fused paged-attention kernel call, produced by paged caches
/// and consumed by the model's decode path. Naive (non-paged) caches return
/// `None` from `KvCache::paged_attention_inputs` — the caller falls back to
/// the `get_kv` + `multi_head_attention` path.
pub struct PagedAttentionInputs<'a, T: Tensor> {
    /// Per-layer K pool tensor (device-resident). Shape: `[num_blocks, block_size, num_kv_heads * head_dim]`.
    pub k_pool: &'a T,
    /// Per-layer V pool, same shape as `k_pool`.
    pub v_pool: &'a T,
    /// Row-major `[batch * max_blocks_per_seq]` i32; `-1` is padding.
    pub block_tables: Vec<i32>,
    /// Per-sequence KV length, `[batch]`.
    pub kv_lens: Vec<i32>,
    /// Stride of `block_tables` (the table's second dimension).
    pub max_blocks_per_seq: usize,
}

pub struct CacheUsage {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub block_size: usize,
}

impl CacheUsage {
    pub fn free_blocks(&self) -> usize {
        self.total_blocks.saturating_sub(self.used_blocks)
    }

    pub fn usage_ratio(&self) -> f32 {
        self.used_blocks as f32 / self.total_blocks as f32
    }
}

pub trait KvCache: Send + Sync {
    type T: Tensor;

    /// Allocate cache space for a new sequence.
    fn allocate(&mut self, seq_id: u64, initial_len: usize) -> Result<()>;

    /// Append new KV to cache for a specific layer.
    fn append(&mut self, seq_id: u64, layer: usize, key: &Self::T, value: &Self::T) -> Result<()>;

    /// Retrieve the full cached K and V for a specific layer.
    /// Returns (key, value) where each is [total_cached_len, num_kv_heads * head_dim].
    fn get_kv(&self, seq_id: u64, layer: usize) -> Result<(Self::T, Self::T)>;

    /// Get block table for a sequence (PagedAttention).
    fn get_block_table(&self, seq_id: u64) -> Result<Vec<usize>>;

    /// Get the current sequence length in cache.
    fn get_seq_len(&self, seq_id: u64) -> Result<usize>;

    /// Free cache for a completed sequence.
    fn free(&mut self, seq_id: u64) -> Result<()>;

    /// Current cache usage.
    fn usage(&self) -> CacheUsage;

    /// Check if we can allocate for a given length.
    fn can_allocate(&self, num_tokens: usize) -> bool;

    /// Assemble inputs for the fused paged-attention kernel.
    ///
    /// - `Ok(None)`: this cache doesn't support paged attention — the caller
    ///   falls back to the `get_kv` + `multi_head_attention` path.
    /// - `Ok(Some(inputs))`: paged inputs assembled; caller can dispatch the
    ///   fused single-kernel paged_attention launch.
    /// - `Err(_)`: paged inputs were requested but could not be assembled
    ///   (e.g. unknown seq_id, layer out of bounds).
    ///
    /// `Result<Option<_>>` rather than `Option<Result<_>>` so the call site
    /// can use `?` to propagate errors and then match on the `Option` for
    /// the dispatch decision.
    fn paged_attention_inputs<'a>(
        &'a self,
        _layer: usize,
        _seq_ids: &[u64],
    ) -> Result<Option<PagedAttentionInputs<'a, Self::T>>> {
        Ok(None)
    }

    /// Advance the cache by one decode token per sequence: allocate blocks as
    /// needed and bump per-sequence lengths (host bookkeeping only — no device
    /// KV write). Returns the absolute pool slot for each sequence's new token,
    /// in `seq_ids` order.
    ///
    /// This is the bookkeeping half of an `append`, split out for the
    /// capture-safe decode path: it must run on **every** decode step (so the
    /// cache stays consistent even when the device KV write is a replayed
    /// graph), whereas the device write happens in [`Self::scatter_decode`]
    /// inside the captured region.
    ///
    /// Default: unsupported (only paged caches implement the capture-safe path).
    fn advance_decode(&mut self, _seq_ids: &[u64]) -> Result<Vec<i32>> {
        Err(crate::ForgeError::InvalidArgument(
            "advance_decode: this KV cache does not support the capture-safe decode path".into(),
        ))
    }

    /// Capture-safe device KV write for one layer: scatter the batch's `key`/
    /// `value` rows (`[n_rows, kv_dim]`) into the layer's pool at the slots
    /// from the matching [`Self::advance_decode`] (passed as `slot_mapping`).
    /// Pure device op — the slot mapping was staged on-device beforehand.
    ///
    /// Default: unsupported.
    fn scatter_decode(
        &mut self,
        _layer: usize,
        _key: &Self::T,
        _value: &Self::T,
        _slot_mapping: &[i32],
        _n_rows: usize,
    ) -> Result<()> {
        Err(crate::ForgeError::InvalidArgument(
            "scatter_decode: this KV cache does not support the capture-safe decode path".into(),
        ))
    }
}
