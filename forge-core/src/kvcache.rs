use crate::tensor::Tensor;
use crate::Result;

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
    fn append(
        &mut self,
        seq_id: u64,
        layer: usize,
        key: &Self::T,
        value: &Self::T,
    ) -> Result<()>;

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
    /// Default returns `None` — the caller falls back to the `get_kv` +
    /// `multi_head_attention` path. Paged caches override to return
    /// `Some(Ok(...))` so the decode hot path can use a fused single-kernel
    /// launch.
    ///
    /// Returning `Some(Err(_))` signals that paged inputs were requested but
    /// could not be assembled (e.g. a seq id is missing) — distinct from "this
    /// cache type doesn't support paged attention" (`None`).
    fn paged_attention_inputs<'a>(
        &'a self,
        _layer: usize,
        _seq_ids: &[u64],
    ) -> Option<Result<PagedAttentionInputs<'a, Self::T>>> {
        None
    }
}
