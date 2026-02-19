use crate::tensor::Tensor;
use crate::Result;

pub struct CacheUsage {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub block_size: usize,
}

impl CacheUsage {
    pub fn free_blocks(&self) -> usize {
        self.total_blocks - self.used_blocks
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
}
