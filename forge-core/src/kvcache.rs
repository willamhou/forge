use crate::tensor::Tensor;
use crate::Result;

/// Metadata for paged attention kernel dispatch.
///
/// Contains raw GPU device pointers (as u64) to avoid CUDA type dependencies
/// in forge-core. The kernel reads KV data from block pools using block table
/// indirection instead of requiring contiguous per-sequence KV tensors.
pub struct PagedAttentionMeta {
    /// GPU pointer to block tables: [num_seqs, max_blocks_per_seq] i32
    pub block_tables_ptr: u64,
    /// GPU pointer to per-sequence KV lengths: [num_seqs] i32
    pub kv_lens_ptr: u64,
    /// GPU pointer to base of K pool for this layer
    pub k_pool_ptr: u64,
    /// GPU pointer to base of V pool for this layer
    pub v_pool_ptr: u64,
    /// Maximum number of blocks per sequence (padding width of block_tables)
    pub max_blocks_per_seq: usize,
    /// Tokens per block
    pub block_size: usize,
    /// num_kv_heads * head_dim
    pub kv_dim: usize,
    /// Number of sequences in this batch
    pub num_seqs: usize,
    /// Opaque handles that keep the GPU allocations backing `block_tables_ptr`
    /// and `kv_lens_ptr` alive until this struct is dropped. Without these,
    /// the device memory would be freed before the kernel reads it.
    pub _block_tables_keepalive: Box<dyn std::any::Any + Send + Sync>,
    pub _kv_lens_keepalive: Box<dyn std::any::Any + Send + Sync>,
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
        if self.total_blocks == 0 {
            return 0.0;
        }
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

    /// Whether this cache supports the paged attention kernel path.
    /// When true, callers can use `paged_attention_meta` instead of `get_kv`
    /// during decode to avoid materializing contiguous KV tensors.
    fn supports_paged_attention(&self) -> bool {
        false
    }

    /// Build metadata for the paged attention decode kernel.
    ///
    /// Uploads block tables and KV lengths to the GPU and returns raw device
    /// pointers that the kernel can consume directly.
    fn paged_attention_meta(
        &mut self,
        _seq_ids: &[u64],
        _layer: usize,
    ) -> Result<PagedAttentionMeta> {
        Err(crate::ForgeError::InvalidArgument(
            "paged attention not supported by this cache".into(),
        ))
    }
}
