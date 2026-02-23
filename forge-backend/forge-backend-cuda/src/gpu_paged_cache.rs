//! GPU-resident paged KV cache.
//!
//! Pre-allocates per-layer K and V block pools as flat GPU buffers.
//! Append writes directly into block slots via `memcpy_dtod` (O(1)).
//! `get_kv` gathers from GPU blocks into contiguous tensors (for backward compat).
//! `paged_attention_meta` builds block tables on host, uploads to GPU, and returns
//! raw device pointers for the paged decode attention kernel.
//!
//! # Safety
//!
//! `PagedAttentionMeta` carries raw `u64` device pointers. These are valid because:
//! - **Pool pointers**: valid for the lifetime of this cache (pools are never reallocated).
//!   Cached at construction time to avoid repeated `device_ptr()` calls.
//! - **Block table / kv_lens pointers**: backed by `Arc<CudaSlice<i32>>` shared between
//!   the cache (for reuse across layers) and the `PagedAttentionMeta` (via `_keepalive`
//!   fields). The GPU memory remains valid as long as any `Arc` clone exists.
//!
//! This relies on `sizeof(u64) == sizeof(*void)` on 64-bit platforms, which is
//! enforced by a compile-time assertion below. CUDA device pointers are 64-bit
//! unsigned integers on all supported platforms; passing a `u64` via cudarc's
//! `PushKernelArg` pushes the raw 8-byte value, which the kernel interprets as
//! a typed device pointer (`const float*`, `const int*`, etc.).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use forge_core::{CacheUsage, DType, ForgeError, KvCache, PagedAttentionMeta, Result, Tensor};

use crate::tensor::CudaTensor;

// Compile-time assertion: device pointers must be representable as u64.
// CUDA device pointers are always 64-bit. If this fires, the raw u64 pointer
// passing pattern in PagedAttentionMeta will not work correctly.
const _: () = assert!(
    std::mem::size_of::<usize>() == std::mem::size_of::<u64>(),
    "PagedAttention requires a 64-bit platform (pointer size must equal u64 size)"
);

/// Per-layer GPU block pool storage, typed by dtype.
///
/// Layout: `[total_blocks * block_size * kv_dim]` elements.
/// Addressing: block `b`, slot `s` -> offset `(b * block_size + s) * kv_dim`.
enum PoolStorage {
    F32 {
        k_pool: CudaSlice<f32>,
        v_pool: CudaSlice<f32>,
    },
    F16 {
        k_pool: CudaSlice<half::f16>,
        v_pool: CudaSlice<half::f16>,
    },
}

/// Per-layer GPU block pool with cached base device pointers.
struct GpuBlockPool {
    storage: PoolStorage,
    /// Cached device pointer to K pool base (set once at construction).
    k_base_ptr: u64,
    /// Cached device pointer to V pool base (set once at construction).
    v_base_ptr: u64,
}

/// Per-sequence metadata.
struct SeqInfo {
    block_ids: Vec<usize>,
    num_tokens: usize,
}

/// GPU-resident paged KV cache.
///
/// Stores KV data directly on GPU in fixed-size blocks. Append is O(1) via
/// device-to-device memcpy into pre-allocated block slots. The paged attention
/// kernel reads KV data by following block table indirections.
pub struct GpuPagedKvCache {
    stream: Arc<CudaStream>,
    /// Per-layer K/V pools on GPU.
    pools: Vec<GpuBlockPool>,
    num_layers: usize,
    block_size: usize,
    total_blocks: usize,
    kv_dim: usize,
    /// Data type of the KV pools (F32 or F16). Exposed via `pool_dtype()`.
    dtype: DType,
    /// Free block IDs (stack).
    free_blocks: Vec<usize>,
    /// Per-sequence metadata.
    sequences: HashMap<u64, SeqInfo>,
    /// Monotonic counter incremented on any mutation (append layer 0, allocate, free).
    /// Used to detect when cached block tables need re-uploading.
    generation: u64,
    /// Cached block table upload, shared via Arc for reuse across layers.
    cached_generation: u64,
    cached_seq_ids: Vec<u64>,
    cached_block_tables: Option<Arc<CudaSlice<i32>>>,
    cached_kv_lens: Option<Arc<CudaSlice<i32>>>,
    cached_bt_ptr: u64,
    cached_kl_ptr: u64,
    cached_max_blocks_per_seq: usize,
}

impl GpuPagedKvCache {
    /// Create a new GPU paged KV cache.
    ///
    /// - `total_blocks`: total number of KV blocks per layer
    /// - `block_size`: tokens per block (typically 16)
    /// - `num_layers`: number of transformer layers
    /// - `num_kv_heads`: number of KV attention heads
    /// - `head_dim`: dimension per attention head
    /// - `dtype`: data type for KV storage (F32 or F16)
    pub fn new(
        backend: crate::backend::CudaBackend,
        total_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        // Block table indices are i32 on the GPU side.
        if total_blocks > i32::MAX as usize {
            return Err(ForgeError::InvalidArgument(
                "total_blocks exceeds i32::MAX; block table requires i32 indices".into(),
            ));
        }

        let kv_dim = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| ForgeError::InvalidArgument("kv_dim overflow".into()))?;
        let pool_size = total_blocks
            .checked_mul(block_size)
            .and_then(|x| x.checked_mul(kv_dim))
            .ok_or_else(|| ForgeError::InvalidArgument("pool_size overflow".into()))?;
        let stream = backend.stream.clone();

        let mut pools = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let (storage, k_base_ptr, v_base_ptr) = match dtype {
                DType::F32 => {
                    let k_pool = stream
                        .alloc_zeros::<f32>(pool_size)
                        .map_err(|e| ForgeError::Cuda(format!("alloc k_pool: {e}")))?;
                    let v_pool = stream
                        .alloc_zeros::<f32>(pool_size)
                        .map_err(|e| ForgeError::Cuda(format!("alloc v_pool: {e}")))?;
                    // Cache base pointers once (pools are never reallocated).
                    let k_ptr = {
                        let (ptr, _guard) = k_pool.device_ptr(&stream);
                        ptr
                    };
                    let v_ptr = {
                        let (ptr, _guard) = v_pool.device_ptr(&stream);
                        ptr
                    };
                    (PoolStorage::F32 { k_pool, v_pool }, k_ptr, v_ptr)
                }
                DType::F16 => {
                    let k_pool = stream
                        .alloc_zeros::<half::f16>(pool_size)
                        .map_err(|e| ForgeError::Cuda(format!("alloc k_pool: {e}")))?;
                    let v_pool = stream
                        .alloc_zeros::<half::f16>(pool_size)
                        .map_err(|e| ForgeError::Cuda(format!("alloc v_pool: {e}")))?;
                    let k_ptr = {
                        let (ptr, _guard) = k_pool.device_ptr(&stream);
                        ptr
                    };
                    let v_ptr = {
                        let (ptr, _guard) = v_pool.device_ptr(&stream);
                        ptr
                    };
                    (PoolStorage::F16 { k_pool, v_pool }, k_ptr, v_ptr)
                }
                other => {
                    return Err(ForgeError::UnsupportedDtype(other));
                }
            };
            pools.push(GpuBlockPool {
                storage,
                k_base_ptr,
                v_base_ptr,
            });
        }

        Ok(Self {
            stream,
            pools,
            num_layers,
            block_size,
            total_blocks,
            kv_dim,
            dtype,
            free_blocks: (0..total_blocks).rev().collect(),
            sequences: HashMap::new(),
            generation: 0,
            cached_generation: u64::MAX, // force initial cache miss
            cached_seq_ids: Vec::new(),
            cached_block_tables: None,
            cached_kv_lens: None,
            cached_bt_ptr: 0,
            cached_kl_ptr: 0,
            cached_max_blocks_per_seq: 0,
        })
    }

    /// Allocate `n` blocks from the free list.
    fn alloc_blocks(&mut self, n: usize) -> Result<Vec<usize>> {
        if self.free_blocks.len() < n {
            return Err(ForgeError::OutOfMemory(format!(
                "need {n} blocks, only {} free",
                self.free_blocks.len()
            )));
        }
        Ok((0..n).map(|_| self.free_blocks.pop().unwrap()).collect())
    }

    /// Return blocks to the free list.
    fn return_blocks(&mut self, blocks: &[usize]) {
        self.free_blocks.extend(blocks);
    }

    /// Compute element offset into the pool for a given block and slot.
    ///
    /// Returns an element index (not byte offset). Used with `CudaSlice::slice()`
    /// which also operates on element indices.
    fn pool_offset(&self, block_id: usize, slot: usize) -> usize {
        (block_id * self.block_size + slot) * self.kv_dim
    }

    /// The data type used for KV pool storage.
    pub fn pool_dtype(&self) -> DType {
        self.dtype
    }

    /// Invalidate the cached block table upload (called on any sequence mutation).
    ///
    /// Wrapping is safe: 2^64 mutations is unreachable in practice. The initial
    /// `cached_generation = u64::MAX` with `generation = 0` guarantees the first
    /// call to `ensure_block_table_cached` always misses.
    fn invalidate_cache(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Ensure the block table cache is up-to-date for the given seq_ids.
    ///
    /// If the cache is valid (same generation and same seq_ids), this is a no-op.
    /// Otherwise, rebuilds the padded block table on host and uploads to GPU.
    /// The upload is shared via Arc so it can be reused across all layers.
    fn ensure_block_table_cached(&mut self, seq_ids: &[u64]) -> Result<()> {
        // Check if cache is still valid
        if self.generation == self.cached_generation && seq_ids == self.cached_seq_ids.as_slice() {
            return Ok(());
        }

        let num_seqs = seq_ids.len();

        // Find max blocks across all sequences for padding
        let mut max_blocks_per_seq = 0usize;
        let mut kv_lens = Vec::with_capacity(num_seqs);
        for &sid in seq_ids {
            let seq = self
                .sequences
                .get(&sid)
                .ok_or(ForgeError::SeqNotFound(sid))?;
            if seq.block_ids.len() > max_blocks_per_seq {
                max_blocks_per_seq = seq.block_ids.len();
            }
            kv_lens.push(seq.num_tokens as i32);
        }

        // Build padded block table on host: [num_seqs * max_blocks_per_seq] i32
        let mut block_tables_host = vec![0i32; num_seqs * max_blocks_per_seq];
        for (i, &sid) in seq_ids.iter().enumerate() {
            let seq = self.sequences.get(&sid).unwrap();
            for (j, &bid) in seq.block_ids.iter().enumerate() {
                block_tables_host[i * max_blocks_per_seq + j] = bid as i32;
            }
        }

        // Upload to GPU
        let block_tables_dev = Arc::new(
            self.stream
                .memcpy_stod(&block_tables_host)
                .map_err(|e| ForgeError::Cuda(format!("upload block_tables: {e}")))?,
        );
        let kv_lens_dev = Arc::new(
            self.stream
                .memcpy_stod(&kv_lens)
                .map_err(|e| ForgeError::Cuda(format!("upload kv_lens: {e}")))?,
        );

        // Extract raw pointers (safe because Arc keeps memory alive).
        let bt_ptr = {
            let (ptr, _guard) = block_tables_dev.device_ptr(&self.stream);
            ptr
        };
        let kl_ptr = {
            let (ptr, _guard) = kv_lens_dev.device_ptr(&self.stream);
            ptr
        };

        self.cached_generation = self.generation;
        self.cached_seq_ids = seq_ids.to_vec();
        self.cached_block_tables = Some(block_tables_dev);
        self.cached_kv_lens = Some(kv_lens_dev);
        self.cached_bt_ptr = bt_ptr;
        self.cached_kl_ptr = kl_ptr;
        self.cached_max_blocks_per_seq = max_blocks_per_seq;

        Ok(())
    }
}

impl KvCache for GpuPagedKvCache {
    type T = CudaTensor;

    fn allocate(&mut self, seq_id: u64, initial_len: usize) -> Result<()> {
        if let Some(old) = self.sequences.remove(&seq_id) {
            self.return_blocks(&old.block_ids);
        }

        let num_blocks = ((initial_len + self.block_size - 1) / self.block_size).max(1);
        let block_ids = self.alloc_blocks(num_blocks)?;
        self.sequences.insert(
            seq_id,
            SeqInfo {
                block_ids,
                num_tokens: 0,
            },
        );
        self.invalidate_cache();
        Ok(())
    }

    fn append(
        &mut self,
        seq_id: u64,
        layer: usize,
        key: &CudaTensor,
        value: &CudaTensor,
    ) -> Result<()> {
        if layer >= self.num_layers {
            return Err(ForgeError::InvalidArgument(format!(
                "layer {layer} exceeds num_layers {}",
                self.num_layers
            )));
        }

        let key_shape = key.shape();
        let new_tokens = key_shape[0];
        let kv_dim = if key_shape.len() > 1 {
            key_shape[1]
        } else {
            1
        };
        if kv_dim != self.kv_dim {
            return Err(ForgeError::InvalidArgument(format!(
                "kv_dim mismatch: cache expects {}, got {kv_dim}",
                self.kv_dim
            )));
        }

        // Determine write start position.
        // IMPORTANT: Layers must be processed in order 0..num_layers for each step.
        // Layer 0's append increments num_tokens; subsequent layers use the updated count.
        let write_start = {
            let seq = self
                .sequences
                .get(&seq_id)
                .ok_or(ForgeError::SeqNotFound(seq_id))?;
            if layer == 0 {
                seq.num_tokens
            } else {
                seq.num_tokens - new_tokens
            }
        };

        // Pre-compute required blocks and allocate all at once (avoids fragile
        // alternating borrows on self.sequences and self.free_blocks in the loop).
        if new_tokens > 0 {
            let last_token_pos = write_start + new_tokens - 1;
            let max_block_idx = last_token_pos / self.block_size;
            let current_blocks = self
                .sequences
                .get(&seq_id)
                .unwrap()
                .block_ids
                .len();
            if max_block_idx >= current_blocks {
                let needed = max_block_idx - current_blocks + 1;
                let new_blocks = self.alloc_blocks(needed)?;
                self.sequences
                    .get_mut(&seq_id)
                    .unwrap()
                    .block_ids
                    .extend(new_blocks);
            }
        }

        // Collect block_ids into a local vec to avoid holding a borrow on
        // self.sequences during the memcpy loop (which borrows self.stream
        // and self.pools).
        let block_ids: Vec<usize> = self
            .sequences
            .get(&seq_id)
            .unwrap()
            .block_ids
            .clone();

        // Perform memcpy, dispatching on pool dtype.
        let stream = &self.stream;
        let pool = &mut self.pools[layer];
        let kv_dim = self.kv_dim;
        let block_size = self.block_size;

        match &mut pool.storage {
            PoolStorage::F32 { k_pool, v_pool } => {
                let k_slice = key.f32_slice()?;
                let v_slice = value.f32_slice()?;
                for t in 0..new_tokens {
                    let token_pos = write_start + t;
                    let block_idx = token_pos / block_size;
                    let slot = token_pos % block_size;
                    let block_id = block_ids[block_idx];
                    let dst_offset = (block_id * block_size + slot) * kv_dim;
                    let src_offset = t * kv_dim;

                    stream
                        .memcpy_dtod(
                            &k_slice.slice(src_offset..src_offset + kv_dim),
                            &mut k_pool.slice_mut(dst_offset..dst_offset + kv_dim),
                        )
                        .map_err(|e| ForgeError::Cuda(format!("memcpy k: {e}")))?;

                    stream
                        .memcpy_dtod(
                            &v_slice.slice(src_offset..src_offset + kv_dim),
                            &mut v_pool.slice_mut(dst_offset..dst_offset + kv_dim),
                        )
                        .map_err(|e| ForgeError::Cuda(format!("memcpy v: {e}")))?;
                }
            }
            PoolStorage::F16 { k_pool, v_pool } => {
                let k_slice = key.f16_slice()?;
                let v_slice = value.f16_slice()?;
                for t in 0..new_tokens {
                    let token_pos = write_start + t;
                    let block_idx = token_pos / block_size;
                    let slot = token_pos % block_size;
                    let block_id = block_ids[block_idx];
                    let dst_offset = (block_id * block_size + slot) * kv_dim;
                    let src_offset = t * kv_dim;

                    stream
                        .memcpy_dtod(
                            &k_slice.slice(src_offset..src_offset + kv_dim),
                            &mut k_pool.slice_mut(dst_offset..dst_offset + kv_dim),
                        )
                        .map_err(|e| ForgeError::Cuda(format!("memcpy k: {e}")))?;

                    stream
                        .memcpy_dtod(
                            &v_slice.slice(src_offset..src_offset + kv_dim),
                            &mut v_pool.slice_mut(dst_offset..dst_offset + kv_dim),
                        )
                        .map_err(|e| ForgeError::Cuda(format!("memcpy v: {e}")))?;
                }
            }
        }

        // Only update token count on layer 0
        if layer == 0 {
            self.sequences.get_mut(&seq_id).unwrap().num_tokens += new_tokens;
            self.invalidate_cache();
        }

        Ok(())
    }

    fn get_kv(&self, seq_id: u64, layer: usize) -> Result<(CudaTensor, CudaTensor)> {
        if layer >= self.num_layers {
            return Err(ForgeError::InvalidArgument(format!(
                "layer {layer} exceeds num_layers {}",
                self.num_layers
            )));
        }

        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;

        if seq.num_tokens == 0 {
            return Err(ForgeError::InvalidArgument(format!(
                "no cached KV for seq {seq_id} layer {layer}"
            )));
        }

        let total_elems = seq.num_tokens * self.kv_dim;
        let shape = vec![seq.num_tokens, self.kv_dim];
        let pool = &self.pools[layer];

        match &pool.storage {
            PoolStorage::F32 { k_pool, v_pool } => {
                let mut k_out = self
                    .stream
                    .alloc_zeros::<f32>(total_elems)
                    .map_err(|e| ForgeError::Cuda(format!("alloc k_out: {e}")))?;
                let mut v_out = self
                    .stream
                    .alloc_zeros::<f32>(total_elems)
                    .map_err(|e| ForgeError::Cuda(format!("alloc v_out: {e}")))?;

                let mut remaining = seq.num_tokens;
                let mut dst_offset = 0usize;

                for &block_id in &seq.block_ids {
                    let fill = remaining.min(self.block_size);
                    let src_offset = self.pool_offset(block_id, 0);
                    let copy_len = fill * self.kv_dim;

                    self.stream
                        .memcpy_dtod(
                            &k_pool.slice(src_offset..src_offset + copy_len),
                            &mut k_out.slice_mut(dst_offset..dst_offset + copy_len),
                        )
                        .map_err(|e| ForgeError::Cuda(format!("gather k: {e}")))?;

                    self.stream
                        .memcpy_dtod(
                            &v_pool.slice(src_offset..src_offset + copy_len),
                            &mut v_out.slice_mut(dst_offset..dst_offset + copy_len),
                        )
                        .map_err(|e| ForgeError::Cuda(format!("gather v: {e}")))?;

                    dst_offset += copy_len;
                    remaining -= fill;
                }

                Ok((
                    CudaTensor::f32_data(k_out, shape.clone()),
                    CudaTensor::f32_data(v_out, shape),
                ))
            }
            PoolStorage::F16 { k_pool, v_pool } => {
                let mut k_out = self
                    .stream
                    .alloc_zeros::<half::f16>(total_elems)
                    .map_err(|e| ForgeError::Cuda(format!("alloc k_out: {e}")))?;
                let mut v_out = self
                    .stream
                    .alloc_zeros::<half::f16>(total_elems)
                    .map_err(|e| ForgeError::Cuda(format!("alloc v_out: {e}")))?;

                let mut remaining = seq.num_tokens;
                let mut dst_offset = 0usize;

                for &block_id in &seq.block_ids {
                    let fill = remaining.min(self.block_size);
                    let src_offset = self.pool_offset(block_id, 0);
                    let copy_len = fill * self.kv_dim;

                    self.stream
                        .memcpy_dtod(
                            &k_pool.slice(src_offset..src_offset + copy_len),
                            &mut k_out.slice_mut(dst_offset..dst_offset + copy_len),
                        )
                        .map_err(|e| ForgeError::Cuda(format!("gather k: {e}")))?;

                    self.stream
                        .memcpy_dtod(
                            &v_pool.slice(src_offset..src_offset + copy_len),
                            &mut v_out.slice_mut(dst_offset..dst_offset + copy_len),
                        )
                        .map_err(|e| ForgeError::Cuda(format!("gather v: {e}")))?;

                    dst_offset += copy_len;
                    remaining -= fill;
                }

                Ok((
                    CudaTensor::f16_data(k_out, shape.clone()),
                    CudaTensor::f16_data(v_out, shape),
                ))
            }
        }
    }

    fn get_block_table(&self, seq_id: u64) -> Result<Vec<usize>> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(seq.block_ids.clone())
    }

    fn get_seq_len(&self, seq_id: u64) -> Result<usize> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(seq.num_tokens)
    }

    fn free(&mut self, seq_id: u64) -> Result<()> {
        let seq = self
            .sequences
            .remove(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        self.return_blocks(&seq.block_ids);
        self.invalidate_cache();
        Ok(())
    }

    fn usage(&self) -> CacheUsage {
        CacheUsage {
            total_blocks: self.total_blocks,
            used_blocks: self.total_blocks - self.free_blocks.len(),
            block_size: self.block_size,
        }
    }

    fn can_allocate(&self, num_tokens: usize) -> bool {
        let blocks_needed = ((num_tokens + self.block_size - 1) / self.block_size).max(1);
        self.free_blocks.len() >= blocks_needed
    }

    fn supports_paged_attention(&self) -> bool {
        true
    }

    fn paged_attention_meta(
        &mut self,
        seq_ids: &[u64],
        layer: usize,
    ) -> Result<PagedAttentionMeta> {
        if layer >= self.num_layers {
            return Err(ForgeError::InvalidArgument(format!(
                "layer {layer} exceeds num_layers {}",
                self.num_layers
            )));
        }

        // Ensure block table is cached (no-op if already current for these seq_ids).
        // Block tables and kv_lens are identical across layers; only pool pointers differ.
        self.ensure_block_table_cached(seq_ids)?;

        let pool = &self.pools[layer];

        Ok(PagedAttentionMeta {
            block_tables_ptr: self.cached_bt_ptr,
            kv_lens_ptr: self.cached_kl_ptr,
            k_pool_ptr: pool.k_base_ptr,
            v_pool_ptr: pool.v_base_ptr,
            max_blocks_per_seq: self.cached_max_blocks_per_seq,
            block_size: self.block_size,
            kv_dim: self.kv_dim,
            num_seqs: seq_ids.len(),
            // Arc clones keep the GPU allocations alive until this meta is dropped.
            // The cache also holds Arc clones for reuse across layers.
            _block_tables_keepalive: Box::new(Arc::clone(
                self.cached_block_tables.as_ref().unwrap(),
            )),
            _kv_lens_keepalive: Box::new(Arc::clone(self.cached_kv_lens.as_ref().unwrap())),
        })
    }
}
