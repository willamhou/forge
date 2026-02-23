//! GPU-resident paged KV cache.
//!
//! Pre-allocates per-layer K and V block pools as flat GPU buffers.
//! Append writes directly into block slots via `memcpy_dtod` (O(1)).
//! `get_kv` gathers from GPU blocks into contiguous tensors (for backward compat).
//! `paged_attention_meta` builds block tables on host, uploads to GPU, and returns
//! raw device pointers for the paged decode attention kernel.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use forge_core::{CacheUsage, ForgeError, KvCache, PagedAttentionMeta, Result, Tensor};

use crate::backend::CudaBackend;
use crate::tensor::CudaTensor;

/// Per-layer GPU block pool: flat CudaSlice for K and V.
///
/// Layout: `[total_blocks * block_size * kv_dim]` elements.
/// Addressing: block `b`, slot `s` → offset `(b * block_size + s) * kv_dim`.
struct GpuBlockPool {
    k_pool: CudaSlice<f32>,
    v_pool: CudaSlice<f32>,
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
    /// Free block IDs (stack).
    free_blocks: Vec<usize>,
    /// Per-sequence metadata.
    sequences: HashMap<u64, SeqInfo>,
}

impl GpuPagedKvCache {
    /// Create a new GPU paged KV cache.
    ///
    /// - `total_blocks`: total number of KV blocks per layer
    /// - `block_size`: tokens per block (typically 16)
    /// - `num_layers`: number of transformer layers
    /// - `num_kv_heads`: number of KV attention heads
    /// - `head_dim`: dimension per attention head
    pub fn new(
        backend: CudaBackend,
        total_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let kv_dim = num_kv_heads * head_dim;
        let pool_size = total_blocks * block_size * kv_dim;
        let stream = backend.stream.clone();

        let mut pools = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k_pool = stream
                .alloc_zeros::<f32>(pool_size)
                .map_err(|e| ForgeError::Cuda(format!("alloc k_pool: {e}")))?;
            let v_pool = stream
                .alloc_zeros::<f32>(pool_size)
                .map_err(|e| ForgeError::Cuda(format!("alloc v_pool: {e}")))?;
            pools.push(GpuBlockPool { k_pool, v_pool });
        }

        Ok(Self {
            stream,
            pools,
            num_layers,
            block_size,
            total_blocks,
            kv_dim,
            free_blocks: (0..total_blocks).rev().collect(),
            sequences: HashMap::new(),
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
    fn pool_offset(&self, block_id: usize, slot: usize) -> usize {
        (block_id * self.block_size + slot) * self.kv_dim
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
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        let write_start = if layer == 0 {
            seq.num_tokens
        } else {
            seq.num_tokens - new_tokens
        };

        // We need the key/value as f32 device slices for memcpy_dtod.
        // The current implementation works with F32 tensors.
        let k_slice = key.f32_slice()?;
        let v_slice = value.f32_slice()?;

        for t in 0..new_tokens {
            let token_pos = write_start + t;
            let block_idx = token_pos / self.block_size;
            let slot = token_pos % self.block_size;

            // Ensure we have enough blocks
            let seq = self.sequences.get(&seq_id).unwrap();
            if block_idx >= seq.block_ids.len() {
                let new_block = self.alloc_blocks(1)?;
                self.sequences
                    .get_mut(&seq_id)
                    .unwrap()
                    .block_ids
                    .extend(new_block);
            }

            let block_id = self.sequences.get(&seq_id).unwrap().block_ids[block_idx];
            let dst_offset = self.pool_offset(block_id, slot);
            let src_offset = t * self.kv_dim;

            // Copy one token's K data: device-to-device memcpy
            let pool = &mut self.pools[layer];
            self.stream
                .memcpy_dtod(
                    &k_slice.slice(src_offset..src_offset + self.kv_dim),
                    &mut pool.k_pool.slice_mut(dst_offset..dst_offset + self.kv_dim),
                )
                .map_err(|e| ForgeError::Cuda(format!("memcpy k: {e}")))?;

            self.stream
                .memcpy_dtod(
                    &v_slice.slice(src_offset..src_offset + self.kv_dim),
                    &mut pool.v_pool.slice_mut(dst_offset..dst_offset + self.kv_dim),
                )
                .map_err(|e| ForgeError::Cuda(format!("memcpy v: {e}")))?;
        }

        // Only update token count on layer 0
        if layer == 0 {
            self.sequences.get_mut(&seq_id).unwrap().num_tokens += new_tokens;
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

        let total = seq.num_tokens * self.kv_dim;
        let shape = vec![seq.num_tokens, self.kv_dim];

        // Allocate contiguous output buffers
        let mut k_out = self
            .stream
            .alloc_zeros::<f32>(total)
            .map_err(|e| ForgeError::Cuda(format!("alloc k_out: {e}")))?;
        let mut v_out = self
            .stream
            .alloc_zeros::<f32>(total)
            .map_err(|e| ForgeError::Cuda(format!("alloc v_out: {e}")))?;

        let pool = &self.pools[layer];
        let mut remaining = seq.num_tokens;
        let mut dst_offset = 0usize;

        for &block_id in &seq.block_ids {
            let fill = remaining.min(self.block_size);
            let src_offset = self.pool_offset(block_id, 0);
            let copy_len = fill * self.kv_dim;

            self.stream
                .memcpy_dtod(
                    &pool.k_pool.slice(src_offset..src_offset + copy_len),
                    &mut k_out.slice_mut(dst_offset..dst_offset + copy_len),
                )
                .map_err(|e| ForgeError::Cuda(format!("gather k: {e}")))?;

            self.stream
                .memcpy_dtod(
                    &pool.v_pool.slice(src_offset..src_offset + copy_len),
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

        let num_seqs = seq_ids.len();

        // Find max blocks across all sequences for padding
        let mut max_blocks_per_seq = 0usize;
        let mut kv_lens = Vec::with_capacity(num_seqs);
        for &sid in seq_ids {
            let seq = self
                .sequences
                .get(&sid)
                .ok_or(ForgeError::SeqNotFound(sid))?;
            let n_blocks = seq.block_ids.len();
            if n_blocks > max_blocks_per_seq {
                max_blocks_per_seq = n_blocks;
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
        let block_tables_dev = self
            .stream
            .memcpy_stod(&block_tables_host)
            .map_err(|e| ForgeError::Cuda(format!("upload block_tables: {e}")))?;
        let kv_lens_dev = self
            .stream
            .memcpy_stod(&kv_lens)
            .map_err(|e| ForgeError::Cuda(format!("upload kv_lens: {e}")))?;

        // Extract raw device pointers.
        // Pool pointers are safe because they live as long as the cache.
        // Block tables and kv_lens CudaSlice are moved into keepalive fields
        // of PagedAttentionMeta so the GPU memory remains valid until after
        // the kernel launch.
        let pool = &self.pools[layer];
        let (k_pool_ptr, _k_guard) = pool.k_pool.device_ptr(&self.stream);
        let (v_pool_ptr, _v_guard) = pool.v_pool.device_ptr(&self.stream);

        // Get raw pointers before moving the slices. We use a block scope so
        // the SyncOnDrop guards are released before we move the slices.
        let bt_ptr = {
            let (ptr, _guard) = block_tables_dev.device_ptr(&self.stream);
            ptr
        };
        let kl_ptr = {
            let (ptr, _guard) = kv_lens_dev.device_ptr(&self.stream);
            ptr
        };

        Ok(PagedAttentionMeta {
            block_tables_ptr: bt_ptr,
            kv_lens_ptr: kl_ptr,
            k_pool_ptr,
            v_pool_ptr,
            max_blocks_per_seq,
            block_size: self.block_size,
            kv_dim: self.kv_dim,
            num_seqs,
            // Keep GPU allocations alive until this meta is dropped.
            // The CudaSlice owns the device memory; dropping it frees it.
            _block_tables_keepalive: Box::new(block_tables_dev),
            _kv_lens_keepalive: Box::new(kv_lens_dev),
        })
    }
}
