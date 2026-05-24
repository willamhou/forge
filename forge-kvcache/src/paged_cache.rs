//! Paged KV cache: block-organized KV storage using BlockManager.
//!
//! Stores K/V data in fixed-size blocks of a backend-allocated pool tensor.
//! Each pool has shape `[num_blocks, block_size, kv_dim]`; one (K, V) pair per layer.
//!
//! The block table maps sequences to their blocks, enabling:
//! - Fine-grained memory management (no per-sequence max length)
//! - Block table export for PagedAttention kernels
//! - Future prefix caching via block sharing
//!
//! The pool is backend-managed via `Backend::paged_write_kv` / `paged_gather_kv`.
//! The default (host-fallback) trait impl is correct but slow; CUDA overrides
//! with device-to-device memcpy so pool addresses stay stable for CUDA Graph
//! capture.

use std::collections::HashMap;

use forge_core::{Backend, CacheUsage, DType, ForgeError, KvCache, Result, Tensor};

/// Block pool: backend-allocated K/V tensors, one (K, V) pair per layer.
struct BlockPool<B: Backend> {
    /// `layers[layer] = (k_pool, v_pool)`; each pool shape `[num_blocks, block_size, kv_dim]`.
    layers: Vec<(B::Tensor, B::Tensor)>,
    kv_dim: usize,
}

impl<B: Backend> BlockPool<B> {
    fn new(
        backend: &B,
        num_layers: usize,
        total_blocks: usize,
        block_size: usize,
        kv_dim: usize,
    ) -> Result<Self> {
        let shape = [total_blocks, block_size, kv_dim];
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = backend.allocate_zeros(&shape, DType::F32)?;
            let v = backend.allocate_zeros(&shape, DType::F32)?;
            layers.push((k, v));
        }
        Ok(Self { layers, kv_dim })
    }

    /// Write K/V rows for a layer at the given slot positions.
    fn write_tokens(
        &mut self,
        backend: &B,
        layer: usize,
        key: &B::Tensor,
        value: &B::Tensor,
        slot_mapping: &[i32],
    ) -> Result<()> {
        let (k_pool, v_pool) = &mut self.layers[layer];
        backend.paged_write_kv(k_pool, key, slot_mapping)?;
        backend.paged_write_kv(v_pool, value, slot_mapping)?;
        Ok(())
    }

    /// Read filled tokens from a sequence's blocks for a given layer.
    fn read_seq(
        &self,
        backend: &B,
        layer: usize,
        block_ids: &[usize],
        total_tokens: usize,
    ) -> Result<(B::Tensor, B::Tensor)> {
        let (k_pool, v_pool) = &self.layers[layer];
        let key = backend.paged_gather_kv(k_pool, block_ids, total_tokens)?;
        let value = backend.paged_gather_kv(v_pool, block_ids, total_tokens)?;
        Ok((key, value))
    }
}

/// Per-sequence metadata: block table + token count.
struct SeqInfo {
    /// Ordered list of block IDs assigned to this sequence.
    block_ids: Vec<usize>,
    /// Total number of tokens stored across all blocks.
    num_tokens: usize,
}

/// Paged KV cache with block-based memory management.
///
/// Uses fixed-size blocks for KV storage. Each sequence has an ordered
/// list of blocks, and tokens fill blocks sequentially. When a block
/// is full, a new one is allocated.
///
/// Storage lives in backend-allocated pool tensors, written via
/// `Backend::paged_write_kv` and read via `Backend::paged_gather_kv`.
/// CUDA backends override those ops to keep the pool device-resident
/// with stable addresses (required for CUDA Graph capture).
pub struct PagedKvCache<B: Backend> {
    backend: B,
    pool: BlockPool<B>,
    num_layers: usize,
    block_size: usize,
    total_blocks: usize,
    /// Free block IDs available for allocation.
    free_blocks: Vec<usize>,
    /// Per-sequence metadata.
    sequences: HashMap<u64, SeqInfo>,
}

impl<B: Backend> PagedKvCache<B> {
    /// Create a new paged KV cache.
    ///
    /// - `total_blocks`: total number of KV blocks in the pool
    /// - `block_size`: tokens per block (typically 16 or 32)
    /// - `num_layers`: number of transformer layers
    /// - `num_kv_heads`: number of KV attention heads
    /// - `head_dim`: dimension per attention head
    pub fn new(
        backend: B,
        total_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let kv_dim = num_kv_heads * head_dim;
        let pool = BlockPool::new(&backend, num_layers, total_blocks, block_size, kv_dim)?;
        Ok(Self {
            backend,
            pool,
            num_layers,
            block_size,
            total_blocks,
            free_blocks: (0..total_blocks).rev().collect(),
            sequences: HashMap::new(),
        })
    }

    /// Borrow the K pool tensor for a layer (shape `[num_blocks, block_size, kv_dim]`).
    ///
    /// The returned reference's device address is stable across `append` calls
    /// — the property the future paged_attention kernel + CUDA Graph capture
    /// rely on. Callers must not invalidate this stability (e.g. by replacing
    /// the cache or reallocating layers).
    pub fn k_pool(&self, layer: usize) -> Result<&B::Tensor> {
        self.pool
            .layers
            .get(layer)
            .map(|(k, _)| k)
            .ok_or_else(|| {
                ForgeError::InvalidArgument(format!(
                    "k_pool: layer {layer} out of bounds ({} layers)",
                    self.num_layers
                ))
            })
    }

    /// Borrow the V pool tensor for a layer. See [`Self::k_pool`].
    pub fn v_pool(&self, layer: usize) -> Result<&B::Tensor> {
        self.pool
            .layers
            .get(layer)
            .map(|(_, v)| v)
            .ok_or_else(|| {
                ForgeError::InvalidArgument(format!(
                    "v_pool: layer {layer} out of bounds ({} layers)",
                    self.num_layers
                ))
            })
    }

    /// Assemble batch-shaped i32 metadata for the given sequences.
    ///
    /// Returns `(block_tables_flat, kv_lens, max_blocks_per_seq)`:
    /// - `block_tables_flat`: row-major `[batch, max_blocks_per_seq]`, padding = `-1`
    /// - `kv_lens`: `[batch]` — current KV length per sequence
    /// - `max_blocks_per_seq`: stride of `block_tables_flat`
    ///
    /// Host-side on purpose. The engine uploads this to a persistent device
    /// scratch tensor on each scheduler step (one H2D memcpy of ~kilobytes,
    /// dwarfed by per-step kernel work). The pool tensors retrieved via
    /// [`Self::k_pool`] / [`Self::v_pool`] stay device-resident — only the
    /// per-step indirection metadata moves.
    pub fn batch_block_tables(
        &self,
        seq_ids: &[u64],
    ) -> Result<(Vec<i32>, Vec<i32>, usize)> {
        let mut kv_lens = Vec::with_capacity(seq_ids.len());
        let mut max_blocks = 0usize;
        for id in seq_ids {
            let seq = self
                .sequences
                .get(id)
                .ok_or(ForgeError::SeqNotFound(*id))?;
            kv_lens.push(seq.num_tokens as i32);
            if seq.block_ids.len() > max_blocks {
                max_blocks = seq.block_ids.len();
            }
        }

        let mut block_tables = vec![-1i32; seq_ids.len() * max_blocks];
        for (i, id) in seq_ids.iter().enumerate() {
            let seq = &self.sequences[id];
            for (j, &b) in seq.block_ids.iter().enumerate() {
                if b > i32::MAX as usize {
                    return Err(ForgeError::InvalidArgument(format!(
                        "block_id {b} does not fit in i32"
                    )));
                }
                block_tables[i * max_blocks + j] = b as i32;
            }
        }

        Ok((block_tables, kv_lens, max_blocks))
    }

    /// Allocate `n` blocks from the free list.
    fn alloc_blocks(&mut self, n: usize) -> Result<Vec<usize>> {
        if self.free_blocks.len() < n {
            return Err(ForgeError::OutOfMemory(format!(
                "Need {n} blocks, only {} free",
                self.free_blocks.len()
            )));
        }
        Ok((0..n).map(|_| self.free_blocks.pop().unwrap()).collect())
    }

    /// Return blocks to the free list.
    fn release_blocks(&mut self, blocks: &[usize]) {
        self.free_blocks.extend(blocks);
    }
}

impl<B: Backend + Clone> KvCache for PagedKvCache<B> {
    type T = B::Tensor;

    fn allocate(&mut self, seq_id: u64, initial_len: usize) -> Result<()> {
        // Free existing allocation if present
        if let Some(old) = self.sequences.remove(&seq_id) {
            self.release_blocks(&old.block_ids);
        }

        let num_blocks = ((initial_len + self.block_size - 1) / self.block_size).max(1);
        let block_ids = self.alloc_blocks(num_blocks)?;
        self.sequences.insert(
            seq_id,
            SeqInfo {
                block_ids,
                num_tokens: 0, // filled by append()
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

        let key_shape = key.shape();
        let new_tokens = key_shape[0];
        let kv_dim = if key_shape.len() > 1 { key_shape[1] } else { 1 };

        if kv_dim != self.pool.kv_dim {
            return Err(ForgeError::InvalidArgument(format!(
                "kv_dim mismatch: cache expects {}, got {kv_dim}",
                self.pool.kv_dim
            )));
        }

        // Determine write start position.
        // Layer 0 writes at current num_tokens and then advances the count.
        // Layers > 0 write at the same positions (count was already advanced by layer 0).
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        let write_start = if layer == 0 {
            seq.num_tokens
        } else {
            seq.num_tokens - new_tokens
        };

        // Build slot_mapping: absolute slot = block_id * block_size + slot_in_block.
        // Grow block list during prefill / decode if needed (layer 0 only — later
        // layers reuse the blocks allocated for layer 0 at the same token positions).
        let block_size = self.block_size;
        let mut slot_mapping = Vec::with_capacity(new_tokens);
        for t in 0..new_tokens {
            let token_pos = write_start + t;
            let block_idx = token_pos / block_size;
            let slot = token_pos % block_size;

            if block_idx >= self.sequences.get(&seq_id).unwrap().block_ids.len() {
                let new_block = self.alloc_blocks(1)?;
                self.sequences
                    .get_mut(&seq_id)
                    .unwrap()
                    .block_ids
                    .extend(new_block);
            }

            let block_id = self.sequences.get(&seq_id).unwrap().block_ids[block_idx];
            slot_mapping.push((block_id * block_size + slot) as i32);
        }

        self.pool
            .write_tokens(&self.backend, layer, key, value, &slot_mapping)?;

        // Only update token count on layer 0
        if layer == 0 {
            self.sequences.get_mut(&seq_id).unwrap().num_tokens += new_tokens;
        }

        Ok(())
    }

    fn get_kv(&self, seq_id: u64, layer: usize) -> Result<(B::Tensor, B::Tensor)> {
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
                "No cached KV for seq {seq_id} layer {layer}"
            )));
        }

        self.pool
            .read_seq(&self.backend, layer, &seq.block_ids, seq.num_tokens)
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
        self.release_blocks(&seq.block_ids);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::DType;

    // A minimal CPU backend for testing
    struct TestBackend;

    #[derive(Clone, Debug)]
    struct TestTensor {
        data: Vec<f32>,
        shape: Vec<usize>,
    }

    impl Tensor for TestTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
    }

    impl Backend for TestBackend {
        type Tensor = TestTensor;

        fn name(&self) -> &str {
            "test"
        }
        fn device_count(&self) -> usize {
            1
        }

        fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<TestTensor> {
            Ok(TestTensor {
                data: data.to_vec(),
                shape: shape.to_vec(),
            })
        }
        fn copy_from_host_f16(&self, _: &[half::f16], _: &[usize]) -> Result<TestTensor> {
            unimplemented!()
        }
        fn copy_from_host_bf16(&self, _: &[half::bf16], _: &[usize]) -> Result<TestTensor> {
            unimplemented!()
        }
        fn copy_to_host_f32(&self, tensor: &TestTensor) -> Result<Vec<f32>> {
            Ok(tensor.data.clone())
        }
        fn allocate(&self, shape: &[usize], _: DType) -> Result<TestTensor> {
            let n: usize = shape.iter().product();
            Ok(TestTensor {
                data: vec![0.0; n],
                shape: shape.to_vec(),
            })
        }
        fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<TestTensor> {
            self.allocate(shape, dtype)
        }
        fn reshape(&self, t: &TestTensor, shape: &[usize]) -> Result<TestTensor> {
            Ok(TestTensor {
                data: t.data.clone(),
                shape: shape.to_vec(),
            })
        }
        fn add(&self, _: &TestTensor, _: &TestTensor) -> Result<TestTensor> {
            unimplemented!()
        }
        fn mul(&self, _: &TestTensor, _: &TestTensor) -> Result<TestTensor> {
            unimplemented!()
        }
        fn mul_scalar(&self, _: &TestTensor, _: f32) -> Result<TestTensor> {
            unimplemented!()
        }
        fn silu(&self, _: &TestTensor) -> Result<TestTensor> {
            unimplemented!()
        }
        fn rms_norm(&self, _: &TestTensor, _: &TestTensor, _: f32) -> Result<TestTensor> {
            unimplemented!()
        }
        fn softmax(&self, _: &TestTensor, _: i32) -> Result<TestTensor> {
            unimplemented!()
        }
        fn matmul(&self, _: &TestTensor, _: &TestTensor) -> Result<TestTensor> {
            unimplemented!()
        }
        fn embedding(&self, _: &TestTensor, _: &[u32]) -> Result<TestTensor> {
            unimplemented!()
        }
        fn rope(&self, _: &TestTensor, _: &TestTensor, _: &TestTensor) -> Result<TestTensor> {
            unimplemented!()
        }
        fn transpose(&self, _: &TestTensor, _: usize, _: usize) -> Result<TestTensor> {
            unimplemented!()
        }
        fn cat(&self, _: &[&TestTensor], _: usize) -> Result<TestTensor> {
            unimplemented!()
        }
        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
        fn cast(&self, t: &TestTensor, _: DType) -> Result<TestTensor> {
            Ok(t.clone())
        }
    }

    impl Clone for TestBackend {
        fn clone(&self) -> Self {
            TestBackend
        }
    }

    fn make_tensor(data: &[f32], shape: &[usize]) -> TestTensor {
        TestTensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn test_allocate_and_free() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 8, 4, 2, 2, 4).unwrap();
        // kv_dim = 2 * 4 = 8

        cache.allocate(1, 3).unwrap();
        assert_eq!(cache.get_seq_len(1).unwrap(), 0);
        assert!(cache.get_block_table(1).unwrap().len() >= 1);

        cache.free(1).unwrap();
        assert!(cache.get_seq_len(1).is_err());
    }

    #[test]
    fn test_append_and_get_kv() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 8, 4, 2, 1, 4).unwrap();
        // kv_dim = 1 * 4 = 4, block_size = 4

        cache.allocate(1, 2).unwrap();

        // Append 2 tokens, layer 0
        let k = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let v = make_tensor(
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            &[2, 4],
        );
        cache.append(1, 0, &k, &v).unwrap();

        // Append 2 tokens, layer 1
        let k1 = make_tensor(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4]);
        let v1 = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        cache.append(1, 1, &k1, &v1).unwrap();

        assert_eq!(cache.get_seq_len(1).unwrap(), 2);

        // Retrieve layer 0
        let (key0, val0) = cache.get_kv(1, 0).unwrap();
        assert_eq!(key0.shape(), &[2, 4]);
        assert_eq!(key0.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            val0.data,
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        );

        // Retrieve layer 1
        let (key1, val1) = cache.get_kv(1, 1).unwrap();
        assert_eq!(key1.data, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        assert_eq!(val1.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_append_incremental_decode() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 8, 2, 1, 1, 2).unwrap();
        // kv_dim = 2, block_size = 2

        cache.allocate(1, 3).unwrap();

        // Prefill: 3 tokens
        let k = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v = make_tensor(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
        cache.append(1, 0, &k, &v).unwrap();
        assert_eq!(cache.get_seq_len(1).unwrap(), 3);

        // Decode: 1 token
        let k = make_tensor(&[7.0, 8.0], &[1, 2]);
        let v = make_tensor(&[70.0, 80.0], &[1, 2]);
        cache.append(1, 0, &k, &v).unwrap();
        assert_eq!(cache.get_seq_len(1).unwrap(), 4);

        // Verify full cache content
        let (key, val) = cache.get_kv(1, 0).unwrap();
        assert_eq!(key.shape(), &[4, 2]);
        assert_eq!(key.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            val.data,
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        );
    }

    #[test]
    fn test_block_boundary_crossing() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 8, 2, 1, 1, 2).unwrap();
        // kv_dim = 2, block_size = 2 → each block holds 2 tokens

        cache.allocate(1, 1).unwrap(); // allocate 1 block

        // Append 1 token (fills half of block 0)
        let k = make_tensor(&[1.0, 2.0], &[1, 2]);
        let v = make_tensor(&[10.0, 20.0], &[1, 2]);
        cache.append(1, 0, &k, &v).unwrap();

        // Append 1 token (fills block 0)
        let k = make_tensor(&[3.0, 4.0], &[1, 2]);
        let v = make_tensor(&[30.0, 40.0], &[1, 2]);
        cache.append(1, 0, &k, &v).unwrap();

        // Append 1 token (needs new block)
        let k = make_tensor(&[5.0, 6.0], &[1, 2]);
        let v = make_tensor(&[50.0, 60.0], &[1, 2]);
        cache.append(1, 0, &k, &v).unwrap();

        assert_eq!(cache.get_seq_len(1).unwrap(), 3);
        assert_eq!(cache.get_block_table(1).unwrap().len(), 2); // 2 blocks used

        let (key, _) = cache.get_kv(1, 0).unwrap();
        assert_eq!(key.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_out_of_memory() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 1, 2, 1, 1, 2).unwrap();
        // Only 1 block, 2 tokens capacity

        cache.allocate(1, 2).unwrap();

        let k = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let v = make_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        cache.append(1, 0, &k, &v).unwrap();

        // Third token should fail — no more blocks
        let k = make_tensor(&[5.0, 6.0], &[1, 2]);
        let v = make_tensor(&[50.0, 60.0], &[1, 2]);
        assert!(cache.append(1, 0, &k, &v).is_err());
    }

    #[test]
    fn test_usage_tracking() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 4, 2, 1, 1, 2).unwrap();

        assert_eq!(cache.usage().used_blocks, 0);
        assert!(cache.can_allocate(4));

        cache.allocate(1, 3).unwrap(); // needs 2 blocks
        assert_eq!(cache.usage().used_blocks, 2);
        assert!(cache.can_allocate(4)); // 2 free blocks → 4 tokens
        assert!(!cache.can_allocate(5)); // would need 3 blocks

        cache.free(1).unwrap();
        assert_eq!(cache.usage().used_blocks, 0);
    }

    #[test]
    fn test_multiple_sequences() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 8, 2, 1, 1, 2).unwrap();

        cache.allocate(1, 2).unwrap();
        cache.allocate(2, 2).unwrap();

        let k1 = make_tensor(&[1.0, 1.0, 2.0, 2.0], &[2, 2]);
        let v1 = make_tensor(&[10.0, 10.0, 20.0, 20.0], &[2, 2]);
        cache.append(1, 0, &k1, &v1).unwrap();

        let k2 = make_tensor(&[3.0, 3.0], &[1, 2]);
        let v2 = make_tensor(&[30.0, 30.0], &[1, 2]);
        cache.append(2, 0, &k2, &v2).unwrap();

        assert_eq!(cache.get_seq_len(1).unwrap(), 2);
        assert_eq!(cache.get_seq_len(2).unwrap(), 1);

        // Free seq 1, seq 2 should still work
        cache.free(1).unwrap();
        let (key2, _) = cache.get_kv(2, 0).unwrap();
        assert_eq!(key2.data, vec![3.0, 3.0]);
    }

    #[test]
    fn test_pool_accessors() {
        let backend = TestBackend;
        // 4 blocks x 3 tokens/block, kv_dim = 2*4 = 8, 2 layers
        let cache = PagedKvCache::new(backend, 4, 3, 2, 2, 4).unwrap();

        // Each layer's pool tensor has shape [num_blocks, block_size, kv_dim]
        let k0 = cache.k_pool(0).unwrap();
        let v0 = cache.v_pool(0).unwrap();
        let k1 = cache.k_pool(1).unwrap();
        assert_eq!(k0.shape(), &[4, 3, 8]);
        assert_eq!(v0.shape(), &[4, 3, 8]);
        assert_eq!(k1.shape(), &[4, 3, 8]);

        // Out-of-bounds layer rejected
        assert!(cache.k_pool(2).is_err());
        assert!(cache.v_pool(99).is_err());
    }

    #[test]
    fn test_batch_block_tables() {
        let backend = TestBackend;
        // 16 blocks, 2 tokens/block, 1 layer, kv_dim = 1*2 = 2
        let mut cache = PagedKvCache::new(backend, 16, 2, 1, 1, 2).unwrap();

        // seq 1: 5 tokens → 3 blocks (ceil(5/2))
        cache.allocate(1, 5).unwrap();
        let k = make_tensor(&[1.0; 10], &[5, 2]);
        let v = make_tensor(&[2.0; 10], &[5, 2]);
        cache.append(1, 0, &k, &v).unwrap();

        // seq 2: 3 tokens → 2 blocks
        cache.allocate(2, 3).unwrap();
        let k = make_tensor(&[3.0; 6], &[3, 2]);
        let v = make_tensor(&[4.0; 6], &[3, 2]);
        cache.append(2, 0, &k, &v).unwrap();

        // seq 3: 1 token → 1 block
        cache.allocate(3, 1).unwrap();
        let k = make_tensor(&[5.0, 5.0], &[1, 2]);
        let v = make_tensor(&[6.0, 6.0], &[1, 2]);
        cache.append(3, 0, &k, &v).unwrap();

        // Batch query [1, 2, 3]: max_blocks = 3
        let (block_tables, kv_lens, max_blocks) =
            cache.batch_block_tables(&[1, 2, 3]).unwrap();
        assert_eq!(max_blocks, 3);
        assert_eq!(kv_lens, vec![5, 3, 1]);
        assert_eq!(block_tables.len(), 9);

        // seq 1's row uses all 3 slots
        assert!(block_tables[0..3].iter().all(|&b| b >= 0));
        // seq 2's row uses first 2 slots, last is padding
        assert!(block_tables[3..5].iter().all(|&b| b >= 0));
        assert_eq!(block_tables[5], -1);
        // seq 3's row uses 1 slot, rest is padding
        assert!(block_tables[6] >= 0);
        assert_eq!(&block_tables[7..9], &[-1, -1]);

        // Block IDs must be distinct across all rows (no aliasing)
        let mut used: Vec<i32> = block_tables.iter().copied().filter(|&b| b >= 0).collect();
        used.sort();
        let unique_count = used.windows(2).filter(|w| w[0] != w[1]).count() + 1;
        assert_eq!(unique_count, used.len(), "block IDs aliased: {used:?}");

        // Batch with one seq → max_blocks reflects just that seq
        let (bt1, lens1, max1) = cache.batch_block_tables(&[3]).unwrap();
        assert_eq!(max1, 1);
        assert_eq!(lens1, vec![1]);
        assert_eq!(bt1.len(), 1);
        assert!(bt1[0] >= 0);

        // Empty batch returns empty vecs
        let (bt_empty, lens_empty, max_empty) =
            cache.batch_block_tables(&[]).unwrap();
        assert_eq!(bt_empty, Vec::<i32>::new());
        assert_eq!(lens_empty, Vec::<i32>::new());
        assert_eq!(max_empty, 0);

        // Unknown seq_id surfaces as SeqNotFound
        assert!(cache.batch_block_tables(&[1, 999]).is_err());
    }
}
