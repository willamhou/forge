//! Paged KV cache: block-organized KV storage using BlockManager.
//!
//! Stores K/V data in fixed-size blocks managed by BlockManager.
//! Data is stored as CPU-side f32 (same as NaiveKvCache) but block-organized
//! for memory efficiency and future PagedAttention integration.
//!
//! Each block holds `block_size` tokens worth of K/V data per layer.
//! The block table maps sequences to their blocks, enabling:
//! - Fine-grained memory management (no per-sequence max length)
//! - Block table export for PagedAttention kernels
//! - Future prefix caching via block sharing

use std::collections::HashMap;

use forge_core::{Backend, CacheUsage, ForgeError, KvCache, Result, Tensor};

/// Per-block K/V storage for a single layer.
struct BlockData {
    /// Key data: [block_size * kv_dim] flattened
    key: Vec<f32>,
    /// Value data: [block_size * kv_dim] flattened
    value: Vec<f32>,
}

/// Block pool: pre-allocated storage indexed by (layer, block_id).
struct BlockPool {
    /// blocks[layer][block_id] -> BlockData
    blocks: Vec<Vec<BlockData>>,
    block_size: usize,
    kv_dim: usize,
}

impl BlockPool {
    fn new(num_layers: usize, total_blocks: usize, block_size: usize, kv_dim: usize) -> Self {
        let block_capacity = block_size * kv_dim;
        let blocks = (0..num_layers)
            .map(|_| {
                (0..total_blocks)
                    .map(|_| BlockData {
                        key: vec![0.0f32; block_capacity],
                        value: vec![0.0f32; block_capacity],
                    })
                    .collect()
            })
            .collect();
        Self {
            blocks,
            block_size,
            kv_dim,
        }
    }

    /// Write a row of K/V data into a specific block slot.
    fn write_token(
        &mut self,
        layer: usize,
        block_id: usize,
        slot: usize,
        key_row: &[f32],
        value_row: &[f32],
    ) {
        let offset = slot * self.kv_dim;
        let block = &mut self.blocks[layer][block_id];
        block.key[offset..offset + self.kv_dim].copy_from_slice(key_row);
        block.value[offset..offset + self.kv_dim].copy_from_slice(value_row);
    }

    /// Read filled tokens from a sequence's blocks for a given layer.
    fn read_seq(
        &self,
        layer: usize,
        block_ids: &[usize],
        total_tokens: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut key_data = Vec::with_capacity(total_tokens * self.kv_dim);
        let mut value_data = Vec::with_capacity(total_tokens * self.kv_dim);
        let mut remaining = total_tokens;

        for &block_id in block_ids {
            let fill = remaining.min(self.block_size);
            let block = &self.blocks[layer][block_id];
            let bytes = fill * self.kv_dim;
            key_data.extend_from_slice(&block.key[..bytes]);
            value_data.extend_from_slice(&block.value[..bytes]);
            remaining -= fill;
        }

        (key_data, value_data)
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
/// Currently stores data on CPU as f32 (like NaiveKvCache) but with
/// block-granular allocation. This enables:
/// - Bounded memory usage via block limits
/// - Block table export for PagedAttention kernels
/// - Future prefix caching via block sharing
pub struct PagedKvCache<B: Backend> {
    backend: B,
    pool: BlockPool,
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
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        Self {
            backend,
            pool: BlockPool::new(num_layers, total_blocks, block_size, kv_dim),
            num_layers,
            block_size,
            total_blocks,
            free_blocks: (0..total_blocks).rev().collect(),
            sequences: HashMap::new(),
        }
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
    fn free_blocks(&mut self, blocks: &[usize]) {
        self.free_blocks.extend(blocks);
    }
}

impl<B: Backend + Clone> KvCache for PagedKvCache<B> {
    type T = B::Tensor;

    fn allocate(&mut self, seq_id: u64, initial_len: usize) -> Result<()> {
        // Free existing allocation if present
        if let Some(old) = self.sequences.remove(&seq_id) {
            self.free_blocks(&old.block_ids);
        }

        let num_blocks = ((initial_len + self.block_size - 1) / self.block_size).max(1);
        let block_ids = self.alloc_blocks(num_blocks)?;
        self.sequences.insert(seq_id, SeqInfo {
            block_ids,
            num_tokens: 0, // filled by append()
        });
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

        let key_f32 = self.backend.copy_to_host_f32(key)?;
        let val_f32 = self.backend.copy_to_host_f32(value)?;

        // Determine write start position.
        // Layer 0 writes at current num_tokens and then advances the count.
        // Layers > 0 write at the same positions (count was already advanced by layer 0).
        let seq = self.sequences.get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        let write_start = if layer == 0 {
            seq.num_tokens
        } else {
            seq.num_tokens - new_tokens
        };

        for t in 0..new_tokens {
            let token_pos = write_start + t;
            let block_idx = token_pos / self.block_size;
            let slot = token_pos % self.block_size;

            // Ensure we have enough blocks (may need to grow during decode)
            if block_idx >= self.sequences.get(&seq_id).unwrap().block_ids.len() {
                let new_block = self.alloc_blocks(1)?;
                self.sequences.get_mut(&seq_id).unwrap().block_ids.extend(new_block);
            }

            let block_id = self.sequences.get(&seq_id).unwrap().block_ids[block_idx];

            let row_start = t * kv_dim;
            let key_row = &key_f32[row_start..row_start + kv_dim];
            let val_row = &val_f32[row_start..row_start + kv_dim];

            self.pool.write_token(layer, block_id, slot, key_row, val_row);
        }

        // Only update token count on layer 0
        if layer == 0 {
            self.sequences.get_mut(&seq_id).unwrap().num_tokens += new_tokens;
        }

        Ok(())
    }

    fn get_kv(&self, seq_id: u64, layer: usize) -> Result<(B::Tensor, B::Tensor)> {
        let seq = self.sequences.get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;

        if seq.num_tokens == 0 {
            return Err(ForgeError::InvalidArgument(format!(
                "No cached KV for seq {seq_id} layer {layer}"
            )));
        }

        let (key_data, value_data) = self.pool.read_seq(layer, &seq.block_ids, seq.num_tokens);

        let shape = &[seq.num_tokens, self.pool.kv_dim];
        let key = self.backend.copy_from_host_f32(&key_data, shape)?;
        let value = self.backend.copy_from_host_f32(&value_data, shape)?;
        Ok((key, value))
    }

    fn get_block_table(&self, seq_id: u64) -> Result<Vec<usize>> {
        let seq = self.sequences.get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(seq.block_ids.clone())
    }

    fn get_seq_len(&self, seq_id: u64) -> Result<usize> {
        let seq = self.sequences.get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(seq.num_tokens)
    }

    fn free(&mut self, seq_id: u64) -> Result<()> {
        let seq = self.sequences.remove(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        self.free_blocks(&seq.block_ids);
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

        fn name(&self) -> &str { "test" }
        fn device_count(&self) -> usize { 1 }

        fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<TestTensor> {
            Ok(TestTensor { data: data.to_vec(), shape: shape.to_vec() })
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
            Ok(TestTensor { data: vec![0.0; n], shape: shape.to_vec() })
        }
        fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<TestTensor> {
            self.allocate(shape, dtype)
        }
        fn reshape(&self, t: &TestTensor, shape: &[usize]) -> Result<TestTensor> {
            Ok(TestTensor { data: t.data.clone(), shape: shape.to_vec() })
        }
        fn add(&self, _: &TestTensor, _: &TestTensor) -> Result<TestTensor> { unimplemented!() }
        fn mul(&self, _: &TestTensor, _: &TestTensor) -> Result<TestTensor> { unimplemented!() }
        fn mul_scalar(&self, _: &TestTensor, _: f32) -> Result<TestTensor> { unimplemented!() }
        fn silu(&self, _: &TestTensor) -> Result<TestTensor> { unimplemented!() }
        fn rms_norm(&self, _: &TestTensor, _: &TestTensor, _: f32) -> Result<TestTensor> { unimplemented!() }
        fn softmax(&self, _: &TestTensor, _: i32) -> Result<TestTensor> { unimplemented!() }
        fn matmul(&self, _: &TestTensor, _: &TestTensor) -> Result<TestTensor> { unimplemented!() }
        fn embedding(&self, _: &TestTensor, _: &[u32]) -> Result<TestTensor> { unimplemented!() }
        fn rope(&self, _: &TestTensor, _: &TestTensor, _: &TestTensor) -> Result<TestTensor> { unimplemented!() }
        fn transpose(&self, _: &TestTensor, _: usize, _: usize) -> Result<TestTensor> { unimplemented!() }
        fn cat(&self, _: &[&TestTensor], _: usize) -> Result<TestTensor> { unimplemented!() }
        fn synchronize(&self) -> Result<()> { Ok(()) }
        fn cast(&self, t: &TestTensor, _: DType) -> Result<TestTensor> {
            Ok(t.clone())
        }
    }

    impl Clone for TestBackend {
        fn clone(&self) -> Self { TestBackend }
    }

    fn make_tensor(data: &[f32], shape: &[usize]) -> TestTensor {
        TestTensor { data: data.to_vec(), shape: shape.to_vec() }
    }

    #[test]
    fn test_allocate_and_free() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 8, 4, 2, 2, 4);
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
        let mut cache = PagedKvCache::new(backend, 8, 4, 2, 1, 4);
        // kv_dim = 1 * 4 = 4, block_size = 4

        cache.allocate(1, 2).unwrap();

        // Append 2 tokens, layer 0
        let k = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let v = make_tensor(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], &[2, 4]);
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
        assert_eq!(val0.data, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);

        // Retrieve layer 1
        let (key1, val1) = cache.get_kv(1, 1).unwrap();
        assert_eq!(key1.data, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        assert_eq!(val1.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_append_incremental_decode() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 8, 2, 1, 1, 2);
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
        assert_eq!(val.data, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn test_block_boundary_crossing() {
        let backend = TestBackend;
        let mut cache = PagedKvCache::new(backend, 8, 2, 1, 1, 2);
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
        let mut cache = PagedKvCache::new(backend, 1, 2, 1, 1, 2);
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
        let mut cache = PagedKvCache::new(backend, 4, 2, 1, 1, 2);

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
        let mut cache = PagedKvCache::new(backend, 8, 2, 1, 1, 2);

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
}
