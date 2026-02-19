use std::collections::HashMap;

use forge_core::{CacheUsage, ForgeError, Result};

pub struct BlockManager {
    total_blocks: usize,
    block_size: usize,
    free_blocks: Vec<usize>,
    /// seq_id -> list of (block_id, fill_count)
    seq_blocks: HashMap<u64, Vec<(usize, usize)>>,
}

impl BlockManager {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            total_blocks,
            block_size,
            free_blocks: (0..total_blocks).rev().collect(),
            seq_blocks: HashMap::new(),
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Result<Vec<usize>> {
        if self.free_blocks.len() < num_blocks {
            return Err(ForgeError::OutOfMemory(format!(
                "Need {} blocks, only {} free",
                num_blocks,
                self.free_blocks.len()
            )));
        }
        let blocks: Vec<usize> = (0..num_blocks)
            .map(|_| self.free_blocks.pop().unwrap())
            .collect();
        Ok(blocks)
    }

    pub fn free(&mut self, blocks: &[usize]) {
        self.free_blocks.extend(blocks);
    }

    pub fn free_count(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        let blocks_needed = (num_tokens + self.block_size - 1) / self.block_size;
        self.free_blocks.len() >= blocks_needed
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn usage(&self) -> CacheUsage {
        CacheUsage {
            total_blocks: self.total_blocks,
            used_blocks: self.total_blocks - self.free_blocks.len(),
            block_size: self.block_size,
        }
    }

    // Sequence-level operations

    pub fn allocate_seq(&mut self, seq_id: u64, initial_tokens: usize) -> Result<()> {
        let num_blocks = ((initial_tokens + self.block_size - 1) / self.block_size).max(1);
        let blocks = self.allocate(num_blocks)?;
        let seq_blocks = blocks.into_iter().map(|b| (b, 0)).collect();
        self.seq_blocks.insert(seq_id, seq_blocks);
        Ok(())
    }

    pub fn free_seq(&mut self, seq_id: u64) -> Result<()> {
        let blocks = self
            .seq_blocks
            .remove(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        let block_ids: Vec<usize> = blocks.into_iter().map(|(id, _)| id).collect();
        self.free(&block_ids);
        Ok(())
    }

    pub fn get_block_table(&self, seq_id: u64) -> Result<Vec<usize>> {
        let blocks = self
            .seq_blocks
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(blocks.iter().map(|(id, _)| *id).collect())
    }

    pub fn append_token(&mut self, seq_id: u64) -> Result<usize> {
        let needs_new_block = {
            let blocks = self
                .seq_blocks
                .get(&seq_id)
                .ok_or(ForgeError::SeqNotFound(seq_id))?;
            let (_, fill) = blocks.last().unwrap();
            *fill >= self.block_size
        };

        if needs_new_block {
            let new_block_id = self
                .free_blocks
                .pop()
                .ok_or_else(|| ForgeError::OutOfMemory("No free blocks".into()))?;
            let blocks = self.seq_blocks.get_mut(&seq_id).unwrap();
            blocks.push((new_block_id, 1));
            Ok(new_block_id)
        } else {
            let blocks = self.seq_blocks.get_mut(&seq_id).unwrap();
            let (_, fill) = blocks.last_mut().unwrap();
            *fill += 1;
            Ok(blocks.last().unwrap().0)
        }
    }

    pub fn seq_len(&self, seq_id: u64) -> Result<usize> {
        let blocks = self
            .seq_blocks
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(blocks.iter().map(|(_, fill)| fill).sum())
    }
}
