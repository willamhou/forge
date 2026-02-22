//! Radix tree for prefix caching.
use std::collections::HashMap;

#[allow(dead_code)]
struct RadixNode {
    children: HashMap<u32, Box<RadixNode>>,
    block_id: Option<usize>,
    ref_count: usize,
    last_access: u64,
    token: u32,
}

impl RadixNode {
    fn new(token: u32) -> Self {
        Self {
            children: HashMap::new(),
            block_id: None,
            ref_count: 0,
            last_access: 0,
            token,
        }
    }
}
/// Radix tree for prefix caching.
///
/// Maps token sequences to KV-cache block IDs so that requests sharing a common
/// prefix (e.g. a system prompt) can reuse already-computed KV blocks.
///
/// Matching and insertion are block-aligned: only complete blocks (of
/// `block_size` tokens) are tracked.
pub struct RadixTree {
    root: RadixNode,
    block_size: usize,
    access_counter: u64,
}

impl RadixTree {
    /// Create a new radix tree with the given block size.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            root: RadixNode::new(0),
            block_size,
            access_counter: 0,
        }
    }

    /// Walk the tree following `tokens` and return the longest block-aligned
    /// prefix match as `(matched_token_count, block_ids)`.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<usize>) {
        self.access_counter += 1;
        let ts = self.access_counter;
        let mut node = &mut self.root;
        let mut block_ids: Vec<usize> = Vec::new();
        let mut depth: usize = 0;
        for &tok in tokens {
            if !node.children.contains_key(&tok) {
                break;
            }
            node = node.children.get_mut(&tok).unwrap();
            node.last_access = ts;
            depth += 1;
            if depth % self.block_size == 0 {
                if let Some(bid) = node.block_id {
                    block_ids.push(bid);
                } else {
                    break;
                }
            }
        }
        (block_ids.len() * self.block_size, block_ids)
    }

    /// Insert a token sequence with associated block IDs into the tree.
    ///
    /// `tokens.len()` must be >= `block_ids.len() * block_size`.
    pub fn insert(&mut self, tokens: &[u32], block_ids: &[usize]) {
        self.access_counter += 1;
        let ts = self.access_counter;
        let covered = block_ids.len() * self.block_size;
        assert!(tokens.len() >= covered, "tokens too short");
        let mut node = &mut self.root;
        for (i, &tok) in tokens.iter().enumerate().take(covered) {
            node = node
                .children
                .entry(tok)
                .or_insert_with(|| Box::new(RadixNode::new(tok)));
            node.ref_count += 1;
            node.last_access = ts;
            let d = i + 1;
            if d % self.block_size == 0 {
                node.block_id = Some(block_ids[d / self.block_size - 1]);
            }
        }
    }

    pub fn release(&mut self, tokens: &[u32]) {
        let mut node = &mut self.root;
        for &tok in tokens {
            if !node.children.contains_key(&tok) {
                break;
            }
            node = node.children.get_mut(&tok).unwrap();
            node.ref_count = node.ref_count.saturating_sub(1);
        }
    }

    pub fn evict_lru(&mut self, n: usize) -> Vec<usize> {
        if n == 0 { return Vec::new(); }
        let mut ev = Vec::new();
        loop {
            let mut cs = Vec::new();
            Self::collect_evictable(&self.root, &mut Vec::new(), &mut cs);
            if cs.is_empty() { break; }
            cs.sort_by_key(|(t, _)| *t);
            for (_, p) in cs {
                if ev.len() >= n { return ev; }
                if let Some(b) = Self::try_evict_leaf(&mut self.root, &p) { ev.push(b); }
            }
            if ev.len() >= n { break; }
        }
        ev
    }

    pub fn num_cached_blocks(&self) -> usize {
        Self::count_blocks(&self.root)
    }

    pub fn num_evictable_blocks(&self) -> usize {
        Self::count_evictable(&self.root)
    }

    fn collect_evictable(
        node: &RadixNode,
        path: &mut Vec<u32>,
        out: &mut Vec<(u64, Vec<u32>)>,
    ) {
        for (tok, child) in &node.children {
            path.push(*tok);
            if child.children.is_empty()
                && child.ref_count == 0
                && child.block_id.is_some()
            {
                out.push((child.last_access, path.clone()));
            }
            Self::collect_evictable(child, path, out);
            path.pop();
        }
    }

    fn try_evict_leaf(root: &mut RadixNode, path: &[u32]) -> Option<usize> {
        if path.is_empty() { return None; }
        // Check eligibility first without moving root.
        {
            let leaf = Self::nav_parent(root, path)?;
            if !leaf.children.is_empty() || leaf.ref_count != 0 || leaf.block_id.is_none() {
                return None;
            }
        }
        // Remove the leaf from its parent.
        let parent = Self::nav_parent(root, &path[..path.len() - 1])?;
        let last_tok = path[path.len() - 1];
        let removed = parent.children.remove(&last_tok)?;
        let bid = removed.block_id;
        // Prune dead ancestors (no children, no block_id, ref_count 0).
        for depth in (1..path.len()).rev() {
            let ancestor = match Self::nav_parent(root, &path[..depth]) {
                Some(a) => a,
                None => break,
            };
            if ancestor.children.is_empty() && ancestor.block_id.is_none() && ancestor.ref_count == 0 {
                // Remove this dead node from its parent.
                let par = Self::nav_parent(root, &path[..depth - 1]).unwrap();
                par.children.remove(&path[depth - 1]);
            } else {
                break;
            }
        }
        bid
    }

    fn nav_parent<'a>(node: &'a mut RadixNode, path: &[u32]) -> Option<&'a mut RadixNode> {
        let mut cur = node;
        for &tok in path {
            cur = cur.children.get_mut(&tok)?;
        }
        Some(cur)
    }

    fn count_blocks(node: &RadixNode) -> usize {
        let mut c = if node.block_id.is_some() { 1 } else { 0 };
        for child in node.children.values() {
            c += Self::count_blocks(child);
        }
        c
    }

    fn count_evictable(node: &RadixNode) -> usize {
        let mut c = 0;
        for child in node.children.values() {
            if child.children.is_empty()
                && child.ref_count == 0
                && child.block_id.is_some()
            {
                c += 1;
            }
            c += Self::count_evictable(child);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let mut tree = RadixTree::new(4);
        let (matched, blocks) = tree.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(matched, 0);
        assert!(blocks.is_empty());
        assert_eq!(tree.num_cached_blocks(), 0);
        assert_eq!(tree.num_evictable_blocks(), 0);
    }

    #[test]
    fn test_basic_insert_and_match() {
        let mut tree = RadixTree::new(4);
        let tokens = vec![10, 20, 30, 40, 50, 60, 70, 80];
        tree.insert(&tokens, &[100, 200]);
        let (matched, blocks) = tree.match_prefix(&tokens);
        assert_eq!(matched, 8);
        assert_eq!(blocks, vec![100, 200]);
    }

    #[test]
    fn test_partial_match() {
        let mut tree = RadixTree::new(4);
        tree.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);
        let (matched, blocks) = tree.match_prefix(&[1, 2, 3, 4, 99, 6, 7, 8]);
        assert_eq!(matched, 4);
        assert_eq!(blocks, vec![10]);
    }

    #[test]
    fn test_block_alignment() {
        let mut tree = RadixTree::new(4);
        tree.insert(&[1, 2, 3, 4], &[10]);
        let query = vec![1, 2, 3, 4, 5, 6];
        let (matched, blocks) = tree.match_prefix(&query);
        assert_eq!(matched, 4);
        assert_eq!(blocks, vec![10]);
    }

    #[test]
    fn test_prefix_reuse() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2, 3, 4], &[10, 20]);
        tree.insert(&[1, 2, 5, 6], &[10, 30]);
        let (m1, b1) = tree.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(m1, 4);
        assert_eq!(b1, vec![10, 20]);
        let (m2, b2) = tree.match_prefix(&[1, 2, 5, 6]);
        assert_eq!(m2, 4);
        assert_eq!(b2, vec![10, 30]);
    }

    #[test]
    fn test_release_decrements_ref_count() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2, 3, 4], &[10, 20]);
        assert_eq!(tree.num_evictable_blocks(), 0);
        tree.release(&[1, 2, 3, 4]);
        assert_eq!(tree.num_evictable_blocks(), 1);
    }

    #[test]
    fn test_evict_lru_basic() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2, 3, 4], &[10, 20]);
        tree.release(&[1, 2, 3, 4]);
        let evicted = tree.evict_lru(1);
        assert_eq!(evicted, vec![20]);
        assert_eq!(tree.num_cached_blocks(), 1);
        let evicted2 = tree.evict_lru(1);
        assert_eq!(evicted2, vec![10]);
        assert_eq!(tree.num_cached_blocks(), 0);
    }

    #[test]
    fn test_evict_lru_ordering() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2], &[10]);
        tree.insert(&[3, 4], &[20]);
        tree.release(&[1, 2]);
        tree.release(&[3, 4]);
        let evicted = tree.evict_lru(1);
        assert_eq!(evicted, vec![10]);
    }

    #[test]
    fn test_evict_zero() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2], &[10]);
        tree.release(&[1, 2]);
        let evicted = tree.evict_lru(0);
        assert!(evicted.is_empty());
        assert_eq!(tree.num_cached_blocks(), 1);
    }

    #[test]
    fn test_evict_more_than_available() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2], &[10]);
        tree.release(&[1, 2]);
        let evicted = tree.evict_lru(100);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], 10);
    }

    #[test]
    fn test_no_evict_with_refs() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2], &[10]);
        let evicted = tree.evict_lru(10);
        assert!(evicted.is_empty());
        assert_eq!(tree.num_cached_blocks(), 1);
    }

    #[test]
    fn test_num_cached_blocks() {
        let mut tree = RadixTree::new(2);
        assert_eq!(tree.num_cached_blocks(), 0);
        tree.insert(&[1, 2, 3, 4], &[10, 20]);
        assert_eq!(tree.num_cached_blocks(), 2);
        tree.insert(&[1, 2, 5, 6], &[10, 30]);
        assert_eq!(tree.num_cached_blocks(), 3);
    }

    #[test]
    fn test_overlapping_prefixes() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2, 3, 4], &[10, 20]);
        tree.insert(&[1, 2, 3, 4, 5, 6], &[10, 20, 30]);
        let (matched, blocks) = tree.match_prefix(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(matched, 6);
        assert_eq!(blocks, vec![10, 20, 30]);
        let (matched2, blocks2) = tree.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(matched2, 4);
        assert_eq!(blocks2, vec![10, 20]);
    }

    #[test]
    fn test_match_no_tokens() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2], &[10]);
        let (matched, blocks) = tree.match_prefix(&[]);
        assert_eq!(matched, 0);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_insert_updates_block_id() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2], &[10]);
        let (_, b1) = tree.match_prefix(&[1, 2]);
        assert_eq!(b1, vec![10]);
        tree.insert(&[1, 2], &[99]);
        let (_, b2) = tree.match_prefix(&[1, 2]);
        assert_eq!(b2, vec![99]);
    }

    #[test]
    fn test_release_saturates_at_zero() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2], &[10]);
        tree.release(&[1, 2]);
        tree.release(&[1, 2]);
        assert_eq!(tree.num_evictable_blocks(), 1);
    }

    #[test]
    fn test_multiple_inserts_accumulate_refs() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2], &[10]);
        tree.insert(&[1, 2], &[10]);
        tree.release(&[1, 2]);
        assert_eq!(tree.num_evictable_blocks(), 0);
        tree.release(&[1, 2]);
        assert_eq!(tree.num_evictable_blocks(), 1);
    }

    #[test]
    fn test_block_size_one() {
        let mut tree = RadixTree::new(1);
        tree.insert(&[5, 10, 15], &[100, 200, 300]);
        let (matched, blocks) = tree.match_prefix(&[5, 10, 15]);
        assert_eq!(matched, 3);
        assert_eq!(blocks, vec![100, 200, 300]);
        let (matched2, blocks2) = tree.match_prefix(&[5, 10]);
        assert_eq!(matched2, 2);
        assert_eq!(blocks2, vec![100, 200]);
    }

    #[test]
    fn test_large_block_size() {
        let mut tree = RadixTree::new(8);
        let tokens: Vec<u32> = (0..16).collect();
        tree.insert(&tokens, &[1, 2]);
        let (matched, blocks) = tree.match_prefix(&tokens);
        assert_eq!(matched, 16);
        assert_eq!(blocks, vec![1, 2]);
        let short: Vec<u32> = (0..7).collect();
        let (matched2, _) = tree.match_prefix(&short);
        assert_eq!(matched2, 0);
    }

    #[test]
    fn test_evict_cascading() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2, 3, 4], &[10, 20]);
        tree.release(&[1, 2, 3, 4]);
        let evicted = tree.evict_lru(10);
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&10));
        assert!(evicted.contains(&20));
        assert_eq!(tree.num_cached_blocks(), 0);
    }

    #[test]
    fn test_match_after_eviction() {
        let mut tree = RadixTree::new(2);
        tree.insert(&[1, 2, 3, 4], &[10, 20]);
        tree.release(&[1, 2, 3, 4]);
        tree.evict_lru(1);
        let (matched, blocks) = tree.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(matched, 2);
        assert_eq!(blocks, vec![10]);
    }

    #[test]
    #[should_panic(expected = "block_size must be positive")]
    fn test_zero_block_size_panics() {
        RadixTree::new(0);
    }

    #[test]
    #[should_panic(expected = "tokens too short")]
    fn test_insert_tokens_too_short_panics() {
        let mut tree = RadixTree::new(4);
        tree.insert(&[1, 2], &[10]);
    }
}
