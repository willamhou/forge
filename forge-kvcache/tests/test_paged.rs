use forge_kvcache::paged::BlockManager;

#[test]
fn test_block_allocation() {
    let mut mgr = BlockManager::new(100, 16); // 100 blocks, 16 tokens each
    let blocks = mgr.allocate(3).unwrap(); // allocate 3 blocks
    assert_eq!(blocks.len(), 3);
    assert_eq!(mgr.free_count(), 97);
}

#[test]
fn test_block_free() {
    let mut mgr = BlockManager::new(100, 16);
    let blocks = mgr.allocate(5).unwrap();
    mgr.free(&blocks);
    assert_eq!(mgr.free_count(), 100);
}

#[test]
fn test_allocation_failure() {
    let mut mgr = BlockManager::new(2, 16);
    let _ = mgr.allocate(2).unwrap();
    assert!(mgr.allocate(1).is_err()); // no free blocks
}

#[test]
fn test_can_allocate() {
    let mgr = BlockManager::new(10, 16);
    assert!(mgr.can_allocate(160)); // 10 blocks * 16 = 160 tokens
    assert!(!mgr.can_allocate(161));
}

#[test]
fn test_seq_allocate_and_free() {
    let mut mgr = BlockManager::new(100, 16);
    mgr.allocate_seq(1, 32).unwrap(); // 2 blocks for 32 tokens
    assert_eq!(mgr.free_count(), 98);
    assert_eq!(mgr.seq_len(1).unwrap(), 32); // fill_counts should reflect initial_tokens

    let table = mgr.get_block_table(1).unwrap();
    assert_eq!(table.len(), 2);

    mgr.free_seq(1).unwrap();
    assert_eq!(mgr.free_count(), 100);
}

#[test]
fn test_seq_allocate_initial_tokens() {
    let mut mgr = BlockManager::new(100, 4); // block_size = 4
    mgr.allocate_seq(1, 6).unwrap(); // 6 tokens -> 2 blocks (4 + 2)
    assert_eq!(mgr.seq_len(1).unwrap(), 6);
    assert_eq!(mgr.free_count(), 98);
}

#[test]
fn test_append_token() {
    let mut mgr = BlockManager::new(100, 4); // block_size = 4
    mgr.allocate_seq(1, 1).unwrap(); // 1 block, fill_count=1

    assert_eq!(mgr.seq_len(1).unwrap(), 1);

    // Fill the rest of the first block (3 more tokens)
    for _ in 0..3 {
        mgr.append_token(1).unwrap();
    }
    assert_eq!(mgr.seq_len(1).unwrap(), 4);

    // Next append should allocate a new block
    let initial_free = mgr.free_count();
    mgr.append_token(1).unwrap();
    assert_eq!(mgr.free_count(), initial_free - 1);
    assert_eq!(mgr.seq_len(1).unwrap(), 5);
}

#[test]
fn test_usage() {
    let mut mgr = BlockManager::new(10, 16);
    let usage = mgr.usage();
    assert_eq!(usage.total_blocks, 10);
    assert_eq!(usage.used_blocks, 0);

    mgr.allocate(3).unwrap();
    let usage = mgr.usage();
    assert_eq!(usage.used_blocks, 3);
}

#[test]
fn test_seq_not_found() {
    let mgr = BlockManager::new(10, 16);
    assert!(mgr.get_block_table(999).is_err());
}

#[test]
fn test_allocate_seq_duplicate_frees_old_blocks() {
    let mut mgr = BlockManager::new(10, 4);
    mgr.allocate_seq(1, 8).unwrap(); // 2 blocks
    assert_eq!(mgr.free_count(), 8);

    // Re-allocate same seq_id — old 2 blocks should be freed before allocating new ones
    mgr.allocate_seq(1, 4).unwrap(); // 1 block
    assert_eq!(mgr.free_count(), 9); // 10 - 1 = 9, not 10 - 3 = 7
    assert_eq!(mgr.seq_len(1).unwrap(), 4);
}
