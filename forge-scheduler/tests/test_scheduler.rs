use forge_core::{
    CacheUsage, FinishReason, InferenceRequest, SamplingParams, Scheduler,
};
use forge_scheduler::{ContinuousBatchingScheduler, SchedulerConfig};

fn default_cache() -> CacheUsage {
    CacheUsage {
        total_blocks: 100,
        used_blocks: 0,
        block_size: 16,
    }
}

fn make_request(id: &str, tokens: Vec<u32>) -> InferenceRequest {
    InferenceRequest {
        request_id: id.to_string(),
        prompt_tokens: tokens,
        sampling_params: SamplingParams::default(),
    }
}

#[test]
fn test_enqueue_and_schedule() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache = default_cache();

    scheduler.enqueue(make_request("req-1", vec![1, 2, 3, 4, 5])).unwrap();

    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 1);
    assert_eq!(batch.decode_seqs.len(), 0);
    assert_eq!(batch.prefill_seqs[0].token_ids, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_prefill_then_decode() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache = default_cache();

    let handle = scheduler.enqueue(make_request("req-1", vec![1, 2, 3])).unwrap();

    // First schedule: prefill
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 1);

    // Simulate token generation
    scheduler.append_token(handle.seq_id, 10).unwrap();

    // Second schedule: decode
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 0);
    assert_eq!(batch.decode_seqs.len(), 1);
    assert_eq!(batch.decode_seqs[0].token_ids, vec![10]);
}

#[test]
fn test_finish_removes_sequence() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache = default_cache();

    let handle = scheduler.enqueue(make_request("req-1", vec![1, 2, 3])).unwrap();
    let _ = scheduler.schedule(&cache).unwrap();

    scheduler.finish(handle.seq_id, FinishReason::EosToken).unwrap();

    let batch = scheduler.schedule(&cache).unwrap();
    assert!(batch.is_empty());
}

#[test]
fn test_max_batch_size() {
    let config = SchedulerConfig {
        max_batch_size: 2,
        ..Default::default()
    };
    let mut scheduler = ContinuousBatchingScheduler::new(config);
    let cache = default_cache();

    for i in 0..5 {
        scheduler.enqueue(make_request(&format!("req-{i}"), vec![1, 2, 3])).unwrap();
    }

    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 2); // limited by max_batch_size
}

#[test]
fn test_generated_tokens() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache = default_cache();

    let handle = scheduler.enqueue(make_request("req-1", vec![1, 2, 3])).unwrap();
    let _ = scheduler.schedule(&cache).unwrap();

    scheduler.append_token(handle.seq_id, 10).unwrap();
    scheduler.append_token(handle.seq_id, 20).unwrap();
    scheduler.append_token(handle.seq_id, 30).unwrap();

    let tokens = scheduler.get_generated_tokens(handle.seq_id).unwrap();
    assert_eq!(tokens, vec![10, 20, 30]);
}

#[test]
fn test_cancel_sequence() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache = default_cache();

    let handle = scheduler.enqueue(make_request("req-1", vec![1, 2, 3])).unwrap();
    let _ = scheduler.schedule(&cache).unwrap();

    scheduler.cancel(handle.seq_id).unwrap();

    let batch = scheduler.schedule(&cache).unwrap();
    assert!(batch.is_empty());
}

#[test]
fn test_seq_not_found() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    assert!(scheduler.cancel(999).is_err());
    assert!(scheduler.append_token(999, 1).is_err());
    assert!(scheduler.get_generated_tokens(999).is_err());
    assert!(scheduler.finish(999, FinishReason::MaxTokens).is_err());
}

#[test]
fn test_cache_pressure_limits_prefill() {
    let config = SchedulerConfig {
        max_batch_size: 10,
        ..Default::default()
    };
    let mut scheduler = ContinuousBatchingScheduler::new(config);

    // Cache with only 2 free blocks of size 16 = 32 tokens
    let cache = CacheUsage {
        total_blocks: 4,
        used_blocks: 2,
        block_size: 16,
    };

    // Enqueue a request needing 48 tokens (3 blocks) — won't fit
    scheduler.enqueue(make_request("req-1", vec![0; 48])).unwrap();

    let batch = scheduler.schedule(&cache).unwrap();
    assert!(batch.is_empty()); // can't schedule: not enough cache
}

#[test]
fn test_decode_before_new_prefill() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache = default_cache();

    // Prefill first request
    let h1 = scheduler.enqueue(make_request("req-1", vec![1, 2, 3])).unwrap();
    let _ = scheduler.schedule(&cache).unwrap();
    scheduler.append_token(h1.seq_id, 10).unwrap();

    // Enqueue second request
    scheduler.enqueue(make_request("req-2", vec![4, 5, 6])).unwrap();

    // Next schedule should include both: decode for req-1, prefill for req-2
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.decode_seqs.len(), 1);
    assert_eq!(batch.prefill_seqs.len(), 1);
}

#[test]
fn test_prefill_overcommit_protection() {
    let config = SchedulerConfig {
        max_batch_size: 10,
        max_prefill_tokens: 4096,
        ..Default::default()
    };
    let mut scheduler = ContinuousBatchingScheduler::new(config);

    // Cache with 3 free blocks of size 16 = 48 tokens capacity
    let cache = CacheUsage {
        total_blocks: 5,
        used_blocks: 2,
        block_size: 16,
    };

    // Enqueue 3 requests, each needing 2 blocks (20 tokens each)
    // Total need: 6 blocks, but only 3 are available
    scheduler.enqueue(make_request("req-1", vec![0; 20])).unwrap();
    scheduler.enqueue(make_request("req-2", vec![0; 20])).unwrap();
    scheduler.enqueue(make_request("req-3", vec![0; 20])).unwrap();

    let batch = scheduler.schedule(&cache).unwrap();
    // Only the first request should be scheduled (2 blocks), leaving 1 free block
    // which isn't enough for the second request (needs 2 blocks)
    assert_eq!(batch.prefill_seqs.len(), 1);
}

// --- Chunked Prefill Tests ---

#[test]
fn test_chunked_prefill_splits_long_prompt() {
    let config = SchedulerConfig {
        max_batch_size: 10,
        max_prefill_tokens: 4096,
        prefill_chunk_size: Some(4),
    };
    let mut scheduler = ContinuousBatchingScheduler::new(config);
    let cache = default_cache();

    // Prompt of 10 tokens, chunk size 4 → chunks: [0..4], [4..8], [8..10]
    scheduler
        .enqueue(make_request("req-1", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        .unwrap();

    // Round 1: first chunk [1,2,3,4]
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 1);
    assert_eq!(batch.prefill_seqs[0].token_ids, vec![1, 2, 3, 4]);
    assert_eq!(batch.prefill_seqs[0].position_offset, 0);
    assert!(!batch.prefill_seqs[0].is_last_prefill_chunk);
    assert_eq!(batch.decode_seqs.len(), 0);

    // Round 2: second chunk [5,6,7,8]
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 1);
    assert_eq!(batch.prefill_seqs[0].token_ids, vec![5, 6, 7, 8]);
    assert_eq!(batch.prefill_seqs[0].position_offset, 4);
    assert!(!batch.prefill_seqs[0].is_last_prefill_chunk);

    // Round 3: final chunk [9,10]
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 1);
    assert_eq!(batch.prefill_seqs[0].token_ids, vec![9, 10]);
    assert_eq!(batch.prefill_seqs[0].position_offset, 8);
    assert!(batch.prefill_seqs[0].is_last_prefill_chunk);

    // Round 4: should now be in decode mode (no prefill, no decode until token appended)
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 0);
    assert_eq!(batch.decode_seqs.len(), 0); // no generated token yet
}

#[test]
fn test_chunked_prefill_exact_chunk_boundary() {
    let config = SchedulerConfig {
        max_batch_size: 10,
        max_prefill_tokens: 4096,
        prefill_chunk_size: Some(5),
    };
    let mut scheduler = ContinuousBatchingScheduler::new(config);
    let cache = default_cache();

    // Prompt of exactly 10 tokens, chunk size 5 → [0..5], [5..10]
    scheduler
        .enqueue(make_request("req-1", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        .unwrap();

    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs[0].token_ids.len(), 5);
    assert!(!batch.prefill_seqs[0].is_last_prefill_chunk);

    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs[0].token_ids.len(), 5);
    assert!(batch.prefill_seqs[0].is_last_prefill_chunk);
}

#[test]
fn test_chunked_prefill_short_prompt_no_chunking() {
    let config = SchedulerConfig {
        max_batch_size: 10,
        max_prefill_tokens: 4096,
        prefill_chunk_size: Some(512),
    };
    let mut scheduler = ContinuousBatchingScheduler::new(config);
    let cache = default_cache();

    // Prompt shorter than chunk size → single prefill, is_last = true
    scheduler
        .enqueue(make_request("req-1", vec![1, 2, 3]))
        .unwrap();

    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 1);
    assert_eq!(batch.prefill_seqs[0].token_ids, vec![1, 2, 3]);
    assert!(batch.prefill_seqs[0].is_last_prefill_chunk);
}

#[test]
fn test_chunked_prefill_decode_interleaved() {
    let config = SchedulerConfig {
        max_batch_size: 10,
        max_prefill_tokens: 4096,
        prefill_chunk_size: Some(3),
    };
    let mut scheduler = ContinuousBatchingScheduler::new(config);
    let cache = default_cache();

    // Enqueue a short request first, then a long one
    let h1 = scheduler
        .enqueue(make_request("req-1", vec![1, 2]))
        .unwrap();
    scheduler
        .enqueue(make_request("req-2", vec![10, 20, 30, 40, 50, 60]))
        .unwrap();

    // Round 1: prefill req-1 (complete) + first chunk of req-2
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 2);
    assert!(batch.prefill_seqs[0].is_last_prefill_chunk); // req-1 done
    assert!(!batch.prefill_seqs[1].is_last_prefill_chunk); // req-2 chunk 1

    // Append a token to req-1 so it enters decode
    scheduler.append_token(h1.seq_id, 99).unwrap();

    // Round 2: decode for req-1 + second chunk of req-2
    let batch = scheduler.schedule(&cache).unwrap();
    assert_eq!(batch.decode_seqs.len(), 1); // req-1 decode
    assert_eq!(batch.prefill_seqs.len(), 1); // req-2 chunk 2
    assert!(batch.prefill_seqs[0].is_last_prefill_chunk); // req-2 done
}

#[test]
fn test_chunked_prefill_no_reject_long_prompts() {
    // With chunking enabled, prompts exceeding max_prefill_tokens should NOT be rejected
    // because they can be split across multiple rounds.
    let config = SchedulerConfig {
        max_batch_size: 10,
        max_prefill_tokens: 8,
        prefill_chunk_size: Some(4),
    };
    let mut scheduler = ContinuousBatchingScheduler::new(config);
    let cache = default_cache();

    // Prompt of 12 tokens > max_prefill_tokens=8, but chunk_size=4 allows it
    scheduler
        .enqueue(make_request("req-1", (1..=12).collect()))
        .unwrap();

    let batch = scheduler.schedule(&cache).unwrap();
    assert!(batch.rejected.is_empty());
    assert_eq!(batch.prefill_seqs.len(), 1);
    assert_eq!(batch.prefill_seqs[0].token_ids.len(), 4);
}
