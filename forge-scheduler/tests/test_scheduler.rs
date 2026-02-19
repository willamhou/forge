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
