use crate::kvcache::CacheUsage;
use crate::{FinishReason, Result, SamplingParams};

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
}

#[derive(Debug, Clone)]
pub struct RequestHandle {
    pub request_id: String,
    pub seq_id: u64,
}

#[derive(Debug, Default)]
pub struct ScheduleBatch {
    pub prefill_seqs: Vec<ScheduledSeq>,
    pub decode_seqs: Vec<ScheduledSeq>,
    /// Sequence IDs rejected because their prompts exceed `max_prefill_tokens`.
    pub rejected_seq_ids: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct ScheduledSeq {
    pub seq_id: u64,
    pub token_ids: Vec<u32>,
    pub position_offset: usize,
    pub sampling_params: SamplingParams,
    pub is_prefill: bool,
}

impl ScheduleBatch {
    pub fn is_empty(&self) -> bool {
        self.prefill_seqs.is_empty() && self.decode_seqs.is_empty()
    }

    pub fn total_seqs(&self) -> usize {
        self.prefill_seqs.len() + self.decode_seqs.len()
    }
}

pub trait Scheduler: Send + Sync {
    fn enqueue(&mut self, request: InferenceRequest) -> Result<RequestHandle>;
    fn cancel(&mut self, seq_id: u64) -> Result<()>;
    fn schedule(&mut self, cache_usage: &CacheUsage) -> Result<ScheduleBatch>;
    fn finish(&mut self, seq_id: u64, reason: FinishReason) -> Result<()>;
    fn append_token(&mut self, seq_id: u64, token_id: u32) -> Result<()>;
    fn get_generated_tokens(&self, seq_id: u64) -> Result<Vec<u32>>;
}
