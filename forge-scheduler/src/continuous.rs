use std::collections::{HashMap, VecDeque};

use forge_core::{
    CacheUsage, FinishReason, ForgeError, InferenceRequest, RequestHandle, Result, ScheduleBatch,
    ScheduledSeq, Scheduler,
};

use crate::sequence::{SeqState, SequenceState};

/// Configuration for the continuous batching scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_prefill_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_prefill_tokens: 4096,
        }
    }
}

pub struct ContinuousBatchingScheduler {
    config: SchedulerConfig,
    /// Sequences waiting to be prefilled (FCFS queue).
    waiting: VecDeque<u64>,
    /// All sequences by ID.
    sequences: HashMap<u64, SequenceState>,
    /// Currently running sequence IDs.
    running: Vec<u64>,
    /// Counter for generating unique seq IDs.
    next_seq_id: u64,
}

impl ContinuousBatchingScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            waiting: VecDeque::new(),
            sequences: HashMap::new(),
            running: Vec::new(),
            next_seq_id: 1,
        }
    }
}

impl Scheduler for ContinuousBatchingScheduler {
    fn enqueue(&mut self, request: InferenceRequest) -> Result<RequestHandle> {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let seq = SequenceState::new(
            seq_id,
            request.request_id.clone(),
            request.prompt_tokens,
            request.sampling_params,
        );
        self.sequences.insert(seq_id, seq);
        self.waiting.push_back(seq_id);

        Ok(RequestHandle {
            request_id: request.request_id,
            seq_id,
        })
    }

    fn cancel(&mut self, seq_id: u64) -> Result<()> {
        if self.sequences.remove(&seq_id).is_none() {
            return Err(ForgeError::SeqNotFound(seq_id));
        }
        self.running.retain(|&id| id != seq_id);
        self.waiting.retain(|&id| id != seq_id);
        Ok(())
    }

    fn schedule(&mut self, cache_usage: &CacheUsage) -> Result<ScheduleBatch> {
        let mut batch = ScheduleBatch::default();

        // Step 1: Schedule decode steps for running sequences.
        let mut continuing: Vec<u64> = Vec::new();
        for &seq_id in &self.running {
            if let Some(seq) = self.sequences.get(&seq_id) {
                if seq.state != SeqState::Running {
                    continue;
                }
                if batch.total_seqs() >= self.config.max_batch_size {
                    break;
                }
                // Decode: send the last generated token (skip if none yet)
                let last_token = match seq.generated_tokens.last() {
                    Some(&t) => t,
                    None => continue, // no token generated yet, skip decode
                };
                batch.decode_seqs.push(ScheduledSeq {
                    seq_id,
                    token_ids: vec![last_token],
                    position_offset: seq.total_len() - 1,
                    sampling_params: seq.sampling_params.clone(),
                    is_prefill: false,
                });
                continuing.push(seq_id);
            }
        }

        // Step 2: Schedule prefill for waiting sequences (FCFS, within budget).
        let mut prefill_token_budget = self.config.max_prefill_tokens;
        let mut newly_running = Vec::new();
        // Track blocks already promised to earlier prefills in this scheduling pass
        // to avoid overcommitting cache capacity.
        let mut blocks_committed: usize = 0;

        let mut to_reject: Vec<u64> = Vec::new();

        while let Some(&seq_id) = self.waiting.front() {
            if batch.total_seqs() >= self.config.max_batch_size {
                break;
            }

            let seq = match self.sequences.get(&seq_id) {
                Some(s) => s,
                None => {
                    self.waiting.pop_front();
                    continue;
                }
            };

            let prompt_len = seq.prompt_tokens.len();

            // Reject prompts that can never fit in the prefill budget.
            if prompt_len > self.config.max_prefill_tokens {
                self.waiting.pop_front();
                to_reject.push(seq_id);
                continue;
            }

            // Skip if remaining budget is insufficient (try next request).
            if prompt_len > prefill_token_budget {
                break;
            }

            // Check if cache can accommodate (accounting for blocks already committed)
            let blocks_needed =
                (prompt_len + cache_usage.block_size - 1) / cache_usage.block_size;
            let available = cache_usage.free_blocks().saturating_sub(blocks_committed);
            if blocks_needed > available {
                break;
            }

            self.waiting.pop_front();
            prefill_token_budget -= prompt_len;
            blocks_committed += blocks_needed;

            batch.prefill_seqs.push(ScheduledSeq {
                seq_id,
                token_ids: seq.prompt_tokens.clone(),
                position_offset: 0,
                sampling_params: seq.sampling_params.clone(),
                is_prefill: true,
            });

            newly_running.push(seq_id);
        }

        // Update states
        for &seq_id in &newly_running {
            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.state = SeqState::Running;
                seq.is_prefilled = true;
            }
        }
        self.running.extend(newly_running);

        // Remove rejected sequences so they don't block future scheduling.
        for &seq_id in &to_reject {
            self.sequences.remove(&seq_id);
        }
        batch.rejected_seq_ids = to_reject;

        Ok(batch)
    }

    fn finish(&mut self, seq_id: u64, _reason: FinishReason) -> Result<()> {
        if self.sequences.remove(&seq_id).is_none() {
            return Err(ForgeError::SeqNotFound(seq_id));
        }
        self.running.retain(|&id| id != seq_id);
        Ok(())
    }

    fn append_token(&mut self, seq_id: u64, token_id: u32) -> Result<()> {
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        seq.generated_tokens.push(token_id);
        Ok(())
    }

    fn get_generated_tokens(&self, seq_id: u64) -> Result<Vec<u32>> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(seq.generated_tokens.clone())
    }
}
