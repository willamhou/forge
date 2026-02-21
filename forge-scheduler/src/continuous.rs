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
    /// Maximum tokens to prefill per chunk. Prompts longer than this are split
    /// into multiple scheduling rounds so decode latency stays stable.
    /// `None` disables chunking (prefill the whole prompt at once).
    pub prefill_chunk_size: Option<usize>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_prefill_tokens: 4096,
            prefill_chunk_size: None,
        }
    }
}

pub struct ContinuousBatchingScheduler {
    config: SchedulerConfig,
    /// Sequences waiting to be prefilled (FCFS queue).
    waiting: VecDeque<u64>,
    /// All sequences by ID.
    sequences: HashMap<u64, SequenceState>,
    /// Currently running sequence IDs (fully prefilled, in decode phase).
    running: Vec<u64>,
    /// Sequences mid-prefill (chunked prefill in progress).
    prefilling: Vec<u64>,
    /// Counter for generating unique seq IDs.
    next_seq_id: u64,
}

impl ContinuousBatchingScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        // Treat prefill_chunk_size == 0 as disabled (None) to prevent
        // infinite-loop where zero-sized chunks never advance offset.
        let config = if config.prefill_chunk_size == Some(0) {
            SchedulerConfig {
                prefill_chunk_size: None,
                ..config
            }
        } else {
            config
        };
        Self {
            config,
            waiting: VecDeque::new(),
            sequences: HashMap::new(),
            running: Vec::new(),
            prefilling: Vec::new(),
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
        self.prefilling.retain(|&id| id != seq_id);
        Ok(())
    }

    fn schedule(&mut self, cache_usage: &CacheUsage) -> Result<ScheduleBatch> {
        let mut batch = ScheduleBatch::default();

        // Step 1: Schedule decode steps for running sequences.
        for &seq_id in &self.running {
            if let Some(seq) = self.sequences.get(&seq_id) {
                if seq.state != SeqState::Running {
                    continue;
                }
                if batch.total_seqs() >= self.config.max_batch_size {
                    break;
                }
                let last_token = match seq.generated_tokens.last() {
                    Some(&t) => t,
                    None => continue,
                };
                batch.decode_seqs.push(ScheduledSeq {
                    seq_id,
                    token_ids: vec![last_token],
                    position_offset: seq.total_len() - 1,
                    sampling_params: seq.sampling_params.clone(),
                    is_prefill: false,
                    is_last_prefill_chunk: false,
                    total_prompt_len: seq.prompt_tokens.len(),
                });
            }
        }

        // Step 2: Continue chunked prefills that are already in progress.
        let mut completed_prefills = Vec::new();
        let prefilling_snapshot: Vec<u64> = self.prefilling.clone();
        for &seq_id in &prefilling_snapshot {
            if batch.total_seqs() >= self.config.max_batch_size {
                break;
            }
            if let Some(seq) = self.sequences.get(&seq_id) {
                let prompt_len = seq.prompt_tokens.len();
                let offset = seq.prefill_offset;
                let remaining = prompt_len - offset;
                let chunk_size = self
                    .config
                    .prefill_chunk_size
                    .unwrap_or(remaining)
                    .min(remaining);

                let chunk_tokens = seq.prompt_tokens[offset..offset + chunk_size].to_vec();
                let is_last = offset + chunk_size >= prompt_len;

                batch.prefill_seqs.push(ScheduledSeq {
                    seq_id,
                    token_ids: chunk_tokens,
                    position_offset: offset,
                    sampling_params: seq.sampling_params.clone(),
                    is_prefill: true,
                    is_last_prefill_chunk: is_last,
                    total_prompt_len: prompt_len,
                });

                if is_last {
                    completed_prefills.push(seq_id);
                }
            }
        }

        // Update offsets for prefilling sequences
        for scheduled in &batch.prefill_seqs {
            if let Some(seq) = self.sequences.get_mut(&scheduled.seq_id) {
                seq.prefill_offset = scheduled.position_offset + scheduled.token_ids.len();
            }
        }

        // Move completed prefills from prefilling to running
        for &seq_id in &completed_prefills {
            self.prefilling.retain(|&id| id != seq_id);
            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.state = SeqState::Running;
                seq.is_prefilled = true;
            }
            self.running.push(seq_id);
        }

        // Step 3: Schedule prefill for new waiting sequences (FCFS, within budget).
        let mut prefill_token_budget = self.config.max_prefill_tokens;
        // Subtract tokens already committed to continuing prefills above
        for scheduled in &batch.prefill_seqs {
            prefill_token_budget = prefill_token_budget.saturating_sub(scheduled.token_ids.len());
        }

        let mut newly_running = Vec::new();
        let mut newly_prefilling = Vec::new();
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

            // Reject prompts that can never fit in the max prefill budget
            // (only when chunking is disabled — with chunking any prompt can be split).
            if self.config.prefill_chunk_size.is_none() && prompt_len > self.config.max_prefill_tokens
            {
                self.waiting.pop_front();
                to_reject.push(seq_id);
                continue;
            }

            // Determine how many tokens to prefill this round
            let chunk_size = match self.config.prefill_chunk_size {
                Some(cs) => cs.min(prompt_len),
                None => prompt_len,
            };

            // Check budget
            if chunk_size > prefill_token_budget {
                break;
            }

            // Check cache: we need to allocate blocks for the full prompt, not just the chunk,
            // because KV cache is allocated on first prefill chunk.
            let blocks_needed = if cache_usage.block_size == 0 {
                usize::MAX
            } else {
                (prompt_len + cache_usage.block_size - 1) / cache_usage.block_size
            };
            let available = cache_usage.free_blocks().saturating_sub(blocks_committed);
            if blocks_needed > available {
                // If the prompt can never fit in the total cache, reject it
                // so it doesn't block all subsequent requests (head-of-line).
                if blocks_needed > cache_usage.total_blocks {
                    self.waiting.pop_front();
                    to_reject.push(seq_id);
                    continue;
                }
                break;
            }

            self.waiting.pop_front();
            prefill_token_budget -= chunk_size;
            blocks_committed += blocks_needed;

            let is_last = chunk_size >= prompt_len;
            let chunk_tokens = seq.prompt_tokens[..chunk_size].to_vec();

            batch.prefill_seqs.push(ScheduledSeq {
                seq_id,
                token_ids: chunk_tokens,
                position_offset: 0,
                sampling_params: seq.sampling_params.clone(),
                is_prefill: true,
                is_last_prefill_chunk: is_last,
                total_prompt_len: prompt_len,
            });

            if is_last {
                newly_running.push(seq_id);
            } else {
                newly_prefilling.push(seq_id);
            }
        }

        // Update states for newly started sequences
        for &seq_id in &newly_running {
            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.state = SeqState::Running;
                seq.is_prefilled = true;
                seq.prefill_offset = seq.prompt_tokens.len();
            }
        }
        self.running.extend(newly_running);

        for &seq_id in &newly_prefilling {
            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.state = SeqState::Running;
                seq.prefill_offset = match self.config.prefill_chunk_size {
                    Some(cs) => cs.min(seq.prompt_tokens.len()),
                    None => seq.prompt_tokens.len(),
                };
            }
        }
        self.prefilling.extend(newly_prefilling);

        // Remove rejected sequences
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
        self.prefilling.retain(|&id| id != seq_id);
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
