//! Engine: main inference runtime loop that orchestrates scheduling,
//! model forward passes, and token sampling.

use tokio::sync::mpsc;
use tracing::error;

use forge_core::{
    Backend, FinishReason, KvCache, Model, ModelInput, Result, ScheduledSeq, Scheduler,
    SeqMetadata,
};

use crate::sampling::CpuSampler;

/// Events emitted by the engine to consumers (HTTP layer, transport).
#[derive(Debug, Clone)]
pub enum EngineEvent {
    Token {
        seq_id: u64,
        token_id: u32,
        text: Option<String>,
    },
    Finish {
        seq_id: u64,
        reason: FinishReason,
    },
    Error {
        seq_id: u64,
        error: String,
    },
}

/// The inference engine: ties together model, scheduler, KV cache, and sampler.
pub struct Engine<B: Backend, M: Model<T = B::Tensor>> {
    model: M,
    backend: B,
    scheduler: Box<dyn Scheduler>,
    kv_cache: Box<dyn KvCache<T = B::Tensor>>,
    sampler: CpuSampler,
    event_tx: mpsc::Sender<EngineEvent>,
}

impl<B: Backend + Clone, M: Model<T = B::Tensor>> Engine<B, M> {
    pub fn new(
        model: M,
        backend: B,
        scheduler: Box<dyn Scheduler>,
        kv_cache: Box<dyn KvCache<T = B::Tensor>>,
        event_tx: mpsc::Sender<EngineEvent>,
    ) -> Self {
        Self {
            model,
            backend,
            scheduler,
            kv_cache,
            sampler: CpuSampler,
            event_tx,
        }
    }

    /// Access the scheduler (e.g., for enqueuing requests from the API layer).
    pub fn scheduler_mut(&mut self) -> &mut dyn Scheduler {
        &mut *self.scheduler
    }

    /// Main engine loop. Runs until the sender side is dropped.
    pub async fn run(&mut self) -> Result<()> {
        loop {
            let cache_usage = self.kv_cache.usage();
            let batch = self.scheduler.schedule(&cache_usage)?;

            if batch.is_empty() {
                tokio::time::sleep(std::time::Duration::from_micros(100)).await;
                continue;
            }

            // Allocate KV cache for new prefill sequences
            for seq in &batch.prefill_seqs {
                self.kv_cache.allocate(seq.seq_id, seq.token_ids.len())?;
            }

            // Phase 1: process one sequence at a time
            let all_seqs: Vec<&ScheduledSeq> = batch
                .prefill_seqs
                .iter()
                .chain(batch.decode_seqs.iter())
                .collect();

            for seq in &all_seqs {
                if let Err(e) = self.process_sequence(seq).await {
                    error!(seq_id = seq.seq_id, error = %e, "sequence processing failed");
                    let _ = self
                        .event_tx
                        .send(EngineEvent::Error {
                            seq_id: seq.seq_id,
                            error: e.to_string(),
                        })
                        .await;
                }
            }
        }
    }

    /// Process a single sequence: forward pass + sample + emit event.
    async fn process_sequence(&mut self, seq: &ScheduledSeq) -> Result<()> {
        let input = self.build_input(seq);

        let output = self.model.forward(&input, &mut *self.kv_cache)?;
        self.backend.synchronize()?;

        // Copy logits to host for CPU sampling
        let logits_host = self.backend.copy_to_host_f32(&output.logits)?;

        // Only look at the last token's logits (for both prefill and decode)
        let vocab_size = self.model.config().vocab_size;
        let num_tokens = logits_host.len() / vocab_size;
        let last_logits = &logits_host[(num_tokens - 1) * vocab_size..];

        let generated = self.scheduler.get_generated_tokens(seq.seq_id)?;
        let result =
            self.sampler
                .sample_single(last_logits, &seq.sampling_params, &generated)?;

        let token_id = result.token_id;
        self.scheduler.append_token(seq.seq_id, token_id)?;

        // Check stop conditions
        let should_finish = seq.sampling_params.stop_token_ids.contains(&token_id)
            || generated.len() + 1 >= seq.sampling_params.max_tokens;

        if should_finish {
            let reason = if seq.sampling_params.stop_token_ids.contains(&token_id) {
                FinishReason::EosToken
            } else {
                FinishReason::MaxTokens
            };
            self.scheduler.finish(seq.seq_id, reason)?;
            self.kv_cache.free(seq.seq_id)?;
            let _ = self
                .event_tx
                .send(EngineEvent::Finish {
                    seq_id: seq.seq_id,
                    reason,
                })
                .await;
        } else {
            let _ = self
                .event_tx
                .send(EngineEvent::Token {
                    seq_id: seq.seq_id,
                    token_id,
                    text: None, // decoded by the server/transport layer
                })
                .await;
        }

        Ok(())
    }

    /// Build a single-sequence ModelInput from a ScheduledSeq.
    fn build_input(&self, seq: &ScheduledSeq) -> ModelInput {
        let gen_count = self.get_generated_count(seq.seq_id);

        let (prompt_len, generated_len) = if seq.is_prefill {
            (seq.token_ids.len(), 0)
        } else {
            // position_offset = prompt_len + generated_len - 1
            // generated_len = gen_count (tokens generated so far, before this step)
            let prompt_len = seq.position_offset + 1 - gen_count;
            (prompt_len, gen_count)
        };

        ModelInput {
            token_ids: vec![seq.token_ids.clone()],
            positions: vec![(0..seq.token_ids.len() as u32).collect()],
            seq_metadata: vec![SeqMetadata {
                seq_id: seq.seq_id,
                prompt_len,
                generated_len,
                is_prefill: seq.is_prefill,
            }],
        }
    }

    fn get_generated_count(&self, seq_id: u64) -> usize {
        self.scheduler
            .get_generated_tokens(seq_id)
            .map(|t| t.len())
            .unwrap_or(0)
    }
}
