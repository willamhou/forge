//! Engine: main inference runtime loop that orchestrates scheduling,
//! model forward passes, and token sampling.

use std::collections::HashMap;

use tokio::sync::mpsc;
use tracing::{error, warn};

use forge_core::{
    Backend, FinishReason, InferenceRequest, KvCache, Model, ModelInput, Result, ScheduledSeq,
    Scheduler, SeqMetadata,
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

/// A request submitted to the engine via the request channel.
pub struct EngineRequest {
    pub inference_req: InferenceRequest,
    /// Per-request channel for sending events back to the caller.
    pub event_tx: mpsc::Sender<EngineEvent>,
}

/// The inference engine: ties together model, scheduler, KV cache, and sampler.
pub struct Engine<B: Backend, M: Model<T = B::Tensor>> {
    model: M,
    backend: B,
    scheduler: Box<dyn Scheduler>,
    kv_cache: Box<dyn KvCache<T = B::Tensor>>,
    sampler: CpuSampler,
    /// Incoming request channel.
    request_rx: mpsc::Receiver<EngineRequest>,
    /// Per-sequence event senders (seq_id → event_tx).
    event_senders: HashMap<u64, mpsc::Sender<EngineEvent>>,
}

impl<B: Backend + Clone, M: Model<T = B::Tensor>> Engine<B, M> {
    pub fn new(
        model: M,
        backend: B,
        scheduler: Box<dyn Scheduler>,
        kv_cache: Box<dyn KvCache<T = B::Tensor>>,
        request_rx: mpsc::Receiver<EngineRequest>,
    ) -> Self {
        Self {
            model,
            backend,
            scheduler,
            kv_cache,
            sampler: CpuSampler,
            request_rx,
            event_senders: HashMap::new(),
        }
    }

    /// Main engine loop. Runs until the request channel is closed.
    pub async fn run(&mut self) -> Result<()> {
        loop {
            // Drain incoming requests (non-blocking)
            let drained = self.drain_requests();

            let cache_usage = self.kv_cache.usage();
            let batch = self.scheduler.schedule(&cache_usage)?;

            if batch.is_empty() {
                if drained == 0 {
                    // No work and no new requests — wait for a request or short timeout
                    tokio::select! {
                        req = self.request_rx.recv() => {
                            match req {
                                Some(r) => self.enqueue_request(r),
                                None => return Ok(()), // channel closed, shutdown
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => {}
                    }
                }
                continue;
            }

            // Allocate KV cache for new prefill sequences
            for seq in &batch.prefill_seqs {
                self.kv_cache.allocate(seq.seq_id, seq.token_ids.len())?;
            }

            // Phase 1: process one sequence at a time
            let all_seqs: Vec<ScheduledSeq> = batch
                .prefill_seqs
                .into_iter()
                .chain(batch.decode_seqs)
                .collect();

            for seq in &all_seqs {
                if let Err(e) = self.process_sequence(seq).await {
                    error!(seq_id = seq.seq_id, error = %e, "sequence processing failed");
                    self.send_event(
                        seq.seq_id,
                        EngineEvent::Error {
                            seq_id: seq.seq_id,
                            error: e.to_string(),
                        },
                    )
                    .await;
                    // Clean up failed sequence
                    let _ = self.scheduler.cancel(seq.seq_id);
                    let _ = self.kv_cache.free(seq.seq_id);
                    self.event_senders.remove(&seq.seq_id);
                }
            }
        }
    }

    /// Drain all pending requests from the channel (non-blocking).
    fn drain_requests(&mut self) -> usize {
        let mut count = 0;
        while let Ok(req) = self.request_rx.try_recv() {
            self.enqueue_request(req);
            count += 1;
        }
        count
    }

    /// Enqueue a request into the scheduler and register its event sender.
    fn enqueue_request(&mut self, req: EngineRequest) {
        match self.scheduler.enqueue(req.inference_req) {
            Ok(handle) => {
                self.event_senders.insert(handle.seq_id, req.event_tx);
            }
            Err(e) => {
                // Best-effort error notification
                let _ = req.event_tx.try_send(EngineEvent::Error {
                    seq_id: 0,
                    error: format!("failed to enqueue: {e}"),
                });
            }
        }
    }

    /// Send an event to a specific sequence's consumer. If the consumer has
    /// disconnected, cancel the sequence and clean up resources.
    async fn send_event(&mut self, seq_id: u64, event: EngineEvent) {
        if let Some(tx) = self.event_senders.get(&seq_id) {
            if tx.send(event).await.is_err() {
                warn!(seq_id, "event receiver dropped, cancelling sequence");
                let _ = self.scheduler.cancel(seq_id);
                let _ = self.kv_cache.free(seq_id);
                self.event_senders.remove(&seq_id);
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
            self.send_event(
                seq.seq_id,
                EngineEvent::Finish {
                    seq_id: seq.seq_id,
                    reason,
                },
            )
            .await;
            self.event_senders.remove(&seq.seq_id);
        } else {
            self.send_event(
                seq.seq_id,
                EngineEvent::Token {
                    seq_id: seq.seq_id,
                    token_id,
                    text: None, // decoded by the server layer
                },
            )
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
            // Invariant: position_offset = prompt_len + generated_len - 1
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
