//! Engine: main inference runtime loop that orchestrates scheduling,
//! model forward passes, and token sampling.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{error, warn};

use forge_core::{
    Backend, FinishReason, InferenceRequest, KvCache, Model, ModelInput, Result, ScheduledSeq,
    Scheduler, SeqMetadata,
};

use crate::constraints::fsm::FsmConstraint;
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
    /// Optional FSM constraint for structured output.
    pub constraint: Option<Box<dyn FsmConstraint>>,
}

/// Per-sequence FSM constraint state.
struct SeqConstraint {
    fsm: Box<dyn FsmConstraint>,
    state: u32,
}

/// Optional decode function for stop_strings checking.
/// Takes a slice of token IDs and returns the decoded text.
pub type DecodeFn = Arc<dyn Fn(&[u32]) -> Option<String> + Send + Sync>;

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
    /// Per-sequence FSM constraints (seq_id → constraint + state).
    constraints: HashMap<u64, SeqConstraint>,
    /// Optional token decoder for stop_strings enforcement.
    decode_fn: Option<DecodeFn>,
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
            constraints: HashMap::new(),
            decode_fn: None,
        }
    }

    /// Set a decode function for stop_strings enforcement.
    pub fn with_decode_fn(mut self, f: DecodeFn) -> Self {
        self.decode_fn = Some(f);
        self
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

            // Allocate KV cache for new prefill sequences.
            // Allocation errors are handled per-sequence rather than killing the
            // engine loop, so transient OOM only drops the affected request.
            let mut failed_seq_ids = Vec::new();
            for seq in &batch.prefill_seqs {
                if let Err(e) = self.kv_cache.allocate(seq.seq_id, seq.token_ids.len()) {
                    error!(seq_id = seq.seq_id, error = %e, "KV cache allocation failed");
                    self.send_event(
                        seq.seq_id,
                        EngineEvent::Error {
                            seq_id: seq.seq_id,
                            error: format!("cache allocation failed: {e}"),
                        },
                    )
                    .await;
                    let _ = self.scheduler.cancel(seq.seq_id);
                    self.event_senders.remove(&seq.seq_id);
                    self.constraints.remove(&seq.seq_id);
                    failed_seq_ids.push(seq.seq_id);
                }
            }

            // Phase 1: process one sequence at a time
            let all_seqs: Vec<ScheduledSeq> = batch
                .prefill_seqs
                .into_iter()
                .chain(batch.decode_seqs)
                .filter(|s| !failed_seq_ids.contains(&s.seq_id))
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
                    self.constraints.remove(&seq.seq_id);
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
                if let Some(fsm) = req.constraint {
                    let initial = fsm.initial_state();
                    self.constraints.insert(
                        handle.seq_id,
                        SeqConstraint {
                            fsm,
                            state: initial,
                        },
                    );
                }
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
                self.constraints.remove(&seq_id);
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

        // Build constraint reference for sampling
        let constraint_ref = self
            .constraints
            .get(&seq.seq_id)
            .map(|c| (&*c.fsm as &dyn FsmConstraint, c.state));

        let result = self.sampler.sample_with_constraint(
            last_logits,
            &seq.sampling_params,
            &generated,
            constraint_ref,
        )?;

        let token_id = result.token_id;
        self.scheduler.append_token(seq.seq_id, token_id)?;

        // Advance FSM state if constraint is active
        if let Some(seq_constraint) = self.constraints.get_mut(&seq.seq_id) {
            match seq_constraint.fsm.next_state(seq_constraint.state, token_id) {
                Some(next) => seq_constraint.state = next,
                None => {
                    warn!(
                        seq_id = seq.seq_id,
                        token_id,
                        fsm_state = seq_constraint.state,
                        "FSM transition invalid for sampled token; removing constraint"
                    );
                    self.constraints.remove(&seq.seq_id);
                }
            }
        }

        // Check stop conditions
        let stop_token_hit = seq.sampling_params.stop_token_ids.contains(&token_id);
        let max_tokens_hit = generated.len() + 1 >= seq.sampling_params.max_tokens;

        // Check stop_strings against decoded text (if decoder available and strings configured)
        let stop_string_hit = if !seq.sampling_params.stop_strings.is_empty() {
            self.decode_fn.as_ref().and_then(|decode| {
                // Include the just-sampled token in the decode
                let mut all_tokens = generated.clone();
                all_tokens.push(token_id);
                let text = decode(&all_tokens)?;
                seq.sampling_params
                    .stop_strings
                    .iter()
                    .any(|s| text.ends_with(s))
                    .then_some(true)
            })
        } else {
            None
        };

        let should_finish = stop_token_hit || max_tokens_hit || stop_string_hit.unwrap_or(false);

        if should_finish {
            let reason = if stop_token_hit {
                FinishReason::EosToken
            } else if stop_string_hit.unwrap_or(false) {
                FinishReason::StopString
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
            self.constraints.remove(&seq.seq_id);
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
