//! Engine: main inference runtime loop that orchestrates scheduling,
//! model forward passes, and token sampling.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{error, warn};

use forge_core::{
    Backend, DecodeGraphDispatch, FinishReason, InferenceRequest, KvCache, Model, ModelInput,
    Result, ScheduledSeq, Scheduler, SeqMetadata,
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

/// Result of a batched decode forward: either the per-sequence token ids
/// (device-argmax fast path, only N ids leave the GPU) or the full host
/// `[N, vocab_size]` logits (CPU sampler path).
enum DecodeOut {
    Ids(Vec<u32>),
    Logits(Vec<f32>),
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
    /// Per-sequence token buffer for stop_string look-ahead.
    /// Tokens are held here until we confirm they don't form part of a
    /// stop string, then flushed to the event channel.
    stop_buffers: HashMap<u64, Vec<u32>>,
    /// Optional CUDA-Graph decode dispatcher (None on CPU / when disabled).
    decode_runner: Option<Box<dyn DecodeGraphDispatch>>,
    /// Configured CUDA-Graph batch-size buckets (sorted, deduped).
    decode_buckets: Vec<u32>,
    /// Per-bucket persistent decode state (allocated lazily on first use).
    decode_states: HashMap<u32, M::DecodeState>,
    /// Last observed backend capture epoch; a change invalidates all graphs.
    last_capture_epoch: u64,
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
            stop_buffers: HashMap::new(),
            decode_runner: None,
            decode_buckets: Vec::new(),
            decode_states: HashMap::new(),
            last_capture_epoch: 0,
        }
    }

    /// Set a decode function for stop_strings enforcement.
    pub fn with_decode_fn(mut self, f: DecodeFn) -> Self {
        self.decode_fn = Some(f);
        self
    }

    /// Enable CUDA-Graph decode capture for the given batch-size `buckets`.
    ///
    /// No-op (logs a warning, stays on the eager path) when the backend has no
    /// graph support — `Backend::make_decode_graph_runner` returns `None` for
    /// CPU and for naive (non-paged) KV caches. The default is off; the server
    /// opts in via `--cuda-graph`.
    pub fn with_cuda_graph(mut self, mut buckets: Vec<u32>) -> Self {
        buckets.retain(|&b| b > 0);
        buckets.sort_unstable();
        buckets.dedup();
        match self.backend.make_decode_graph_runner(&buckets) {
            Some(runner) => {
                self.decode_runner = Some(runner);
                self.decode_buckets = buckets;
                self.last_capture_epoch = self.backend.decode_capture_epoch();
            }
            None => {
                warn!("CUDA graph requested but backend has no graph support — staying eager");
            }
        }
        self
    }

    /// True if `batch_size` exactly matches a configured CUDA-Graph bucket.
    fn is_decode_bucket(&self, batch_size: u32) -> bool {
        self.decode_buckets.binary_search(&batch_size).is_ok()
    }

    /// Main engine loop. Runs until the request channel is closed.
    pub async fn run(&mut self) -> Result<()> {
        loop {
            // Drain incoming requests (non-blocking)
            let drained = self.drain_requests();

            let cache_usage = self.kv_cache.usage();
            let batch = self.scheduler.schedule(&cache_usage)?;

            // Notify clients of rejected sequences before checking whether
            // the batch is empty.  `is_empty()` ignores `rejected`, so
            // a scheduling round that only rejects requests would otherwise skip
            // error emission entirely, leaving request handlers hanging.
            for (seq_id, reason) in &batch.rejected {
                self.send_event(
                    *seq_id,
                    EngineEvent::Error {
                        seq_id: *seq_id,
                        error: reason.clone(),
                    },
                );
                self.event_senders.remove(seq_id);
                self.constraints.remove(seq_id);
                self.stop_buffers.remove(seq_id);
            }

            if batch.is_empty() {
                if drained == 0 && batch.rejected.is_empty() {
                    // No work, no rejections, no new requests — wait for a request
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

            // Allocate KV cache for new prefill sequences (first chunk only).
            // Allocation errors are handled per-sequence rather than killing the
            // engine loop, so transient OOM only drops the affected request.
            let mut failed_seq_ids = Vec::new();
            for seq in &batch.prefill_seqs {
                // Only allocate on the first prefill chunk (offset == 0).
                // Subsequent chunks append to an already-allocated cache entry.
                if seq.position_offset != 0 {
                    continue;
                }
                if let Err(e) = self.kv_cache.allocate(seq.seq_id, seq.total_prompt_len) {
                    error!(seq_id = seq.seq_id, error = %e, "KV cache allocation failed");
                    self.send_event(
                        seq.seq_id,
                        EngineEvent::Error {
                            seq_id: seq.seq_id,
                            error: format!("cache allocation failed: {e}"),
                        },
                    );
                    let _ = self.scheduler.cancel(seq.seq_id);
                    self.event_senders.remove(&seq.seq_id);
                    self.constraints.remove(&seq.seq_id);
                    self.stop_buffers.remove(&seq.seq_id);
                    failed_seq_ids.push(seq.seq_id);
                }
            }

            // Prefill: sequential (variable-length, can't concatenate)
            for seq in batch
                .prefill_seqs
                .iter()
                .filter(|s| !failed_seq_ids.contains(&s.seq_id))
            {
                if let Err(e) = self.process_sequence(seq) {
                    error!(seq_id = seq.seq_id, error = %e, "prefill failed");
                    self.send_event(
                        seq.seq_id,
                        EngineEvent::Error {
                            seq_id: seq.seq_id,
                            error: e.to_string(),
                        },
                    );
                    let _ = self.scheduler.cancel(seq.seq_id);
                    let _ = self.kv_cache.free(seq.seq_id);
                    self.event_senders.remove(&seq.seq_id);
                    self.constraints.remove(&seq.seq_id);
                    self.stop_buffers.remove(&seq.seq_id);
                }
            }

            // Decode: batched when > 1 sequence, single otherwise
            let decode_seqs: Vec<ScheduledSeq> = batch
                .decode_seqs
                .into_iter()
                .filter(|s| !failed_seq_ids.contains(&s.seq_id))
                .collect();

            // Route through the batched path when graph capture is on (so
            // batch=1 decode also benefits) or whenever there's more than one
            // decode sequence. Single-sequence decode without graph capture
            // stays on the simpler `process_sequence` path.
            let route_batched =
                !decode_seqs.is_empty() && (self.decode_runner.is_some() || decode_seqs.len() > 1);

            if route_batched {
                if let Err(e) = self.process_decode_batch(&decode_seqs) {
                    // Forward pass failed — clean up all sequences in the batch
                    error!(error = %e, "batch decode forward failed");
                    for seq in &decode_seqs {
                        self.send_event(
                            seq.seq_id,
                            EngineEvent::Error {
                                seq_id: seq.seq_id,
                                error: e.to_string(),
                            },
                        );
                        let _ = self.scheduler.cancel(seq.seq_id);
                        let _ = self.kv_cache.free(seq.seq_id);
                        self.event_senders.remove(&seq.seq_id);
                        self.constraints.remove(&seq.seq_id);
                        self.stop_buffers.remove(&seq.seq_id);
                    }
                }
            } else {
                for seq in &decode_seqs {
                    if let Err(e) = self.process_sequence(seq) {
                        error!(seq_id = seq.seq_id, error = %e, "decode failed");
                        self.send_event(
                            seq.seq_id,
                            EngineEvent::Error {
                                seq_id: seq.seq_id,
                                error: e.to_string(),
                            },
                        );
                        let _ = self.scheduler.cancel(seq.seq_id);
                        let _ = self.kv_cache.free(seq.seq_id);
                        self.event_senders.remove(&seq.seq_id);
                        self.constraints.remove(&seq.seq_id);
                        self.stop_buffers.remove(&seq.seq_id);
                    }
                }
            }

            // Yield to the runtime after each scheduling step. The per-token
            // path is fully synchronous (forward + blocking `synchronize` +
            // `try_send`) and never awaits, so without this the loop monopolises
            // its worker and the SSE handler task isn't polled until generation
            // ends — making streaming clients receive every token at once. One
            // cooperative yield per step lets the handler drain events live.
            tokio::task::yield_now().await;
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

    /// Send an event to a specific sequence's consumer. Uses `try_send` to
    /// avoid blocking the engine loop when a consumer is slow. If the channel
    /// is full or disconnected, cancel the sequence and clean up resources.
    fn send_event(&mut self, seq_id: u64, event: EngineEvent) {
        if let Some(tx) = self.event_senders.get(&seq_id) {
            match tx.try_send(event) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    warn!(seq_id, "event channel full, cancelling slow consumer");
                    let _ = self.scheduler.cancel(seq_id);
                    let _ = self.kv_cache.free(seq_id);
                    self.event_senders.remove(&seq_id);
                    self.constraints.remove(&seq_id);
                    self.stop_buffers.remove(&seq_id);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    warn!(seq_id, "event receiver dropped, cancelling sequence");
                    let _ = self.scheduler.cancel(seq_id);
                    let _ = self.kv_cache.free(seq_id);
                    self.event_senders.remove(&seq_id);
                    self.constraints.remove(&seq_id);
                    self.stop_buffers.remove(&seq_id);
                }
            }
        }
    }

    /// Process a single sequence: forward pass + sample + emit event.
    /// For chunked prefill, non-final chunks only run the forward pass (no sampling).
    fn process_sequence(&mut self, seq: &ScheduledSeq) -> Result<()> {
        // Short-circuit: if max_tokens is already reached (e.g. max_tokens=0),
        // finish immediately without running the forward pass.
        let generated = self.scheduler.get_generated_tokens(seq.seq_id)?;
        if generated.len() >= seq.sampling_params.max_tokens {
            self.scheduler.finish(seq.seq_id, FinishReason::MaxTokens)?;
            self.kv_cache.free(seq.seq_id)?;
            self.send_event(
                seq.seq_id,
                EngineEvent::Finish {
                    seq_id: seq.seq_id,
                    reason: FinishReason::MaxTokens,
                },
            );
            self.event_senders.remove(&seq.seq_id);
            self.constraints.remove(&seq.seq_id);
            self.stop_buffers.remove(&seq.seq_id);
            return Ok(());
        }

        let input = self.build_input(seq);

        let output = self.model.forward(&input, &mut *self.kv_cache)?;
        self.backend.synchronize()?;

        // For non-final prefill chunks, skip sampling — just populate KV cache.
        if seq.is_prefill && !seq.is_last_prefill_chunk {
            return Ok(());
        }

        // Copy logits to host for CPU sampling
        let logits_host = self.backend.copy_to_host_f32(&output.logits)?;

        // Only look at the last token's logits (for both prefill and decode)
        let vocab_size = self.model.config().vocab_size;
        if vocab_size == 0 || logits_host.len() < vocab_size {
            return Err(forge_core::ForgeError::InvalidArgument(format!(
                "empty or undersized logits: got {} elements, vocab_size={}",
                logits_host.len(),
                vocab_size
            )));
        }
        let num_tokens = logits_host.len() / vocab_size;
        let last_logits = &logits_host[(num_tokens - 1) * vocab_size..];

        self.sample_and_emit(seq, last_logits)
    }

    /// Process multiple decode sequences in a single batched forward pass.
    ///
    /// Each sequence contributes exactly 1 token. Tokens are concatenated into
    /// `[N, hidden_size]` for the forward pass, then logits are split per-sequence
    /// for independent sampling.
    fn process_decode_batch(&mut self, seqs: &[ScheduledSeq]) -> Result<()> {
        // Filter out sequences that have already hit max_tokens
        let mut active_seqs = Vec::with_capacity(seqs.len());
        for seq in seqs {
            let generated = self.scheduler.get_generated_tokens(seq.seq_id)?;
            if generated.len() >= seq.sampling_params.max_tokens {
                self.scheduler.finish(seq.seq_id, FinishReason::MaxTokens)?;
                self.kv_cache.free(seq.seq_id)?;
                self.send_event(
                    seq.seq_id,
                    EngineEvent::Finish {
                        seq_id: seq.seq_id,
                        reason: FinishReason::MaxTokens,
                    },
                );
                self.event_senders.remove(&seq.seq_id);
                self.constraints.remove(&seq.seq_id);
                self.stop_buffers.remove(&seq.seq_id);
                continue;
            }
            active_seqs.push(seq);
        }

        if active_seqs.is_empty() {
            return Ok(());
        }

        // Build batched ModelInput (N sequences, 1 token each)
        let input = self.build_batch_input(&active_seqs);
        let n = active_seqs.len() as u32;

        // Device-side sampling plan. Greedy (temp ≤ 0) + unconstrained seqs use
        // argmax; temp > 0 seqs without top-k/p/min-p use on-device Gumbel-max.
        // Either way only N token ids leave the GPU. Any seq needing the CPU
        // sampler (top-k/p/min-p, penalties, FSM mask) forces the full
        // host-logits path for the whole batch.
        let all_argmax = active_seqs.iter().all(|s| self.decode_argmax_eligible(s));
        let all_device = all_argmax
            || active_seqs
                .iter()
                .all(|s| self.decode_argmax_eligible(s) || self.decode_gumbel_eligible(s));

        // Per-row sampling params, built once (they don't depend on the logits).
        // Only needed for the mixed greedy/Gumbel `sample` path; the all-argmax
        // path uploads nothing.
        let sample_params: Option<(Vec<f32>, Vec<u64>, Vec<u32>)> = if all_device && !all_argmax {
            let mut temps = Vec::with_capacity(active_seqs.len());
            let mut seeds = Vec::with_capacity(active_seqs.len());
            let mut steps = Vec::with_capacity(active_seqs.len());
            for s in &active_seqs {
                temps.push(s.sampling_params.temperature);
                // seed: explicit per-request seed, else derive from seq_id so
                // each sequence still gets an independent, deterministic stream.
                seeds.push(s.sampling_params.seed.unwrap_or(s.seq_id));
                steps.push(self.get_generated_count(s.seq_id) as u32);
            }
            Some((temps, seeds, steps))
        } else {
            None
        };

        // Run the decode forward via the CUDA-Graph capture/replay path
        // (bucketed batch sizes) or the eager forward, then produce either the
        // per-seq token ids (fast path) or the host [N, vocab_size] logits.
        let use_graph = self.decode_runner.is_some() && self.is_decode_bucket(n);
        let decode_out: DecodeOut = if use_graph {
            // Lazily allocate this bucket's persistent decode state.
            if !self.decode_states.contains_key(&n) {
                let st = self.model.make_decode_state(n as usize)?;
                self.decode_states.insert(n, st);
            }

            // Stage this step's inputs into persistent scratch (outside any
            // captured region) so a replayed graph reads fresh data.
            {
                let state = self.decode_states.get_mut(&n).unwrap();
                self.model
                    .stage_decode(&input, &mut *self.kv_cache, state)?;
            }

            // Capture-epoch check AFTER staging: staging is what grows the
            // scratch/pool buffers, so any device/host pointer a captured graph
            // baked may have moved during this very step. Checking here (rather
            // than before `stage_decode`) guarantees a staging-time grow drops
            // the stale graphs before `dispatch` could replay one.
            let epoch = self.backend.decode_capture_epoch();
            if epoch != self.last_capture_epoch {
                if let Some(r) = self.decode_runner.as_mut() {
                    r.invalidate_all();
                }
                self.last_capture_epoch = epoch;
            }

            // Capture-or-replay the pure-kernel compute. Disjoint-field borrows
            // keep the dispatcher and the closure's captures separate.
            {
                let model = &self.model;
                let kv = &mut *self.kv_cache;
                let state = self.decode_states.get_mut(&n).unwrap();
                let runner = self
                    .decode_runner
                    .as_mut()
                    .expect("decode_runner is Some in the graph branch");
                let mut fwd = || model.compute_decode(kv, state);
                runner.dispatch(n, &mut fwd)?;
            }
            self.backend.synchronize()?;

            let state = self.decode_states.get(&n).unwrap();
            let logits = self.model.decode_logits(state);
            if all_argmax {
                DecodeOut::Ids(self.backend.argmax(logits)?)
            } else if let Some((temps, seeds, steps)) = &sample_params {
                DecodeOut::Ids(self.backend.sample(logits, temps, seeds, steps)?)
            } else {
                DecodeOut::Logits(self.backend.copy_to_host_f32(logits)?)
            }
        } else {
            let output = self.model.forward(&input, &mut *self.kv_cache)?;
            self.backend.synchronize()?;
            if all_argmax {
                DecodeOut::Ids(self.backend.argmax(&output.logits)?)
            } else if let Some((temps, seeds, steps)) = &sample_params {
                DecodeOut::Ids(self.backend.sample(&output.logits, temps, seeds, steps)?)
            } else {
                DecodeOut::Logits(self.backend.copy_to_host_f32(&output.logits)?)
            }
        };
        let vocab_size = self.model.config().vocab_size;

        // Per-sequence emit. The fast path skips the sampler entirely (eligible
        // seqs have no FSM, so `emit_token` handles append/stop/finish); the
        // full path runs the CPU sampler over each seq's logits slice.
        for (i, seq) in active_seqs.iter().enumerate() {
            let res = match &decode_out {
                DecodeOut::Ids(ids) => {
                    let token_id = ids[i];
                    match self.scheduler.get_generated_tokens(seq.seq_id) {
                        Ok(generated) => self.emit_token(seq, token_id, &generated),
                        Err(e) => Err(e),
                    }
                }
                DecodeOut::Logits(host) => {
                    self.sample_and_emit(seq, &host[i * vocab_size..(i + 1) * vocab_size])
                }
            };
            if let Err(e) = res {
                error!(seq_id = seq.seq_id, error = %e, "sampling failed in batch");
                self.send_event(
                    seq.seq_id,
                    EngineEvent::Error {
                        seq_id: seq.seq_id,
                        error: e.to_string(),
                    },
                );
                let _ = self.scheduler.cancel(seq.seq_id);
                let _ = self.kv_cache.free(seq.seq_id);
                self.event_senders.remove(&seq.seq_id);
                self.constraints.remove(&seq.seq_id);
                self.stop_buffers.remove(&seq.seq_id);
            }
        }

        Ok(())
    }

    /// Whether `seq` can be decoded with a pure device-side argmax: greedy
    /// (temperature ≤ 0), no penalties (which would reshape the logits), and no
    /// FSM constraint (which masks logits before sampling). top-k/top-p/min-p
    /// are irrelevant under greedy. These seqs need only the argmax token id,
    /// never the full host logits.
    fn decode_argmax_eligible(&self, seq: &ScheduledSeq) -> bool {
        let p = &seq.sampling_params;
        p.temperature <= 0.0
            && p.repetition_penalty == 1.0
            && p.presence_penalty == 0.0
            && p.frequency_penalty == 0.0
            && !self.constraints.contains_key(&seq.seq_id)
    }

    /// Whether `seq` can be sampled with on-device Gumbel-max: temperature > 0,
    /// no top-k/top-p/min-p filtering (not yet on the device path), no penalties
    /// (which need the token history on host), and no FSM constraint. The
    /// device `sample` op draws from `softmax(logits / temperature)`.
    fn decode_gumbel_eligible(&self, seq: &ScheduledSeq) -> bool {
        let p = &seq.sampling_params;
        p.temperature > 0.0
            && p.top_k.is_none()
            && p.top_p >= 1.0
            && p.min_p.is_none()
            && p.repetition_penalty == 1.0
            && p.presence_penalty == 0.0
            && p.frequency_penalty == 0.0
            && !self.constraints.contains_key(&seq.seq_id)
    }

    /// Sample a token from logits, advance FSM, handle stop conditions, emit events.
    ///
    /// Shared by `process_sequence` (single) and `process_decode_batch` (batched).
    fn sample_and_emit(&mut self, seq: &ScheduledSeq, last_logits: &[f32]) -> Result<()> {
        let generated = self.scheduler.get_generated_tokens(seq.seq_id)?;

        // Build constraint reference for sampling
        let constraint_ref = self
            .constraints
            .get(&seq.seq_id)
            .map(|c| (&*c.fsm as &dyn FsmConstraint, c.state));

        // If FSM is in a terminal state with no allowed tokens, finish cleanly
        // instead of sampling (which would pick an arbitrary token and fail).
        if let Some((fsm, state)) = constraint_ref {
            let no_tokens = fsm.allowed_tokens(state).is_none_or(|t| t.is_empty());
            if no_tokens && fsm.is_final_state(state) {
                self.scheduler
                    .finish(seq.seq_id, FinishReason::StopString)?;
                self.kv_cache.free(seq.seq_id)?;
                self.send_event(
                    seq.seq_id,
                    EngineEvent::Finish {
                        seq_id: seq.seq_id,
                        reason: FinishReason::StopString,
                    },
                );
                self.event_senders.remove(&seq.seq_id);
                self.constraints.remove(&seq.seq_id);
                self.stop_buffers.remove(&seq.seq_id);
                return Ok(());
            }
        }

        // When the FSM is in an accepting state, unmask EOS/stop tokens so the
        // model can choose to stop at a valid completion boundary instead of being
        // forced to continue until max_tokens.
        let result = if let Some((fsm, state)) = constraint_ref {
            if fsm.is_final_state(state) {
                // Apply the FSM mask manually, then restore EOS tokens.
                let mut masked_logits = last_logits.to_vec();
                fsm.mask_logits(state, &mut masked_logits);
                for &eos_id in &seq.sampling_params.stop_token_ids {
                    if (eos_id as usize) < masked_logits.len() {
                        masked_logits[eos_id as usize] = last_logits[eos_id as usize];
                    }
                }
                self.sampler
                    .sample_single(&masked_logits, &seq.sampling_params, &generated)?
            } else {
                self.sampler.sample_with_constraint(
                    last_logits,
                    &seq.sampling_params,
                    &generated,
                    constraint_ref,
                )?
            }
        } else {
            self.sampler
                .sample_single(last_logits, &seq.sampling_params, &generated)?
        };

        self.emit_token(seq, result.token_id, &generated)
    }

    /// Append `token_id` to `seq`, advance any FSM constraint, run stop /
    /// EOS / max-token checks, and emit the resulting events. Shared by the
    /// host sampler path ([`Self::sample_and_emit`]) and the device-argmax
    /// fast path in [`Self::process_decode_batch`]. `generated` is the
    /// sequence's tokens BEFORE this one.
    fn emit_token(&mut self, seq: &ScheduledSeq, token_id: u32, generated: &[u32]) -> Result<()> {
        self.scheduler.append_token(seq.seq_id, token_id)?;

        // Advance FSM state if constraint is active.
        // We use a two-phase approach: first compute the result inside the
        // borrow, then handle abort outside to avoid double-borrow of `self`.
        let mut fsm_finished = false;
        let mut fsm_abort: Option<(u32, u32)> = None; // (fsm_state, token_id)
        if let Some(seq_constraint) = self.constraints.get_mut(&seq.seq_id) {
            match seq_constraint
                .fsm
                .next_state(seq_constraint.state, token_id)
            {
                Some(next) => {
                    seq_constraint.state = next;
                    if seq_constraint.fsm.is_final_state(next) {
                        let has_transitions = seq_constraint
                            .fsm
                            .allowed_tokens(next)
                            .is_some_and(|t| !t.is_empty());
                        if !has_transitions {
                            fsm_finished = true;
                        }
                    }
                }
                None => {
                    if seq_constraint.fsm.is_final_state(seq_constraint.state)
                        && seq.sampling_params.stop_token_ids.contains(&token_id)
                    {
                        fsm_finished = true;
                    } else {
                        fsm_abort = Some((seq_constraint.state, token_id));
                    }
                }
            }
        }

        // Handle FSM abort outside the borrow scope.
        if let Some((fsm_state, bad_token)) = fsm_abort {
            error!(
                seq_id = seq.seq_id,
                token_id = bad_token,
                fsm_state,
                "FSM transition invalid for sampled token; aborting sequence"
            );
            self.constraints.remove(&seq.seq_id);
            self.stop_buffers.remove(&seq.seq_id);
            self.scheduler.finish(seq.seq_id, FinishReason::Cancelled)?;
            self.kv_cache.free(seq.seq_id)?;
            self.send_event(
                seq.seq_id,
                EngineEvent::Error {
                    seq_id: seq.seq_id,
                    error: format!(
                        "structured output constraint violated: \
                         no valid FSM transition from state {fsm_state} for token {bad_token}"
                    ),
                },
            );
            self.event_senders.remove(&seq.seq_id);
            return Ok(());
        }

        // Check stop conditions
        let stop_token_hit = seq.sampling_params.stop_token_ids.contains(&token_id);
        let max_tokens_hit = generated.len() + 1 >= seq.sampling_params.max_tokens;

        let has_stop_strings = !seq.sampling_params.stop_strings.is_empty();
        let stop_string_hit = if has_stop_strings {
            self.stop_buffers
                .entry(seq.seq_id)
                .or_default()
                .push(token_id);

            self.decode_fn.as_ref().and_then(|decode| {
                let mut all_tokens = generated.to_vec();
                all_tokens.push(token_id);
                let text = decode(&all_tokens)?;
                seq.sampling_params
                    .stop_strings
                    .iter()
                    .any(|s| text.contains(s))
                    .then_some(true)
            })
        } else {
            None
        };

        let should_finish =
            stop_token_hit || max_tokens_hit || stop_string_hit.unwrap_or(false) || fsm_finished;

        if stop_string_hit.unwrap_or(false) {
            self.stop_buffers.remove(&seq.seq_id);
        } else if has_stop_strings {
            let flush_all = should_finish;
            let prefix_match = if !flush_all {
                self.decode_fn.as_ref().and_then(|decode| {
                    let mut all_tokens = generated.to_vec();
                    all_tokens.push(token_id);
                    let text = decode(&all_tokens)?;
                    let is_prefix = seq.sampling_params.stop_strings.iter().any(|s| {
                        s.char_indices()
                            .skip(1)
                            .any(|(byte_pos, _)| text.ends_with(&s[..byte_pos]))
                            || text.ends_with(s)
                    });
                    Some(is_prefix)
                })
            } else {
                Some(false)
            };

            if !prefix_match.unwrap_or(false) || flush_all {
                if let Some(buf) = self.stop_buffers.remove(&seq.seq_id) {
                    for &tid in &buf {
                        self.send_event(
                            seq.seq_id,
                            EngineEvent::Token {
                                seq_id: seq.seq_id,
                                token_id: tid,
                                text: None,
                            },
                        );
                    }
                }
            }
        } else {
            self.send_event(
                seq.seq_id,
                EngineEvent::Token {
                    seq_id: seq.seq_id,
                    token_id,
                    text: None,
                },
            );
        }

        if should_finish {
            let reason = if stop_token_hit {
                FinishReason::EosToken
            } else if stop_string_hit.unwrap_or(false) {
                FinishReason::StopString
            } else if fsm_finished {
                FinishReason::EosToken
            } else {
                FinishReason::MaxTokens
            };
            self.scheduler.finish(seq.seq_id, reason)?;
            self.kv_cache.free(seq.seq_id)?;
            self.stop_buffers.remove(&seq.seq_id);
            self.send_event(
                seq.seq_id,
                EngineEvent::Finish {
                    seq_id: seq.seq_id,
                    reason,
                },
            );
            self.event_senders.remove(&seq.seq_id);
            self.constraints.remove(&seq.seq_id);
        }

        Ok(())
    }

    /// Build a batched ModelInput from multiple decode sequences (1 token each).
    fn build_batch_input(&self, seqs: &[&ScheduledSeq]) -> ModelInput {
        let mut token_ids = Vec::with_capacity(seqs.len());
        let mut positions = Vec::with_capacity(seqs.len());
        let mut seq_metadata = Vec::with_capacity(seqs.len());

        for seq in seqs {
            let gen_count = self.get_generated_count(seq.seq_id);
            let prompt_len = (seq.position_offset + 1).saturating_sub(gen_count).max(1);
            let offset = seq.position_offset as u32;
            let pos: Vec<u32> = (offset..offset + seq.token_ids.len() as u32).collect();

            token_ids.push(seq.token_ids.clone());
            positions.push(pos);
            seq_metadata.push(SeqMetadata {
                seq_id: seq.seq_id,
                prompt_len,
                generated_len: gen_count,
                is_prefill: false,
            });
        }

        ModelInput {
            token_ids,
            positions,
            seq_metadata,
        }
    }

    /// Build a single-sequence ModelInput from a ScheduledSeq.
    fn build_input(&self, seq: &ScheduledSeq) -> ModelInput {
        let gen_count = self.get_generated_count(seq.seq_id);

        let (prompt_len, generated_len) = if seq.is_prefill {
            // For chunked prefill, position_offset is the starting position of this chunk.
            // prompt_len reflects the total tokens fed so far (offset + chunk size).
            (seq.position_offset + seq.token_ids.len(), 0)
        } else {
            // Invariant: position_offset = prompt_len + generated_len - 1
            let prompt_len = (seq.position_offset + 1).saturating_sub(gen_count).max(1);
            (prompt_len, gen_count)
        };

        // Positions are absolute: [offset, offset+1, ..., offset+len-1]
        let offset = seq.position_offset as u32;
        let positions: Vec<u32> = (offset..offset + seq.token_ids.len() as u32).collect();

        ModelInput {
            token_ids: vec![seq.token_ids.clone()],
            positions: vec![positions],
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
