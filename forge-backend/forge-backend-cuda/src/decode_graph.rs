//! Decode-side CUDA Graph dispatcher.
//!
//! Routes batched decode through a [`CudaGraphCache`] keyed by batch_size.
//! Buckets are pre-configured (typically `[1, 2, 4, 8, ..., max_batch]`);
//! exact-match dispatches go through capture/replay, non-bucket batch
//! sizes fall through to the uncaptured path.
//!
//! ## Scope of this task (4)
//!
//! Library only — no engine wiring. The runner is correct in isolation but
//! requires Task 5 (persistent input/output buffers throughout the decode
//! forward pass) before the captured graph paths produce correct results
//! for full-model decode. Today the only kernel forge has end-to-end
//! capture-safe is `paged_attention_into` (Task 2.5b), which is what the
//! integration test in `tests/test_decode_graph.rs` captures.
//!
//! ## Bucketing
//!
//! Caller supplies the bucket list at construction. The runner does NOT pad
//! non-bucket sizes up to the nearest bucket — that's Task 7's job (it
//! requires the model + cache to accept padded batches with masked-out
//! "filler" rows). Until then, non-bucket sizes silently fall back.
//!
//! ## Capture failure handling
//!
//! If `cache.run_or_capture` fails (e.g. closure errored inside capture),
//! the error propagates. A failed capture leaves no entry in the cache, so
//! a retry will re-capture.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use forge_core::Result;

use crate::cuda_graph::CudaGraphCache;

/// Per-bucket decode dispatcher.
pub struct DecodeGraphRunner {
    cache: CudaGraphCache,
    /// Sorted, deduplicated bucket list. Bucket = `batch_size as u32`.
    buckets: Vec<u32>,
}

impl DecodeGraphRunner {
    /// Construct from an existing `CudaGraphCache` and a bucket list.
    /// Bucket list is sorted + deduplicated on the way in.
    pub fn new(cache: CudaGraphCache, mut buckets: Vec<u32>) -> Self {
        buckets.sort_unstable();
        buckets.dedup();
        Self { cache, buckets }
    }

    /// Convenience: build a fresh `CudaGraphCache` and wrap.
    pub fn with_buckets(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        buckets: Vec<u32>,
    ) -> Self {
        Self::new(CudaGraphCache::new(ctx, stream), buckets)
    }

    /// Configured bucket list.
    pub fn buckets(&self) -> &[u32] {
        &self.buckets
    }

    /// True if `batch_size` exactly matches a configured bucket.
    pub fn is_bucket(&self, batch_size: u32) -> bool {
        self.buckets.binary_search(&batch_size).is_ok()
    }

    /// Borrow the underlying graph cache. Mostly for tests / diagnostics.
    pub fn cache(&self) -> &CudaGraphCache {
        &self.cache
    }

    /// Number of buckets currently captured.
    pub fn captured_count(&self) -> usize {
        self.cache.len()
    }

    /// Run `fwd` for a decode batch of `batch_size`.
    ///
    /// - **Bucket exact match**: dispatched through `cache.run_or_capture`.
    ///   First call captures + launches; subsequent calls replay.
    /// - **Non-bucket batch size**: `fwd` invoked directly (uncaptured).
    ///   Task 7 will lift this by padding up to the nearest bucket.
    ///
    /// Caller's closure must produce kernel launches whose every device-pointer
    /// argument is stable across calls (persistent pool tensors, persistent
    /// scratches, caller-provided output buffers). Otherwise the replay reads
    /// from stale / freed memory.
    ///
    /// ## Host-side side effects are NOT replayed
    ///
    /// On bucket cache hit the closure is NOT invoked — cudarc replays only
    /// the recorded kernel launches + memcpys. Any host-side bookkeeping the
    /// closure performs (counters, tracing, metrics, log lines) only runs on
    /// the first call (during capture) and is silently skipped on every
    /// subsequent replay. Callers that need per-call host-side bookkeeping
    /// must hoist it OUT of the closure (i.e. perform it around `dispatch`,
    /// not inside `fwd`).
    pub fn dispatch<F>(&mut self, batch_size: u32, fwd: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        if self.is_bucket(batch_size) {
            self.cache.run_or_capture(batch_size, fwd)
        } else {
            fwd()
        }
    }

    /// Drop all captured graphs. Use when persistent device pointers move
    /// (e.g. paged-attention scratch grew — check
    /// `CudaBackend::paged_scratch_versions` and call this on version bump).
    pub fn invalidate_all(&mut self) {
        self.cache.clear();
    }

    /// Invalidate just one bucket's captured graph.
    pub fn invalidate(&mut self, batch_size: u32) -> bool {
        self.cache.invalidate(batch_size)
    }
}
