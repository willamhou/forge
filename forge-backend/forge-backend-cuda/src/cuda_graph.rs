//! Per-bucket CUDA Graph capture / replay cache.
//!
//! Captures a user closure once per `bucket` key (typically a batch-shape
//! discriminator), then replays the captured graph on subsequent hits. The
//! captured graph baked the device pointers and kernel arg values present
//! at capture time — the caller is responsible for ensuring those pointers
//! stay valid (see Task 2.5a / 2.5b: persistent i32 scratch +
//! `paged_attention_into` are forge's answer to that requirement for the
//! decode path).
//!
//! Built on top of cudarc 0.17.8's `CudaGraph` which already owns
//! `cu_graph` + `cu_graph_exec` via RAII; this module only manages the
//! per-bucket lifecycle.
//!
//! Stream / context requirements (validated in Task 1 spike, enforced by
//! `CudaBackend::new`):
//! - Stream must be `ctx.new_stream()` — `ctx.default_stream()` (NULL
//!   legacy stream) cannot be captured (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
//! - `unsafe { ctx.disable_event_tracking() }` must be in effect — otherwise
//!   cudarc's auto-inserted `cuStreamWaitEvent` calls reference events
//!   outside the capture region and invalidate the graph.
//!
//! ## Threading
//!
//! `&mut self` on `run_or_capture` prevents concurrent capture on the same
//! cache instance. However, the captured stream is `Arc<CudaStream>` — any
//! other holder (e.g. another `CudaBackend` op called from a background
//! task) that issues launches against that stream BETWEEN `begin_capture`
//! and `end_capture` will have those launches accidentally recorded into
//! the graph, or worse, invalidate the capture.
//!
//! This is safe today because `CudaBackend`'s contract (see its struct doc)
//! is single-threaded engine access. If that contract is ever relaxed,
//! `CudaGraphCache` callers must serialize all stream usage during
//! `run_or_capture` themselves.
//!
//! No engine integration here — that's Task 4 (`DecodeGraphRunner`).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, CudaGraph, CudaStream};

use forge_core::{ForgeError, Result};

/// Per-bucket captured graph. Owns the instantiated executable; dropping
/// releases both the recorded graph and the instantiated exec (cudarc RAII).
struct CapturedGraph {
    graph: CudaGraph,
}

/// Capture-once / replay-many cache, keyed by `u32` bucket.
///
/// The bucket value is opaque to the cache — callers typically derive it
/// from the batch-shape (e.g. `active_seqs.len() as u32`), but any
/// deterministic discriminator works as long as the closure passed to
/// `run_or_capture` produces the same kernel sequence + arg-pointer pattern
/// for the same bucket.
pub struct CudaGraphCache {
    /// Held only to document the context-stream binding contract; not used
    /// directly (capture works through `stream`).
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    /// The stream all captured ops must launch on (see module docs).
    stream: Arc<CudaStream>,
    graphs: HashMap<u32, CapturedGraph>,
}

impl CudaGraphCache {
    /// Construct an empty cache bound to `ctx` and `stream`. The stream is
    /// expected to be the same one the engine uses for its kernel launches —
    /// otherwise captured kernels would run on a different stream than the
    /// rest of the pipeline.
    pub fn new(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self {
            ctx,
            stream,
            graphs: HashMap::new(),
        }
    }

    /// True if a captured graph exists for `bucket`.
    pub fn has(&self, bucket: u32) -> bool {
        self.graphs.contains_key(&bucket)
    }

    /// Number of captured graphs currently cached.
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Run `fwd`, capturing on cache miss and replaying on hit.
    ///
    /// - **Miss**: enter capture mode → run `fwd` (which issues kernel
    ///   launches; capture records them without executing) → exit capture
    ///   mode → launch the resulting graph once (this is what actually
    ///   produces the kernel side effects for this call) → insert into cache.
    /// - **Hit**: launch the cached graph.
    ///
    /// The closure must not synchronize the stream, perform device-host
    /// copies that wait on completion, or do any other operation that is
    /// incompatible with capture mode. Allocations during capture are
    /// dangerous (they would be baked into the graph as graph-internal
    /// allocations); forge avoids this by routing through persistent
    /// scratches + caller-provided output buffers.
    ///
    /// `fwd` errors are surfaced even when `end_capture` also errors —
    /// the user-side cause is reported first, the downstream
    /// `STREAM_CAPTURE_INVALIDATED` is swallowed.
    pub fn run_or_capture<F>(&mut self, bucket: u32, fwd: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        if let Some(g) = self.graphs.get(&bucket) {
            return g
                .graph
                .launch()
                .map_err(|e| ForgeError::Cuda(format!("graph.launch (cache hit): {e}")));
        }

        // Cache miss: capture, instantiate, then launch once.
        //
        // RELAXED capture mode (vs GLOBAL or THREAD_LOCAL) is the one the
        // Task 1 spike validated against cuBLAS + custom kernels. GLOBAL
        // would reject any concurrent operation in the same process —
        // unnecessarily strict for our single-stream usage.
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| ForgeError::Cuda(format!("begin_capture: {e}")))?;

        let fwd_result = fwd();

        // Always end capture, even if fwd failed — capture state must be
        // drained from the stream regardless. The Task 1 spike found that
        // AUTO_FREE_ON_LAUNCH (=1) is the only safe `_NONE`-substitute flag:
        // cudarc 0.17.8 bindgen omits a `_NONE` (=0) enum variant, and
        // UPLOAD (=2) requires the WithParams API. AUTO_FREE_ON_LAUNCH is a
        // genuine no-op for graphs that contain no graph-internal allocs.
        let instantiate_flag = CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph_result = self.stream.end_capture(instantiate_flag);

        // Surface fwd error first — the end_capture failure is downstream
        // (STREAM_CAPTURE_INVALIDATED) and would mask the real cause.
        fwd_result?;
        let graph = graph_result
            .map_err(|e| ForgeError::Cuda(format!("end_capture: {e}")))?
            .ok_or_else(|| {
                ForgeError::Cuda("end_capture returned None — closure issued no captureable ops".into())
            })?;

        graph
            .launch()
            .map_err(|e| ForgeError::Cuda(format!("graph.launch (initial): {e}")))?;

        self.graphs.insert(bucket, CapturedGraph { graph });
        Ok(())
    }

    /// Replay a previously-captured graph. Errors if `bucket` was never
    /// captured (does not implicitly capture). Used by benchmarks and tests
    /// that want to isolate replay overhead from capture overhead.
    pub fn replay(&self, bucket: u32) -> Result<()> {
        let g = self.graphs.get(&bucket).ok_or_else(|| {
            ForgeError::Cuda(format!(
                "replay: no captured graph for bucket {bucket} (call run_or_capture first)"
            ))
        })?;
        g.graph
            .launch()
            .map_err(|e| ForgeError::Cuda(format!("graph.launch (replay): {e}")))
    }

    /// Drop the captured graph for `bucket`, if any. The next
    /// `run_or_capture(bucket, ...)` will re-capture. Used to invalidate
    /// stale captures when persistent device pointers move (e.g. paged
    /// attention scratch grew — see `CudaBackend::paged_scratch_versions`).
    pub fn invalidate(&mut self, bucket: u32) -> bool {
        self.graphs.remove(&bucket).is_some()
    }

    /// Drop all captured graphs.
    pub fn clear(&mut self) {
        self.graphs.clear();
    }
}
