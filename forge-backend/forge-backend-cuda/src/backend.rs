use std::sync::{Arc, Mutex};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use forge_core::{
    Backend, DType, DecodeGraphDispatch, DecodeStageInputs, ForgeError, Result, Tensor,
};

use crate::decode_graph::DecodeGraphRunner;
use crate::tensor::CudaTensor;

struct KernelFunctions {
    add_f32: CudaFunction,
    add_bias_f32: CudaFunction,
    mul_f32: CudaFunction,
    mul_scalar_f32: CudaFunction,
    silu_f32: CudaFunction,
    fused_silu_mul_f32: CudaFunction,
    rms_norm_f32: CudaFunction,
    fused_residual_rms_norm_f32: CudaFunction,
    softmax_f32: CudaFunction,
    embedding_f32: CudaFunction,
    rope_f32: CudaFunction,
    transpose_f32: CudaFunction,
    // FP16 variants
    add_f16: CudaFunction,
    add_bias_f16: CudaFunction,
    mul_f16: CudaFunction,
    mul_scalar_f16: CudaFunction,
    silu_f16: CudaFunction,
    fused_silu_mul_f16: CudaFunction,
    rms_norm_f16: CudaFunction,
    fused_residual_rms_norm_f16: CudaFunction,
    softmax_f16: CudaFunction,
    embedding_f16: CudaFunction,
    rope_f16: CudaFunction,
    transpose_f16: CudaFunction,
    split_qkv_f32: CudaFunction,
    cast_f16_to_f32: CudaFunction,
    cast_f32_to_f16: CudaFunction,
    split_qkv_f16: CudaFunction,
    // Attention helpers
    extract_head_f32: CudaFunction,
    apply_causal_mask_f32: CudaFunction,
    interleave_heads_f32: CudaFunction,
    extract_head_f16: CudaFunction,
    apply_causal_mask_f16: CudaFunction,
    interleave_heads_f16: CudaFunction,
    // Batched decode attention
    batched_decode_attention_f32: CudaFunction,
    batched_decode_attention_f16: CudaFunction,
    // Paged attention (decode)
    paged_attention_f32: CudaFunction,
    paged_attention_f16: CudaFunction,

    scatter_kv_f32: CudaFunction,
    scatter_kv_f16: CudaFunction,
    // Sampling
    argmax_f32: CudaFunction,
    argmax_f16: CudaFunction,
    sample_gumbel_f32: CudaFunction,
    sample_gumbel_f16: CudaFunction,
}

// CudaBackend is Clone for sharing with components like NaiveKvCache,
// but cuBLAS/kernel calls are NOT thread-safe for concurrent use.
// The engine must ensure single-threaded access to the backend.
#[allow(dead_code)]
#[derive(Clone)]
pub struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) blas: Arc<CudaBlas>,
    kernels: Arc<KernelFunctions>,
    _module_f32: Arc<CudaModule>,
    _module_f16: Arc<CudaModule>,
    /// Persistent device scratch for paged_attention's i32 block_tables.
    /// Grows monotonically. Replaying a captured CUDA Graph requires the
    /// scratch's device pointer to be stable across calls; a `grow` event
    /// invalidates any captured graph that baked the previous pointer
    /// (Task 3's capture cache will react to the version bump).
    paged_block_tables: Arc<Mutex<I32Scratch>>,
    /// Persistent device scratch for paged_attention's i32 kv_lens. See above.
    paged_kv_lens: Arc<Mutex<I32Scratch>>,
    /// Persistent device scratch for embedding's u32 indices. Replaces the
    /// per-call `memcpy_stod(indices)` (fresh alloc) with `memcpy_htod`
    /// into a stable buffer — Task 5c.3.
    embedding_indices: Arc<Mutex<U32Scratch>>,
    /// Persistent device scratch for RoPE's f32 cos table (per-call upload
    /// from cached host gather, into stable device pointer).
    rope_cos: Arc<Mutex<F32Scratch>>,
    /// Persistent device scratch for RoPE's f32 sin table. See `rope_cos`.
    rope_sin: Arc<Mutex<F32Scratch>>,
    /// Persistent device scratch for the KV-scatter slot_mapping (i32). Lets
    /// `scatter_kv` read write destinations from a stable device pointer so the
    /// op can be recorded in a captured graph (the old `paged_write_kv`
    /// host-loop bakes destinations into memcpy nodes — not replay-safe).
    scatter_slot_mapping: Arc<Mutex<I32Scratch>>,
}

/// Monotonically-growing i32 device buffer. `version` increments on every
/// reallocation so callers (e.g. CUDA Graph capture cache) can detect when
/// the underlying device pointer has changed.
struct I32Scratch {
    buf: CudaSlice<i32>,
    version: u64,
    /// Persistent HOST mirror used as the `memcpy_htod` source. A captured
    /// graph bakes the host source pointer of an H2D copy; sourcing from this
    /// stable backend-owned buffer (instead of a per-call Vec that drops when
    /// the caller returns) keeps that pointer valid across capture + replay.
    /// Sized to match `buf`; never shrinks, so the pointer is stable once
    /// warmed to the steady-state size.
    host: Vec<i32>,
}

/// u32 sibling of `I32Scratch` — for kernels that read indices as u32
/// (embedding's `indices` array, etc.).
struct U32Scratch {
    buf: CudaSlice<u32>,
    version: u64,
    host: Vec<u32>,
}

/// f32 sibling of `I32Scratch` — for kernels that need a small per-call
/// f32 staging buffer (RoPE cos/sin tables gathered from host indices).
struct F32Scratch {
    buf: CudaSlice<f32>,
    version: u64,
    host: Vec<f32>,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx =
            CudaContext::new(ordinal).map_err(|e| ForgeError::Cuda(format!("context: {e}")))?;
        // Two CUDA-Graph-capture preconditions (validated in Task 1 spike):
        //
        // 1. `ctx.default_stream()` returns the NULL / legacy stream — CUDA
        //    Graphs cannot capture it (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
        //    Use a real non-blocking stream instead.
        //
        // 2. cudarc's `launch_builder` in multi-stream contexts auto-inserts
        //    `cuStreamWaitEvent` dependencies for read/write tracking. Those
        //    waits reference events recorded outside the capture region and
        //    would invalidate the graph (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED).
        //    We use one stream FIFO-ordered, so disable event tracking.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx
            .new_stream()
            .map_err(|e| ForgeError::Cuda(format!("new_stream: {e}")))?;
        let blas =
            CudaBlas::new(stream.clone()).map_err(|e| ForgeError::Cuda(format!("cublas: {e}")))?;

        // Compile F32 kernels (concatenate all module sources)
        let f32_src = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            forge_kernels::elementwise::F32_SRC,
            forge_kernels::norm::F32_SRC,
            forge_kernels::positional::F32_SRC,
            forge_kernels::memory::F32_SRC,
            forge_kernels::attention::F32_SRC,
            forge_kernels::decode_attention::F32_SRC,
            forge_kernels::paged_attention::F32_SRC,
            forge_kernels::sampling::F32_SRC,
        );
        let ptx_f32 =
            compile_ptx(&f32_src).map_err(|e| ForgeError::Cuda(format!("nvrtc f32: {e}")))?;
        let module_f32 = ctx
            .load_module(ptx_f32)
            .map_err(|e| ForgeError::Cuda(format!("module load f32: {e}")))?;

        // Compile F16 kernels — requires cuda_fp16.h from CUDA toolkit
        let f16_src = format!(
            "#include <cuda_fp16.h>\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            forge_kernels::elementwise::F16_SRC,
            forge_kernels::norm::F16_SRC,
            forge_kernels::positional::F16_SRC,
            forge_kernels::memory::F16_SRC,
            forge_kernels::attention::F16_SRC,
            forge_kernels::decode_attention::F16_SRC,
            forge_kernels::paged_attention::F16_SRC,
            forge_kernels::sampling::F16_SRC,
        );
        let cuda_include = Self::find_cuda_include()?;
        let ptx_f16 = cudarc::nvrtc::compile_ptx_with_opts(
            &f16_src,
            cudarc::nvrtc::CompileOptions {
                use_fast_math: Some(true),
                include_paths: vec![cuda_include],
                ..Default::default()
            },
        )
        .map_err(|e| ForgeError::Cuda(format!("nvrtc f16: {e}")))?;
        let module_f16 = ctx
            .load_module(ptx_f16)
            .map_err(|e| ForgeError::Cuda(format!("module load f16: {e}")))?;

        let load_f32 = |name: &str| -> Result<CudaFunction> {
            module_f32
                .load_function(name)
                .map_err(|e| ForgeError::Cuda(format!("load {name}: {e}")))
        };
        let load_f16 = |name: &str| -> Result<CudaFunction> {
            module_f16
                .load_function(name)
                .map_err(|e| ForgeError::Cuda(format!("load {name}: {e}")))
        };

        let kernels = KernelFunctions {
            add_f32: load_f32("add_f32")?,
            add_bias_f32: load_f32("add_bias_f32")?,
            mul_f32: load_f32("mul_f32")?,
            mul_scalar_f32: load_f32("mul_scalar_f32")?,
            silu_f32: load_f32("silu_f32")?,
            fused_silu_mul_f32: load_f32("fused_silu_mul_f32")?,
            rms_norm_f32: load_f32("rms_norm_f32")?,
            fused_residual_rms_norm_f32: load_f32("fused_residual_rms_norm_f32")?,
            softmax_f32: load_f32("softmax_f32")?,
            embedding_f32: load_f32("embedding_f32")?,
            rope_f32: load_f32("rope_f32")?,
            transpose_f32: load_f32("transpose_f32")?,
            split_qkv_f32: load_f32("split_qkv_f32")?,
            // F16 kernels
            add_f16: load_f16("add_f16")?,
            add_bias_f16: load_f16("add_bias_f16")?,
            mul_f16: load_f16("mul_f16")?,
            mul_scalar_f16: load_f16("mul_scalar_f16")?,
            silu_f16: load_f16("silu_f16")?,
            fused_silu_mul_f16: load_f16("fused_silu_mul_f16")?,
            rms_norm_f16: load_f16("rms_norm_f16")?,
            fused_residual_rms_norm_f16: load_f16("fused_residual_rms_norm_f16")?,
            softmax_f16: load_f16("softmax_f16")?,
            embedding_f16: load_f16("embedding_f16")?,
            rope_f16: load_f16("rope_f16")?,
            transpose_f16: load_f16("transpose_f16")?,
            cast_f16_to_f32: load_f16("cast_f16_to_f32")?,
            cast_f32_to_f16: load_f16("cast_f32_to_f16")?,
            split_qkv_f16: load_f16("split_qkv_f16")?,
            // Attention helpers
            extract_head_f32: load_f32("extract_head_f32")?,
            apply_causal_mask_f32: load_f32("apply_causal_mask_f32")?,
            interleave_heads_f32: load_f32("interleave_heads_f32")?,
            extract_head_f16: load_f16("extract_head_f16")?,
            apply_causal_mask_f16: load_f16("apply_causal_mask_f16")?,
            interleave_heads_f16: load_f16("interleave_heads_f16")?,
            // Batched decode attention
            batched_decode_attention_f32: load_f32("batched_decode_attention_f32")?,
            batched_decode_attention_f16: load_f16("batched_decode_attention_f16")?,
            // Paged attention (decode)
            paged_attention_f32: load_f32("paged_attention_f32")?,
            paged_attention_f16: load_f16("paged_attention_f16")?,
            scatter_kv_f32: load_f32("scatter_kv_f32")?,
            scatter_kv_f16: load_f16("scatter_kv_f16")?,
            argmax_f32: load_f32("argmax_f32")?,
            argmax_f16: load_f16("argmax_f16")?,
            sample_gumbel_f32: load_f32("sample_gumbel_f32")?,
            sample_gumbel_f16: load_f16("sample_gumbel_f16")?,
        };

        // Initial scratch capacity. Grown on demand.
        let initial_scratch_cap = 16;
        let block_tables = stream
            .alloc_zeros::<i32>(initial_scratch_cap)
            .map_err(|e| ForgeError::Cuda(format!("alloc paged_block_tables: {e}")))?;
        let kv_lens = stream
            .alloc_zeros::<i32>(initial_scratch_cap)
            .map_err(|e| ForgeError::Cuda(format!("alloc paged_kv_lens: {e}")))?;
        let emb_indices = stream
            .alloc_zeros::<u32>(initial_scratch_cap)
            .map_err(|e| ForgeError::Cuda(format!("alloc embedding_indices: {e}")))?;
        let rope_cos_buf = stream
            .alloc_zeros::<f32>(initial_scratch_cap)
            .map_err(|e| ForgeError::Cuda(format!("alloc rope_cos: {e}")))?;
        let rope_sin_buf = stream
            .alloc_zeros::<f32>(initial_scratch_cap)
            .map_err(|e| ForgeError::Cuda(format!("alloc rope_sin: {e}")))?;
        let slot_mapping_buf = stream
            .alloc_zeros::<i32>(initial_scratch_cap)
            .map_err(|e| ForgeError::Cuda(format!("alloc scatter_slot_mapping: {e}")))?;

        Ok(Self {
            ctx,
            stream,
            blas: Arc::new(blas),
            kernels: Arc::new(kernels),
            _module_f32: module_f32,
            _module_f16: module_f16,
            // Host mirrors are sized to match their device buffers (invariant:
            // `host.len() == buf.len()`), so `upload_*` never resizes the host
            // on the hot path — only `ensure_*` grows both together (bumping
            // `version`), keeping the captured H2D source pointer stable.
            paged_block_tables: Arc::new(Mutex::new(I32Scratch {
                buf: block_tables,
                version: 0,
                host: vec![0i32; initial_scratch_cap],
            })),
            paged_kv_lens: Arc::new(Mutex::new(I32Scratch {
                buf: kv_lens,
                version: 0,
                host: vec![0i32; initial_scratch_cap],
            })),
            embedding_indices: Arc::new(Mutex::new(U32Scratch {
                buf: emb_indices,
                version: 0,
                host: vec![0u32; initial_scratch_cap],
            })),
            rope_cos: Arc::new(Mutex::new(F32Scratch {
                buf: rope_cos_buf,
                version: 0,
                host: vec![0.0f32; initial_scratch_cap],
            })),
            rope_sin: Arc::new(Mutex::new(F32Scratch {
                buf: rope_sin_buf,
                version: 0,
                host: vec![0.0f32; initial_scratch_cap],
            })),
            scatter_slot_mapping: Arc::new(Mutex::new(I32Scratch {
                buf: slot_mapping_buf,
                version: 0,
                host: vec![0i32; initial_scratch_cap],
            })),
        })
    }

    /// Shared handle to the CUDA context. Used by callers that need to
    /// build secondary resources (e.g. `CudaGraphCache`) on the same
    /// context. The context is configured with `disable_event_tracking`
    /// (capture prereq); creating additional streams from it that are then
    /// captured is supported.
    pub fn ctx(&self) -> Arc<CudaContext> {
        self.ctx.clone()
    }

    /// Shared handle to the backend's primary stream. All ops in this
    /// backend launch on this stream — kernels, memcpys, cuBLAS. CUDA Graph
    /// capture must capture this same stream so the captured graph replays
    /// exactly the launch sequence the engine would have issued.
    pub fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    /// Current `(block_tables_version, kv_lens_version)` for the paged
    /// attention scratch buffers. Each value bumps every time the
    /// corresponding scratch reallocates (and its device pointer changes).
    ///
    /// CUDA Graph capture caches use this to detect when a captured graph
    /// referencing an old scratch pointer becomes invalid.
    pub fn paged_scratch_versions(&self) -> (u64, u64) {
        let bt = self
            .paged_block_tables
            .lock()
            .map(|g| g.version)
            .unwrap_or(0);
        let kv = self.paged_kv_lens.lock().map(|g| g.version).unwrap_or(0);
        (bt, kv)
    }

    /// Version of the embedding indices scratch (bumps on grow). See
    /// `paged_scratch_versions` for the consumer semantics.
    pub fn embedding_scratch_version(&self) -> u64 {
        self.embedding_indices
            .lock()
            .map(|g| g.version)
            .unwrap_or(0)
    }

    /// `(rope_cos_version, rope_sin_version)`.
    pub fn rope_scratch_versions(&self) -> (u64, u64) {
        let c = self.rope_cos.lock().map(|g| g.version).unwrap_or(0);
        let s = self.rope_sin.lock().map(|g| g.version).unwrap_or(0);
        (c, s)
    }

    /// Version of the scatter slot_mapping scratch (bumps on grow).
    pub fn scatter_slot_mapping_version(&self) -> u64 {
        self.scatter_slot_mapping
            .lock()
            .map(|g| g.version)
            .unwrap_or(0)
    }

    /// Upload `slot_mapping` into the persistent device scratch. This is the
    /// capture-UNSAFE half of the KV scatter (a `memcpy_htod`) and must run
    /// OUTSIDE any captured region — call it in the stage phase, before
    /// `graph.launch`. The captured `scatter_kv` then reads the staged buffer.
    ///
    /// Returns the staged length (= `slot_mapping.len()`), which the caller
    /// passes to `scatter_kv` as `n_rows`.
    pub fn stage_slot_mapping(&self, slot_mapping: &[i32]) -> Result<usize> {
        let mut scratch = self
            .scatter_slot_mapping
            .lock()
            .map_err(|_| ForgeError::Cuda("scatter_slot_mapping mutex poisoned".into()))?;
        self.ensure_i32_scratch(&mut scratch, slot_mapping.len().max(1))?;
        if !slot_mapping.is_empty() {
            self.stream
                .memcpy_htod(slot_mapping, &mut scratch.buf)
                .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        }
        Ok(slot_mapping.len())
    }

    /// Upload one decode step's per-step inputs (token ids, RoPE cos/sin,
    /// paged block tables, KV lengths) into their persistent device scratches
    /// (each via the host-mirror `upload_*` path, so the captured `memcpy_htod`
    /// nodes keep a stable host source pointer). Capture-UNSAFE — call OUTSIDE
    /// any captured region; the captured `compute_decode` then re-reads these
    /// scratches on replay.
    pub fn stage_decode_inputs_impl(
        &self,
        token_indices: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
        block_tables: &[i32],
        kv_lens: &[i32],
    ) -> Result<()> {
        {
            let mut s = self
                .embedding_indices
                .lock()
                .map_err(|_| ForgeError::Cuda("embedding_indices mutex poisoned".into()))?;
            self.upload_u32_scratch(&mut s, token_indices)?;
        }
        {
            let mut s = self
                .rope_cos
                .lock()
                .map_err(|_| ForgeError::Cuda("rope_cos mutex poisoned".into()))?;
            self.upload_f32_scratch(&mut s, rope_cos)?;
        }
        {
            let mut s = self
                .rope_sin
                .lock()
                .map_err(|_| ForgeError::Cuda("rope_sin mutex poisoned".into()))?;
            self.upload_f32_scratch(&mut s, rope_sin)?;
        }
        {
            let mut s = self
                .paged_block_tables
                .lock()
                .map_err(|_| ForgeError::Cuda("paged_block_tables mutex poisoned".into()))?;
            self.upload_i32_scratch(&mut s, block_tables)?;
        }
        {
            let mut s = self
                .paged_kv_lens
                .lock()
                .map_err(|_| ForgeError::Cuda("paged_kv_lens mutex poisoned".into()))?;
            self.upload_i32_scratch(&mut s, kv_lens)?;
        }
        Ok(())
    }

    /// Capture-safe KV scatter: writes the first `n_rows` rows of `src` into
    /// `pool` at the slots previously uploaded via `stage_slot_mapping`. Pure
    /// kernel launch (reads the device slot_mapping scratch) — no host upload,
    /// so it can be recorded inside a captured CUDA Graph.
    ///
    /// `pool` is rank-3 `[num_blocks, block_size, kv_dim]`; `src` is
    /// `[>= n_rows, kv_dim]`. Bounds are enforced defensively in-kernel
    /// (OOB/negative slots are skipped); the host should have validated during
    /// staging.
    pub fn scatter_kv(&self, pool: &mut CudaTensor, src: &CudaTensor, n_rows: usize) -> Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        if pool.dtype() != src.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "scatter_kv: pool/src dtype mismatch ({:?} vs {:?})",
                pool.dtype(),
                src.dtype()
            )));
        }
        let pool_shape = pool.shape().to_vec();
        if pool_shape.len() != 3 {
            return Err(ForgeError::InvalidArgument(format!(
                "scatter_kv: pool must be rank-3 [num_blocks, block_size, kv_dim], got {pool_shape:?}"
            )));
        }
        let block_size = pool_shape[1];
        let kv_dim = pool_shape[2];
        let total_slots = pool_shape[0] * block_size;
        let src_rows = src.shape().first().copied().unwrap_or(0);
        if src_rows < n_rows {
            return Err(ForgeError::InvalidArgument(format!(
                "scatter_kv: src has {src_rows} rows but n_rows={n_rows}"
            )));
        }

        let scratch = self
            .scatter_slot_mapping
            .lock()
            .map_err(|_| ForgeError::Cuda("scatter_slot_mapping mutex poisoned".into()))?;
        if scratch.buf.len() < n_rows {
            return Err(ForgeError::InvalidArgument(format!(
                "scatter_kv: slot_mapping scratch holds {} entries but n_rows={n_rows} (call stage_slot_mapping first)",
                scratch.buf.len()
            )));
        }
        let slot_dev = &scratch.buf;

        let n_rows_u32 = n_rows as u32;
        let kv_dim_u32 = kv_dim as u32;
        let total_slots_u32 = total_slots as u32;
        let launch_cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (256.min(kv_dim as u32).max(1), 1, 1),
            shared_mem_bytes: 0,
        };

        match pool.dtype() {
            DType::F32 => {
                let p = pool.f32_slice_mut()?;
                let s = src.f32_slice()?;
                let mut builder = self.stream.launch_builder(&self.kernels.scatter_kv_f32);
                builder.arg(p);
                builder.arg(s);
                builder.arg(slot_dev);
                builder.arg(&n_rows_u32);
                builder.arg(&kv_dim_u32);
                builder.arg(&total_slots_u32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F16 => {
                let p = pool.f16_slice_mut()?;
                let s = src.f16_slice()?;
                let mut builder = self.stream.launch_builder(&self.kernels.scatter_kv_f16);
                builder.arg(p);
                builder.arg(s);
                builder.arg(slot_dev);
                builder.arg(&n_rows_u32);
                builder.arg(&kv_dim_u32);
                builder.arg(&total_slots_u32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    /// Grow `scratch` if its capacity is below `needed`. Bumps `version` on
    /// every grow so CUDA Graph capture caches can detect pointer changes.
    fn ensure_i32_scratch(&self, scratch: &mut I32Scratch, needed: usize) -> Result<()> {
        if scratch.buf.len() < needed {
            // Geometric growth (1.5×) with a floor so we don't churn on small bumps.
            let new_cap = (needed.max(scratch.buf.len() * 3 / 2)).max(16);
            scratch.buf = self
                .stream
                .alloc_zeros::<i32>(new_cap)
                .map_err(|e| ForgeError::Cuda(format!("grow i32 scratch: {e}")))?;
            // Grow the HOST mirror in lockstep: a captured graph bakes the
            // mirror's pointer as its H2D source, so the mirror must only move
            // when `version` bumps (else a bucket-size change reallocs the host
            // Vec without invalidating captures → replay reads freed memory).
            scratch.host.resize(new_cap, 0);
            scratch.version = scratch.version.wrapping_add(1);
        }
        Ok(())
    }

    /// u32 sibling of `ensure_i32_scratch`.
    fn ensure_u32_scratch(&self, scratch: &mut U32Scratch, needed: usize) -> Result<()> {
        if scratch.buf.len() < needed {
            let new_cap = (needed.max(scratch.buf.len() * 3 / 2)).max(16);
            scratch.buf = self
                .stream
                .alloc_zeros::<u32>(new_cap)
                .map_err(|e| ForgeError::Cuda(format!("grow u32 scratch: {e}")))?;
            // Host mirror grows with the device buffer — see ensure_i32_scratch.
            scratch.host.resize(new_cap, 0);
            scratch.version = scratch.version.wrapping_add(1);
        }
        Ok(())
    }

    /// f32 sibling of `ensure_i32_scratch`.
    fn ensure_f32_scratch(&self, scratch: &mut F32Scratch, needed: usize) -> Result<()> {
        if scratch.buf.len() < needed {
            let new_cap = (needed.max(scratch.buf.len() * 3 / 2)).max(16);
            scratch.buf = self
                .stream
                .alloc_zeros::<f32>(new_cap)
                .map_err(|e| ForgeError::Cuda(format!("grow f32 scratch: {e}")))?;
            // Host mirror grows with the device buffer — see ensure_i32_scratch.
            scratch.host.resize(new_cap, 0.0);
            scratch.version = scratch.version.wrapping_add(1);
        }
        Ok(())
    }

    /// Upload `src` into `scratch`'s device buffer, sourcing the `memcpy_htod`
    /// from the scratch's persistent HOST mirror (not `src` directly). A
    /// captured CUDA Graph bakes the H2D source pointer; the mirror is a
    /// stable backend-owned buffer, so the baked pointer stays valid across
    /// capture + replay even after the caller's `src` Vec is dropped. See the
    /// `I32Scratch::host` doc. The mirror never shrinks → pointer is stable
    /// once warmed to steady-state size (a `grow` bumps `version`, which the
    /// capture cache uses to invalidate).
    fn upload_i32_scratch(&self, scratch: &mut I32Scratch, src: &[i32]) -> Result<()> {
        // `ensure_*` grows the host mirror with the device buffer (bumping
        // `version`), so the mirror is always large enough here — never resize
        // it on this path, which would move the pointer without a version bump.
        self.ensure_i32_scratch(scratch, src.len().max(1))?;
        scratch.host[..src.len()].copy_from_slice(src);
        self.stream
            .memcpy_htod(&scratch.host[..src.len()], &mut scratch.buf)
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    /// u32 sibling of `upload_i32_scratch`.
    fn upload_u32_scratch(&self, scratch: &mut U32Scratch, src: &[u32]) -> Result<()> {
        // See upload_i32_scratch: the mirror is pre-sized by `ensure_*`.
        self.ensure_u32_scratch(scratch, src.len().max(1))?;
        scratch.host[..src.len()].copy_from_slice(src);
        self.stream
            .memcpy_htod(&scratch.host[..src.len()], &mut scratch.buf)
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    /// f32 sibling of `upload_i32_scratch`.
    fn upload_f32_scratch(&self, scratch: &mut F32Scratch, src: &[f32]) -> Result<()> {
        // See upload_i32_scratch: the mirror is pre-sized by `ensure_*`.
        self.ensure_f32_scratch(scratch, src.len().max(1))?;
        scratch.host[..src.len()].copy_from_slice(src);
        self.stream
            .memcpy_htod(&scratch.host[..src.len()], &mut scratch.buf)
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    /// Inherent implementation of paged_attention_into. The `Backend` trait
    /// method `paged_attention_into` (in `impl Backend for CudaBackend`)
    /// delegates here. Keeping the implementation inherent lets unit tests
    /// call it directly with the CudaTensor concrete type.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_into_impl(
        &self,
        out: &mut CudaTensor,
        q: &CudaTensor,
        k_pool: &CudaTensor,
        v_pool: &CudaTensor,
        block_tables: &[i32],
        kv_lens: &[i32],
        max_blocks_per_seq: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let batch_size = kv_lens.len();
        if block_tables.len() != batch_size * max_blocks_per_seq {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_attention_into: block_tables.len()={} != batch_size={batch_size} * max_blocks_per_seq={max_blocks_per_seq}",
                block_tables.len()
            )));
        }
        let pool_shape = k_pool.shape();
        if pool_shape.len() != 3 {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_attention_into: k_pool must be rank-3, got {pool_shape:?}"
            )));
        }
        if v_pool.shape() != pool_shape {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_attention_into: k_pool/v_pool shape mismatch ({pool_shape:?} vs {:?})",
                v_pool.shape()
            )));
        }
        if num_kv_heads == 0 {
            return Err(ForgeError::InvalidArgument(
                "paged_attention_into: num_kv_heads must be > 0".into(),
            ));
        }
        let kv_dim_expected = num_kv_heads * head_dim;
        if pool_shape[2] != kv_dim_expected {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_attention_into: pool kv_dim {} != num_kv_heads {} * head_dim {} = {}",
                pool_shape[2], num_kv_heads, head_dim, kv_dim_expected
            )));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_attention_into: num_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}"
            )));
        }
        let q_shape = q.shape();
        if q_shape.len() != 2 || q_shape[0] != batch_size || q_shape[1] != num_heads * head_dim {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![batch_size, num_heads * head_dim],
                got: q_shape.to_vec(),
            });
        }
        let out_shape = out.shape();
        let expected_out = vec![batch_size, num_heads * head_dim];
        if out_shape != expected_out.as_slice() {
            return Err(ForgeError::ShapeMismatch {
                expected: expected_out,
                got: out_shape.to_vec(),
            });
        }
        if out.dtype() != q.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_attention_into: out dtype {:?} != q dtype {:?}",
                out.dtype(),
                q.dtype()
            )));
        }
        if kv_lens.iter().any(|&l| l < 0) {
            return Err(ForgeError::InvalidArgument(
                "paged_attention_into: kv_lens contains negative value".into(),
            ));
        }
        if batch_size == 0 {
            return Ok(()); // out is already correctly-shaped (empty)
        }
        if q.dtype() != k_pool.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_attention_into: q dtype {:?} != k_pool dtype {:?}",
                q.dtype(),
                k_pool.dtype()
            )));
        }

        let num_blocks = pool_shape[0];
        let block_size = pool_shape[1];

        // Per-seq bounds: enough block-table entries for kv_len, and every
        // entry the kernel will dereference (the first blocks_needed of each
        // row) must be a valid block id. Padding entries past blocks_needed
        // are not touched by the kernel.
        for (b, &kv_len_i32) in kv_lens.iter().enumerate() {
            let kv_len = kv_len_i32 as usize;
            let blocks_needed = kv_len.div_ceil(block_size);
            if blocks_needed > max_blocks_per_seq {
                return Err(ForgeError::InvalidArgument(format!(
                    "paged_attention_into: seq[{b}] kv_len={kv_len} needs {blocks_needed} blocks but max_blocks_per_seq={max_blocks_per_seq}"
                )));
            }
            let row_start = b * max_blocks_per_seq;
            for j in 0..blocks_needed {
                let id = block_tables[row_start + j];
                if id < 0 || (id as usize) >= num_blocks {
                    return Err(ForgeError::InvalidArgument(format!(
                        "paged_attention_into: seq[{b}] block_tables[{j}]={id} invalid (num_blocks={num_blocks})"
                    )));
                }
            }
        }

        // Upload i32 metadata into persistent device scratch (stable address,
        // version bumps on grow — see `paged_scratch_versions`).
        //
        // Both scratch mutexes are held through the kernel launch below.
        // Safe today because CudaBackend's contract (`see struct doc`) is
        // single-threaded engine access — no recursion, no contention. Lock
        // order is stable (block_tables → kv_lens) so even if that contract
        // changes there's no deadlock risk between paged_attention calls.
        let mut bt_scratch = self
            .paged_block_tables
            .lock()
            .map_err(|_| ForgeError::Cuda("paged_block_tables mutex poisoned".into()))?;
        self.upload_i32_scratch(&mut bt_scratch, block_tables)?;

        let mut kv_scratch = self
            .paged_kv_lens
            .lock()
            .map_err(|_| ForgeError::Cuda("paged_kv_lens mutex poisoned".into()))?;
        self.upload_i32_scratch(&mut kv_scratch, kv_lens)?;

        let block_tables_dev = &bt_scratch.buf;
        let kv_lens_dev = &kv_scratch.buf;

        let block_dim = next_power_of_2(128u32.min(head_dim as u32));
        // Shared mem: reduction scratch (block_dim floats) + output accumulator
        // (head_dim floats). Always f32 regardless of tensor dtype.
        let shared_mem = (block_dim + head_dim as u32) * 4;

        let num_heads_i32 = num_heads as i32;
        let num_kv_heads_i32 = num_kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = block_size as i32;
        let max_blocks_i32 = max_blocks_per_seq as i32;

        let launch_cfg = LaunchConfig {
            grid_dim: (batch_size as u32, num_heads as u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        match q.dtype() {
            DType::F32 => {
                let q_slice = q.f32_slice()?;
                let k_slice = k_pool.f32_slice()?;
                let v_slice = v_pool.f32_slice()?;
                let out_slice = out.f32_slice_mut()?;
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.paged_attention_f32);
                builder.arg(out_slice);
                builder.arg(q_slice);
                builder.arg(k_slice);
                builder.arg(v_slice);
                builder.arg(block_tables_dev);
                builder.arg(kv_lens_dev);
                builder.arg(&scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&head_dim_i32);
                builder.arg(&block_size_i32);
                builder.arg(&max_blocks_i32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F16 => {
                let q_slice = q.f16_slice()?;
                let k_slice = k_pool.f16_slice()?;
                let v_slice = v_pool.f16_slice()?;
                let out_slice = out.f16_slice_mut()?;
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.paged_attention_f16);
                builder.arg(out_slice);
                builder.arg(q_slice);
                builder.arg(k_slice);
                builder.arg(v_slice);
                builder.arg(block_tables_dev);
                builder.arg(kv_lens_dev);
                builder.arg(&scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&head_dim_i32);
                builder.arg(&block_size_i32);
                builder.arg(&max_blocks_i32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    /// Locate the CUDA toolkit include directory containing `cuda_fp16.h`.
    ///
    /// Search order: `$CUDA_HOME/include`, `$CUDA_PATH/include`, `/usr/local/cuda/include`.
    fn find_cuda_include() -> Result<String> {
        let candidates: Vec<std::path::PathBuf> = std::env::var("CUDA_HOME")
            .into_iter()
            .chain(std::env::var("CUDA_PATH"))
            .map(|p| std::path::PathBuf::from(p).join("include"))
            .chain(std::iter::once(std::path::PathBuf::from(
                "/usr/local/cuda/include",
            )))
            .collect();

        for path in &candidates {
            if path.join("cuda_fp16.h").exists() {
                return Ok(path.to_string_lossy().into_owned());
            }
        }

        Err(ForgeError::Cuda(format!(
            "cuda_fp16.h not found; searched: {}. Set CUDA_HOME to your CUDA toolkit.",
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }
}

/// Round up to the next power of 2 (minimum 32 for warp size).
fn next_power_of_2(n: u32) -> u32 {
    let n = n.max(32);
    1u32 << (32 - (n - 1).leading_zeros())
}

fn validate_shape(data_len: usize, shape: &[usize]) -> Result<()> {
    let expected: usize = shape.iter().product();
    if data_len != expected {
        return Err(ForgeError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![data_len],
        });
    }
    Ok(())
}

fn validate_same_shape(a: &CudaTensor, b: &CudaTensor) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(ForgeError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

impl CudaBackend {
    /// Cast an F16 tensor to F32 on the GPU, returning a host Vec<f32>.
    fn cast_f16_to_f32_host(&self, tensor: &CudaTensor) -> Result<Vec<f32>> {
        let n = tensor.len() as u32;
        let mut out = self
            .stream
            .alloc_zeros::<f32>(n as usize)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        let mut builder = self.stream.launch_builder(&self.kernels.cast_f16_to_f32);
        builder.arg(&mut out);
        builder.arg(tensor.f16_slice()?);
        builder.arg(&n);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n))
                .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        }

        self.stream
            .memcpy_dtov(&out)
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }
}

impl Backend for CudaBackend {
    type Tensor = CudaTensor;

    fn name(&self) -> &str {
        "cuda"
    }

    fn device_count(&self) -> usize {
        CudaContext::device_count().unwrap_or(0) as usize
    }

    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<CudaTensor> {
        let numel = CudaTensor::numel_from_shape(shape);
        match dtype {
            DType::F32 => {
                let data = self
                    .stream
                    .alloc_zeros::<f32>(numel)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(CudaTensor::f32_data(data, shape.to_vec()))
            }
            DType::F16 => {
                let data = self
                    .stream
                    .alloc_zeros::<half::f16>(numel)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(CudaTensor::f16_data(data, shape.to_vec()))
            }
            DType::BF16 => {
                let data = self
                    .stream
                    .alloc_zeros::<half::bf16>(numel)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(CudaTensor::bf16_data(data, shape.to_vec()))
            }
            _ => Err(ForgeError::UnsupportedDtype(dtype)),
        }
    }

    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<CudaTensor> {
        self.allocate(shape, dtype)
    }

    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<CudaTensor> {
        validate_shape(data.len(), shape)?;
        let slice = self
            .stream
            .memcpy_stod(data)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        Ok(CudaTensor::f32_data(slice, shape.to_vec()))
    }

    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<CudaTensor> {
        validate_shape(data.len(), shape)?;
        let slice = self
            .stream
            .memcpy_stod(data)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        Ok(CudaTensor::f16_data(slice, shape.to_vec()))
    }

    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<CudaTensor> {
        validate_shape(data.len(), shape)?;
        let slice = self
            .stream
            .memcpy_stod(data)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        Ok(CudaTensor::bf16_data(slice, shape.to_vec()))
    }

    fn copy_to_host_f32(&self, tensor: &CudaTensor) -> Result<Vec<f32>> {
        match tensor.dtype() {
            DType::F32 => self
                .stream
                .memcpy_dtov(tensor.f32_slice()?)
                .map_err(|e| ForgeError::Cuda(e.to_string())),
            DType::F16 => self.cast_f16_to_f32_host(tensor),
            DType::BF16 => {
                // No GPU cast kernel yet — download bf16 to host and convert.
                let bf16_host: Vec<half::bf16> = self
                    .stream
                    .memcpy_dtov(tensor.bf16_slice()?)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(bf16_host.iter().map(|v| v.to_f32()).collect())
            }
            other => Err(ForgeError::InvalidArgument(format!(
                "copy_to_host_f32 not supported for {:?}",
                other
            ))),
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    fn matmul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "matmul requires 2D tensors".into(),
            ));
        }
        let m = a_shape[0];
        let n = b_shape[1];
        if a.dtype() != b.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "matmul dtype mismatch: {:?} vs {:?}",
                a.dtype(),
                b.dtype()
            )));
        }
        // Allocate output, then delegate to matmul_into (which does the full
        // gemm logic). Keeps a single source of truth for the kernel.
        let mut out = self.allocate_zeros(&[m, n], a.dtype())?;
        self.matmul_into(&mut out, a, b)?;
        Ok(out)
    }

    /// In-place matmul into a caller-provided buffer.
    ///
    /// Same gemm logic as `matmul`, just doesn't allocate the output.
    /// Used by the engine's persistent-buffer / CUDA-Graph capture path
    /// so the captured kernel's output pointer is stable across replays.
    fn matmul_into(&self, out: &mut CudaTensor, a: &CudaTensor, b: &CudaTensor) -> Result<()> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "matmul_into: requires 2D tensors".into(),
            ));
        }
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        if b_shape[0] != k {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![k, n],
                got: b_shape.to_vec(),
            });
        }
        if a.dtype() != b.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "matmul_into: a/b dtype mismatch ({:?} vs {:?})",
                a.dtype(),
                b.dtype()
            )));
        }
        let expected_out = vec![m, n];
        if out.shape() != expected_out.as_slice() {
            return Err(ForgeError::ShapeMismatch {
                expected: expected_out,
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != a.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "matmul_into: out dtype {:?} != a dtype {:?}",
                out.dtype(),
                a.dtype()
            )));
        }

        // Row-major `A * B` in cuBLAS (which is column-major) is computed
        // as `B^T * A^T` from cuBLAS's POV: pass (b, a) and swap (m, n)
        // and (lda, ldb) accordingly. The result lands row-major in `out`
        // without any explicit transpose. This is load-bearing — touching
        // the swap or arg order will produce a silently transposed result.
        match a.dtype() {
            DType::F32 => {
                let a_slice = a.f32_slice()?;
                let b_slice = b.f32_slice()?;
                let c_slice = out.f32_slice_mut()?;
                unsafe {
                    self.blas
                        .gemm(
                            GemmConfig {
                                transa: cublasOperation_t::CUBLAS_OP_N,
                                transb: cublasOperation_t::CUBLAS_OP_N,
                                m: n as i32,
                                n: m as i32,
                                k: k as i32,
                                alpha: 1.0f32,
                                lda: n as i32,
                                ldb: k as i32,
                                beta: 0.0f32,
                                ldc: n as i32,
                            },
                            b_slice,
                            a_slice,
                            c_slice,
                        )
                        .map_err(|e| ForgeError::Cuda(format!("gemm f32: {e}")))?;
                }
                Ok(())
            }
            DType::F16 => {
                let a_slice = a.f16_slice()?;
                let b_slice = b.f16_slice()?;
                let c_slice = out.f16_slice_mut()?;
                unsafe {
                    self.blas
                        .gemm(
                            GemmConfig {
                                transa: cublasOperation_t::CUBLAS_OP_N,
                                transb: cublasOperation_t::CUBLAS_OP_N,
                                m: n as i32,
                                n: m as i32,
                                k: k as i32,
                                alpha: half::f16::from_f32(1.0),
                                lda: n as i32,
                                ldb: k as i32,
                                beta: half::f16::from_f32(0.0),
                                ldc: n as i32,
                            },
                            b_slice,
                            a_slice,
                            c_slice,
                        )
                        .map_err(|e| ForgeError::Cuda(format!("gemm f16: {e}")))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn add(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        validate_same_shape(a, b)?;
        let mut out = self.allocate_zeros(&a.shape, a.dtype())?;
        self.add_into(&mut out, a, b)?;
        Ok(out)
    }

    /// Broadcast bias add: `out[r,c] = x[r,c] + bias[c]`. x is `[rows, cols]`,
    /// bias is `[cols]`. Used for Qwen2 QKV projection bias.
    fn add_bias(&self, x: &CudaTensor, bias: &CudaTensor) -> Result<CudaTensor> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "add_bias: x must be 2D [rows, cols]".into(),
            ));
        }
        let rows = shape[0];
        let cols = shape[1];
        if bias.shape() != [cols] {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![cols],
                got: bias.shape().to_vec(),
            });
        }
        if x.dtype() != bias.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "add_bias: x dtype {:?} != bias dtype {:?}",
                x.dtype(),
                bias.dtype()
            )));
        }
        let rows_u = rows as u32;
        let cols_u = cols as u32;
        let n = (rows * cols) as u32;
        let mut out = self.allocate_zeros(shape, x.dtype())?;

        match x.dtype() {
            DType::F32 => {
                let x_s = x.f32_slice()?;
                let b_s = bias.f32_slice()?;
                let o = out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.add_bias_f32);
                builder.arg(o);
                builder.arg(x_s);
                builder.arg(b_s);
                builder.arg(&rows_u);
                builder.arg(&cols_u);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
            }
            DType::F16 => {
                let x_s = x.f16_slice()?;
                let b_s = bias.f16_slice()?;
                let o = out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.add_bias_f16);
                builder.arg(o);
                builder.arg(x_s);
                builder.arg(b_s);
                builder.arg(&rows_u);
                builder.arg(&cols_u);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
            }
            other => return Err(ForgeError::UnsupportedDtype(other)),
        }
        Ok(out)
    }

    /// In-place add into a caller-provided buffer. See `matmul_into`.
    fn add_into(&self, out: &mut CudaTensor, a: &CudaTensor, b: &CudaTensor) -> Result<()> {
        validate_same_shape(a, b)?;
        if out.shape() != a.shape() {
            return Err(ForgeError::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != a.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "add_into: out dtype {:?} != a dtype {:?}",
                out.dtype(),
                a.dtype()
            )));
        }
        let n = a.len() as u32;

        match a.dtype() {
            DType::F16 => {
                let a_slice = a.f16_slice()?;
                let b_slice = b.f16_slice()?;
                let out_slice = out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.add_f16);
                builder.arg(out_slice);
                builder.arg(a_slice);
                builder.arg(b_slice);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F32 => {
                let a_slice = a.f32_slice()?;
                let b_slice = b.f32_slice()?;
                let out_slice = out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.add_f32);
                builder.arg(out_slice);
                builder.arg(a_slice);
                builder.arg(b_slice);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn mul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        validate_same_shape(a, b)?;
        let n = a.len() as u32;

        match a.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.mul_f16);
                builder.arg(&mut out);
                builder.arg(a.f16_slice()?);
                builder.arg(b.f16_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, a.shape.clone()))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.mul_f32);
                builder.arg(&mut out);
                builder.arg(a.f32_slice()?);
                builder.arg(b.f32_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, a.shape.clone()))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn mul_scalar(&self, a: &CudaTensor, scalar: f32) -> Result<CudaTensor> {
        let n = a.len() as u32;

        match a.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.mul_scalar_f16);
                builder.arg(&mut out);
                builder.arg(a.f16_slice()?);
                builder.arg(&scalar);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, a.shape.clone()))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.mul_scalar_f32);
                builder.arg(&mut out);
                builder.arg(a.f32_slice()?);
                builder.arg(&scalar);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, a.shape.clone()))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn silu(&self, a: &CudaTensor) -> Result<CudaTensor> {
        let n = a.len() as u32;

        match a.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.silu_f16);
                builder.arg(&mut out);
                builder.arg(a.f16_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, a.shape.clone()))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.silu_f32);
                builder.arg(&mut out);
                builder.arg(a.f32_slice()?);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, a.shape.clone()))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn fused_silu_mul(&self, gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
        validate_same_shape(gate, up)?;
        let mut out = self.allocate_zeros(&gate.shape, gate.dtype())?;
        self.fused_silu_mul_into(&mut out, gate, up)?;
        Ok(out)
    }

    /// In-place fused SiLU(gate)*up. See `matmul_into` for the
    /// capture-stability contract.
    fn fused_silu_mul_into(
        &self,
        out: &mut CudaTensor,
        gate: &CudaTensor,
        up: &CudaTensor,
    ) -> Result<()> {
        validate_same_shape(gate, up)?;
        if out.shape() != gate.shape() {
            return Err(ForgeError::ShapeMismatch {
                expected: gate.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != gate.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "fused_silu_mul_into: out dtype {:?} != gate dtype {:?}",
                out.dtype(),
                gate.dtype()
            )));
        }
        let n = gate.len() as u32;
        match gate.dtype() {
            DType::F16 => {
                let g = gate.f16_slice()?;
                let u = up.f16_slice()?;
                let o = out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.fused_silu_mul_f16);
                builder.arg(o);
                builder.arg(g);
                builder.arg(u);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F32 => {
                let g = gate.f32_slice()?;
                let u = up.f32_slice()?;
                let o = out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.fused_silu_mul_f32);
                builder.arg(o);
                builder.arg(g);
                builder.arg(u);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn rms_norm(&self, x: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        let mut out = self.allocate_zeros(&x.shape, x.dtype())?;
        self.rms_norm_into(&mut out, x, weight, eps)?;
        Ok(out)
    }

    /// In-place RMS normalization. See `matmul_into` for the
    /// capture-stability contract.
    fn rms_norm_into(
        &self,
        out: &mut CudaTensor,
        x: &CudaTensor,
        weight: &CudaTensor,
        eps: f32,
    ) -> Result<()> {
        let shape = x.shape();
        let cols = *shape.last().unwrap();
        if weight.len() != cols {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![cols],
                got: weight.shape().to_vec(),
            });
        }
        if out.shape() != shape {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != x.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "rms_norm_into: out dtype {:?} != x dtype {:?}",
                out.dtype(),
                x.dtype()
            )));
        }
        let rows = x.len() / cols;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        let block_dim = next_power_of_2(256u32.min(cols as u32));
        // Shared memory uses f32 for both F16 and F32 paths (reduction in f32)
        let shared_mem = block_dim * 4;

        match x.dtype() {
            DType::F16 => {
                let x_s = x.f16_slice()?;
                let w_s = weight.f16_slice()?;
                let o = out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.rms_norm_f16);
                builder.arg(o);
                builder.arg(x_s);
                builder.arg(w_s);
                builder.arg(&eps);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F32 => {
                let x_s = x.f32_slice()?;
                let w_s = weight.f32_slice()?;
                let o = out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.rms_norm_f32);
                builder.arg(o);
                builder.arg(x_s);
                builder.arg(w_s);
                builder.arg(&eps);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn fused_residual_rms_norm(
        &self,
        x: &CudaTensor,
        residual: &CudaTensor,
        weight: &CudaTensor,
        eps: f32,
    ) -> Result<(CudaTensor, CudaTensor)> {
        validate_same_shape(x, residual)?;
        let mut normed = self.allocate_zeros(&x.shape, x.dtype())?;
        let mut residual_out = self.allocate_zeros(&x.shape, x.dtype())?;
        self.fused_residual_rms_norm_into(
            &mut normed,
            &mut residual_out,
            x,
            residual,
            weight,
            eps,
        )?;
        Ok((normed, residual_out))
    }

    /// In-place fused residual + RMS norm. See `matmul_into` for the
    /// capture-stability contract.
    fn fused_residual_rms_norm_into(
        &self,
        normed_out: &mut CudaTensor,
        residual_out: &mut CudaTensor,
        x: &CudaTensor,
        residual: &CudaTensor,
        weight: &CudaTensor,
        eps: f32,
    ) -> Result<()> {
        validate_same_shape(x, residual)?;
        let shape = x.shape();
        let cols = *shape.last().unwrap();
        if weight.len() != cols {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![cols],
                got: weight.shape().to_vec(),
            });
        }
        if normed_out.shape() != shape || residual_out.shape() != shape {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: normed_out.shape().to_vec(),
            });
        }
        if normed_out.dtype() != x.dtype() || residual_out.dtype() != x.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "fused_residual_rms_norm_into: out dtypes (norm={:?}, residual={:?}) != x dtype {:?}",
                normed_out.dtype(),
                residual_out.dtype(),
                x.dtype()
            )));
        }
        let rows = x.len() / cols;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        let block_dim = next_power_of_2(256u32.min(cols as u32));
        let shared_mem = block_dim * 4;

        match x.dtype() {
            DType::F16 => {
                let x_s = x.f16_slice()?;
                let r_s = residual.f16_slice()?;
                let w_s = weight.f16_slice()?;
                let n_o = normed_out.f16_slice_mut()?;
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.fused_residual_rms_norm_f16);
                builder.arg(n_o);
                // Re-borrow residual_out's slice — but we still hold n_o mut borrow on normed_out;
                // they're distinct tensors so separate borrows are fine.
                let r_o = residual_out.f16_slice_mut()?;
                builder.arg(r_o);
                builder.arg(x_s);
                builder.arg(r_s);
                builder.arg(w_s);
                builder.arg(&eps);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F32 => {
                let x_s = x.f32_slice()?;
                let r_s = residual.f32_slice()?;
                let w_s = weight.f32_slice()?;
                let n_o = normed_out.f32_slice_mut()?;
                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.fused_residual_rms_norm_f32);
                builder.arg(n_o);
                let r_o = residual_out.f32_slice_mut()?;
                builder.arg(r_o);
                builder.arg(x_s);
                builder.arg(r_s);
                builder.arg(w_s);
                builder.arg(&eps);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn rope(
        &self,
        x: &CudaTensor,
        freqs_cos: &CudaTensor,
        freqs_sin: &CudaTensor,
    ) -> Result<CudaTensor> {
        let mut out = self.allocate_zeros(&x.shape, x.dtype())?;
        self.rope_into(&mut out, x, freqs_cos, freqs_sin)?;
        Ok(out)
    }

    /// In-place RoPE with host cos/sin — overrides the default trait impl
    /// to upload through the persistent `rope_cos`/`rope_sin` scratches
    /// (stable device pointers across replays). See `paged_attention_into`'s
    /// `paged_block_tables` scratch for the same pattern.
    ///
    /// Both scratch locks are held through the kernel launch. Safe under
    /// the single-threaded engine contract (see CudaBackend struct doc);
    /// lock order is stable (cos → sin → kernel) so even if the contract
    /// loosens there's no deadlock between rope_with_host_freqs_into calls.
    fn rope_with_host_freqs_into(
        &self,
        out: &mut CudaTensor,
        x: &CudaTensor,
        cos_host: &[f32],
        sin_host: &[f32],
    ) -> Result<()> {
        // Reuse rope_into's full validation by computing the expected shapes here.
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(ForgeError::InvalidArgument(
                "rope_with_host_freqs_into: x must be rank-4".into(),
            ));
        }
        let seq_len = shape[1];
        let head_dim = shape[3];
        if head_dim % 2 != 0 {
            return Err(ForgeError::InvalidArgument(
                "rope_with_host_freqs_into: head_dim must be even".into(),
            ));
        }
        let half = head_dim / 2;
        let expected = seq_len * half;
        if cos_host.len() != expected || sin_host.len() != expected {
            return Err(ForgeError::InvalidArgument(format!(
                "rope_with_host_freqs_into: cos/sin host slices must be {expected} elements (got cos={}, sin={})",
                cos_host.len(),
                sin_host.len()
            )));
        }

        // Cross-check output buffer matches x. (rope_into would do this too,
        // but we bypass it to keep the scratch borrows live across the
        // kernel launch — CudaSlice::clone is device-to-device copy in
        // cudarc 0.17.8, NOT Arc-share, so wrapping the scratch in a fresh
        // CudaTensor would defeat the whole persistent-scratch point.)
        if out.shape() != shape {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != x.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "rope_with_host_freqs_into: out dtype {:?} != x dtype {:?}",
                out.dtype(),
                x.dtype()
            )));
        }

        // Stage cos/sin into persistent scratch.
        let mut cos_scratch = self
            .rope_cos
            .lock()
            .map_err(|_| ForgeError::Cuda("rope_cos mutex poisoned".into()))?;
        self.upload_f32_scratch(&mut cos_scratch, cos_host)?;

        let mut sin_scratch = self
            .rope_sin
            .lock()
            .map_err(|_| ForgeError::Cuda("rope_sin mutex poisoned".into()))?;
        self.upload_f32_scratch(&mut sin_scratch, sin_host)?;

        let cos_dev = &cos_scratch.buf;
        let sin_dev = &sin_scratch.buf;

        let batch = shape[0] as u32;
        let seq_len_u = shape[1] as u32;
        let num_heads = shape[2] as u32;
        let head_dim_u = shape[3] as u32;
        let total = batch * seq_len_u * num_heads * (head_dim_u / 2);

        match x.dtype() {
            DType::F16 => {
                let x_s = x.f16_slice()?;
                let o = out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.rope_f16);
                builder.arg(o);
                builder.arg(x_s);
                builder.arg(cos_dev);
                builder.arg(sin_dev);
                builder.arg(&batch);
                builder.arg(&seq_len_u);
                builder.arg(&num_heads);
                builder.arg(&head_dim_u);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F32 => {
                let x_s = x.f32_slice()?;
                let o = out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.rope_f32);
                builder.arg(o);
                builder.arg(x_s);
                builder.arg(cos_dev);
                builder.arg(sin_dev);
                builder.arg(&batch);
                builder.arg(&seq_len_u);
                builder.arg(&num_heads);
                builder.arg(&head_dim_u);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    /// In-place RoPE kernel. See `matmul_into` for the capture-stability
    /// contract.
    fn rope_into(
        &self,
        out: &mut CudaTensor,
        x: &CudaTensor,
        freqs_cos: &CudaTensor,
        freqs_sin: &CudaTensor,
    ) -> Result<()> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(ForgeError::InvalidArgument(
                "rope_into expects 4D tensor [batch, seq_len, heads, head_dim]".into(),
            ));
        }
        let batch = shape[0] as u32;
        let seq_len = shape[1] as u32;
        let num_heads = shape[2] as u32;
        let head_dim = shape[3] as u32;
        if head_dim % 2 != 0 {
            return Err(ForgeError::InvalidArgument(
                "rope_into requires even head_dim".into(),
            ));
        }
        let half_dim = head_dim / 2;
        let expected_freq_len = (seq_len * half_dim) as usize;
        if freqs_cos.len() < expected_freq_len || freqs_sin.len() < expected_freq_len {
            return Err(ForgeError::InvalidArgument(format!(
                "rope_into: freq tensors need at least {} elements, got cos={} sin={}",
                expected_freq_len,
                freqs_cos.len(),
                freqs_sin.len()
            )));
        }
        if out.shape() != shape {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != x.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "rope_into: out dtype {:?} != x dtype {:?}",
                out.dtype(),
                x.dtype()
            )));
        }
        let total = batch * seq_len * num_heads * half_dim;

        match x.dtype() {
            DType::F16 => {
                let x_s = x.f16_slice()?;
                let c_s = freqs_cos.f32_slice()?;
                let s_s = freqs_sin.f32_slice()?;
                let o = out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.rope_f16);
                builder.arg(o);
                builder.arg(x_s);
                builder.arg(c_s);
                builder.arg(s_s);
                builder.arg(&batch);
                builder.arg(&seq_len);
                builder.arg(&num_heads);
                builder.arg(&head_dim);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F32 => {
                let x_s = x.f32_slice()?;
                let c_s = freqs_cos.f32_slice()?;
                let s_s = freqs_sin.f32_slice()?;
                let o = out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.rope_f32);
                builder.arg(o);
                builder.arg(x_s);
                builder.arg(c_s);
                builder.arg(s_s);
                builder.arg(&batch);
                builder.arg(&seq_len);
                builder.arg(&num_heads);
                builder.arg(&head_dim);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(total))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn softmax(&self, x: &CudaTensor, dim: i32) -> Result<CudaTensor> {
        let shape = x.shape();
        let ndim = shape.len() as i32;
        let normalized_dim = if dim < 0 { ndim + dim } else { dim };
        if normalized_dim != ndim - 1 {
            return Err(ForgeError::InvalidArgument(format!(
                "softmax only supports last dimension (got dim={dim}, ndim={ndim})"
            )));
        }

        let cols = *shape.last().unwrap();
        let rows = x.len() / cols;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        let block_dim = next_power_of_2(256u32.min(cols as u32));
        let shared_mem = block_dim * 4;

        match x.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.softmax_f16);
                builder.arg(&mut out);
                builder.arg(x.f16_slice()?);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, shape.to_vec()))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.softmax_f32);
                builder.arg(&mut out);
                builder.arg(x.f32_slice()?);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, shape.to_vec()))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn embedding(&self, weight: &CudaTensor, indices: &[u32]) -> Result<CudaTensor> {
        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "embedding weight must be 2D [vocab_size, embedding_dim]".into(),
            ));
        }
        let embedding_dim = weight_shape[1];
        let mut out = self.allocate_zeros(&[indices.len(), embedding_dim], weight.dtype())?;
        self.embedding_into(&mut out, weight, indices)?;
        Ok(out)
    }

    /// In-place embedding lookup. See `matmul_into` for the
    /// capture-stability contract.
    fn embedding_into(
        &self,
        out: &mut CudaTensor,
        weight: &CudaTensor,
        indices: &[u32],
    ) -> Result<()> {
        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "embedding_into: weight must be 2D [vocab_size, embedding_dim]".into(),
            ));
        }
        let vocab_size = weight_shape[0];
        let embedding_dim = weight_shape[1];
        let num_indices = indices.len();

        let expected_out = vec![num_indices, embedding_dim];
        if out.shape() != expected_out.as_slice() {
            return Err(ForgeError::ShapeMismatch {
                expected: expected_out,
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != weight.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "embedding_into: out dtype {:?} != weight dtype {:?}",
                out.dtype(),
                weight.dtype()
            )));
        }

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(ForgeError::InvalidArgument(format!(
                    "embedding_into: index {idx} out of range (vocab_size={vocab_size})"
                )));
            }
        }

        let num_indices_u32 = num_indices as u32;
        let embedding_dim_u32 = embedding_dim as u32;
        let vocab_size_u32 = vocab_size as u32;

        // Upload indices into the persistent scratch — stable device pointer
        // across calls (until the scratch grows; see embedding_scratch_version).
        // Replaces the previous per-call `memcpy_stod` (fresh alloc).
        //
        // The MutexGuard is held through the kernel launch below — same
        // pattern as paged_attention_into (see CudaBackend struct doc for
        // the single-threaded-engine contract that makes this safe).
        let mut indices_scratch = self
            .embedding_indices
            .lock()
            .map_err(|_| ForgeError::Cuda("embedding_indices mutex poisoned".into()))?;
        self.upload_u32_scratch(&mut indices_scratch, indices)?;
        let indices_dev = &indices_scratch.buf;

        let launch_cfg = LaunchConfig {
            grid_dim: (num_indices as u32, 1, 1),
            block_dim: (256.min(embedding_dim as u32), 1, 1),
            shared_mem_bytes: 0,
        };

        match weight.dtype() {
            DType::F16 => {
                let w = weight.f16_slice()?;
                let o = out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.embedding_f16);
                builder.arg(o);
                builder.arg(w);
                builder.arg(indices_dev);
                builder.arg(&num_indices_u32);
                builder.arg(&embedding_dim_u32);
                builder.arg(&vocab_size_u32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F32 => {
                let w = weight.f32_slice()?;
                let o = out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.embedding_f32);
                builder.arg(o);
                builder.arg(w);
                builder.arg(indices_dev);
                builder.arg(&num_indices_u32);
                builder.arg(&embedding_dim_u32);
                builder.arg(&vocab_size_u32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    /// Note: cudarc 0.17.8's `CudaSlice::clone` is a device-to-device
    /// copy, so this reshape allocates + copies (NOT a zero-copy view).
    /// For capture-stable persistent buffers, use `reshape_into` instead.
    fn reshape(&self, x: &CudaTensor, shape: &[usize]) -> Result<CudaTensor> {
        let numel: usize = shape.iter().product();
        if numel != x.len() {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: x.shape.clone(),
            });
        }
        Ok(CudaTensor {
            data: x.data.clone(),
            shape: shape.to_vec(),
            dtype: x.dtype,
        })
    }

    /// In-place reshape via `memcpy_dtod` into the caller-provided buffer.
    /// Output device pointer is stable across calls — capture-safe.
    fn reshape_into(&self, out: &mut CudaTensor, x: &CudaTensor, shape: &[usize]) -> Result<()> {
        let want_numel: usize = shape.iter().product();
        if want_numel != x.len() {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: x.shape.clone(),
            });
        }
        if out.shape() != shape {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != x.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "reshape_into: out dtype {:?} != x dtype {:?}",
                out.dtype(),
                x.dtype()
            )));
        }
        match x.dtype() {
            DType::F32 => {
                let src = x.f32_slice()?;
                let dst = out.f32_slice_mut()?;
                self.stream
                    .memcpy_dtod(src, dst)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
            }
            DType::F16 => {
                let src = x.f16_slice()?;
                let dst = out.f16_slice_mut()?;
                self.stream
                    .memcpy_dtod(src, dst)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
            }
            DType::BF16 => {
                let src = x.bf16_slice()?;
                let dst = out.bf16_slice_mut()?;
                self.stream
                    .memcpy_dtod(src, dst)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
            }
            other => return Err(ForgeError::UnsupportedDtype(other)),
        }
        Ok(())
    }

    fn transpose(&self, x: &CudaTensor, dim0: usize, dim1: usize) -> Result<CudaTensor> {
        let shape = x.shape();
        if shape.len() != 2 || !((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0)) {
            return Err(ForgeError::InvalidArgument(
                "transpose currently only supports 2D tensors with dims (0,1)".into(),
            ));
        }
        let rows = shape[0];
        let cols = shape[1];
        let n = (rows * cols) as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        match x.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.transpose_f16);
                builder.arg(&mut out);
                builder.arg(x.f16_slice()?);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, vec![cols, rows]))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(rows * cols)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.transpose_f32);
                builder.arg(&mut out);
                builder.arg(x.f32_slice()?);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, vec![cols, rows]))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn cat(&self, tensors: &[&CudaTensor], dim: usize) -> Result<CudaTensor> {
        if tensors.is_empty() {
            return Err(ForgeError::InvalidArgument("empty tensor list".into()));
        }
        if dim != 0 {
            return Err(ForgeError::InvalidArgument(
                "cat currently only supports dim=0".into(),
            ));
        }

        let ndim = tensors[0].shape().len();
        let inner_size: usize = if ndim > 1 {
            tensors[0].shape()[1..].iter().product()
        } else {
            1
        };

        for t in tensors.iter().skip(1) {
            if t.shape().len() != ndim {
                return Err(ForgeError::ShapeMismatch {
                    expected: tensors[0].shape().to_vec(),
                    got: t.shape().to_vec(),
                });
            }
            for d in 1..ndim {
                if t.shape()[d] != tensors[0].shape()[d] {
                    return Err(ForgeError::ShapeMismatch {
                        expected: tensors[0].shape().to_vec(),
                        got: t.shape().to_vec(),
                    });
                }
            }
        }

        let mut total_first_dim = 0usize;
        for t in tensors {
            total_first_dim += t.shape()[0];
        }
        let total_elems = total_first_dim * inner_size;

        let mut out_shape = tensors[0].shape().to_vec();
        out_shape[0] = total_first_dim;

        match tensors[0].dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(total_elems)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut offset = 0usize;
                for t in tensors {
                    let len = t.len();
                    let src = t.f16_slice()?;
                    self.stream
                        .memcpy_dtod(src, &mut out.slice_mut(offset..offset + len))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    offset += len;
                }

                Ok(CudaTensor::f16_data(out, out_shape))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(total_elems)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut offset = 0usize;
                for t in tensors {
                    let len = t.len();
                    let src = t.f32_slice()?;
                    self.stream
                        .memcpy_dtod(src, &mut out.slice_mut(offset..offset + len))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    offset += len;
                }

                Ok(CudaTensor::f32_data(out, out_shape))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn cast(&self, x: &CudaTensor, dtype: DType) -> Result<CudaTensor> {
        if x.dtype() == dtype {
            return Ok(x.clone());
        }
        let mut out = self.allocate_zeros(&x.shape, dtype)?;
        self.cast_into(&mut out, x)?;
        Ok(out)
    }

    /// In-place dtype cast. See `matmul_into` for the capture-stability
    /// contract. If `out.dtype() == x.dtype()`, this is a device-to-device
    /// memcpy (still inside `out`'s existing buffer).
    fn cast_into(&self, out: &mut CudaTensor, x: &CudaTensor) -> Result<()> {
        if out.shape() != x.shape() {
            return Err(ForgeError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        let n = x.len() as u32;
        match (x.dtype(), out.dtype()) {
            (a, b) if a == b => {
                // Same dtype — d2d copy into out's existing buffer.
                match a {
                    DType::F32 => {
                        let src = x.f32_slice()?;
                        let dst = out.f32_slice_mut()?;
                        self.stream
                            .memcpy_dtod(src, dst)
                            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    }
                    DType::F16 => {
                        let src = x.f16_slice()?;
                        let dst = out.f16_slice_mut()?;
                        self.stream
                            .memcpy_dtod(src, dst)
                            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    }
                    other => return Err(ForgeError::UnsupportedDtype(other)),
                }
                Ok(())
            }
            (DType::F16, DType::F32) => {
                let src = x.f16_slice()?;
                let dst = out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.cast_f16_to_f32);
                builder.arg(dst);
                builder.arg(src);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            (DType::F32, DType::F16) => {
                let src = x.f32_slice()?;
                let dst = out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.cast_f32_to_f16);
                builder.arg(dst);
                builder.arg(src);
                builder.arg(&n);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            (from, to) => Err(ForgeError::InvalidArgument(format!(
                "cast_into from {:?} to {:?} not supported",
                from, to
            ))),
        }
    }

    /// Device-to-device write into the paged pool.
    ///
    /// Coalesces consecutive slot mappings into a single `memcpy_dtod` per run.
    /// For decode (q_len=1) this is one call per layer; for contiguous prefill
    /// (slot_mapping = [base, base+1, ...]) it collapses to one call regardless
    /// of token count.
    ///
    /// The pool tensor is mutated in place — its device pointer does not change.
    /// This is the property CUDA Graph capture relies on.
    fn paged_write_kv(
        &self,
        pool: &mut CudaTensor,
        src: &CudaTensor,
        slot_mapping: &[i32],
    ) -> Result<()> {
        if slot_mapping.is_empty() {
            return Ok(());
        }
        if pool.dtype() != src.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_write_kv: pool/src dtype mismatch ({:?} vs {:?})",
                pool.dtype(),
                src.dtype()
            )));
        }
        let pool_shape = pool.shape().to_vec();
        if pool_shape.len() != 3 {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_write_kv: pool must be rank-3 [num_blocks, block_size, kv_dim], got {pool_shape:?}"
            )));
        }
        let block_size = pool_shape[1];
        let kv_dim = pool_shape[2];
        let total_capacity_slots = pool_shape[0] * block_size;

        for &slot in slot_mapping {
            if slot < 0 || (slot as usize) >= total_capacity_slots {
                return Err(ForgeError::InvalidArgument(format!(
                    "paged_write_kv: slot {slot} out of bounds (capacity {total_capacity_slots})"
                )));
            }
        }

        let src_rows = src.shape().first().copied().unwrap_or(0);
        if src_rows < slot_mapping.len() {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_write_kv: src has {src_rows} rows but slot_mapping has {} entries",
                slot_mapping.len()
            )));
        }

        // Coalesce consecutive slots into a single memcpy per run. Shared by
        // F32 and F16 paths — only the slice accessor and element type differ.
        macro_rules! run_dtod {
            ($get_src:ident, $get_dst:ident) => {{
                let src_slice = src.$get_src()?;
                let pool_slice = pool.$get_dst()?;
                let mut run_start = 0usize;
                while run_start < slot_mapping.len() {
                    let mut run_end = run_start + 1;
                    while run_end < slot_mapping.len()
                        && slot_mapping[run_end] == slot_mapping[run_end - 1] + 1
                    {
                        run_end += 1;
                    }
                    let run_len = run_end - run_start;
                    let dst_off = (slot_mapping[run_start] as usize) * kv_dim;
                    let src_off = run_start * kv_dim;
                    let elems = run_len * kv_dim;
                    self.stream
                        .memcpy_dtod(
                            &src_slice.slice(src_off..src_off + elems),
                            &mut pool_slice.slice_mut(dst_off..dst_off + elems),
                        )
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    run_start = run_end;
                }
            }};
        }

        match pool.dtype() {
            DType::F32 => run_dtod!(f32_slice, f32_slice_mut),
            DType::F16 => run_dtod!(f16_slice, f16_slice_mut),
            DType::BF16 => run_dtod!(bf16_slice, bf16_slice_mut),
            other => {
                return Err(ForgeError::UnsupportedDtype(other));
            }
        }
        Ok(())
    }

    /// Device-to-device gather from the paged pool.
    ///
    /// One `memcpy_dtod` per (possibly partial) block. Output tensor is freshly
    /// allocated; the pool tensor is read-only.
    fn paged_gather_kv(
        &self,
        pool: &CudaTensor,
        block_ids: &[usize],
        total_tokens: usize,
    ) -> Result<CudaTensor> {
        let pool_shape = pool.shape();
        if pool_shape.len() != 3 {
            return Err(ForgeError::InvalidArgument(format!(
                "paged_gather_kv: pool must be rank-3, got {pool_shape:?}"
            )));
        }
        let num_blocks = pool_shape[0];
        let block_size = pool_shape[1];
        let kv_dim = pool_shape[2];
        let out_shape = vec![total_tokens, kv_dim];

        // Bounds-check all block_ids first, regardless of dtype.
        for &block_id in block_ids {
            if block_id >= num_blocks {
                return Err(ForgeError::InvalidArgument(format!(
                    "paged_gather_kv: block_id {block_id} out of bounds ({num_blocks})"
                )));
            }
        }

        macro_rules! gather_dtod {
            ($T:ty, $get_pool:ident, $ctor:ident) => {{
                let pool_slice = pool.$get_pool()?;
                let mut out = self
                    .stream
                    .alloc_zeros::<$T>(total_tokens * kv_dim)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                let mut remaining = total_tokens;
                let mut out_off = 0usize;
                for &block_id in block_ids {
                    if remaining == 0 {
                        break;
                    }
                    let fill = remaining.min(block_size);
                    let src_off = block_id * block_size * kv_dim;
                    let elems = fill * kv_dim;
                    self.stream
                        .memcpy_dtod(
                            &pool_slice.slice(src_off..src_off + elems),
                            &mut out.slice_mut(out_off..out_off + elems),
                        )
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    out_off += elems;
                    remaining -= fill;
                }
                Ok(CudaTensor::$ctor(out, out_shape))
            }};
        }

        match pool.dtype() {
            DType::F32 => gather_dtod!(f32, f32_slice, f32_data),
            DType::F16 => gather_dtod!(half::f16, f16_slice, f16_data),
            DType::BF16 => gather_dtod!(half::bf16, bf16_slice, bf16_data),
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    /// Paged attention decode kernel — allocating variant.
    ///
    /// Convenience wrapper around [`Self::paged_attention_into`] that
    /// allocates a fresh output tensor of shape `[batch, num_heads * head_dim]`.
    /// For CUDA-Graph capture, prefer `paged_attention_into` with a
    /// pre-allocated, persistent output buffer so the captured graph
    /// references a stable device pointer.
    fn paged_attention(
        &self,
        q: &CudaTensor,
        k_pool: &CudaTensor,
        v_pool: &CudaTensor,
        block_tables: &[i32],
        kv_lens: &[i32],
        max_blocks_per_seq: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<CudaTensor> {
        let batch_size = kv_lens.len();
        if batch_size == 0 {
            // Match the trait default impl semantics for empty batch.
            return self.allocate(&[0, num_heads * head_dim], q.dtype());
        }
        let mut out = self.allocate_zeros(&[batch_size, num_heads * head_dim], q.dtype())?;
        self.paged_attention_into(
            &mut out,
            q,
            k_pool,
            v_pool,
            block_tables,
            kv_lens,
            max_blocks_per_seq,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )?;
        Ok(out)
    }

    /// Trait override — delegates to the inherent `paged_attention_into_impl`
    /// which holds the real body. Splitting this way lets the trait method
    /// (callable via generic `B: Backend`) share an implementation with
    /// any direct CudaBackend-typed callers (e.g. unit tests).
    fn paged_attention_into(
        &self,
        out: &mut CudaTensor,
        q: &CudaTensor,
        k_pool: &CudaTensor,
        v_pool: &CudaTensor,
        block_tables: &[i32],
        kv_lens: &[i32],
        max_blocks_per_seq: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        self.paged_attention_into_impl(
            out,
            q,
            k_pool,
            v_pool,
            block_tables,
            kv_lens,
            max_blocks_per_seq,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
    }

    fn split_qkv(
        &self,
        qkv: &CudaTensor,
        q_size: usize,
        kv_size: usize,
    ) -> Result<(CudaTensor, CudaTensor, CudaTensor)> {
        let shape = qkv.shape();
        if shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "split_qkv requires 2D tensor".into(),
            ));
        }
        let rows = shape[0];
        let mut q_out = self.allocate_zeros(&[rows, q_size], qkv.dtype())?;
        let mut k_out = self.allocate_zeros(&[rows, kv_size], qkv.dtype())?;
        let mut v_out = self.allocate_zeros(&[rows, kv_size], qkv.dtype())?;
        self.split_qkv_into(&mut q_out, &mut k_out, &mut v_out, qkv, q_size, kv_size)?;
        Ok((q_out, k_out, v_out))
    }

    /// In-place QKV split. See `matmul_into` for the capture-stability contract.
    fn split_qkv_into(
        &self,
        q_out: &mut CudaTensor,
        k_out: &mut CudaTensor,
        v_out: &mut CudaTensor,
        qkv: &CudaTensor,
        q_size: usize,
        kv_size: usize,
    ) -> Result<()> {
        let shape = qkv.shape();
        if shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "split_qkv_into: qkv must be 2D".into(),
            ));
        }
        let rows = shape[0];
        let total_cols = q_size + 2 * kv_size;
        if shape[1] != total_cols {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![rows, total_cols],
                got: shape.to_vec(),
            });
        }
        for (out, want) in [
            (q_out.shape(), q_size),
            (k_out.shape(), kv_size),
            (v_out.shape(), kv_size),
        ] {
            if out != [rows, want] {
                return Err(ForgeError::ShapeMismatch {
                    expected: vec![rows, want],
                    got: out.to_vec(),
                });
            }
        }
        if q_out.dtype() != qkv.dtype()
            || k_out.dtype() != qkv.dtype()
            || v_out.dtype() != qkv.dtype()
        {
            return Err(ForgeError::InvalidArgument(format!(
                "split_qkv_into: out dtypes ({:?}, {:?}, {:?}) != qkv dtype {:?}",
                q_out.dtype(),
                k_out.dtype(),
                v_out.dtype(),
                qkv.dtype()
            )));
        }

        let rows_u32 = rows as u32;
        let q_cols_u32 = q_size as u32;
        let kv_cols_u32 = kv_size as u32;
        let block_dim = next_power_of_2((256u32).min(q_size.max(kv_size) as u32));

        let launch_cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        match qkv.dtype() {
            DType::F16 => {
                let qkv_s = qkv.f16_slice()?;
                let q_s = q_out.f16_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.split_qkv_f16);
                builder.arg(q_s);
                let k_s = k_out.f16_slice_mut()?;
                builder.arg(k_s);
                let v_s = v_out.f16_slice_mut()?;
                builder.arg(v_s);
                builder.arg(qkv_s);
                builder.arg(&rows_u32);
                builder.arg(&q_cols_u32);
                builder.arg(&kv_cols_u32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            DType::F32 => {
                let qkv_s = qkv.f32_slice()?;
                let q_s = q_out.f32_slice_mut()?;
                let mut builder = self.stream.launch_builder(&self.kernels.split_qkv_f32);
                builder.arg(q_s);
                let k_s = k_out.f32_slice_mut()?;
                builder.arg(k_s);
                let v_s = v_out.f32_slice_mut()?;
                builder.arg(v_s);
                builder.arg(qkv_s);
                builder.arg(&rows_u32);
                builder.arg(&q_cols_u32);
                builder.arg(&kv_cols_u32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn slice_rows(
        &self,
        tensor: &CudaTensor,
        start_row: usize,
        num_rows: usize,
    ) -> Result<CudaTensor> {
        let mut out_shape = tensor.shape().to_vec();
        out_shape[0] = num_rows;
        let mut out = self.allocate_zeros(&out_shape, tensor.dtype())?;
        self.slice_rows_into(&mut out, tensor, start_row, num_rows)?;
        Ok(out)
    }

    /// In-place row slice via device-to-device memcpy. See `matmul_into`.
    fn slice_rows_into(
        &self,
        out: &mut CudaTensor,
        tensor: &CudaTensor,
        start_row: usize,
        num_rows: usize,
    ) -> Result<()> {
        let shape = tensor.shape();
        if shape.is_empty() {
            return Err(ForgeError::InvalidArgument(
                "slice_rows_into: input must be non-empty".into(),
            ));
        }
        if start_row + num_rows > shape[0] {
            return Err(ForgeError::InvalidArgument(format!(
                "slice_rows_into: start_row {start_row} + num_rows {num_rows} > tensor rows {}",
                shape[0]
            )));
        }
        let cols: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            1
        };
        let mut expected = shape.to_vec();
        expected[0] = num_rows;
        if out.shape() != expected.as_slice() {
            return Err(ForgeError::ShapeMismatch {
                expected,
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != tensor.dtype() {
            return Err(ForgeError::InvalidArgument(format!(
                "slice_rows_into: out dtype {:?} != tensor dtype {:?}",
                out.dtype(),
                tensor.dtype()
            )));
        }
        let offset = start_row * cols;
        let len = num_rows * cols;

        match tensor.dtype() {
            DType::F32 => {
                let src = tensor.f32_slice()?;
                let dst = out.f32_slice_mut()?;
                self.stream
                    .memcpy_dtod(&src.slice(offset..offset + len), dst)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(())
            }
            DType::F16 => {
                let src = tensor.f16_slice()?;
                let dst = out.f16_slice_mut()?;
                self.stream
                    .memcpy_dtod(&src.slice(offset..offset + len), dst)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(())
            }
            DType::BF16 => {
                let src = tensor.bf16_slice()?;
                let dst = out.bf16_slice_mut()?;
                self.stream
                    .memcpy_dtod(&src.slice(offset..offset + len), dst)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                Ok(())
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn extract_head(
        &self,
        tensor: &CudaTensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        head: usize,
    ) -> Result<CudaTensor> {
        let n = (seq_len * head_dim) as u32;
        let seq_len_u32 = seq_len as u32;
        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let head_idx_u32 = head as u32;

        match tensor.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.extract_head_f16);
                builder.arg(&mut out);
                builder.arg(tensor.f16_slice()?);
                builder.arg(&seq_len_u32);
                builder.arg(&num_heads_u32);
                builder.arg(&head_dim_u32);
                builder.arg(&head_idx_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, vec![seq_len, head_dim]))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self.stream.launch_builder(&self.kernels.extract_head_f32);
                builder.arg(&mut out);
                builder.arg(tensor.f32_slice()?);
                builder.arg(&seq_len_u32);
                builder.arg(&num_heads_u32);
                builder.arg(&head_dim_u32);
                builder.arg(&head_idx_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, vec![seq_len, head_dim]))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn apply_causal_mask(
        &self,
        scores: &CudaTensor,
        seq_len: usize,
        kv_len: usize,
    ) -> Result<CudaTensor> {
        let n = (seq_len * kv_len) as u32;
        let seq_len_u32 = seq_len as u32;
        let kv_len_u32 = kv_len as u32;

        match scores.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.apply_causal_mask_f16);
                builder.arg(&mut out);
                builder.arg(scores.f16_slice()?);
                builder.arg(&seq_len_u32);
                builder.arg(&kv_len_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(out, scores.shape.clone()))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(n as usize)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.apply_causal_mask_f32);
                builder.arg(&mut out);
                builder.arg(scores.f32_slice()?);
                builder.arg(&seq_len_u32);
                builder.arg(&kv_len_u32);
                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n))
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(out, scores.shape.clone()))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn interleave_heads(
        &self,
        heads: &[&CudaTensor],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<CudaTensor> {
        let num_heads = heads.len();
        if num_heads == 0 {
            return Err(ForgeError::InvalidArgument("empty heads list".into()));
        }
        let dtype = heads[0].dtype();
        let n = (seq_len * head_dim) as u32;
        let seq_len_u32 = seq_len as u32;
        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let out_shape = vec![seq_len, num_heads * head_dim];

        match dtype {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(seq_len * num_heads * head_dim)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                for (h_idx, head) in heads.iter().enumerate() {
                    let head_idx_u32 = h_idx as u32;
                    let mut builder = self
                        .stream
                        .launch_builder(&self.kernels.interleave_heads_f16);
                    builder.arg(&mut out);
                    builder.arg(head.f16_slice()?);
                    builder.arg(&seq_len_u32);
                    builder.arg(&num_heads_u32);
                    builder.arg(&head_dim_u32);
                    builder.arg(&head_idx_u32);
                    unsafe {
                        builder
                            .launch(LaunchConfig::for_num_elems(n))
                            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    }
                }

                Ok(CudaTensor::f16_data(out, out_shape))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(seq_len * num_heads * head_dim)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                for (h_idx, head) in heads.iter().enumerate() {
                    let head_idx_u32 = h_idx as u32;
                    let mut builder = self
                        .stream
                        .launch_builder(&self.kernels.interleave_heads_f32);
                    builder.arg(&mut out);
                    builder.arg(head.f32_slice()?);
                    builder.arg(&seq_len_u32);
                    builder.arg(&num_heads_u32);
                    builder.arg(&head_dim_u32);
                    builder.arg(&head_idx_u32);
                    unsafe {
                        builder
                            .launch(LaunchConfig::for_num_elems(n))
                            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                    }
                }

                Ok(CudaTensor::f32_data(out, out_shape))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    fn multi_head_attention(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        num_heads: usize,
        _num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<CudaTensor> {
        let seq_len = q.shape()[1];

        // Route through attention_fwd (FA2 when feature enabled, naive otherwise)
        let result_4d = crate::flash_attention::attention_fwd(self, q, k, v, scale, is_causal)?;

        // Flatten from [1, seq_len, num_heads, head_dim] → [seq_len, num_heads * head_dim]
        self.reshape(&result_4d, &[seq_len, num_heads * head_dim])
    }

    fn batched_decode_attention(
        &self,
        q: &CudaTensor,
        k_caches: &[CudaTensor],
        v_caches: &[CudaTensor],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<CudaTensor> {
        let num_seqs = k_caches.len();
        if num_seqs == 0 {
            return self.allocate(&[0, num_heads * head_dim], q.dtype());
        }
        if v_caches.len() != num_seqs {
            return Err(ForgeError::InvalidArgument(format!(
                "k_caches.len()={} != v_caches.len()={}",
                num_seqs,
                v_caches.len()
            )));
        }
        if q.shape()[0] != num_seqs {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![num_seqs, num_heads * head_dim],
                got: q.shape().to_vec(),
            });
        }

        // Build pointer tables and kv_lens on host.
        // device_ptr() returns (CUdeviceptr, SyncOnDrop). CUdeviceptr is u64.
        // We collect the device pointers into host arrays, then upload them.
        let mut k_ptrs_host: Vec<u64> = Vec::with_capacity(num_seqs);
        let mut v_ptrs_host: Vec<u64> = Vec::with_capacity(num_seqs);
        let mut kv_lens_host: Vec<i32> = Vec::with_capacity(num_seqs);

        // We need to keep the SyncOnDrop guards alive until after the kernel launch.
        let mut _k_guards = Vec::with_capacity(num_seqs);
        let mut _v_guards = Vec::with_capacity(num_seqs);

        match q.dtype() {
            DType::F16 => {
                for i in 0..num_seqs {
                    let k_slice = k_caches[i].f16_slice()?;
                    let v_slice = v_caches[i].f16_slice()?;
                    let (k_ptr, k_guard) = k_slice.device_ptr(&self.stream);
                    let (v_ptr, v_guard) = v_slice.device_ptr(&self.stream);
                    k_ptrs_host.push(k_ptr);
                    v_ptrs_host.push(v_ptr);
                    _k_guards.push(k_guard);
                    _v_guards.push(v_guard);
                    kv_lens_host.push(k_caches[i].shape()[0] as i32);
                }
            }
            DType::F32 => {
                for i in 0..num_seqs {
                    let k_slice = k_caches[i].f32_slice()?;
                    let v_slice = v_caches[i].f32_slice()?;
                    let (k_ptr, k_guard) = k_slice.device_ptr(&self.stream);
                    let (v_ptr, v_guard) = v_slice.device_ptr(&self.stream);
                    k_ptrs_host.push(k_ptr);
                    v_ptrs_host.push(v_ptr);
                    _k_guards.push(k_guard);
                    _v_guards.push(v_guard);
                    kv_lens_host.push(k_caches[i].shape()[0] as i32);
                }
            }
            other => return Err(ForgeError::UnsupportedDtype(other)),
        }

        // Upload pointer tables and kv_lens to GPU
        let k_ptrs_dev = self
            .stream
            .memcpy_stod(&k_ptrs_host)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        let v_ptrs_dev = self
            .stream
            .memcpy_stod(&v_ptrs_host)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        let kv_lens_dev = self
            .stream
            .memcpy_stod(&kv_lens_host)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        let block_dim = next_power_of_2(128u32.min(head_dim as u32));
        let shared_mem = (block_dim + head_dim as u32) * 4; // scratch + output accumulator

        let num_heads_i32 = num_heads as i32;
        let num_kv_heads_i32 = num_kv_heads as i32;
        let head_dim_i32 = head_dim as i32;

        match q.dtype() {
            DType::F16 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<half::f16>(num_seqs * num_heads * head_dim)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.batched_decode_attention_f16);
                builder.arg(&mut out);
                builder.arg(q.f16_slice()?);
                builder.arg(&k_ptrs_dev);
                builder.arg(&v_ptrs_dev);
                builder.arg(&kv_lens_dev);
                builder.arg(&scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&head_dim_i32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (num_seqs as u32, num_heads as u32, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f16_data(
                    out,
                    vec![num_seqs, num_heads * head_dim],
                ))
            }
            DType::F32 => {
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(num_seqs * num_heads * head_dim)
                    .map_err(|e| ForgeError::Cuda(e.to_string()))?;

                let mut builder = self
                    .stream
                    .launch_builder(&self.kernels.batched_decode_attention_f32);
                builder.arg(&mut out);
                builder.arg(q.f32_slice()?);
                builder.arg(&k_ptrs_dev);
                builder.arg(&v_ptrs_dev);
                builder.arg(&kv_lens_dev);
                builder.arg(&scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&head_dim_i32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (num_seqs as u32, num_heads as u32, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: shared_mem,
                        })
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }

                Ok(CudaTensor::f32_data(
                    out,
                    vec![num_seqs, num_heads * head_dim],
                ))
            }
            other => Err(ForgeError::UnsupportedDtype(other)),
        }
    }

    /// On-device per-row argmax: one block reduces one row of `[rows, cols]`
    /// logits to its max-value index, so only `rows` ids cross PCIe (vs the
    /// full logits in the default impl). Tie-break matches the CPU sampler —
    /// highest index among equal maxima (see [`Backend::argmax`]).
    fn argmax(&self, logits: &CudaTensor) -> Result<Vec<u32>> {
        let shape = logits.shape();
        let (rows, cols) = match shape.len() {
            1 => (1usize, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(ForgeError::InvalidArgument(format!(
                    "argmax: expected 1-D or 2-D logits, got {shape:?}"
                )));
            }
        };
        if cols == 0 {
            return Err(ForgeError::InvalidArgument(
                "argmax: empty logits (cols == 0)".into(),
            ));
        }

        let mut out_ids = self
            .stream
            .alloc_zeros::<u32>(rows)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let block_dim = next_power_of_2(256u32.min(cols as u32)).max(1);
        // Reduction scratch: one f32 value + one u32 index per thread.
        let shared_mem = block_dim * 8;
        let launch_cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        match logits.dtype() {
            DType::F32 => {
                let l = logits.f32_slice()?;
                let mut builder = self.stream.launch_builder(&self.kernels.argmax_f32);
                builder.arg(&mut out_ids);
                builder.arg(l);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
            }
            DType::F16 => {
                let l = logits.f16_slice()?;
                let mut builder = self.stream.launch_builder(&self.kernels.argmax_f16);
                builder.arg(&mut out_ids);
                builder.arg(l);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
            }
            other => return Err(ForgeError::UnsupportedDtype(other)),
        }

        self.stream
            .memcpy_dtov(&out_ids)
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    /// On-device Gumbel-max multinomial sample (see [`Backend::sample_gumbel`]).
    /// One block per row perturbs each logit with Gumbel noise and reduces to
    /// the argmax; only `rows` ids cross PCIe.
    fn sample_gumbel(
        &self,
        logits: &CudaTensor,
        temperature: f32,
        seed: u64,
        step: u32,
    ) -> Result<Vec<u32>> {
        let shape = logits.shape();
        let (rows, cols) = match shape.len() {
            1 => (1usize, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(ForgeError::InvalidArgument(format!(
                    "sample_gumbel: expected 1-D or 2-D logits, got {shape:?}"
                )));
            }
        };
        if cols == 0 {
            return Err(ForgeError::InvalidArgument(
                "sample_gumbel: empty logits (cols == 0)".into(),
            ));
        }
        if !(temperature > 0.0) {
            return Err(ForgeError::InvalidArgument(
                "sample_gumbel: temperature must be > 0 (use argmax for greedy)".into(),
            ));
        }

        let mut out_ids = self
            .stream
            .alloc_zeros::<u32>(rows)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let inv_temp = 1.0f32 / temperature;
        let block_dim = next_power_of_2(256u32.min(cols as u32)).max(1);
        let shared_mem = block_dim * 8; // f32 value + u32 index per thread
        let launch_cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        match logits.dtype() {
            DType::F32 => {
                let l = logits.f32_slice()?;
                let mut builder = self.stream.launch_builder(&self.kernels.sample_gumbel_f32);
                builder.arg(&mut out_ids);
                builder.arg(l);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                builder.arg(&inv_temp);
                builder.arg(&seed);
                builder.arg(&step);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
            }
            DType::F16 => {
                let l = logits.f16_slice()?;
                let mut builder = self.stream.launch_builder(&self.kernels.sample_gumbel_f16);
                builder.arg(&mut out_ids);
                builder.arg(l);
                builder.arg(&rows_u32);
                builder.arg(&cols_u32);
                builder.arg(&inv_temp);
                builder.arg(&seed);
                builder.arg(&step);
                unsafe {
                    builder
                        .launch(launch_cfg)
                        .map_err(|e| ForgeError::Cuda(e.to_string()))?;
                }
            }
            other => return Err(ForgeError::UnsupportedDtype(other)),
        }

        self.stream
            .memcpy_dtov(&out_ids)
            .map_err(|e| ForgeError::Cuda(e.to_string()))
    }

    fn stage_decode_inputs(&self, inputs: &DecodeStageInputs) -> Result<()> {
        self.stage_decode_inputs_impl(
            inputs.token_indices,
            inputs.rope_cos,
            inputs.rope_sin,
            inputs.block_tables,
            inputs.kv_lens,
        )
    }

    fn stage_slot_mapping(&self, slot_mapping: &[i32]) -> Result<usize> {
        // Inherent method: uploads to the persistent device scratch.
        CudaBackend::stage_slot_mapping(self, slot_mapping)
    }

    fn scatter_kv(
        &self,
        pool: &mut CudaTensor,
        src: &CudaTensor,
        _slot_mapping: &[i32],
        n_rows: usize,
    ) -> Result<()> {
        // Capture-safe: ignore the host slot_mapping and read the device
        // scratch staged by `stage_slot_mapping` — no host pointer is baked
        // into the captured graph.
        CudaBackend::scatter_kv(self, pool, src, n_rows)
    }

    fn make_decode_graph_runner(&self, buckets: &[u32]) -> Option<Box<dyn DecodeGraphDispatch>> {
        Some(Box::new(DecodeGraphRunner::with_buckets(
            self.ctx(),
            self.stream(),
            buckets.to_vec(),
        )))
    }

    fn decode_capture_epoch(&self) -> u64 {
        // Sum of every persistent decode scratch's grow-counter. Counters only
        // increment, so any scratch reallocation (which moves the device
        // pointer a captured graph baked) changes the sum and triggers
        // re-capture in the engine.
        let (bt, kv) = self.paged_scratch_versions();
        let (cos, sin) = self.rope_scratch_versions();
        bt.wrapping_add(kv)
            .wrapping_add(cos)
            .wrapping_add(sin)
            .wrapping_add(self.embedding_scratch_version())
            .wrapping_add(self.scatter_slot_mapping_version())
    }
}
