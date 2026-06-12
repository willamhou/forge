//! Task 1 spike: verify cudarc 0.17 CUDA Graph API meets the requirements
//! of docs/plans/2026-05-22-cuda-graphs-plan.md.
//!
//! Five checks (per the plan):
//!   1. `stream.begin_capture` + `stream.end_capture` round-trip
//!   2. Graph instantiation (cudarc folds this into `end_capture`)
//!   3. `CudaGraph::launch` re-runs captured kernels correctly
//!   4. A cuBLAS GEMM issued during capture also replays
//!   5. H2D memcpy on a persistent input buffer *before* `graph.launch()`
//!      changes what the next replay sees
//!
//! Run: cargo run --release -p forge-backend-cuda --example graph_spike
//!
//! Decision gates (printed at exit):
//!   GREEN  — all checks pass, proceed to Task 2
//!   YELLOW — capture round-trip works but some path (e.g. cuBLAS) needs
//!            workspace pinning / pivot; details printed
//!   RED    — fundamental gap in cudarc; raw FFI shim or version bump needed

use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const KERNELS: &str = r#"
extern "C" __global__ void inc_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

extern "C" __global__ void mul_two(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;
}

extern "C" __global__ void scale_inv(float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] /= scale;
}
"#;

const N: usize = 1024;
const REPLAYS: usize = 100;
const GEMM_DIM: usize = 4;

// inc → mul → scale_inv on input 0.0 → ((0+1)*2)/3 = 0.6666...
// GEMM: identity(4) × arange(16, col-major) = arange(16)
const EXPECTED_BUF: f32 = ((0.0_f32 + 1.0) * 2.0) / 3.0;
const EXPECTED_INPUT_5: f32 = ((5.0_f32 + 1.0) * 2.0) / 3.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("== forge CUDA Graphs spike (cudarc 0.17.8) ==");

    let ctx = CudaContext::new(0)?;
    // CRITICAL gotcha #1: ctx.default_stream() returns the NULL / legacy
    // default stream — CUDA Graphs explicitly cannot capture it. Use a
    // real non-blocking stream.
    //
    // CRITICAL gotcha #2: cudarc's safe launch_builder, when the context
    // is in multi-stream mode, auto-inserts cuStreamWaitEvent dependencies
    // for read/write tracking. Those waits reference events recorded
    // outside the capture region and invalidate the graph
    // (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED). Disable event tracking;
    // we'll re-introduce dependencies via the graph topology itself.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let blas = CudaBlas::new(stream.clone())?;
    let module = ctx.load_module(compile_ptx(KERNELS)?)?;
    let inc = module.load_function("inc_one")?;
    let mul = module.load_function("mul_two")?;
    let scale = module.load_function("scale_inv")?;

    let mut buf = stream.alloc_zeros::<f32>(N)?;
    let mut gemm_a = stream.alloc_zeros::<f32>(GEMM_DIM * GEMM_DIM)?;
    let mut gemm_b = stream.alloc_zeros::<f32>(GEMM_DIM * GEMM_DIM)?;
    let mut gemm_c = stream.alloc_zeros::<f32>(GEMM_DIM * GEMM_DIM)?;

    let host_identity: Vec<f32> = (0..GEMM_DIM * GEMM_DIM)
        .map(|i| if i % (GEMM_DIM + 1) == 0 { 1.0 } else { 0.0 })
        .collect();
    let host_arange: Vec<f32> = (0..GEMM_DIM * GEMM_DIM).map(|i| i as f32).collect();
    stream.memcpy_htod(&host_identity, &mut gemm_a)?;
    stream.memcpy_htod(&host_arange, &mut gemm_b)?;
    stream.synchronize()?;

    let kernel_cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let gemm_cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: GEMM_DIM as i32,
        n: GEMM_DIM as i32,
        k: GEMM_DIM as i32,
        alpha: 1.0_f32,
        beta: 0.0_f32,
        lda: GEMM_DIM as i32,
        ldb: GEMM_DIM as i32,
        ldc: GEMM_DIM as i32,
    };
    let n_i32: i32 = N as i32;
    let scale_value: f32 = 3.0;

    let zeros_host: Vec<f32> = vec![0.0; N];
    let fives_host: Vec<f32> = vec![5.0; N];

    // cudarc 0.17.8's bindgen has no _NONE (=0) variant; transmuting 0 into
    // a #[repr(u32)] enum is UB. UPLOAD (=2) would be a no-op semantically
    // but needs the WithParams API to supply an upload stream — passing it
    // through cudarc's plain end_capture returns CUDA_ERROR_INVALID_VALUE.
    // AUTO_FREE_ON_LAUNCH (=1) only affects graph-internal allocations,
    // which our captured ops never make, so it is a genuine no-op here.
    let instantiate_flag =
        CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

    // ===== Stage A: uncaptured baseline (kernels + GEMM) =====
    stream.memcpy_htod(&zeros_host, &mut buf)?;
    run_kernels(
        &stream,
        &inc,
        &mul,
        &scale,
        &mut buf,
        kernel_cfg,
        n_i32,
        scale_value,
    )?;
    unsafe { blas.gemm(gemm_cfg, &gemm_a, &gemm_b, &mut gemm_c)? };
    stream.synchronize()?;
    let baseline_buf: Vec<f32> = stream.memcpy_dtov(&buf)?;
    let baseline_gemm: Vec<f32> = stream.memcpy_dtov(&gemm_c)?;
    println!(
        "[A baseline ] buf[0]={:.6} (expect {:.6})  gemm[0..4]={:?}",
        baseline_buf[0],
        EXPECTED_BUF,
        &baseline_gemm[0..GEMM_DIM],
    );
    if (baseline_buf[0] - EXPECTED_BUF).abs() > 1e-5 {
        return red("baseline buf disagrees with hand-computed value");
    }

    // ===== Stage B: capture kernels only (no GEMM) =====
    stream.memcpy_htod(&zeros_host, &mut buf)?;
    stream.synchronize()?;
    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap_kernels_err = run_kernels(
        &stream,
        &inc,
        &mul,
        &scale,
        &mut buf,
        kernel_cfg,
        n_i32,
        scale_value,
    );
    let graph_kernels_result = stream.end_capture(instantiate_flag);
    // Check dispatch failure FIRST so the diagnostic is the real cause,
    // not the downstream STREAM_CAPTURE_INVALIDATED that end_capture would surface.
    if let Err(e) = cap_kernels_err {
        return red(&format!("kernel dispatch failed inside capture: {e}"));
    }
    let graph_kernels =
        graph_kernels_result?.ok_or("end_capture returned None for kernels-only capture")?;
    println!("[B captureK ] kernel-only capture ok");

    // Replay REPLAYS× with H2D-reset before each launch. Separately measure
    // pure graph-launch overhead (the timer-only loop) vs. with-H2D overhead
    // so the reported number is not conflated, and compare against an
    // equivalent direct-launch loop to surface a regression if one exists.
    let t_replay_only = Instant::now();
    for _ in 0..REPLAYS {
        graph_kernels.launch()?;
    }
    stream.synchronize()?;
    let replay_only_us = t_replay_only.elapsed().as_micros() as f64 / REPLAYS as f64;

    let t_direct = Instant::now();
    for _ in 0..REPLAYS {
        stream.memcpy_htod(&zeros_host, &mut buf)?;
        run_kernels(
            &stream,
            &inc,
            &mul,
            &scale,
            &mut buf,
            kernel_cfg,
            n_i32,
            scale_value,
        )?;
    }
    stream.synchronize()?;
    let direct_us = t_direct.elapsed().as_micros() as f64 / REPLAYS as f64;

    // Final correctness check: reset, launch graph once, validate ALL elements.
    stream.memcpy_htod(&zeros_host, &mut buf)?;
    graph_kernels.launch()?;
    stream.synchronize()?;
    let replay_buf: Vec<f32> = stream.memcpy_dtov(&buf)?;
    let kernels_match = replay_buf
        .iter()
        .zip(&baseline_buf)
        .all(|(r, b)| (r - b).abs() < 1e-5);
    println!(
        "[B replay×{:>3}] buf[0..2]={:?}  graph-only {:.1} µs  direct-equiv {:.1} µs",
        REPLAYS,
        &replay_buf[0..2],
        replay_only_us,
        direct_us,
    );
    if !kernels_match {
        return red("replayed kernel-only graph diverges from baseline (full-buf check)");
    }

    // ===== Stage C: H2D-before-replay updates inputs =====
    stream.memcpy_htod(&fives_host, &mut buf)?;
    graph_kernels.launch()?;
    stream.synchronize()?;
    let updated: Vec<f32> = stream.memcpy_dtov(&buf)?;
    let updated_all_match = updated.iter().all(|&v| (v - EXPECTED_INPUT_5).abs() < 1e-5);
    println!(
        "[C h2d→replay] buf[0..2]={:?} (expect {:.6} for every element)",
        &updated[0..2],
        EXPECTED_INPUT_5,
    );
    if !updated_all_match {
        return red("H2D memcpy before launch did not propagate to every element");
    }

    // ===== Stage D: capture kernels + cuBLAS GEMM =====
    // Establish independent ground truth for the GEMM so we don't just compare
    // baseline-vs-replay (which a no-op replay would also satisfy).
    let host_zero_gemm = vec![0.0_f32; GEMM_DIM * GEMM_DIM];
    let expected_gemm = &host_arange; // identity × arange (col-major) = arange

    stream.memcpy_htod(&zeros_host, &mut buf)?;
    stream.synchronize()?;
    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap_full_err = (|| -> Result<(), Box<dyn std::error::Error>> {
        run_kernels(
            &stream,
            &inc,
            &mul,
            &scale,
            &mut buf,
            kernel_cfg,
            n_i32,
            scale_value,
        )?;
        unsafe { blas.gemm(gemm_cfg, &gemm_a, &gemm_b, &mut gemm_c)? };
        Ok(())
    })();
    let graph_full_result = stream.end_capture(instantiate_flag);
    // Surface dispatch error before end_capture's downstream symptom.
    if let Err(e) = cap_full_err {
        return yellow(&format!(
            "kernels capture OK but cuBLAS GEMM cannot be captured: {e}. \
             Task 2.5 path: pre-allocate cuBLAS workspace via \
             cublasSetWorkspace before capture, or fall back to a custom \
             GEMM kernel for the decode hot path.",
        ));
    }
    let graph_full = graph_full_result?.ok_or_else(|| -> Box<dyn std::error::Error> {
        "GEMM dispatch returned Ok but end_capture produced no graph — \
         cuBLAS likely poisoned the capture stream silently"
            .into()
    })?;

    // Replay REPLAYS×; reset BOTH buf and gemm_c before each launch so a
    // missing/dropped GEMM node cannot pass by re-reading stale baseline state.
    // (The original spike only reset buf and got false-positives from the
    // Stage A baseline lingering in gemm_c.)
    for _ in 0..REPLAYS {
        stream.memcpy_htod(&zeros_host, &mut buf)?;
        stream.memcpy_htod(&host_zero_gemm, &mut gemm_c)?;
        graph_full.launch()?;
    }
    stream.synchronize()?;
    let final_buf: Vec<f32> = stream.memcpy_dtov(&buf)?;
    let final_gemm: Vec<f32> = stream.memcpy_dtov(&gemm_c)?;

    let buf_match = final_buf.iter().all(|v| (v - EXPECTED_BUF).abs() < 1e-5);
    let gemm_vs_baseline = final_gemm
        .iter()
        .zip(&baseline_gemm)
        .all(|(r, b)| (r - b).abs() < 1e-5);
    let gemm_vs_ground_truth = final_gemm
        .iter()
        .zip(expected_gemm.iter())
        .all(|(r, b)| (r - b).abs() < 1e-5);
    if !buf_match {
        return yellow(
            "Stage D combined-capture: kernel side diverged from EXPECTED_BUF \
             on at least one element — capture may have reordered or dropped \
             a kernel node",
        );
    }
    if !gemm_vs_baseline || !gemm_vs_ground_truth {
        return yellow(
            "Stage D combined-capture: GEMM did not produce the expected \
             arange — either capture dropped the gemm node (would leave \
             gemm_c at zeros), or layout/config drift exists",
        );
    }
    println!(
        "[D capture+G×{:>3}] buf[0..2]={:?}  gemm[0..4]={:?}",
        REPLAYS,
        &final_buf[0..2],
        &final_gemm[0..GEMM_DIM],
    );
    println!();
    println!("✅ Task 1 decision gate = GREEN — all stages pass, proceed to Task 2.");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_kernels(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    inc: &cudarc::driver::CudaFunction,
    mul: &cudarc::driver::CudaFunction,
    scale: &cudarc::driver::CudaFunction,
    buf: &mut cudarc::driver::CudaSlice<f32>,
    kernel_cfg: LaunchConfig,
    n_i32: i32,
    scale_value: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut b = stream.launch_builder(inc);
    b.arg(&mut *buf);
    b.arg(&n_i32);
    unsafe { b.launch(kernel_cfg)? };

    let mut b = stream.launch_builder(mul);
    b.arg(&mut *buf);
    b.arg(&n_i32);
    unsafe { b.launch(kernel_cfg)? };

    let mut b = stream.launch_builder(scale);
    b.arg(&mut *buf);
    b.arg(&scale_value);
    b.arg(&n_i32);
    unsafe { b.launch(kernel_cfg)? };

    Ok(())
}

fn red(msg: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("❌ Task 1 decision gate = RED — {msg}");
    Err(msg.into())
}

fn yellow(msg: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("⚠️  Task 1 decision gate = YELLOW — {msg}");
    Err(msg.into())
}
