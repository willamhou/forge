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

    // cudarc 0.17.8's bindgen omits the implicit CUDA_GRAPH_INSTANTIATE_FLAG_NONE
    // (=0) variant; flags are u32 bit-or values. Transmute 0 to get default
    // instantiation. Upstream issue worth filing for a NONE alias.
    let none_flag: CUgraphInstantiate_flags = unsafe { std::mem::transmute::<u32, _>(0) };

    // ===== Stage A: uncaptured baseline (kernels + GEMM) =====
    stream.memcpy_htod(&zeros_host, &mut buf)?;
    run_kernels(&stream, &inc, &mul, &scale, &mut buf, kernel_cfg, n_i32, scale_value)?;
    unsafe { blas.gemm(gemm_cfg, &gemm_a, &gemm_b, &mut gemm_c)? };
    stream.synchronize()?;
    let baseline_buf: Vec<f32> = stream.memcpy_dtov(&buf)?;
    let baseline_gemm: Vec<f32> = stream.memcpy_dtov(&gemm_c)?;
    println!(
        "[A baseline ] buf[0]={:.6} (expect {:.6})  gemm[0..4]={:?}",
        baseline_buf[0], EXPECTED_BUF, &baseline_gemm[0..GEMM_DIM],
    );
    if (baseline_buf[0] - EXPECTED_BUF).abs() > 1e-5 {
        return red("baseline buf disagrees with hand-computed value");
    }

    // ===== Stage B: capture kernels only (no GEMM) =====
    stream.memcpy_htod(&zeros_host, &mut buf)?;
    stream.synchronize()?;
    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap_kernels_err =
        run_kernels(&stream, &inc, &mul, &scale, &mut buf, kernel_cfg, n_i32, scale_value);
    let graph_kernels_opt = stream.end_capture(none_flag)?;
    if let Err(e) = cap_kernels_err {
        return red(&format!("kernel dispatch failed inside capture: {e}"));
    }
    let graph_kernels = graph_kernels_opt
        .ok_or("end_capture returned None for kernels-only capture")?;
    println!("[B captureK ] kernel-only capture ok");

    // Replay REPLAYS× with H2D-reset before each launch
    let t = Instant::now();
    for _ in 0..REPLAYS {
        stream.memcpy_htod(&zeros_host, &mut buf)?;
        graph_kernels.launch()?;
    }
    stream.synchronize()?;
    let replay_us_each = t.elapsed().as_micros() as f64 / REPLAYS as f64;
    let replay_buf: Vec<f32> = stream.memcpy_dtov(&buf)?;
    let kernels_match = (replay_buf[0] - baseline_buf[0]).abs() < 1e-5;
    println!(
        "[B replay×{:>3}] buf[0]={:.6} (expect {:.6})  avg {:.1} µs/replay",
        REPLAYS, replay_buf[0], baseline_buf[0], replay_us_each,
    );
    if !kernels_match {
        return red("replayed kernel-only graph diverges from baseline");
    }

    // ===== Stage C: H2D-before-replay updates inputs =====
    stream.memcpy_htod(&fives_host, &mut buf)?;
    graph_kernels.launch()?;
    stream.synchronize()?;
    let updated: Vec<f32> = stream.memcpy_dtov(&buf)?;
    println!(
        "[C h2d→replay] buf[0]={:.6} (expect {:.6} from input=5.0)",
        updated[0], EXPECTED_INPUT_5,
    );
    if (updated[0] - EXPECTED_INPUT_5).abs() > 1e-5 {
        return red("H2D memcpy before launch did not propagate into next replay");
    }

    // ===== Stage D: capture kernels + cuBLAS GEMM =====
    stream.memcpy_htod(&zeros_host, &mut buf)?;
    stream.synchronize()?;
    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap_full_err = (|| -> Result<(), Box<dyn std::error::Error>> {
        run_kernels(&stream, &inc, &mul, &scale, &mut buf, kernel_cfg, n_i32, scale_value)?;
        unsafe { blas.gemm(gemm_cfg, &gemm_a, &gemm_b, &mut gemm_c)? };
        Ok(())
    })();
    let graph_full_opt = stream.end_capture(none_flag)?;

    if let Err(e) = cap_full_err {
        return yellow(&format!(
            "kernels capture OK but cuBLAS GEMM cannot be captured: {e}. \
             Task 2 path: pre-allocate cuBLAS workspace via cublasSetWorkspace \
             before capture, or fall back to a custom GEMM kernel for the \
             decode hot path.",
        ));
    }
    let graph_full = match graph_full_opt {
        Some(g) => g,
        None => {
            return yellow(
                "GEMM dispatch returned Ok but end_capture produced no graph — \
                 cuBLAS likely poisoned the capture stream silently",
            );
        }
    };
    stream.memcpy_htod(&zeros_host, &mut buf)?;
    graph_full.launch()?;
    stream.synchronize()?;
    let rgemm: Vec<f32> = stream.memcpy_dtov(&gemm_c)?;
    let gemm_replay_match = rgemm
        .iter()
        .zip(&baseline_gemm)
        .all(|(r, b)| (r - b).abs() < 1e-5);
    if !gemm_replay_match {
        return yellow(
            "GEMM captured ok but the replayed output differs from the \
             uncaptured baseline — investigate before trusting in production",
        );
    }
    println!("[D capture+G] cuBLAS GEMM captured & replayed ok");
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
