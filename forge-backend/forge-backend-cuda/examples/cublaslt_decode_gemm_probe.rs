//! cuBLAS vs cuBLASLt probe for forge's M=8 decode GEMM shapes (Qwen3-4B).
//!
//! Goal: find out whether cuBLASLt's heuristic picks `nvjet_sm121_*_tmaAB`
//! (Blackwell-native + TMA) instead of `cutlass_80_wmma_*` for the shapes that
//! dominate forge's batch decode. pega's profile shows nvjet on the same
//! workload — same libcublas.so.13.1.1.3 — so the only place the two engines
//! can diverge is in how the call is made.
//!
//! Layout choices probed:
//!   1. classic cuBLAS GEMM with the swap-trick (forge's current code path):
//!        row-major C = A·B → call cuBLAS with (B, A) and swap m/n. Result lands
//!        row-major. cuBLAS sees `transa=N, transb=N` over column-major.
//!   2. cuBLASLt with the same swap trick.
//!   3. cuBLASLt called native column-major (transa=N, transb=N, A is k×m
//!      col-major, B is n×k col-major), output transposed afterwards (skipped —
//!      we just measure the matmul).
//!
//! Run:
//!   cargo run --release -p forge-backend-cuda \
//!     --example cublaslt_decode_gemm_probe
//!
//! To see kernel names:
//!   nsys profile -t cuda --stats=true -o /tmp/lt_probe \
//!     ./target/release/examples/cublaslt_decode_gemm_probe
//!   nsys stats --report cuda_gpu_kern_sum --format table /tmp/lt_probe.nsys-rep

use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::f16;

// Qwen3-4B decode GEMM shapes at M=8 (C=8 batch step).
// Each is (m, k, n) where C[m,n] = A[m,k] · B[k,n].
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (8, 2560, 6144, "qkv_proj (M=8, K=2560 hidden, N=4096+1024+1024)"),
    (8, 4096, 2560, "attn_proj (M=8, K=q_proj_size, N=hidden)"),
    (8, 2560, 9728, "gate_proj/up_proj (M=8, K=hidden, N=intermediate)"),
    (8, 9728, 2560, "down_proj (M=8, K=intermediate, N=hidden)"),
    (8, 2560, 151936, "lm_head (M=8, K=hidden, N=vocab) — largest output"),
    (2, 2560, 6144, "qkv_proj (M=2)"),
    (2, 4096, 2560, "attn_proj (M=2)"),
    (2, 9728, 2560, "down_proj (M=2)"),
    (2, 2560, 151936, "lm_head (M=2)"),
];

fn alloc_f16(stream: &Arc<CudaStream>, n: usize) -> CudaSlice<f16> {
    let host: Vec<f16> = (0..n).map(|i| f16::from_f32(((i % 113) as f32) * 1e-3)).collect();
    stream.memcpy_stod(&host).expect("alloc")
}

unsafe fn classic_gemm_swap(
    blas: &CudaBlas,
    a: &CudaSlice<f16>, // [m, k] row-major
    b: &CudaSlice<f16>, // [k, n] row-major
    c: &mut CudaSlice<f16>, // [m, n] row-major
    m: usize,
    k: usize,
    n: usize,
) {
    // row-major C = A·B  ≡  cuBLAS (col-major) sees B^T · A^T → pass (B, A),
    // swap (m,n), pass leading dims as the row-major widths. Matches
    // forge-backend-cuda/src/backend.rs:matmul_into.
    blas.gemm(
        GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: f16::from_f32(1.0),
            lda: n as i32,
            ldb: k as i32,
            beta: f16::from_f32(0.0),
            ldc: n as i32,
        },
        b,
        a,
        c,
    )
    .expect("classic gemm");
}

unsafe fn lt_gemm_swap(
    lt: &CudaBlasLT,
    a: &CudaSlice<f16>,
    b: &CudaSlice<f16>,
    c: &mut CudaSlice<f16>,
    m: usize,
    k: usize,
    n: usize,
) {
    // Same swap-trick layout, just routed through cuBLASLt's heuristic
    // selection (vs classic cuBLAS's algo table).
    let cfg = MatmulConfig {
        transa: false,
        transb: false,
        transc: false,
        m: n as u64,
        n: m as u64,
        k: k as u64,
        alpha: 1.0,
        lda: n as i64,
        ldb: k as i64,
        beta: 0.0,
        ldc: n as i64,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };
    lt.matmul(cfg, b, a, c, None, None).expect("lt matmul swap");
}

unsafe fn lt_gemm_native_col(
    lt: &CudaBlasLT,
    a: &CudaSlice<f16>,
    b: &CudaSlice<f16>,
    c: &mut CudaSlice<f16>,
    m: usize,
    k: usize,
    n: usize,
) {
    // Treat row-major A[m,k] as a column-major matrix of shape (k, m) and
    // ask cuBLASLt for op(A)=A^T. Same for B and C. That asks cuBLASLt for the
    // mathematical equivalent of A·B in row-major space, but expressed as
    // (A^T)^T · (B^T)^T = A·B over column-major buffers. Doing it this way (vs
    // the swap trick) gives cuBLASLt a "canonical" descriptor — the hypothesis
    // is the heuristic table indexes algos by descriptor shape and the swap-
    // trick descriptor (column-major with N×M dims) hides shapes the nvjet
    // table covers.
    let cfg = MatmulConfig {
        transa: true,
        transb: true,
        transc: false,
        m: m as u64,
        n: n as u64,
        k: k as u64,
        alpha: 1.0,
        // A row-major [m,k] viewed col-major shape (k,m), lda = k.
        lda: k as i64,
        // B row-major [k,n] viewed col-major shape (n,k), ldb = n.
        ldb: n as i64,
        beta: 0.0,
        // C output col-major shape (m,n), ldc = m. CAVEAT: that lays C out
        // column-major; reading it as row-major would be the transpose. We do
        // NOT consume C downstream — only timing the matmul — so this is OK
        // for the probe. (For prod we'd need to handle the layout, which is
        // what makes this approach harder than the swap trick.)
        ldc: m as i64,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };
    lt.matmul(cfg, a, b, c, None, None).expect("lt matmul native");
}

fn bench<F: FnMut()>(label: &str, stream: &Arc<CudaStream>, iters: usize, mut f: F) -> f64 {
    // warmup
    for _ in 0..5 {
        f();
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let per_us = dt * 1e6 / iters as f64;
    println!("  {label:40} {per_us:8.2} μs/call");
    per_us
}

// Correctness check: compute reference output once via classic cuBLAS (which we
// trust), copy back to host, then compare against each Lt variant after a
// suitable read-back / transpose. Print a max-abs diff.
fn host_max_abs_diff(a: &[f16], b: &[f16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).expect("cublas");
    let lt = CudaBlasLT::new(stream.clone()).expect("cublaslt");

    let iters = 500;
    println!("Iters per cell: {iters}");
    println!();

    for &(m, k, n, label) in SHAPES {
        println!("Shape: {label}");
        println!("  ({m} x {k}) x ({k} x {n}) = ({m} x {n})");

        let a = alloc_f16(&stream, m * k);
        let b = alloc_f16(&stream, k * n);
        let mut c = stream.alloc_zeros::<f16>(m * n).unwrap();

        // Correctness: build a reference output with classic, then check each
        // Lt variant matches under the expected output-byte interpretation.
        unsafe { classic_gemm_swap(&blas, &a, &b, &mut c, m, k, n) };
        stream.synchronize().unwrap();
        let c_ref_rm: Vec<f16> = stream.memcpy_dtov(&c).unwrap(); // row-major [m,n]

        // Lt-swap: same swap-trick descriptor → output bytes are row-major [m,n].
        let mut c2 = stream.alloc_zeros::<f16>(m * n).unwrap();
        unsafe { lt_gemm_swap(&lt, &a, &b, &mut c2, m, k, n) };
        stream.synchronize().unwrap();
        let c_lt_swap: Vec<f16> = stream.memcpy_dtov(&c2).unwrap();
        let diff_swap = host_max_abs_diff(&c_ref_rm, &c_lt_swap);

        // Lt-native: descriptor produces col-major (m,n) → bytes read row-major
        // are [n,m] = c_ref_rm^T. Build expected by transposing on host.
        let mut c3 = stream.alloc_zeros::<f16>(m * n).unwrap();
        unsafe { lt_gemm_native_col(&lt, &a, &b, &mut c3, m, k, n) };
        stream.synchronize().unwrap();
        let c_lt_native: Vec<f16> = stream.memcpy_dtov(&c3).unwrap();
        let mut c_ref_rm_T = vec![f16::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                // c_ref_rm^T[j,i] = c_ref_rm[i,j], stored row-major [n,m] at j*m+i.
                c_ref_rm_T[j * m + i] = c_ref_rm[i * n + j];
            }
        }
        let diff_native = host_max_abs_diff(&c_ref_rm_T, &c_lt_native);

        println!(
            "  correctness: Lt-swap maxabs={:.4}  Lt-native (vs ref^T) maxabs={:.4}",
            diff_swap, diff_native
        );

        let classic_us = bench("classic cuBLAS gemm (swap)", &stream, iters, || unsafe {
            classic_gemm_swap(&blas, &a, &b, &mut c, m, k, n);
        });
        let lt_swap_us = bench("cuBLASLt matmul (swap trick)", &stream, iters, || unsafe {
            lt_gemm_swap(&lt, &a, &b, &mut c, m, k, n);
        });
        let lt_native_us = bench(
            "cuBLASLt matmul (native col-major)",
            &stream,
            iters,
            || unsafe {
                lt_gemm_native_col(&lt, &a, &b, &mut c, m, k, n);
            },
        );

        println!(
            "  Lt-swap / classic   = {:.3}x   Lt-native / classic = {:.3}x",
            lt_swap_us / classic_us,
            lt_native_us / classic_us
        );

        // Discourage the optimizer from removing the bufs.
        let (ptr, _) = a.device_ptr(&stream);
        let (ptr2, _) = b.device_ptr(&stream);
        let (ptr3, _) = c.device_ptr_mut(&stream);
        std::hint::black_box((ptr, ptr2, ptr3));

        println!();
    }
}
