//! Microbenchmark: Q8_0 quantized GEMV vs cuBLAS f16 GEMV at decode shapes.
//!
//! Decode is GPU-memory-bound: each projection reads its whole weight matrix to
//! produce M=1 outputs. The bet behind quantized decode is that reading the
//! weight as Q8_0 (~17/16 bytes per element after the per-block scale) instead
//! of f16 (2 bytes per element) cuts the dominant memory traffic ~2× and so
//! roughly halves decode latency — *if* the kernel actually saturates HBM
//! bandwidth. This bench measures whether that holds on the real Qwen3-4B
//! projection shapes.
//!
//! For each shape (weight [n, k], activation [M=1, k], output [1, n]) it times:
//!   - quantized GEMV: `matmul_quant_into(out, a, wq)` with `wq` a Q8_0 [n, k]
//!   - cuBLAS f16 GEMV: `matmul_into(out, a, w_kn)` with `w_kn` the *dequantized*
//!     f16 weight transposed to [k, n] (the existing f16 decode path)
//!
//! and reports per shape: latency (us), effective weight-read bandwidth (GB/s),
//! and the quantized-vs-f16 speedup. Effective bandwidth counts only the weight
//! bytes (the dominant term at M=1): Q8_0 reads n*k*34/32 bytes, f16 reads
//! n*k*2 bytes.
//!
//! Run: cargo run --release -p forge-backend-cuda --example quant_gemv_bench

use std::time::Instant;

use forge_backend_cuda::{CudaBackend, quantize_q8_0};
use forge_core::{Backend, DType};
use half::f16;

/// GB10 (Grace Blackwell) advertised LPDDR5X bandwidth, for context in the
/// printed table — the quantized GEMV should approach this.
const GB10_PEAK_GBPS: f64 = 273.0;

/// Deterministic pseudo-random f32 in roughly [-amp, amp]. (Same splitmix as
/// the correctness test, so weights/activations are reproducible.)
fn pseudo_random(i: usize, salt: u64, amp: f32) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let u = ((x >> 40) as f32) * (1.0 / 16_777_216.0); // [0, 1)
    (u * 2.0 - 1.0) * amp
}

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
    /// lm_head is huge; allow fewer iters to keep the run quick.
    warmup: usize,
    iters: usize,
}

/// Host dequant of Q8_0 bytes to f16 (bit-identical to the kernel reconstruction)
/// — the cuBLAS reference must use the *same* weights the quantized path sees.
fn dequant_q8_0_to_f16(bytes: &[u8], n_elements: usize) -> Vec<f16> {
    const BLOCK_BYTES: usize = 34;
    let mut out = Vec::with_capacity(n_elements);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &q in &block[2..] {
            out.push(f16::from_f32(scale * (q as i8) as f32));
        }
    }
    out.truncate(n_elements);
    out
}

fn time_loop<F: FnMut()>(backend: &CudaBackend, warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    backend.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    backend.synchronize().unwrap();
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1.0e6 / iters as f64 // us per iter
}

fn main() {
    let backend = CudaBackend::new(0).expect("CUDA backend");

    // Qwen3-4B projection shapes, M=1 (single-sequence decode). Weights [n, k];
    // output [1, n].
    let shapes = [
        Shape {
            name: "qkv",
            n: 6144,
            k: 2560,
            warmup: 30,
            iters: 200,
        },
        Shape {
            name: "o",
            n: 2560,
            k: 4096,
            warmup: 30,
            iters: 200,
        },
        Shape {
            name: "gate",
            n: 9728,
            k: 2560,
            warmup: 30,
            iters: 200,
        },
        Shape {
            name: "up",
            n: 9728,
            k: 2560,
            warmup: 30,
            iters: 200,
        },
        Shape {
            name: "down",
            n: 2560,
            k: 9728,
            warmup: 30,
            iters: 200,
        },
        Shape {
            name: "lm_head",
            n: 151936,
            k: 2560,
            warmup: 5,
            iters: 30,
        },
    ];

    println!("Q8_0 quantized GEMV vs cuBLAS f16 GEMV — decode M=1, Qwen3-4B shapes");
    println!("GB10 advertised peak bandwidth: {GB10_PEAK_GBPS:.0} GB/s\n");
    println!(
        "{:<8} {:>7} {:>6} | {:>10} {:>9} | {:>10} {:>9} | {:>8}",
        "shape", "n", "k", "quant us", "GB/s", "cuBLAS us", "GB/s", "speedup"
    );
    println!("{}", "-".repeat(78));

    for s in &shapes {
        let (n, k) = (s.n, s.k);
        assert!(k.is_multiple_of(32), "k must be a multiple of 32");

        // [n, k] f16 weights → quantize → upload Q8_0 [n, k].
        let w_flat: Vec<f16> = (0..n * k)
            .map(|i| f16::from_f32(pseudo_random(i, 0x51A5 ^ n as u64, 1.0)))
            .collect();
        let w_bytes = quantize_q8_0(&w_flat);
        let wq = backend
            .copy_from_host_quant(&w_bytes, &[n, k], DType::Q8_0)
            .expect("copy_from_host_quant");

        // cuBLAS reference uses the SAME (dequantized) weights, transposed to
        // [k, n] for the standard `a · w` matmul.
        let w_deq = dequant_q8_0_to_f16(&w_bytes, n * k);
        let w_nk = backend
            .copy_from_host_f16(&w_deq, &[n, k])
            .expect("upload dequant weights");
        let w_kn = backend.transpose(&w_nk, 0, 1).expect("transpose to [k, n]");

        // [1, k] f16 activation.
        let a_flat: Vec<f16> = (0..k)
            .map(|i| f16::from_f32(pseudo_random(i, 0xABCD, 1.0)))
            .collect();
        let a = backend
            .copy_from_host_f16(&a_flat, &[1, k])
            .expect("upload a");

        let mut out_q = backend.allocate_zeros(&[1, n], DType::F16).unwrap();
        let mut out_b = backend.allocate_zeros(&[1, n], DType::F16).unwrap();

        // Quantized GEMV.
        let quant_us = time_loop(&backend, s.warmup, s.iters, || {
            backend.matmul_quant_into(&mut out_q, &a, &wq).unwrap();
        });
        // cuBLAS f16 GEMV.
        let blas_us = time_loop(&backend, s.warmup, s.iters, || {
            backend.matmul_into(&mut out_b, &a, &w_kn).unwrap();
        });

        // Effective weight-read bandwidth (M=1: weight bytes dominate traffic).
        let quant_bytes = (n * k) as f64 * 34.0 / 32.0;
        let f16_bytes = (n * k) as f64 * 2.0;
        let quant_gbps = quant_bytes / (quant_us * 1.0e3); // bytes / ns = GB/s
        let blas_gbps = f16_bytes / (blas_us * 1.0e3);
        let speedup = blas_us / quant_us;

        println!(
            "{:<8} {:>7} {:>6} | {:>10.1} {:>9.1} | {:>10.1} {:>9.1} | {:>7.2}x",
            s.name, n, k, quant_us, quant_gbps, blas_us, blas_gbps, speedup
        );
    }

    println!("\nNote: GB/s counts weight reads only (Q8_0 = n*k*34/32 bytes, f16 = n*k*2 bytes).");
    println!("At M=1 decode the weight read is the dominant memory traffic.");
    println!(
        "Caveat: shapes whose weight fits in L2 (e.g. 'o', ~22MB Q8_0) can show a\n\
         cuBLAS reading >peak bandwidth (L2-resident) — not a memory-bound baseline.\n\
         The honest comparison is the shapes whose weight exceeds L2 (qkv/gate/up/down/lm_head)."
    );
}
