//! A linear projection that keeps an FP16 weight for prefill and, when
//! quantized decode is enabled, an additional Q8_0 copy for the decode GEMV.
//!
//! ## Why two weight copies / two orientations
//!
//! The prefill GEMM and the decode GEMV want the weight in opposite layouts:
//!
//! - **Prefill** uses `matmul(x[m, in], W[in, out]) -> [m, out]`. The loader
//!   transposes HuggingFace's `[out, in]` into `[in, out]` so this works — that
//!   transposed tensor is `w_f16`, and it is byte-for-byte the weight the model
//!   used before this type existed.
//! - **Decode** uses `matmul_quant_into(out[m, out], x[m, in], Wq[out, in])`,
//!   where the kernel computes `out[j] = Σ x · Wq[j, :]` row-by-row. So the
//!   quantized weight is kept in the *original* `[out, in]` orientation (no
//!   transpose), quantized to Q8_0.
//!
//! When quantized decode is **off**, `w_quant` is `None` and
//! [`QuantizedLinear::matmul_decode_into`] falls back to the plain f16
//! `matmul_into` — so the decode path is bit-for-bit identical to before.

use forge_core::{Backend, Result, Tensor};

/// A linear weight with an FP16 prefill copy and an optional Q8_0 decode copy.
pub struct QuantizedLinear<B: Backend> {
    /// `[in, out]` FP16 — the transposed weight used by prefill `matmul`.
    /// Identical to the pre-quantization weight; always present.
    w_f16: B::Tensor,
    /// `[out, in]` Q8_0 — the decode GEMV weight. `Some` only when quantized
    /// decode is enabled and the weight's `in` dim is a multiple of 32.
    w_quant: Option<B::Tensor>,
}

impl<B: Backend> QuantizedLinear<B> {
    /// Construct an FP16-only linear (no quantized decode copy).
    pub fn new_f16(w_f16: B::Tensor) -> Self {
        Self {
            w_f16,
            w_quant: None,
        }
    }

    /// Construct with an optional Q8_0 decode copy.
    pub fn new(w_f16: B::Tensor, w_quant: Option<B::Tensor>) -> Self {
        Self { w_f16, w_quant }
    }

    /// Prefill projection: `x[m, in] · w_f16[in, out] -> [m, out]`.
    pub fn matmul_prefill(&self, x: &B::Tensor, backend: &B) -> Result<B::Tensor> {
        backend.matmul(x, &self.w_f16)
    }

    /// Decode projection into a caller-provided buffer.
    ///
    /// Dispatch by batch size `m` (= `x.shape()[0]`, the number of sequences
    /// decoding this step):
    ///
    /// - **`m == 1` with a quantized copy** → `matmul_quant_into(out, x,
    ///   w_quant[out, in])`. The Q8_0 GEMV reads the weight once and is
    ///   memory-bound, beating cuBLAS f16 at single-stream decode (the quant
    ///   payoff). Launch-only and capture-safe (the quant weight is uploaded
    ///   once at load and never moves or grows).
    /// - **`m > 1`** → the f16 `matmul_into(out, x, w_f16)`. The GEMV does NOT
    ///   amortize the weight across a batch (each row re-reads it, cost scales
    ///   `m×`), so batch decode is dramatically faster on cuBLAS's f16 GEMM,
    ///   which reuses the weight across the batch via tensor cores. Measured on
    ///   GB10/Qwen3-4B: at C=8 the Q8 GEMV degrades to ~173ms TPOT vs the f16
    ///   GEMM's ~67ms. A quantized kernel that wins at batch needs int8 tensor
    ///   cores (W8A8), not this scalar-MAC GEMV.
    /// - **No quantized copy (flag off)** → f16 `matmul_into`, bit-for-bit
    ///   identical to the pre-quantization decode path.
    ///
    /// Note: with quant on, a sequence decoded alone (`m==1`, Q8 weights) vs
    /// batched (`m>1`, f16 weights) sees slightly different numerics — both
    /// valid for a precision-for-speed option, but greedy output can depend on
    /// batch composition.
    pub fn matmul_decode_into(
        &self,
        out: &mut B::Tensor,
        x: &B::Tensor,
        backend: &B,
    ) -> Result<()> {
        let m = x.shape().first().copied().unwrap_or(0);
        match &self.w_quant {
            Some(wq) if m == 1 => backend.matmul_quant_into(out, x, wq),
            _ => backend.matmul_into(out, x, &self.w_f16),
        }
    }

    /// The FP16 weight's dtype (used to cast attention output to match `wo`).
    pub fn dtype(&self) -> forge_core::DType {
        self.w_f16.dtype()
    }
}
