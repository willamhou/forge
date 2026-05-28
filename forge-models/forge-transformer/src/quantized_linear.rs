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
    /// - With a quantized copy: `matmul_quant_into(out[m, out], x[m, in],
    ///   w_quant[out, in])`. This is the launch-only, static-weight decode GEMV
    ///   — capture-safe (the quant weight is uploaded once at load and never
    ///   moves or grows).
    /// - Without one (flag off): falls back to the FP16 `matmul_into(out, x,
    ///   w_f16)`, bit-for-bit identical to the pre-quantization decode path.
    pub fn matmul_decode_into(
        &self,
        out: &mut B::Tensor,
        x: &B::Tensor,
        backend: &B,
    ) -> Result<()> {
        match &self.w_quant {
            Some(wq) => backend.matmul_quant_into(out, x, wq),
            None => backend.matmul_into(out, x, &self.w_f16),
        }
    }

    /// The FP16 weight's dtype (used to cast attention output to match `wo`).
    pub fn dtype(&self) -> forge_core::DType {
        self.w_f16.dtype()
    }
}
