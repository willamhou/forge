//! Host-side weight quantizers for the quantized-decode path.
//!
//! Currently only Q8_0 (GGUF-compatible `block_q8_0`): 32 elements per block,
//! 34 bytes per block = a 2-byte little-endian f16 scale followed by 32 × i8
//! quantized values. The on-device GEMV kernels (`forge-kernels::quantized`)
//! reinterpret exactly this layout, and `forge-loader`'s `dequantize_q8_0`
//! decodes it — keep the three in sync.

use half::f16;

/// Elements packed into one Q8_0 block.
pub const Q8_0_BLOCK_ELEMS: usize = 32;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (i8 quants).
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Quantize an f16 weight buffer into GGUF-compatible Q8_0 blocks.
///
/// Per 32-element block: `scale = amax / 127` where `amax = max(|x_i|)`;
/// `q_i = round(x_i / scale)` clamped to `[-127, 127]`. A zero block
/// (`amax == 0`) yields `scale = 0` and all-zero quants. Dequant is
/// `x_i = scale * q_i`.
///
/// The block format requires the element count to be a multiple of 32; all
/// Qwen3 weight dims (hidden 2560, intermediate 9728, head/kv projections,
/// vocab) satisfy this, so we assert rather than pad.
pub fn quantize_q8_0(weights: &[f16]) -> Vec<u8> {
    assert!(
        weights.len().is_multiple_of(Q8_0_BLOCK_ELEMS),
        "quantize_q8_0: weight count {} is not a multiple of {}",
        weights.len(),
        Q8_0_BLOCK_ELEMS
    );

    let n_blocks = weights.len() / Q8_0_BLOCK_ELEMS;
    let mut out = Vec::with_capacity(n_blocks * Q8_0_BLOCK_BYTES);

    for block in weights.chunks_exact(Q8_0_BLOCK_ELEMS) {
        // amax over the block in f32 (matches the dequant precision domain).
        let amax = block
            .iter()
            .map(|x| x.to_f32().abs())
            .fold(0.0f32, f32::max);

        if amax == 0.0 {
            // All-zero block: zero scale + zero quants.
            out.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
            out.extend(std::iter::repeat_n(0u8, Q8_0_BLOCK_ELEMS));
            continue;
        }

        let scale = amax / 127.0;
        let inv_scale = 1.0 / scale;

        // Store the scale as f16 (GGUF stores delta as f16).
        let scale_f16 = f16::from_f32(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        for x in block {
            let q = (x.to_f32() * inv_scale).round();
            let q = q.clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference dequant identical to `forge-loader`'s `dequantize_q8_0`,
    /// used to check round-trip error is bounded by the per-block scale.
    fn dequant_q8_0(bytes: &[u8], n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for block in bytes.chunks_exact(Q8_0_BLOCK_BYTES) {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            for &q in &block[2..] {
                out.push(scale * (q as i8) as f32);
            }
        }
        out.truncate(n);
        out
    }

    #[test]
    fn q8_0_layout_and_roundtrip() {
        let n = 32 * 5;
        let weights: Vec<f16> = (0..n)
            .map(|i| f16::from_f32(((i as f32) * 0.013).sin() * 3.0))
            .collect();
        let bytes = quantize_q8_0(&weights);
        assert_eq!(bytes.len(), (n / 32) * 34);

        let deq = dequant_q8_0(&bytes, n);
        // Per element error must be within half a quantization step (scale/2)
        // plus f16 rounding of the scale; bound loosely by the global amax.
        let amax = weights.iter().map(|x| x.to_f32().abs()).fold(0.0, f32::max);
        let tol = amax / 127.0 * 0.6;
        for (i, (w, d)) in weights.iter().zip(&deq).enumerate() {
            let e = (w.to_f32() - d).abs();
            assert!(e <= tol, "elem {i}: |{} - {}| = {e} > {tol}", w.to_f32(), d);
        }
    }

    #[test]
    fn q8_0_zero_block() {
        let weights = vec![f16::from_f32(0.0); 32];
        let bytes = quantize_q8_0(&weights);
        assert_eq!(bytes.len(), 34);
        // scale == 0 and all quants 0.
        assert_eq!(&bytes[0..2], &f16::from_f32(0.0).to_le_bytes());
        assert!(bytes[2..].iter().all(|&b| b == 0));
    }
}
