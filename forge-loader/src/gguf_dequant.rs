//! GGUF dequantization routines.
//!
//! Converts quantized weight data to f16 for inference.

use forge_core::{ForgeError, Result};
use half::f16;

/// Dequantize Q8_0 data to f16.
///
/// Q8_0 block layout (34 bytes per 32 elements):
/// - 2 bytes: f16 scale factor (delta)
/// - 32 bytes: 32 × i8 quantized values
///
/// Reconstruction: `value[i] = delta * quant[i]`
pub fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f16>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8 × 32)

    let n_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let expected_bytes = n_blocks * BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(ForgeError::ModelLoad(format!(
            "Q8_0: expected at least {expected_bytes} bytes for {n_elements} elements, got {}",
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];

        // First 2 bytes: f16 scale
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = f16::from_bits(scale_bits).to_f32();

        // Next 32 bytes: i8 quantized values
        let quants = &block[2..BLOCK_BYTES];

        let remaining = n_elements - output.len();
        let count = BLOCK_SIZE.min(remaining);

        for i in 0..count {
            let q = quants[i] as i8;
            let val = scale * q as f32;
            output.push(f16::from_f32(val));
        }
    }

    Ok(output)
}

/// Dequantize Q4_K (Q4_K_M) data to f16.
///
/// Q4_K block layout (144 bytes per 256 elements):
/// - 2 bytes: f16 d (super-block scale)
/// - 2 bytes: f16 dmin (super-block min)
/// - 12 bytes: scales_and_mins (packed 6-bit scales + 6-bit mins for 8 sub-blocks)
/// - 128 bytes: 256 × 4-bit quantized values (packed as nibbles)
///
/// Each super-block contains 8 sub-blocks of 32 elements each.
/// Sub-block reconstruction: `value[i] = d * sc[j] * q[i] - dmin * m[j]`
/// where j = sub-block index, sc = sub-block scale, m = sub-block min.
pub fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f16>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;
    const SUB_BLOCK_SIZE: usize = 32;
    const N_SUB_BLOCKS: usize = 8;

    let n_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let expected_bytes = n_blocks * BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(ForgeError::ModelLoad(format!(
            "Q4_K: expected at least {expected_bytes} bytes for {n_elements} elements, got {}",
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];

        // Super-block scale and min
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();

        // Unpack sub-block scales and mins from the 12-byte packed section.
        // The packing format stores 6-bit values for 8 scales and 8 mins.
        let scales_section = &block[4..16];
        let (scales, mins) = unpack_q4_k_scales(scales_section);

        // Quantized nibbles section: 128 bytes = 256 × 4 bits
        let qs = &block[16..144];

        let remaining = n_elements - output.len();
        let count = BLOCK_SIZE.min(remaining);

        for i in 0..count {
            let sub_block = i / SUB_BLOCK_SIZE;
            let sc = scales[sub_block.min(N_SUB_BLOCKS - 1)] as f32;
            let m = mins[sub_block.min(N_SUB_BLOCKS - 1)] as f32;

            // Extract 4-bit quantized value.
            // GGML Q4_K stores 64 elements per 32 bytes: the first 32 values
            // come from low nibbles, the next 32 from high nibbles.
            let group = i / 64;
            let pos_in_group = i % 64;
            let nibble = if pos_in_group < 32 {
                qs[group * 32 + pos_in_group] & 0x0F
            } else {
                (qs[group * 32 + (pos_in_group - 32)] >> 4) & 0x0F
            };

            let val = d * sc * nibble as f32 - dmin * m;
            output.push(f16::from_f32(val));
        }
    }

    Ok(output)
}

/// Unpack 6-bit scales and mins from the 12-byte packed section of a Q4_K block.
///
/// The 12 bytes encode 8 scales and 8 mins, each 6 bits wide.
/// Layout follows the GGML reference implementation:
/// - Bytes 0-3:  low 6 bits of scales[0..4]
/// - Bytes 4-7:  low 6 bits of mins[0..4]
/// - Bytes 8-11: high bits for scales[4..8] and mins[4..8],
///               combined with the top 2 bits from bytes 0-7
fn unpack_q4_k_scales(packed: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    // First 4 scales: low 6 bits from packed[0..4]
    for i in 0..4 {
        scales[i] = packed[i] & 0x3F;
    }
    // First 4 mins: low 6 bits from packed[4..8]
    for i in 0..4 {
        mins[i] = packed[4 + i] & 0x3F;
    }
    // Last 4 scales: 4 bits from packed[8..12] (low) + 2 bits from packed[0..4] >> 6 (high)
    // Matches GGML get_scale_min_k4: d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
    for i in 0..4 {
        scales[4 + i] = (packed[8 + i] & 0x0F) | ((packed[i] >> 6) << 4);
    }
    // Last 4 mins: 4 bits from packed[8..12] >> 4 (low) + 2 bits from packed[4..8] >> 6 (high)
    // Matches GGML get_scale_min_k4: m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
    for i in 0..4 {
        mins[4 + i] = (packed[8 + i] >> 4) | ((packed[4 + i] >> 6) << 4);
    }

    (scales, mins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Build a single Q8_0 block: scale=1.0, quants=[0, 1, 2, ..., 31]
        let mut block = Vec::new();
        let scale = f16::from_f32(1.0);
        block.extend_from_slice(&scale.to_bits().to_le_bytes());
        for i in 0..32u8 {
            block.push(i);
        }
        assert_eq!(block.len(), 34);

        let result = dequantize_q8_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);

        // value[i] = 1.0 * i
        assert_eq!(result[0].to_f32(), 0.0);
        assert!((result[1].to_f32() - 1.0).abs() < 0.01);
        assert!((result[31].to_f32() - 31.0).abs() < 0.1);
    }

    #[test]
    fn test_dequantize_q8_0_negative() {
        let mut block = Vec::new();
        let scale = f16::from_f32(0.5);
        block.extend_from_slice(&scale.to_bits().to_le_bytes());
        // Put -10 as i8 in the first quant
        block.push((-10i8) as u8);
        for _ in 1..32 {
            block.push(0);
        }

        let result = dequantize_q8_0(&block, 32).unwrap();
        // value[0] = 0.5 * (-10) = -5.0
        assert!((result[0].to_f32() - (-5.0)).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_q8_0_insufficient_data() {
        let block = vec![0u8; 10]; // too short
        assert!(dequantize_q8_0(&block, 32).is_err());
    }

    #[test]
    fn test_dequantize_q4_k_basic() {
        // Build a minimal Q4_K block (144 bytes for 256 elements)
        let mut block = vec![0u8; 144];
        // d = 1.0 (f16)
        let d = f16::from_f32(1.0);
        block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
        // dmin = 0.0
        block[2..4].copy_from_slice(&f16::from_f32(0.0).to_bits().to_le_bytes());
        // scales_and_mins: set first scale to 1 (6-bit value)
        block[4] = 1; // scale[0] = 1

        // First nibble = 5 → value should be d * 1 * 5 - 0 = 5.0
        block[16] = 0x05; // low nibble = 5, high nibble = 0

        let result = dequantize_q4_k(&block, 256).unwrap();
        assert_eq!(result.len(), 256);
        // First element: d(1.0) * scale(1) * nibble(5) - dmin(0) * min(0) = 5.0
        assert!((result[0].to_f32() - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_dequantize_q4_k_insufficient_data() {
        let block = vec![0u8; 10]; // too short
        assert!(dequantize_q4_k(&block, 256).is_err());
    }

    #[test]
    fn test_dequantize_q8_0_two_blocks() {
        // 64 elements = 2 blocks of 32
        let mut data = Vec::new();
        for block_idx in 0..2 {
            let scale = f16::from_f32((block_idx + 1) as f32);
            data.extend_from_slice(&scale.to_bits().to_le_bytes());
            for i in 0..32u8 {
                data.push(i);
            }
        }

        let result = dequantize_q8_0(&data, 64).unwrap();
        assert_eq!(result.len(), 64);
        // Block 0: scale=1.0, quant[0]=0 → 0.0
        assert_eq!(result[0].to_f32(), 0.0);
        // Block 1: scale=2.0, quant[0]=0 → 0.0
        assert_eq!(result[32].to_f32(), 0.0);
        // Block 1: scale=2.0, quant[1]=1 → 2.0
        assert!((result[33].to_f32() - 2.0).abs() < 0.01);
    }
}
