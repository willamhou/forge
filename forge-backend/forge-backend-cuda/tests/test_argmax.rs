//! GPU greedy-sampling (`Backend::argmax`) correctness: per-row argmax,
//! highest-index tie-break (matching the CPU sampler's `max_by`), the F16
//! path, and parity with a plain reference reduction over a wide vocab.

use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, DType};

/// Reference per-row argmax with highest-index tie-break (`>=`), the same
/// contract as `Backend::argmax` and the CPU sampler's `Iterator::max_by`.
fn ref_argmax(data: &[f32], rows: usize, cols: usize) -> Vec<u32> {
    (0..rows)
        .map(|r| {
            let row = &data[r * cols..(r + 1) * cols];
            let mut best = f32::NEG_INFINITY;
            let mut best_i = 0u32;
            for (i, &v) in row.iter().enumerate() {
                if v >= best {
                    best = v;
                    best_i = i as u32;
                }
            }
            best_i
        })
        .collect()
}

#[test]
fn argmax_single_row_f32() {
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[0.1, 0.5, 0.2, 0.9, 0.3], &[1, 5])
        .unwrap();
    assert_eq!(backend.argmax(&logits).unwrap(), vec![3]);
}

#[test]
fn argmax_1d_treated_as_one_row() {
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend.copy_from_host_f32(&[3.0, 1.0, 7.0, 2.0], &[4]).unwrap();
    assert_eq!(backend.argmax(&logits).unwrap(), vec![2]);
}

#[test]
fn argmax_batched_rows_f32() {
    let backend = CudaBackend::new(0).unwrap();
    let data = [
        1.0, 2.0, 9.0, 0.0, // row 0 → 2
        5.0, 1.0, 2.0, 3.0, // row 1 → 0
        0.1, 0.2, 0.3, 4.0, // row 2 → 3
    ];
    let logits = backend.copy_from_host_f32(&data, &[3, 4]).unwrap();
    assert_eq!(backend.argmax(&logits).unwrap(), vec![2, 0, 3]);
}

#[test]
fn argmax_tie_break_picks_highest_index() {
    // Equal maxima at indices 1 and 3 → highest index (3) wins, matching the
    // CPU sampler's `Iterator::max_by` (returns the last maximum).
    let backend = CudaBackend::new(0).unwrap();
    let logits = backend
        .copy_from_host_f32(&[0.0, 5.0, 0.0, 5.0, 1.0], &[1, 5])
        .unwrap();
    assert_eq!(backend.argmax(&logits).unwrap(), vec![3]);
}

#[test]
fn argmax_large_vocab_parity_with_reference() {
    // Wider than one block (block_dim caps at 256) to exercise the strided
    // per-thread scan + shared-memory reduction over many iterations.
    let rows = 4usize;
    let cols = 32_000usize;
    let mut data = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] = ((r * 7 + c * 13) % 1000) as f32 * 0.01;
        }
        // Force a unique, unambiguous maximum at a known column.
        let peak = (r * 4099 + 17) % cols;
        data[r * cols + peak] = 100.0 + r as f32;
    }
    let gpu = CudaBackend::new(0).unwrap();
    let g = gpu.copy_from_host_f32(&data, &[rows, cols]).unwrap();
    assert_eq!(gpu.argmax(&g).unwrap(), ref_argmax(&data, rows, cols));
}

#[test]
fn argmax_f16_matches_f32() {
    let backend = CudaBackend::new(0).unwrap();
    let data = [1.0f32, 2.0, 9.0, 0.0, 5.0, 1.0, 2.0, 3.0];
    let f32_t = backend.copy_from_host_f32(&data, &[2, 4]).unwrap();
    let f16_t = backend.cast(&f32_t, DType::F16).unwrap();
    assert_eq!(
        backend.argmax(&f16_t).unwrap(),
        backend.argmax(&f32_t).unwrap()
    );
}
