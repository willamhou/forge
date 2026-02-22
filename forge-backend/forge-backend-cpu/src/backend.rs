extern crate openblas_src;

use forge_core::{Backend, DType, ForgeError, Result, Tensor};

use crate::tensor::CpuTensor;

/// CPU backend for Forge inference engine.
///
/// All data lives in host memory as `Vec<f32>` wrapped in `Arc`.
/// Uses OpenBLAS for matrix multiplication; all other ops are pure Rust.
#[derive(Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
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

fn validate_same_shape(a: &CpuTensor, b: &CpuTensor) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(ForgeError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;

    fn name(&self) -> &str {
        "cpu"
    }

    fn device_count(&self) -> usize {
        1
    }

    // ── Allocation ──────────────────────────────────────────────

    fn allocate(&self, shape: &[usize], _dtype: DType) -> Result<CpuTensor> {
        // CPU backend stores everything as f32 internally, regardless of requested dtype.
        let numel: usize = shape.iter().product();
        Ok(CpuTensor::new(vec![0.0; numel], shape.to_vec()))
    }

    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<CpuTensor> {
        self.allocate(shape, dtype)
    }

    // ── Data transfer ───────────────────────────────────────────

    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        Ok(CpuTensor::new(data.to_vec(), shape.to_vec()))
    }

    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        Ok(CpuTensor::new(f32_data, shape.to_vec()))
    }

    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<CpuTensor> {
        validate_shape(data.len(), shape)?;
        let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        Ok(CpuTensor::new(f32_data, shape.to_vec()))
    }

    fn copy_to_host_f32(&self, tensor: &CpuTensor) -> Result<Vec<f32>> {
        Ok(tensor.data().to_vec())
    }

    // ── Synchronization ─────────────────────────────────────────

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    // ── Compute ops ────────────────────────────────────────────

    fn matmul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "matmul requires 2D tensors".into(),
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
        if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
            return Err(ForgeError::InvalidArgument(
                "matmul dimensions exceed i32::MAX (BLAS limitation)".into(),
            ));
        }
        let mut c = vec![0.0f32; m * n];
        unsafe {
            cblas::sgemm(
                cblas::Layout::RowMajor,
                cblas::Transpose::None,
                cblas::Transpose::None,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.data(),
                k as i32,
                b.data(),
                n as i32,
                0.0,
                &mut c,
                n as i32,
            );
        }
        Ok(CpuTensor::new(c, vec![m, n]))
    }

    fn add(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        validate_same_shape(a, b)?;
        let data: Vec<f32> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x + y)
            .collect();
        Ok(CpuTensor::new(data, a.shape().to_vec()))
    }

    fn mul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        validate_same_shape(a, b)?;
        let data: Vec<f32> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x * y)
            .collect();
        Ok(CpuTensor::new(data, a.shape().to_vec()))
    }

    fn mul_scalar(&self, a: &CpuTensor, scalar: f32) -> Result<CpuTensor> {
        let data: Vec<f32> = a.data().iter().map(|x| x * scalar).collect();
        Ok(CpuTensor::new(data, a.shape().to_vec()))
    }

    fn silu(&self, a: &CpuTensor) -> Result<CpuTensor> {
        let data: Vec<f32> = a.data().iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        Ok(CpuTensor::new(data, a.shape().to_vec()))
    }

    fn fused_silu_mul(&self, gate: &CpuTensor, up: &CpuTensor) -> Result<CpuTensor> {
        validate_same_shape(gate, up)?;
        let g = gate.data();
        let u = up.data();
        let data: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui)
            .collect();
        Ok(CpuTensor::new(data, gate.shape().to_vec()))
    }

    fn rms_norm(&self, x: &CpuTensor, weight: &CpuTensor, eps: f32) -> Result<CpuTensor> {
        let shape = x.shape();
        let cols = *shape.last().unwrap();
        if weight.len() != cols {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![cols],
                got: weight.shape().to_vec(),
            });
        }
        let rows = x.len() / cols;
        let src = x.data();
        let w = weight.data();
        let mut out = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let row_data = &src[row * cols..(row + 1) * cols];
            let ss: f32 = row_data.iter().map(|v| v * v).sum();
            let rms = (ss / cols as f32 + eps).sqrt().recip();
            for col in 0..cols {
                out[row * cols + col] = row_data[col] * rms * w[col];
            }
        }
        Ok(CpuTensor::new(out, shape.to_vec()))
    }

    fn rope(
        &self,
        x: &CpuTensor,
        freqs_cos: &CpuTensor,
        freqs_sin: &CpuTensor,
    ) -> Result<CpuTensor> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(ForgeError::InvalidArgument(
                "rope expects 4D tensor [batch, seq_len, heads, head_dim]".into(),
            ));
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];
        if head_dim % 2 != 0 {
            return Err(ForgeError::InvalidArgument(
                "rope requires even head_dim".into(),
            ));
        }
        let half_dim = head_dim / 2;
        let expected_freq_len = seq_len * half_dim;
        if freqs_cos.len() < expected_freq_len || freqs_sin.len() < expected_freq_len {
            return Err(ForgeError::InvalidArgument(format!(
                "freq tensors need at least {} elements, got cos={} sin={}",
                expected_freq_len,
                freqs_cos.len(),
                freqs_sin.len()
            )));
        }
        let src = x.data();
        let cos = freqs_cos.data();
        let sin = freqs_sin.data();
        let mut out = vec![0.0f32; src.len()];
        for b in 0..batch {
            for pos in 0..seq_len {
                for head in 0..num_heads {
                    let base = b * seq_len * num_heads * head_dim
                        + pos * num_heads * head_dim
                        + head * head_dim;
                    for h in 0..half_dim {
                        let x0 = src[base + h];
                        let x1 = src[base + h + half_dim];
                        let cos_val = cos[pos * half_dim + h];
                        let sin_val = sin[pos * half_dim + h];
                        out[base + h] = x0 * cos_val - x1 * sin_val;
                        out[base + h + half_dim] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }
        Ok(CpuTensor::new(out, shape.to_vec()))
    }

    fn softmax(&self, x: &CpuTensor, dim: i32) -> Result<CpuTensor> {
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
        let src = x.data();
        let mut out = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let row_data = &src[row * cols..(row + 1) * cols];
            let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for col in 0..cols {
                let v = (row_data[col] - max_val).exp();
                out[row * cols + col] = v;
                sum += v;
            }
            for col in 0..cols {
                out[row * cols + col] /= sum;
            }
        }
        Ok(CpuTensor::new(out, shape.to_vec()))
    }

    fn embedding(&self, weight: &CpuTensor, indices: &[u32]) -> Result<CpuTensor> {
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(ForgeError::InvalidArgument(
                "embedding weight must be 2D [vocab_size, embedding_dim]".into(),
            ));
        }
        let vocab_size = w_shape[0];
        let dim = w_shape[1];
        let mut out = Vec::with_capacity(indices.len() * dim);
        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(ForgeError::InvalidArgument(format!(
                    "embedding index {idx} out of range (vocab_size={vocab_size})"
                )));
            }
            let start = idx as usize * dim;
            out.extend_from_slice(&weight.data()[start..start + dim]);
        }
        Ok(CpuTensor::new(out, vec![indices.len(), dim]))
    }

    fn reshape(&self, x: &CpuTensor, shape: &[usize]) -> Result<CpuTensor> {
        let numel: usize = shape.iter().product();
        if numel != x.len() {
            return Err(ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: x.shape().to_vec(),
            });
        }
        Ok(CpuTensor {
            data: x.data.clone(),
            shape: shape.to_vec(),
            dtype: x.dtype,
        })
    }

    fn transpose(&self, x: &CpuTensor, dim0: usize, dim1: usize) -> Result<CpuTensor> {
        let shape = x.shape();
        if shape.len() != 2 || !((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0)) {
            return Err(ForgeError::InvalidArgument(
                "transpose currently only supports 2D tensors with dims (0,1)".into(),
            ));
        }
        let rows = shape[0];
        let cols = shape[1];
        let src = x.data();
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = src[r * cols + c];
            }
        }
        Ok(CpuTensor::new(out, vec![cols, rows]))
    }

    fn cat(&self, tensors: &[&CpuTensor], dim: usize) -> Result<CpuTensor> {
        if tensors.is_empty() {
            return Err(ForgeError::InvalidArgument("empty tensor list".into()));
        }
        if dim != 0 {
            return Err(ForgeError::InvalidArgument(
                "cat currently only supports dim=0".into(),
            ));
        }
        let ndim = tensors[0].shape().len();
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
        let mut total_first_dim = 0;
        let total_len: usize = tensors.iter().map(|t| t.len()).sum();
        let mut all_data = Vec::with_capacity(total_len);
        for t in tensors {
            total_first_dim += t.shape()[0];
            all_data.extend_from_slice(t.data());
        }
        let mut out_shape = tensors[0].shape().to_vec();
        out_shape[0] = total_first_dim;
        Ok(CpuTensor::new(all_data, out_shape))
    }

    fn slice_rows(
        &self,
        tensor: &CpuTensor,
        start_row: usize,
        num_rows: usize,
    ) -> Result<CpuTensor> {
        let shape = tensor.shape();
        let cols: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            1
        };
        let offset = start_row * cols;
        let len = num_rows * cols;
        let data = tensor.data()[offset..offset + len].to_vec();
        let mut out_shape = shape.to_vec();
        out_shape[0] = num_rows;
        Ok(CpuTensor::new(data, out_shape))
    }

    fn split_qkv(
        &self,
        qkv: &CpuTensor,
        q_size: usize,
        kv_size: usize,
    ) -> Result<(CpuTensor, CpuTensor, CpuTensor)> {
        let rows = qkv.shape()[0];
        let total_cols = q_size + 2 * kv_size;
        if qkv.len() != rows * total_cols {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![rows, total_cols],
                got: qkv.shape().to_vec(),
            });
        }
        let data = qkv.data();
        let mut q = Vec::with_capacity(rows * q_size);
        let mut k = Vec::with_capacity(rows * kv_size);
        let mut v = Vec::with_capacity(rows * kv_size);
        for r in 0..rows {
            let row = &data[r * total_cols..(r + 1) * total_cols];
            q.extend_from_slice(&row[..q_size]);
            k.extend_from_slice(&row[q_size..q_size + kv_size]);
            v.extend_from_slice(&row[q_size + kv_size..]);
        }
        Ok((
            CpuTensor::new(q, vec![rows, q_size]),
            CpuTensor::new(k, vec![rows, kv_size]),
            CpuTensor::new(v, vec![rows, kv_size]),
        ))
    }

    fn fused_residual_rms_norm(
        &self,
        x: &CpuTensor,
        residual: &CpuTensor,
        weight: &CpuTensor,
        eps: f32,
    ) -> Result<(CpuTensor, CpuTensor)> {
        let sum = self.add(x, residual)?;
        let normed = self.rms_norm(&sum, weight, eps)?;
        Ok((normed, sum))
    }

    fn cast(&self, x: &CpuTensor, _dtype: DType) -> Result<CpuTensor> {
        // CPU backend stores everything as f32 internally, so cast is a no-op.
        Ok(x.clone())
    }
}
