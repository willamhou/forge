use crate::tensor::Tensor;
use crate::{DType, Result};

/// Counter-based uniform in (0, 1) keyed on `(seed, step, row, col)`, bit-for-bit
/// identical to `forge_uniform` in `forge-kernels/src/sampling.rs` so the CPU
/// default and CUDA `sample_gumbel` paths draw the same numbers.
fn forge_uniform(seed: u64, step: u32, row: u32, col: u32) -> f32 {
    let mut x = seed
        ^ (step as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (((row as u64) << 32) | (col as u64));
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    let hi = (x >> 40) as u32; // top 24 bits
    (hi as f32 + 0.5) * (1.0 / 16_777_216.0)
}

/// Gumbel noise `-log(-log u)` for the Gumbel-max sampling trick. `u` is
/// clamped into the open interval (same clamp as `forge_gumbel` in
/// `forge-kernels`): at the top of the range the f32 uniform can round to
/// exactly 1.0, which would make the noise `+inf` and pin that token.
fn gumbel_noise(seed: u64, step: u32, row: u32, col: u32) -> f32 {
    let u = forge_uniform(seed, step, row, col)
        .max(1.0e-7_f32)
        .min(1.0_f32 - 1.0e-7_f32);
    -(-(u.ln())).ln()
}

/// Per-step decode inputs the engine stages into backend-persistent scratch
/// buffers before a captured/replayed decode forward (see
/// [`Backend::stage_decode_inputs`]). On replay the captured graph re-reads
/// these scratches, so they must hold the *current* step's data — staging
/// runs every step, outside the captured region.
///
/// All slices are host-side; the backend uploads them to stable device
/// addresses. `slot_mapping` is staged separately via
/// [`Backend::stage_slot_mapping`] because the KV scatter consumes it directly.
pub struct DecodeStageInputs<'a> {
    /// Flattened token ids, one per sequence (`[batch]`).
    pub token_indices: &'a [u32],
    /// Gathered RoPE cos table rows for this step's positions
    /// (`[batch * head_dim/2]`).
    pub rope_cos: &'a [f32],
    /// Gathered RoPE sin table rows, same shape as `rope_cos`.
    pub rope_sin: &'a [f32],
    /// Row-major paged block tables (`[batch * max_blocks_per_seq]`, `-1` pad).
    pub block_tables: &'a [i32],
    /// Per-sequence KV length (`[batch]`).
    pub kv_lens: &'a [i32],
}

/// Backend-specific captured-graph dispatcher for batched decode.
///
/// CPU and other graph-less backends have none ([`Backend::make_decode_graph_runner`]
/// returns `None`); CUDA wraps a per-bucket capture/replay cache. The engine
/// drives it once per decode step.
pub trait DecodeGraphDispatch: Send {
    /// Run a decode forward of `batch_size`:
    /// - **bucket cache hit** → replay the captured graph (`fwd` is NOT called);
    /// - **bucket cold** → capture `fwd` into a graph, then launch it once;
    /// - **non-bucket size** → run `fwd` directly (uncaptured fallback).
    ///
    /// `fwd` must issue only capture-safe work (no host→device copies, stable
    /// device pointers) — i.e. the pre-staged `Model::compute_decode`.
    fn dispatch(&mut self, batch_size: u32, fwd: &mut dyn FnMut() -> Result<()>) -> Result<()>;

    /// Drop all captured graphs. Called by the engine when the backend's
    /// [`Backend::decode_capture_epoch`] changes (a baked device pointer moved).
    fn invalidate_all(&mut self);
}

pub trait Backend: Send + Sync + 'static {
    type Tensor: Tensor;

    fn name(&self) -> &str;
    fn device_count(&self) -> usize;

    // Allocation
    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;
    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;

    // Data transfer
    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_to_host_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>>;

    // Synchronization
    fn synchronize(&self) -> Result<()>;

    // Core ops
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// Broadcast bias add: `out[r, c] = x[r, c] + bias[c]`.
    ///
    /// `x` is `[rows, cols]` (2D), `bias` is `[cols]`. Used for the QKV
    /// projection bias in Qwen2-family models (Llama has no QKV bias).
    /// Default impl round-trips through host f32; GPU backends override with
    /// a broadcast kernel.
    fn add_bias(&self, x: &Self::Tensor, bias: &Self::Tensor) -> Result<Self::Tensor> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(crate::ForgeError::InvalidArgument(
                "add_bias: x must be 2D [rows, cols]".into(),
            ));
        }
        let cols = shape[1];
        if bias.shape() != [cols] {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: vec![cols],
                got: bias.shape().to_vec(),
            });
        }
        if x.dtype() != crate::DType::F32 {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "add_bias default impl is F32-only (got {:?}); backend must override",
                x.dtype()
            )));
        }
        let mut data = self.copy_to_host_f32(x)?;
        let bias_data = self.copy_to_host_f32(bias)?;
        let rows = shape[0];
        for r in 0..rows {
            for c in 0..cols {
                data[r * cols + c] += bias_data[c];
            }
        }
        self.copy_from_host_f32(&data, shape)
    }

    /// Matrix multiply, writing into a caller-provided output buffer.
    ///
    /// `a`: `[m, k]`, `b`: `[k, n]`, `out`: `[m, n]`; out dtype must match
    /// a/b dtype.
    ///
    /// Used by the engine's persistent-buffer + CUDA-Graph capture path
    /// (Task 5+) where the captured kernel's output arg must point at a
    /// stable device address. CUDA backends override to gemm directly into
    /// `out`'s underlying device slice; the default impl just allocates +
    /// reassigns `*out`, which is correct numerically but does NOT preserve
    /// the tensor's underlying device pointer — fine for non-CUDA backends
    /// (they don't capture graphs).
    ///
    /// **Capture-safety contract**: backends that participate in CUDA Graph
    /// capture (or any equivalent "record kernel arg pointers, replay
    /// later" mechanism) MUST override this method to write in place.
    /// The default impl's `*out = ...` reassignment changes the tensor's
    /// underlying storage on every call and would invalidate any captured
    /// graph that baked the previous pointer. The same contract applies
    /// to every other `_into` method on this trait.
    fn matmul_into(
        &self,
        out: &mut Self::Tensor,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<()> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(crate::ForgeError::InvalidArgument(
                "matmul_into: requires 2D tensors".into(),
            ));
        }
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        if b_shape[0] != k {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: vec![k, n],
                got: b_shape.to_vec(),
            });
        }
        let out_shape = out.shape();
        if out_shape != [m, n] {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: vec![m, n],
                got: out_shape.to_vec(),
            });
        }
        if out.dtype() != a.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "matmul_into: out dtype {:?} != a dtype {:?}",
                out.dtype(),
                a.dtype()
            )));
        }
        *out = self.matmul(a, b)?;
        Ok(())
    }

    /// Element-wise add, writing into a caller-provided output buffer.
    /// Out shape + dtype must match both inputs.
    /// See [`Self::matmul_into`] for the capture-stability contract.
    fn add_into(&self, out: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) -> Result<()> {
        if a.shape() != b.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }
        if out.shape() != a.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != a.dtype() || a.dtype() != b.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "add_into: dtype mismatch out={:?} a={:?} b={:?}",
                out.dtype(),
                a.dtype(),
                b.dtype()
            )));
        }
        *out = self.add(a, b)?;
        Ok(())
    }

    /// RMS normalization, writing into a caller-provided output buffer.
    /// `out` shape + dtype must match `x`. See [`Self::matmul_into`] for the
    /// capture-stability contract.
    fn rms_norm_into(
        &self,
        out: &mut Self::Tensor,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<()> {
        if out.shape() != x.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != x.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "rms_norm_into: out dtype {:?} != x dtype {:?}",
                out.dtype(),
                x.dtype()
            )));
        }
        *out = self.rms_norm(x, weight, eps)?;
        Ok(())
    }

    /// Fused residual-add + RMS normalization, writing into caller-provided
    /// buffers. Both outputs share `x`'s shape + dtype.
    fn fused_residual_rms_norm_into(
        &self,
        normed_out: &mut Self::Tensor,
        residual_out: &mut Self::Tensor,
        x: &Self::Tensor,
        residual: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<()> {
        if x.shape() != residual.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: residual.shape().to_vec(),
            });
        }
        if normed_out.shape() != x.shape() || residual_out.shape() != x.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: normed_out.shape().to_vec(),
            });
        }
        if normed_out.dtype() != x.dtype() || residual_out.dtype() != x.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "fused_residual_rms_norm_into: out dtypes (norm={:?}, residual={:?}) != x dtype {:?}",
                normed_out.dtype(),
                residual_out.dtype(),
                x.dtype()
            )));
        }
        let (n, r) = self.fused_residual_rms_norm(x, residual, weight, eps)?;
        *normed_out = n;
        *residual_out = r;
        Ok(())
    }

    /// Fused SiLU(gate) * up, writing into a caller-provided output buffer.
    /// All three tensors share shape + dtype.
    fn fused_silu_mul_into(
        &self,
        out: &mut Self::Tensor,
        gate: &Self::Tensor,
        up: &Self::Tensor,
    ) -> Result<()> {
        if gate.shape() != up.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: gate.shape().to_vec(),
                got: up.shape().to_vec(),
            });
        }
        if out.shape() != gate.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: gate.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != gate.dtype() || gate.dtype() != up.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "fused_silu_mul_into: dtype mismatch out={:?} gate={:?} up={:?}",
                out.dtype(),
                gate.dtype(),
                up.dtype()
            )));
        }
        *out = self.fused_silu_mul(gate, up)?;
        Ok(())
    }

    /// Embedding lookup, writing into a caller-provided output buffer.
    /// `out` shape: `[indices.len(), embedding_dim]`; `out` dtype = `weight` dtype.
    fn embedding_into(
        &self,
        out: &mut Self::Tensor,
        weight: &Self::Tensor,
        indices: &[u32],
    ) -> Result<()> {
        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(crate::ForgeError::InvalidArgument(
                "embedding_into: weight must be 2D [vocab, embed_dim]".into(),
            ));
        }
        let embedding_dim = weight_shape[1];
        let expected = vec![indices.len(), embedding_dim];
        if out.shape() != expected.as_slice() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected,
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != weight.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "embedding_into: out dtype {:?} != weight dtype {:?}",
                out.dtype(),
                weight.dtype()
            )));
        }
        *out = self.embedding(weight, indices)?;
        Ok(())
    }

    /// In-place QKV split. All three outputs share `qkv`'s dtype; their
    /// shapes are `[rows, q_size]`, `[rows, kv_size]`, `[rows, kv_size]`.
    /// See [`Self::matmul_into`] for the capture-stability contract.
    fn split_qkv_into(
        &self,
        q_out: &mut Self::Tensor,
        k_out: &mut Self::Tensor,
        v_out: &mut Self::Tensor,
        qkv: &Self::Tensor,
        q_size: usize,
        kv_size: usize,
    ) -> Result<()> {
        let shape = qkv.shape();
        if shape.len() != 2 {
            return Err(crate::ForgeError::InvalidArgument(
                "split_qkv_into: qkv must be 2D".into(),
            ));
        }
        let rows = shape[0];
        let total_cols = q_size + 2 * kv_size;
        if shape[1] != total_cols {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: vec![rows, total_cols],
                got: shape.to_vec(),
            });
        }
        for (out, want_cols, label) in [
            (&*q_out, q_size, "q_out"),
            (&*k_out, kv_size, "k_out"),
            (&*v_out, kv_size, "v_out"),
        ] {
            if out.shape() != [rows, want_cols] {
                return Err(crate::ForgeError::ShapeMismatch {
                    expected: vec![rows, want_cols],
                    got: out.shape().to_vec(),
                });
            }
            if out.dtype() != qkv.dtype() {
                return Err(crate::ForgeError::InvalidArgument(format!(
                    "split_qkv_into: {label} dtype {:?} != qkv dtype {:?}",
                    out.dtype(),
                    qkv.dtype()
                )));
            }
        }
        let (q, k, v) = self.split_qkv(qkv, q_size, kv_size)?;
        *q_out = q;
        *k_out = k;
        *v_out = v;
        Ok(())
    }

    /// In-place row-slice. `out` shape `[num_rows, cols..]` must match the
    /// corresponding slice of `tensor`. See [`Self::matmul_into`].
    fn slice_rows_into(
        &self,
        out: &mut Self::Tensor,
        tensor: &Self::Tensor,
        start_row: usize,
        num_rows: usize,
    ) -> Result<()> {
        let in_shape = tensor.shape();
        if in_shape.is_empty() {
            return Err(crate::ForgeError::InvalidArgument(
                "slice_rows_into: input tensor must be non-empty".into(),
            ));
        }
        if start_row + num_rows > in_shape[0] {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "slice_rows_into: start_row {start_row} + num_rows {num_rows} > tensor rows {}",
                in_shape[0]
            )));
        }
        let mut expected = in_shape.to_vec();
        expected[0] = num_rows;
        if out.shape() != expected.as_slice() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected,
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != tensor.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "slice_rows_into: out dtype {:?} != tensor dtype {:?}",
                out.dtype(),
                tensor.dtype()
            )));
        }
        *out = self.slice_rows(tensor, start_row, num_rows)?;
        Ok(())
    }

    /// Dtype cast, writing into a caller-provided output buffer.
    /// `out` shape must match `x`; cast direction implied by `x.dtype()` →
    /// `out.dtype()`. If `out.dtype() == x.dtype()`, this is a copy (no
    /// kernel launch on optimized backends).
    fn cast_into(&self, out: &mut Self::Tensor, x: &Self::Tensor) -> Result<()> {
        if out.shape() != x.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        *out = self.cast(x, out.dtype())?;
        Ok(())
    }
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn mul_scalar(&self, a: &Self::Tensor, scalar: f32) -> Result<Self::Tensor>;
    fn silu(&self, a: &Self::Tensor) -> Result<Self::Tensor>;

    /// Fused SiLU activation and element-wise multiply: out = silu(gate) * up
    fn fused_silu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor) -> Result<Self::Tensor> {
        let activated = self.silu(gate)?;
        self.mul(&activated, up)
    }

    fn rms_norm(&self, x: &Self::Tensor, weight: &Self::Tensor, eps: f32) -> Result<Self::Tensor>;

    /// Fused residual addition + RMSNorm.
    /// Returns (normalized, updated_residual) where:
    ///   updated_residual = x + residual_in
    ///   normalized = rms_norm(updated_residual, weight, eps)
    fn fused_residual_rms_norm(
        &self,
        x: &Self::Tensor,
        residual: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<(Self::Tensor, Self::Tensor)> {
        let sum = self.add(x, residual)?;
        let normed = self.rms_norm(&sum, weight, eps)?;
        Ok((normed, sum))
    }

    /// Split a concatenated QKV tensor [rows, q_size + kv_size + kv_size] into
    /// (Q [rows, q_size], K [rows, kv_size], V [rows, kv_size]).
    fn split_qkv(
        &self,
        qkv: &Self::Tensor,
        q_size: usize,
        kv_size: usize,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor)> {
        let rows = qkv.shape()[0];
        let total_cols = q_size + 2 * kv_size;
        let data = self.copy_to_host_f32(qkv)?;
        let mut q_data = Vec::with_capacity(rows * q_size);
        let mut k_data = Vec::with_capacity(rows * kv_size);
        let mut v_data = Vec::with_capacity(rows * kv_size);
        for r in 0..rows {
            let row = &data[r * total_cols..(r + 1) * total_cols];
            q_data.extend_from_slice(&row[..q_size]);
            k_data.extend_from_slice(&row[q_size..q_size + kv_size]);
            v_data.extend_from_slice(&row[q_size + kv_size..]);
        }
        Ok((
            self.copy_from_host_f32(&q_data, &[rows, q_size])?,
            self.copy_from_host_f32(&k_data, &[rows, kv_size])?,
            self.copy_from_host_f32(&v_data, &[rows, kv_size])?,
        ))
    }

    /// Batched decode attention: N sequences x 1 query token each.
    /// Processes Q@K^T -> softmax -> @V for all sequences.
    ///
    /// `q`: [num_seqs, num_heads * head_dim]
    /// `k_caches`: per-sequence K caches, each [kv_len_i, num_kv_heads * head_dim]
    /// `v_caches`: per-sequence V caches, each [kv_len_i, num_kv_heads * head_dim]
    ///
    /// Returns: [num_seqs, num_heads * head_dim]
    fn batched_decode_attention(
        &self,
        q: &Self::Tensor,
        k_caches: &[Self::Tensor],
        v_caches: &[Self::Tensor],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Self::Tensor> {
        let n = k_caches.len();
        let heads_per_group = num_heads / num_kv_heads;
        let mut seq_outputs = Vec::with_capacity(n);

        for i in 0..n {
            let q_row = self.slice_rows(q, i, 1)?;
            let kv_len = k_caches[i].shape()[0];

            // Reshape for per-head extraction: [seq_len, num_heads, head_dim]
            let q_3d = self.reshape(&q_row, &[1, num_heads, head_dim])?;
            let k_3d = self.reshape(&k_caches[i], &[kv_len, num_kv_heads, head_dim])?;
            let v_3d = self.reshape(&v_caches[i], &[kv_len, num_kv_heads, head_dim])?;

            let mut head_outputs = Vec::with_capacity(num_heads);
            for h in 0..num_heads {
                let kv_h = h / heads_per_group;
                let q_head = self.extract_head(&q_3d, 1, num_heads, head_dim, h)?;
                let k_head = self.extract_head(&k_3d, kv_len, num_kv_heads, head_dim, kv_h)?;
                let v_head = self.extract_head(&v_3d, kv_len, num_kv_heads, head_dim, kv_h)?;

                let k_t = self.transpose(&k_head, 0, 1)?;
                let scores = self.matmul(&q_head, &k_t)?;
                let scores = self.mul_scalar(&scores, scale)?;
                // No causal mask needed for decode (seq_len=1)
                let attn = self.softmax(&scores, -1)?;
                head_outputs.push(self.matmul(&attn, &v_head)?);
            }

            let refs: Vec<&Self::Tensor> = head_outputs.iter().collect();
            let seq_out = self.interleave_heads(&refs, 1, head_dim)?;
            seq_outputs.push(seq_out);
        }

        let refs: Vec<&Self::Tensor> = seq_outputs.iter().collect();
        self.cat(&refs, 0)
    }

    /// Multi-head scaled dot-product attention.
    ///
    /// Q: [1, seq_len, num_heads, head_dim]
    /// K: [1, kv_len, num_kv_heads, head_dim]
    /// V: [1, kv_len, num_kv_heads, head_dim]
    ///
    /// Returns: [seq_len, num_heads * head_dim]
    ///
    /// Default impl: per-head loop (extract_head -> matmul -> softmax -> interleave).
    /// CUDA override: FlashAttention v2 when available.
    fn multi_head_attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<Self::Tensor> {
        let q_shape = q.shape();
        let seq_len = q_shape[1];
        let kv_len = k.shape()[1];
        let heads_per_group = num_heads / num_kv_heads;

        // Reshape to 3D for extract_head: strip batch dim (always 1)
        let q = self.reshape(q, &[seq_len, num_heads, head_dim])?;
        let k = self.reshape(k, &[kv_len, num_kv_heads, head_dim])?;
        let v = self.reshape(v, &[kv_len, num_kv_heads, head_dim])?;

        let mut head_outputs = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let kv_h = h / heads_per_group;

            let q_head = self.extract_head(&q, seq_len, num_heads, head_dim, h)?;
            let k_head = self.extract_head(&k, kv_len, num_kv_heads, head_dim, kv_h)?;
            let v_head = self.extract_head(&v, kv_len, num_kv_heads, head_dim, kv_h)?;

            let k_t = self.transpose(&k_head, 0, 1)?;
            let scores = self.matmul(&q_head, &k_t)?;
            let scores = self.mul_scalar(&scores, scale)?;

            let scores = if is_causal && seq_len > 1 {
                self.apply_causal_mask(&scores, seq_len, kv_len)?
            } else {
                scores
            };

            let attn = self.softmax(&scores, -1)?;
            head_outputs.push(self.matmul(&attn, &v_head)?);
        }

        let refs: Vec<&Self::Tensor> = head_outputs.iter().collect();
        self.interleave_heads(&refs, seq_len, head_dim)
    }

    fn rope(
        &self,
        x: &Self::Tensor,
        freqs_cos: &Self::Tensor,
        freqs_sin: &Self::Tensor,
    ) -> Result<Self::Tensor>;

    /// In-place RoPE driven by host-side cos/sin slices.
    ///
    /// The caller passes the F32 cos/sin freq data as host slices; backends
    /// upload them internally. CUDA backends route the upload through a
    /// persistent device scratch so the kernel arg pointers are stable for
    /// CUDA-Graph capture even though the cos/sin values themselves change
    /// per call (positions differ per decode step).
    ///
    /// `cos_host` / `sin_host` are expected length `[seq_len * head_dim/2]`.
    fn rope_with_host_freqs_into(
        &self,
        out: &mut Self::Tensor,
        x: &Self::Tensor,
        cos_host: &[f32],
        sin_host: &[f32],
    ) -> Result<()> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(crate::ForgeError::InvalidArgument(
                "rope_with_host_freqs_into: x must be rank-4".into(),
            ));
        }
        let seq_len = shape[1];
        let head_dim = shape[3];
        if head_dim % 2 != 0 {
            return Err(crate::ForgeError::InvalidArgument(
                "rope_with_host_freqs_into: head_dim must be even".into(),
            ));
        }
        let half = head_dim / 2;
        let expected = seq_len * half;
        if cos_host.len() != expected || sin_host.len() != expected {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "rope_with_host_freqs_into: cos/sin host slices must be {expected} elements (got cos={}, sin={})",
                cos_host.len(),
                sin_host.len()
            )));
        }
        // Default impl: transient upload. CUDA override uses persistent scratch.
        let cos = self.copy_from_host_f32(cos_host, &[seq_len, half])?;
        let sin = self.copy_from_host_f32(sin_host, &[seq_len, half])?;
        self.rope_into(out, x, &cos, &sin)
    }

    /// In-place RoPE. `out` shape + dtype must match `x` (which is rank-4
    /// `[batch, seq_len, num_heads, head_dim]`). cos/sin are F32 freq tables
    /// of length `[seq_len, head_dim/2]`. See [`Self::matmul_into`] for
    /// capture-stability semantics.
    fn rope_into(
        &self,
        out: &mut Self::Tensor,
        x: &Self::Tensor,
        freqs_cos: &Self::Tensor,
        freqs_sin: &Self::Tensor,
    ) -> Result<()> {
        if out.shape() != x.shape() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != x.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "rope_into: out dtype {:?} != x dtype {:?}",
                out.dtype(),
                x.dtype()
            )));
        }
        *out = self.rope(x, freqs_cos, freqs_sin)?;
        Ok(())
    }
    fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor>;
    fn embedding(&self, weight: &Self::Tensor, indices: &[u32]) -> Result<Self::Tensor>;
    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor>;

    /// In-place reshape — copy `x`'s contents into `out` (whose shape acts
    /// as the target shape; numel must match). On CUDA backends today this
    /// is a `memcpy_dtod` into `out`'s existing device buffer, so the
    /// output pointer stays stable across calls (capture-safe).
    ///
    /// **Why not zero-copy?** CudaTensor wraps an owned CudaSlice, and
    /// cudarc 0.17.8's `CudaSlice::clone` is a device-to-device copy, not
    /// an Arc-share — see `rope_with_host_freqs_into` in CudaBackend for
    /// the same finding. Properly view-based reshape would need
    /// CudaTensor to hold either an `Arc<CudaSlice>` or a `CudaView`
    /// variant, both larger refactors. The memcpy is a per-call cost but
    /// dominated by surrounding matmul/norm work; profile shows it's
    /// noise. The capture-stability benefit dominates.
    fn reshape_into(
        &self,
        out: &mut Self::Tensor,
        x: &Self::Tensor,
        shape: &[usize],
    ) -> Result<()> {
        let want_numel: usize = shape.iter().product();
        if want_numel != x.shape().iter().product::<usize>() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: x.shape().to_vec(),
            });
        }
        if out.shape() != shape {
            return Err(crate::ForgeError::ShapeMismatch {
                expected: shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != x.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "reshape_into: out dtype {:?} != x dtype {:?}",
                out.dtype(),
                x.dtype()
            )));
        }
        *out = self.reshape(x, shape)?;
        Ok(())
    }
    fn transpose(&self, x: &Self::Tensor, dim0: usize, dim1: usize) -> Result<Self::Tensor>;
    fn cat(&self, tensors: &[&Self::Tensor], dim: usize) -> Result<Self::Tensor>;

    /// Cast a tensor to a different dtype. Returns the input unchanged if already the target dtype.
    fn cast(&self, x: &Self::Tensor, dtype: DType) -> Result<Self::Tensor>;

    // ── Paged KV pool primitives ────────────────────────────────
    // Used by PagedKvCache. CUDA backend overrides with device-to-device memcpy
    // so the pool's device address stays stable across writes (prereq for
    // CUDA Graph capture — see docs/plans/2026-05-22-cuda-graphs-plan.md).

    /// Write rows from `src` into the paged pool at the given slot positions.
    ///
    /// `pool` shape: `[num_blocks, block_size, kv_dim]`
    /// `src`  shape: `[num_tokens, kv_dim]`
    /// `slot_mapping[t]` = `block_id_t * block_size + slot_in_block_t` (absolute slot index).
    ///
    /// The pool is updated in place. The default impl round-trips through the
    /// host (correct, but allocates a fresh tensor — defeats the "stable address"
    /// goal); GPU backends override with device-to-device memcpy.
    fn paged_write_kv(
        &self,
        pool: &mut Self::Tensor,
        src: &Self::Tensor,
        slot_mapping: &[i32],
    ) -> Result<()> {
        // Reject non-F32 pools: the host-fallback path round-trips through
        // f32 (no `copy_to_host_f16` / `copy_to_host_bf16` exists on Backend),
        // so an F16/BF16 pool would be silently coerced to F32 and corrupt
        // the model's activation dtype. Backends supporting non-F32 pools
        // (CUDA) must override this method.
        if pool.dtype() != crate::DType::F32 {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "paged_write_kv default impl only supports F32 pools (got {:?}); backend must override for non-F32 dtypes",
                pool.dtype()
            )));
        }
        let pool_shape = pool.shape().to_vec();
        let block_size = pool_shape[1];
        let kv_dim = pool_shape[2];
        let pool_capacity_slots = pool_shape[0] * block_size;
        let mut pool_data = self.copy_to_host_f32(pool)?;
        let src_data = self.copy_to_host_f32(src)?;
        for (t, &slot) in slot_mapping.iter().enumerate() {
            let slot_u = slot as usize;
            if slot_u >= pool_capacity_slots {
                return Err(crate::ForgeError::InvalidArgument(format!(
                    "paged_write_kv: slot {slot_u} out of bounds (capacity {pool_capacity_slots})"
                )));
            }
            let dst_off = slot_u * kv_dim;
            let src_off = t * kv_dim;
            pool_data[dst_off..dst_off + kv_dim]
                .copy_from_slice(&src_data[src_off..src_off + kv_dim]);
        }
        *pool = self.copy_from_host_f32(&pool_data, &pool_shape)?;
        Ok(())
    }

    /// Decode attention against a paged K/V pool.
    ///
    /// - `q`: `[batch, num_heads * head_dim]` — one query token per sequence (decode `q_len = 1`)
    /// - `k_pool` / `v_pool`: `[num_blocks, block_size, num_kv_heads * head_dim]` (device-resident,
    ///   stable address) — see [`Self::paged_write_kv`]
    /// - `block_tables`: row-major `[batch * max_blocks_per_seq]` i32 slice, padding = `-1`.
    ///   Host-side on purpose; the trait keeps i32 dtype out of the tensor type system. CUDA
    ///   backends maintain an internal device scratch tensor that they upload to on each call
    ///   (the upload is small — kilobytes — relative to per-step kernel work; the scratch's
    ///   device address is stable for graph capture).
    /// - `kv_lens`: `[batch]` host i32 slice — current KV length per sequence
    ///
    /// Returns: `[batch, num_heads * head_dim]`
    ///
    /// Default impl gathers each sequence's K/V via [`Self::paged_gather_kv`] then dispatches
    /// to [`Self::batched_decode_attention`]. Correct but slow (one gather per sequence per
    /// call); GPU backends override with a single fused paged-attention kernel (Task 2.2).
    fn paged_attention(
        &self,
        q: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_tables: &[i32],
        kv_lens: &[i32],
        max_blocks_per_seq: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Self::Tensor> {
        let batch_size = kv_lens.len();
        if block_tables.len() != batch_size * max_blocks_per_seq {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "paged_attention: block_tables.len()={} != batch_size={batch_size} * max_blocks_per_seq={max_blocks_per_seq}",
                block_tables.len()
            )));
        }
        let pool_shape = k_pool.shape();
        if pool_shape.len() != 3 {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "paged_attention: k_pool must be rank-3, got {pool_shape:?}"
            )));
        }
        if v_pool.shape() != pool_shape {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "paged_attention: k_pool and v_pool shape mismatch ({pool_shape:?} vs {:?})",
                v_pool.shape()
            )));
        }
        if num_kv_heads == 0 {
            return Err(crate::ForgeError::InvalidArgument(
                "paged_attention: num_kv_heads must be > 0".into(),
            ));
        }
        let kv_dim_expected = num_kv_heads * head_dim;
        if pool_shape[2] != kv_dim_expected {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "paged_attention: pool kv_dim {} != num_kv_heads {} * head_dim {} = {}",
                pool_shape[2], num_kv_heads, head_dim, kv_dim_expected
            )));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "paged_attention: num_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}"
            )));
        }

        let num_blocks = pool_shape[0];
        let block_size = pool_shape[1];

        let mut k_caches = Vec::with_capacity(batch_size);
        let mut v_caches = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let kv_len = kv_lens[b];
            if kv_len < 0 {
                return Err(crate::ForgeError::InvalidArgument(format!(
                    "paged_attention: kv_lens[{b}] = {kv_len} < 0"
                )));
            }
            let kv_len = kv_len as usize;
            // Per-seq: must have enough block-table entries to cover kv_len.
            let blocks_needed = kv_len.div_ceil(block_size);
            if blocks_needed > max_blocks_per_seq {
                return Err(crate::ForgeError::InvalidArgument(format!(
                    "paged_attention: seq[{b}] kv_len={kv_len} needs {blocks_needed} blocks but max_blocks_per_seq={max_blocks_per_seq}"
                )));
            }
            let row_start = b * max_blocks_per_seq;
            let row = &block_tables[row_start..row_start + max_blocks_per_seq];
            // Per-seq: the first blocks_needed entries must be valid (no -1, in pool range).
            for (j, &id) in row.iter().take(blocks_needed).enumerate() {
                if id < 0 || (id as usize) >= num_blocks {
                    return Err(crate::ForgeError::InvalidArgument(format!(
                        "paged_attention: seq[{b}] block_tables[{j}]={id} invalid (num_blocks={num_blocks})"
                    )));
                }
            }
            let block_ids: Vec<usize> = row
                .iter()
                .take(blocks_needed)
                .map(|&id| id as usize)
                .collect();
            k_caches.push(self.paged_gather_kv(k_pool, &block_ids, kv_len)?);
            v_caches.push(self.paged_gather_kv(v_pool, &block_ids, kv_len)?);
        }

        self.batched_decode_attention(
            q,
            &k_caches,
            &v_caches,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
    }

    /// Decode attention against a paged K/V pool, writing into a caller-
    /// provided output buffer.
    ///
    /// Same algorithm as [`Self::paged_attention`] but `out` is supplied by
    /// the caller — required for CUDA Graph capture (the captured kernel
    /// arg's device pointer must be stable across replays).
    ///
    /// `out` shape: `[batch, num_heads * head_dim]`; dtype matches `q`.
    ///
    /// Default impl: calls `paged_attention` to allocate the result, then
    /// copies via `reshape_into` into `out` (non-zero-cost on CUDA today;
    /// future iteration can override with an in-place kernel that skips
    /// the intermediate alloc). The CUDA backend overrides directly.
    #[allow(clippy::too_many_arguments)]
    fn paged_attention_into(
        &self,
        out: &mut Self::Tensor,
        q: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_tables: &[i32],
        kv_lens: &[i32],
        max_blocks_per_seq: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let batch = kv_lens.len();
        let expected = vec![batch, num_heads * head_dim];
        if out.shape() != expected.as_slice() {
            return Err(crate::ForgeError::ShapeMismatch {
                expected,
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != q.dtype() {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "paged_attention_into: out dtype {:?} != q dtype {:?}",
                out.dtype(),
                q.dtype()
            )));
        }
        let result = self.paged_attention(
            q,
            k_pool,
            v_pool,
            block_tables,
            kv_lens,
            max_blocks_per_seq,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )?;
        let expected_again = vec![batch, num_heads * head_dim];
        self.reshape_into(out, &result, &expected_again)
    }

    /// Gather `total_tokens` tokens from the given block IDs into a contiguous tensor.
    ///
    /// `pool` shape: `[num_blocks, block_size, kv_dim]`
    /// Returns shape: `[total_tokens, kv_dim]`
    ///
    /// Tokens are read in order from `block_ids[0]` (up to `block_size` tokens), then
    /// `block_ids[1]`, etc., stopping when `total_tokens` have been collected.
    fn paged_gather_kv(
        &self,
        pool: &Self::Tensor,
        block_ids: &[usize],
        total_tokens: usize,
    ) -> Result<Self::Tensor> {
        // Same restriction as `paged_write_kv`: default impl is F32-only.
        // Non-F32 backends must override or the returned tensor's dtype
        // will silently disagree with the pool's.
        if pool.dtype() != crate::DType::F32 {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "paged_gather_kv default impl only supports F32 pools (got {:?}); backend must override for non-F32 dtypes",
                pool.dtype()
            )));
        }
        let pool_shape = pool.shape();
        let block_size = pool_shape[1];
        let kv_dim = pool_shape[2];
        let pool_data = self.copy_to_host_f32(pool)?;
        let mut out = Vec::with_capacity(total_tokens * kv_dim);
        let mut remaining = total_tokens;
        for &block_id in block_ids {
            if remaining == 0 {
                break;
            }
            let fill = remaining.min(block_size);
            let block_off = block_id * block_size * kv_dim;
            let bytes = fill * kv_dim;
            out.extend_from_slice(&pool_data[block_off..block_off + bytes]);
            remaining -= fill;
        }
        self.copy_from_host_f32(&out, &[total_tokens, kv_dim])
    }

    // ── Attention helpers ───────────────────────────────────────
    // Default impls go through host; CudaBackend overrides with GPU kernels.

    /// Extract head `head` from `[seq_len, num_heads, head_dim]` layout → `[seq_len, head_dim]`.
    fn extract_head(
        &self,
        tensor: &Self::Tensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        head: usize,
    ) -> Result<Self::Tensor> {
        let data = self.copy_to_host_f32(tensor)?;
        let stride = num_heads * head_dim;
        let mut out = Vec::with_capacity(seq_len * head_dim);
        for t in 0..seq_len {
            let start = t * stride + head * head_dim;
            out.extend_from_slice(&data[start..start + head_dim]);
        }
        self.copy_from_host_f32(&out, &[seq_len, head_dim])
    }

    /// Apply causal mask: `scores[q][k] = -inf` for `k > kv_len - seq_len + q`.
    ///
    /// Input/output: `[seq_len, kv_len]`.
    fn apply_causal_mask(
        &self,
        scores: &Self::Tensor,
        seq_len: usize,
        kv_len: usize,
    ) -> Result<Self::Tensor> {
        let mut data = self.copy_to_host_f32(scores)?;
        for q_pos in 0..seq_len {
            let abs_pos = kv_len - seq_len + q_pos;
            for k_pos in (abs_pos + 1)..kv_len {
                data[q_pos * kv_len + k_pos] = f32::NEG_INFINITY;
            }
        }
        self.copy_from_host_f32(&data, scores.shape())
    }

    /// Extract rows `[start_row..start_row+num_rows]` from a tensor.
    /// Input: `[total_rows, cols...]`, Output: `[num_rows, cols...]`.
    fn slice_rows(
        &self,
        tensor: &Self::Tensor,
        start_row: usize,
        num_rows: usize,
    ) -> Result<Self::Tensor> {
        let shape = tensor.shape();
        let cols: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            1
        };
        let data = self.copy_to_host_f32(tensor)?;
        let offset = start_row * cols;
        let len = num_rows * cols;
        let mut out_shape = shape.to_vec();
        out_shape[0] = num_rows;
        self.copy_from_host_f32(&data[offset..offset + len], &out_shape)
    }

    /// Interleave per-head `[seq_len, head_dim]` outputs → `[seq_len, num_heads * head_dim]`.
    fn interleave_heads(
        &self,
        heads: &[&Self::Tensor],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Self::Tensor> {
        let num_heads = heads.len();
        let mut head_data: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
        for h in heads {
            head_data.push(self.copy_to_host_f32(h)?);
        }
        let mut result = Vec::with_capacity(seq_len * num_heads * head_dim);
        for t in 0..seq_len {
            for data in &head_data {
                let offset = t * head_dim;
                result.extend_from_slice(&data[offset..offset + head_dim]);
            }
        }
        self.copy_from_host_f32(&result, &[seq_len, num_heads * head_dim])
    }

    /// Per-row argmax of `[rows, cols]` (or `[cols]`) logits → one token id per
    /// row. This is greedy decode. Tie-break: the HIGHEST index among equal
    /// maxima wins, matching the CPU sampler's `Iterator::max_by` (which
    /// returns the last maximum), so the GPU path is bit-for-bit identical.
    ///
    /// Default: copy to host and reduce on CPU. Capture-capable backends
    /// (CUDA) override this with an on-device reduction, so only the `rows`
    /// resulting ids cross PCIe instead of the full `[rows, cols]` logits.
    fn argmax(&self, logits: &Self::Tensor) -> Result<Vec<u32>> {
        let shape = logits.shape();
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(crate::ForgeError::InvalidArgument(format!(
                    "argmax: expected 1-D or 2-D logits, got {shape:?}"
                )));
            }
        };
        if cols == 0 {
            return Err(crate::ForgeError::InvalidArgument(
                "argmax: empty logits (cols == 0)".into(),
            ));
        }
        let host = self.copy_to_host_f32(logits)?;
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = &host[r * cols..(r + 1) * cols];
            let mut best = f32::NEG_INFINITY;
            let mut best_i = 0usize;
            for (i, &v) in row.iter().enumerate() {
                // `>=` keeps the last (highest-index) maximum, matching `max_by`.
                if v >= best {
                    best = v;
                    best_i = i;
                }
            }
            out.push(best_i as u32);
        }
        Ok(out)
    }

    /// Per-row multinomial sample from `softmax(logits / temperature)` via the
    /// Gumbel-max trick: `argmax_i(logits_i / T + g_i)` with
    /// `g_i = -log(-log u_i)` and `u_i` a counter-based uniform keyed on
    /// `(seed, step, row, col)`. This is the on-device draw used by vLLM/SGLang;
    /// it is reproducible run-to-run for a fixed `(seed, step)` on the same
    /// backend, but does NOT reproduce a CPU `StdRng` sequence.
    ///
    /// `temperature` must be > 0 (greedy/`T <= 0` callers use [`Self::argmax`]).
    /// Default: copy to host and reduce on CPU with the identical RNG, so the
    /// CPU and CUDA paths agree; CUDA overrides with an on-device kernel.
    fn sample_gumbel(
        &self,
        logits: &Self::Tensor,
        temperature: f32,
        seed: u64,
        step: u32,
    ) -> Result<Vec<u32>> {
        let shape = logits.shape();
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(crate::ForgeError::InvalidArgument(format!(
                    "sample_gumbel: expected 1-D or 2-D logits, got {shape:?}"
                )));
            }
        };
        if cols == 0 {
            return Err(crate::ForgeError::InvalidArgument(
                "sample_gumbel: empty logits (cols == 0)".into(),
            ));
        }
        if temperature <= 0.0 || temperature.is_nan() {
            return Err(crate::ForgeError::InvalidArgument(
                "sample_gumbel: temperature must be > 0 (use argmax for greedy)".into(),
            ));
        }
        let inv_temp = 1.0 / temperature;
        let host = self.copy_to_host_f32(logits)?;
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = &host[r * cols..(r + 1) * cols];
            let mut best = f32::NEG_INFINITY;
            let mut best_i = 0usize;
            for (i, &v) in row.iter().enumerate() {
                let key = v * inv_temp + gumbel_noise(seed, step, r as u32, i as u32);
                if key >= best {
                    best = key;
                    best_i = i;
                }
            }
            out.push(best_i as u32);
        }
        Ok(out)
    }

    /// Per-row sampling with per-sequence parameters: row `r` decodes greedily
    /// (argmax) when `temps[r] <= 0`, else draws via Gumbel-max at that row's
    /// temperature/seed/step. One call handles a decode batch mixing greedy and
    /// sampled sequences. `temps`, `seeds`, `steps` all have `rows` entries.
    /// Returns one token id per row; only those ids cross PCIe.
    ///
    /// Default: copy to host and reduce on CPU with the identical RNG, so the
    /// CPU and CUDA paths agree; CUDA overrides with an on-device kernel.
    fn sample(
        &self,
        logits: &Self::Tensor,
        temps: &[f32],
        seeds: &[u64],
        steps: &[u32],
    ) -> Result<Vec<u32>> {
        let shape = logits.shape();
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(crate::ForgeError::InvalidArgument(format!(
                    "sample: expected 1-D or 2-D logits, got {shape:?}"
                )));
            }
        };
        if cols == 0 {
            return Err(crate::ForgeError::InvalidArgument(
                "sample: empty logits (cols == 0)".into(),
            ));
        }
        if temps.len() != rows || seeds.len() != rows || steps.len() != rows {
            return Err(crate::ForgeError::InvalidArgument(format!(
                "sample: per-row params must have {rows} entries (got temps={}, seeds={}, steps={})",
                temps.len(),
                seeds.len(),
                steps.len()
            )));
        }
        let host = self.copy_to_host_f32(logits)?;
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = &host[r * cols..(r + 1) * cols];
            let temp = temps[r];
            let do_sample = temp > 0.0;
            let inv_temp = if do_sample { 1.0 / temp } else { 0.0 };
            let mut best = f32::NEG_INFINITY;
            let mut best_i = 0usize;
            for (i, &v) in row.iter().enumerate() {
                let key = if do_sample {
                    v * inv_temp + gumbel_noise(seeds[r], steps[r], r as u32, i as u32)
                } else {
                    v
                };
                if key >= best {
                    best = key;
                    best_i = i;
                }
            }
            out.push(best_i as u32);
        }
        Ok(out)
    }

    /// Stage one decode step's per-step inputs (token ids, RoPE freqs, paged
    /// block tables, KV lengths) into persistent device scratch buffers, so a
    /// replayed captured [`Model::compute_decode`](crate::Model::compute_decode)
    /// graph re-reads fresh data. Runs every decode step, OUTSIDE any captured
    /// region.
    ///
    /// Default: no-op — backends without graph capture (CPU) read these inputs
    /// directly during compute, so nothing needs pre-staging.
    fn stage_decode_inputs(&self, _inputs: &DecodeStageInputs) -> Result<()> {
        Ok(())
    }

    /// Stage the per-step KV-write slot mapping. Capture-safe backends upload
    /// it to a persistent device scratch (call OUTSIDE the captured region);
    /// the captured [`Self::scatter_kv`] then reads that scratch. Returns the
    /// staged length.
    ///
    /// Default: no-op store — the host-fallback [`Self::scatter_kv`] uses the
    /// `slot_mapping` slice it is handed directly.
    fn stage_slot_mapping(&self, slot_mapping: &[i32]) -> Result<usize> {
        Ok(slot_mapping.len())
    }

    /// Capture-safe decode KV write: scatter the first `n_rows` rows of `src`
    /// into `pool` (`[num_blocks, block_size, kv_dim]`) at the given slots.
    ///
    /// `slot_mapping` is the host-side mapping consumed by the default impl;
    /// capture-safe backends (CUDA) ignore it and read the device scratch
    /// previously filled by [`Self::stage_slot_mapping`], so the scatter is a
    /// pure kernel launch with no baked host pointer.
    fn scatter_kv(
        &self,
        pool: &mut Self::Tensor,
        src: &Self::Tensor,
        slot_mapping: &[i32],
        n_rows: usize,
    ) -> Result<()> {
        self.paged_write_kv(pool, src, &slot_mapping[..n_rows])
    }

    /// Build a captured-graph decode dispatcher for the given batch-size
    /// `buckets`, or `None` if this backend has no CUDA-Graph support
    /// (CPU). Default: `None` — the engine then always runs the eager
    /// decode path.
    fn make_decode_graph_runner(&self, _buckets: &[u32]) -> Option<Box<dyn DecodeGraphDispatch>> {
        None
    }

    /// Monotonic epoch that bumps whenever a persistent device pointer a
    /// captured decode graph could have baked may have moved (a scratch or
    /// pool reallocated on grow). The engine invalidates captured graphs
    /// when this value changes. Default: `0` (never invalidates).
    fn decode_capture_epoch(&self) -> u64 {
        0
    }
}
