use forge_core::{Backend, ForgeError, ModelConfig, Result, Tensor};

/// Precomputed RoPE frequency tables.
///
/// Stores both GPU tensors (for offset=0 fast path) and host-side f32 vectors
/// (for offset slicing during decode, avoiding GPU round-trips).
pub struct RopeFreqs<B: Backend> {
    cos: B::Tensor,
    sin: B::Tensor,
    cos_host: Vec<f32>,
    sin_host: Vec<f32>,
    half_dim: usize,
    max_seq_len: usize,
}

impl<B: Backend> RopeFreqs<B> {
    /// Precompute cos/sin tables for the given config.
    ///
    /// Returns tensors of shape [max_seq_len, head_dim/2] that can be
    /// sliced per-sequence during forward.
    pub fn precompute(config: &ModelConfig, max_seq_len: usize, backend: &B) -> Result<Self> {
        let head_dim = config.head_dim;
        let half_dim = head_dim / 2;
        let theta = config.rope_theta;

        let mut cos_data = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin_data = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = (pos as f64)
                    / theta.powf(2.0 * (i as f64) / (head_dim as f64));
                cos_data.push(freq.cos() as f32);
                sin_data.push(freq.sin() as f32);
            }
        }

        let cos = backend.copy_from_host_f32(&cos_data, &[max_seq_len, half_dim])?;
        let sin = backend.copy_from_host_f32(&sin_data, &[max_seq_len, half_dim])?;

        Ok(Self {
            cos,
            sin,
            cos_host: cos_data,
            sin_host: sin_data,
            half_dim,
            max_seq_len,
        })
    }

    /// Apply RoPE with a position offset.
    ///
    /// For prefill, `pos_offset` is 0 (positions 0..seq_len).
    /// For decode, `pos_offset` is the number of cached tokens
    /// so the single new token gets the correct position index.
    ///
    /// `x` shape: [batch, seq_len, num_heads, head_dim]
    pub fn apply_with_offset(
        &self,
        x: &B::Tensor,
        pos_offset: usize,
        backend: &B,
    ) -> Result<B::Tensor> {
        if pos_offset == 0 {
            // Fast path: no offset needed, use full tables directly.
            // The kernel reads only seq_len rows from the table.
            return backend.rope(x, &self.cos, &self.sin);
        }

        let seq_len = x.shape()[1];
        let start = pos_offset * self.half_dim;
        let end = (pos_offset + seq_len) * self.half_dim;

        if end > self.cos_host.len() {
            return Err(ForgeError::InvalidArgument(format!(
                "RoPE offset {} + seq_len {} = {} exceeds precomputed table length {}",
                pos_offset,
                seq_len,
                pos_offset + seq_len,
                self.max_seq_len,
            )));
        }

        // Slice from cached host-side data (no GPU round-trip)
        let cos_slice =
            backend.copy_from_host_f32(&self.cos_host[start..end], &[seq_len, self.half_dim])?;
        let sin_slice =
            backend.copy_from_host_f32(&self.sin_host[start..end], &[seq_len, self.half_dim])?;

        backend.rope(x, &cos_slice, &sin_slice)
    }

    /// Apply RoPE with explicit per-token positions.
    ///
    /// Unlike `apply_with_offset` which assumes consecutive positions
    /// `[offset..offset+seq_len]`, this method allows arbitrary positions
    /// per token — required for batched decode where each token comes from
    /// a different sequence at a different absolute position.
    ///
    /// `x` shape: `[1, seq_len, num_heads, head_dim]`
    /// `positions`: `&[u32]` of length `seq_len`, each entry is the absolute position.
    pub fn apply_with_positions(
        &self,
        x: &B::Tensor,
        positions: &[u32],
        backend: &B,
    ) -> Result<B::Tensor> {
        let mut out = backend.allocate_zeros(x.shape(), x.dtype())?;
        self.apply_with_positions_into(&mut out, x, positions, backend)?;
        Ok(out)
    }

    /// In-place per-token RoPE — writes the rotated tensor into a
    /// caller-provided `out` buffer instead of allocating.
    ///
    /// Used by the CUDA-Graph capture path so the captured kernel arg
    /// pointers stay stable across replays.
    ///
    /// Note: the cos/sin freq tensors uploaded each call are still
    /// transient (allocated fresh from the cached host tables). Wiring
    /// them through persistent device scratches is Task 5c.3 (along with
    /// embedding's indices scratch).
    pub fn apply_with_positions_into(
        &self,
        out: &mut B::Tensor,
        x: &B::Tensor,
        positions: &[u32],
        backend: &B,
    ) -> Result<()> {
        let seq_len = x.shape()[1];
        if positions.len() != seq_len {
            return Err(ForgeError::InvalidArgument(format!(
                "positions length {} != seq_len {}",
                positions.len(),
                seq_len,
            )));
        }

        // Gather cos/sin rows by position index
        let mut cos_data = Vec::with_capacity(seq_len * self.half_dim);
        let mut sin_data = Vec::with_capacity(seq_len * self.half_dim);

        for &pos in positions {
            let pos = pos as usize;
            if pos >= self.max_seq_len {
                return Err(ForgeError::InvalidArgument(format!(
                    "RoPE position {} exceeds precomputed table length {}",
                    pos, self.max_seq_len,
                )));
            }
            let start = pos * self.half_dim;
            let end = start + self.half_dim;
            cos_data.extend_from_slice(&self.cos_host[start..end]);
            sin_data.extend_from_slice(&self.sin_host[start..end]);
        }

        let cos_tensor =
            backend.copy_from_host_f32(&cos_data, &[seq_len, self.half_dim])?;
        let sin_tensor =
            backend.copy_from_host_f32(&sin_data, &[seq_len, self.half_dim])?;

        backend.rope_into(out, x, &cos_tensor, &sin_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_backend_cpu::CpuBackend;
    use forge_core::DType;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            num_hidden_layers: 1,
            vocab_size: 4,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F32,
        }
    }

    #[test]
    fn apply_with_positions_into_matches_alloc() {
        let backend = CpuBackend::new();
        let config = tiny_config();
        let rope = RopeFreqs::precompute(&config, 32, &backend).unwrap();

        // [1, 3, num_heads=2, head_dim=4] = 24 elements
        let x_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05).collect();
        let x = backend
            .copy_from_host_f32(&x_data, &[1, 3, 2, 4])
            .unwrap();
        let positions: Vec<u32> = vec![0, 1, 5];

        let out_alloc = rope.apply_with_positions(&x, &positions, &backend).unwrap();
        let alloc_host = backend.copy_to_host_f32(&out_alloc).unwrap();

        let mut out_into = backend
            .allocate_zeros(&[1, 3, 2, 4], DType::F32)
            .unwrap();
        rope.apply_with_positions_into(&mut out_into, &x, &positions, &backend)
            .unwrap();
        let into_host = backend.copy_to_host_f32(&out_into).unwrap();
        assert_eq!(alloc_host, into_host);
    }

    #[test]
    fn apply_with_positions_into_validates_length() {
        let backend = CpuBackend::new();
        let config = tiny_config();
        let rope = RopeFreqs::precompute(&config, 32, &backend).unwrap();

        let x = backend
            .copy_from_host_f32(&vec![0.0; 24], &[1, 3, 2, 4])
            .unwrap();
        let mut out = backend
            .allocate_zeros(&[1, 3, 2, 4], DType::F32)
            .unwrap();
        // positions length != seq_len → error
        let _ = rope
            .apply_with_positions_into(&mut out, &x, &[0, 1], &backend)
            .expect_err("position length mismatch should error");
    }
}
