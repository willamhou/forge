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
}
