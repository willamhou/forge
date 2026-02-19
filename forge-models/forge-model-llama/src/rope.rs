use forge_core::{Backend, ModelConfig, Result};

/// Precomputed RoPE frequency tables.
pub struct RopeFreqs<B: Backend> {
    cos: B::Tensor,
    sin: B::Tensor,
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

        Ok(Self { cos, sin })
    }

    /// Apply RoPE to a tensor of shape [batch, seq_len, num_heads, head_dim].
    ///
    /// Uses the backend's `rope` op which expects (x, freqs_cos, freqs_sin).
    pub fn apply(&self, x: &B::Tensor, backend: &B) -> Result<B::Tensor> {
        backend.rope(x, &self.cos, &self.sin)
    }
}
