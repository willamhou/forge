//! Persistent device tensors for capture-stable Llama decode forward.
//!
//! All intermediate tensors a decode forward pass needs are pre-allocated
//! once (per `batch_size`) and reused across calls. Combined with the
//! `_into` Backend op variants (Tasks 5a–5c.3) and the persistent scratch
//! buffers on CudaBackend (Task 2.5a + 5c.3), this means a captured CUDA
//! Graph from `LlamaModel::forward_into` references device pointers that
//! are stable across replays.
//!
//! Buffer layout is one-per-logical-role, reused across all decoder layers.
//! Layer-spanning state is in `hidden` (the residual stream), which the
//! final per-layer `add_into(hidden, residual, mlp_out)` overwrites in
//! place for the next iteration to read.
//!
//! Memory cost (approximate) for `batch_size = N`, Llama-7B
//! (hidden=4096, intermediate=11008, vocab=32000, num_heads=32, head_dim=128):
//! - logits: `N * 32000 * 2B` (F16) ≈ N × 64 KiB
//! - intermediates: ~12 × `N * 4096 * 2B` ≈ N × 96 KiB
//! - MLP intermediates: 3 × `N * 11008 * 2B` ≈ N × 66 KiB
//! - per-token k/v slice buffers: `2N × 4096 × 2B` ≈ N² × 16 KiB (small for typical batches)
//!
//! For batch=8 Llama-7B that's roughly 2 MiB total — negligible.

use forge_core::{Backend, DType, ModelConfig, Result};

/// Per-batch persistent buffers for [`LlamaModel::forward_into`].
///
/// Construct once with [`Self::new`] for a fixed `batch_size`; pass the
/// same instance into every `forward_into` call at that batch shape.
/// Different batch sizes need different buffers (different shapes), which
/// is why the future engine integration (Task 5c.5) maintains one
/// `LlamaPersistentBuffers` per CUDA-Graph bucket.
#[allow(dead_code)]
pub struct LlamaPersistentBuffers<B: Backend> {
    /// Embedding lookup output. `[N, hidden]`.
    pub embeddings: B::Tensor,
    /// Residual stream that carries between layers. `[N, hidden]`.
    /// Initialized from `embeddings` at the start; overwritten by the
    /// final `add_into(hidden, residual, mlp_out)` at each layer's end.
    pub hidden: B::Tensor,

    // ── Per-layer iteration intermediates (shared across all layers) ──
    /// Output of `input_layernorm` AND `post_attention_layernorm` (the
    /// two norms within a layer don't overlap, so they share one buffer).
    /// `[N, hidden]`.
    pub normed: B::Tensor,
    /// Output of `wqkv` matmul. `[N, q_size + 2*kv_size]`.
    pub qkv: B::Tensor,
    /// Q split. `[N, q_size]` where `q_size = num_heads * head_dim`.
    pub q_2d: B::Tensor,
    /// K split. `[N, kv_size]` where `kv_size = num_kv_heads * head_dim`.
    pub k_2d: B::Tensor,
    /// V split. `[N, kv_size]`.
    pub v_2d: B::Tensor,
    /// Q reshaped to 4D for RoPE input. `[1, N, num_heads, head_dim]`.
    pub q_4d: B::Tensor,
    /// K reshaped to 4D for RoPE input. `[1, N, num_kv_heads, head_dim]`.
    pub k_4d: B::Tensor,
    /// Q post-RoPE, 4D. `[1, N, num_heads, head_dim]`.
    pub q_rotated_4d: B::Tensor,
    /// K post-RoPE, 4D. `[1, N, num_kv_heads, head_dim]`.
    pub k_rotated_4d: B::Tensor,
    /// Q post-RoPE flattened back for paged_attention. `[N, q_size]`.
    pub q_rotated_2d: B::Tensor,
    /// K post-RoPE flattened back for KV-cache append. `[N, kv_size]`.
    pub k_rotated_2d: B::Tensor,

    /// Per-token K row buffers for the KV-cache append loop. Length `N`,
    /// each `[1, kv_size]`. Replaces the per-iteration `slice_rows` alloc.
    pub k_rows: Vec<B::Tensor>,
    /// Per-token V row buffers. See [`Self::k_rows`].
    pub v_rows: Vec<B::Tensor>,

    /// Output of paged_attention. `[N, q_size]`. Dtype is the model's
    /// activation dtype (`config.dtype`).
    pub attn_out: B::Tensor,
    /// `attn_out` cast to `wo`'s weight dtype (may equal `attn_out`'s
    /// dtype if model is uniform-dtype). Allocated with `wo_dtype` passed
    /// to `new`; the subsequent `matmul_into(attn_proj, attn_out_cast, wo)`
    /// requires all three to share dtype. Mismatching activation + weight
    /// dtypes would fail before this fix.
    pub attn_out_cast: B::Tensor,
    /// Output of `wo` (o_proj) matmul. `[N, hidden]`. Dtype = `wo_dtype`
    /// (matches `attn_out_cast` and `self.wo`).
    pub attn_proj: B::Tensor,

    /// `residual` output of `fused_residual_rms_norm`. Holds `hidden_prev
    /// + attn_proj`. Lives until the post-MLP `add_into` consumes it.
    /// `[N, hidden]`.
    pub residual_after_attn: B::Tensor,

    /// MLP gate projection output. `[N, intermediate]`.
    pub gate: B::Tensor,
    /// MLP up projection output. `[N, intermediate]`.
    pub up: B::Tensor,
    /// `fused_silu_mul(gate, up)` output. `[N, intermediate]`.
    pub silu_mul: B::Tensor,
    /// MLP down_proj output. `[N, hidden]`.
    pub mlp_out: B::Tensor,

    /// Output of the final `norm.forward`. `[N, hidden]`. Dtype = activation dtype.
    pub final_normed: B::Tensor,
    /// `final_normed` cast to `lm_head`'s weight dtype, when they differ.
    /// `matmul_into(logits, final_normed_cast, lm_head)` requires uniform
    /// dtype across inputs + output.
    pub final_normed_cast: B::Tensor,
    /// Output of the lm_head matmul. `[N, vocab_size]`. Dtype = `lm_head_dtype`
    /// (matches `final_normed_cast` and `self.lm_head`).
    pub logits: B::Tensor,

    /// Sanity field — used to validate that `forward_into` is called with
    /// the right batch size.
    pub batch_size: usize,
}

impl<B: Backend> LlamaPersistentBuffers<B> {
    /// Allocate all per-batch buffers for a fixed `batch_size`.
    ///
    /// - **Activation dtype** is taken from `config.dtype` and used for the
    ///   residual stream, embeddings, norms, paged_attention output, and
    ///   MLP intermediates.
    /// - **`wo_dtype`** is the dtype of the attention output projection
    ///   weight (`self_attn.wo`); used for `attn_out_cast` + `attn_proj`.
    ///   Allocate-time selection of these buffer dtypes ensures the
    ///   `cast_into(attn_out_cast, attn_out)` + `matmul_into(attn_proj,
    ///   attn_out_cast, wo)` chain agrees on dtypes regardless of whether
    ///   activation and weight dtypes match.
    /// - **`lm_head_dtype`** is the dtype of `self.lm_head`; used for
    ///   `final_normed_cast` + `logits`.
    ///
    /// For uniform-dtype models (Llama F16 weights + F16 activations),
    /// pass `config.dtype` for both. For mixed-dtype loads, pull from the
    /// loaded weights directly.
    pub fn new(
        backend: &B,
        config: &ModelConfig,
        batch_size: usize,
        wo_dtype: DType,
        lm_head_dtype: DType,
    ) -> Result<Self> {
        let n = batch_size;
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_key_value_heads * config.head_dim;
        let dt = config.dtype;

        let mut k_rows = Vec::with_capacity(n);
        let mut v_rows = Vec::with_capacity(n);
        for _ in 0..n {
            k_rows.push(backend.allocate_zeros(&[1, kv_size], dt)?);
            v_rows.push(backend.allocate_zeros(&[1, kv_size], dt)?);
        }

        Ok(Self {
            embeddings: backend.allocate_zeros(&[n, h], dt)?,
            hidden: backend.allocate_zeros(&[n, h], dt)?,
            normed: backend.allocate_zeros(&[n, h], dt)?,
            qkv: backend.allocate_zeros(&[n, q_size + 2 * kv_size], dt)?,
            q_2d: backend.allocate_zeros(&[n, q_size], dt)?,
            k_2d: backend.allocate_zeros(&[n, kv_size], dt)?,
            v_2d: backend.allocate_zeros(&[n, kv_size], dt)?,
            q_4d: backend.allocate_zeros(
                &[1, n, config.num_attention_heads, config.head_dim],
                dt,
            )?,
            k_4d: backend.allocate_zeros(
                &[1, n, config.num_key_value_heads, config.head_dim],
                dt,
            )?,
            q_rotated_4d: backend.allocate_zeros(
                &[1, n, config.num_attention_heads, config.head_dim],
                dt,
            )?,
            k_rotated_4d: backend.allocate_zeros(
                &[1, n, config.num_key_value_heads, config.head_dim],
                dt,
            )?,
            q_rotated_2d: backend.allocate_zeros(&[n, q_size], dt)?,
            k_rotated_2d: backend.allocate_zeros(&[n, kv_size], dt)?,
            k_rows,
            v_rows,
            attn_out: backend.allocate_zeros(&[n, q_size], dt)?,
            // attn_out_cast + attn_proj live in wo's dtype so the cast+matmul
            // chain agrees regardless of activation/weight dtype mismatch.
            attn_out_cast: backend.allocate_zeros(&[n, q_size], wo_dtype)?,
            attn_proj: backend.allocate_zeros(&[n, h], wo_dtype)?,
            residual_after_attn: backend.allocate_zeros(&[n, h], dt)?,
            gate: backend.allocate_zeros(&[n, inter], dt)?,
            up: backend.allocate_zeros(&[n, inter], dt)?,
            silu_mul: backend.allocate_zeros(&[n, inter], dt)?,
            mlp_out: backend.allocate_zeros(&[n, h], dt)?,
            final_normed: backend.allocate_zeros(&[n, h], dt)?,
            // Same dtype-coherence story as attn_out_cast / attn_proj.
            final_normed_cast: backend.allocate_zeros(&[n, h], lm_head_dtype)?,
            logits: backend.allocate_zeros(&[n, vocab], lm_head_dtype)?,
            batch_size: n,
        })
    }

    /// Convenience constructor for the uniform-dtype case: all
    /// activations + weights share `config.dtype`. The standard Llama
    /// load path produces this (F16 weights + F16 activations is the
    /// production norm).
    ///
    /// Equivalent to [`Self::new`] with `wo_dtype = lm_head_dtype = config.dtype`.
    pub fn new_uniform(
        backend: &B,
        config: &ModelConfig,
        batch_size: usize,
    ) -> Result<Self> {
        Self::new(backend, config, batch_size, config.dtype, config.dtype)
    }
}
