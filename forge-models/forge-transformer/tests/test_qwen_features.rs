//! Smoke tests locking the Qwen2 QKV-bias and Qwen3 QK-norm wiring in
//! `LlamaAttention::forward`. Each test runs attention twice on identical
//! inputs — once plain, once with the optional feature attached — and asserts
//! the outputs differ. A future refactor that drops the `add_bias` or
//! `maybe_q_norm` call in the forward pass would silently pass cargo check;
//! these tests turn that into a visible failure.
//!
//! They are not numerical correctness tests — that lives in end-to-end model
//! parity bench against HuggingFace. The goal here is *path coverage*: both
//! optional features must remain reachable from the public attention API.

use forge_backend_cpu::CpuBackend;
use forge_core::{Backend, DType, KvCache, ModelConfig, Tensor};
use forge_kvcache::naive::NaiveKvCache;
use forge_transformer::layers::{LlamaAttention, RMSNorm};
use forge_transformer::quantized_linear::QuantizedLinear;
use forge_transformer::rope::RopeFreqs;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        intermediate_size: 16,
        num_hidden_layers: 1,
        vocab_size: 4,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        dtype: DType::F32,
    }
}

fn make_tensor(
    backend: &CpuBackend,
    seed: f32,
    rows: usize,
    cols: usize,
) -> <CpuBackend as Backend>::Tensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 * 0.013 + seed) % 1.0) - 0.5)
        .collect();
    backend.copy_from_host_f32(&data, &[rows, cols]).unwrap()
}

fn make_vector(backend: &CpuBackend, seed: f32, n: usize) -> <CpuBackend as Backend>::Tensor {
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07 + seed) % 1.0).collect();
    backend.copy_from_host_f32(&data, &[n]).unwrap()
}

/// Which of the optional Qwen-family norms to attach.
#[derive(Clone, Copy)]
enum NormMode {
    /// No q_norm, no k_norm (Llama / Qwen2 default).
    None,
    /// q_norm only — isolates the Q path.
    QOnly,
    /// k_norm only — isolates the K path.
    KOnly,
    /// Both q_norm and k_norm (the loader's Qwen3 path).
    Both,
}

/// Build a `LlamaAttention` with deterministic weights and optional features.
fn build_attn(
    backend: &CpuBackend,
    config: &ModelConfig,
    bias: bool,
    norm: NormMode,
) -> LlamaAttention<CpuBackend> {
    let h = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;

    // Match the loader pattern: build wq/wk/wv at [hidden, *_dim], cat along
    // dim=0 (after transpose), then transpose back to [hidden, q+2*kv].
    let wq = make_tensor(backend, 0.1, h, q_dim);
    let wk = make_tensor(backend, 0.2, h, kv_dim);
    let wv = make_tensor(backend, 0.3, h, kv_dim);
    let wq_t = backend.transpose(&wq, 0, 1).unwrap();
    let wk_t = backend.transpose(&wk, 0, 1).unwrap();
    let wv_t = backend.transpose(&wv, 0, 1).unwrap();
    let cat_t = backend.cat(&[&wq_t, &wk_t, &wv_t], 0).unwrap();
    let wqkv = backend.transpose(&cat_t, 0, 1).unwrap();
    let wo = make_tensor(backend, 0.4, q_dim, h);

    let qkv_bias = if bias {
        Some(make_vector(backend, 0.5, q_dim + 2 * kv_dim))
    } else {
        None
    };

    let attn = LlamaAttention::new_with_bias(
        QuantizedLinear::new_f16(wqkv),
        qkv_bias,
        QuantizedLinear::new_f16(wo),
        config,
    );

    // Non-unit weights so RMSNorm is not a no-op against any plausible input.
    let mk_q_norm = || {
        RMSNorm::new(
            make_vector(backend, 0.6, config.head_dim),
            config.rms_norm_eps,
        )
    };
    let mk_k_norm = || {
        RMSNorm::new(
            make_vector(backend, 0.7, config.head_dim),
            config.rms_norm_eps,
        )
    };
    match norm {
        NormMode::None => attn,
        NormMode::QOnly => attn.with_qk_norm(Some(mk_q_norm()), None),
        NormMode::KOnly => attn.with_qk_norm(None, Some(mk_k_norm())),
        NormMode::Both => attn.with_qk_norm(Some(mk_q_norm()), Some(mk_k_norm())),
    }
}

fn run_attn(
    backend: &CpuBackend,
    config: &ModelConfig,
    attn: &LlamaAttention<CpuBackend>,
    x: &<CpuBackend as Backend>::Tensor,
) -> Vec<f32> {
    let rope = RopeFreqs::precompute(config, 64, backend).unwrap();
    let mut kv = NaiveKvCache::new(backend.clone(), config.num_hidden_layers, 4);
    let seq_len = x.shape()[0];
    kv.allocate(1, seq_len).unwrap();
    let out = attn.forward(x, &rope, 0, &mut kv, 1, 0, backend).unwrap();
    backend.copy_to_host_f32(&out).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch between runs");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn qwen3_q_norm_alone_is_wired_into_attention_forward() {
    // Isolates the q_norm path: attaches Q-norm only, no K-norm. If
    // `maybe_q_norm` becomes a no-op, output must collapse to the plain case.
    let backend = CpuBackend::new();
    let config = tiny_config();
    let x = make_tensor(&backend, 0.05, 3, config.hidden_size);

    let attn_plain = build_attn(&backend, &config, false, NormMode::None);
    let attn_q_only = build_attn(&backend, &config, false, NormMode::QOnly);

    let out_plain = run_attn(&backend, &config, &attn_plain, &x);
    let out_q_only = run_attn(&backend, &config, &attn_q_only, &x);

    let diff = max_abs_diff(&out_plain, &out_q_only);
    assert!(
        diff > 1e-4,
        "Q-norm alone has no observable effect on attention output \
         (max_abs_diff={diff:.3e}). The `maybe_q_norm` path is likely \
         no longer reachable from `LlamaAttention::forward`."
    );
}

#[test]
fn qwen3_k_norm_alone_is_wired_into_attention_forward() {
    // Isolates the k_norm path. Symmetric to the q_norm test above so that
    // a regression that no-ops one but not the other is still caught.
    let backend = CpuBackend::new();
    let config = tiny_config();
    let x = make_tensor(&backend, 0.05, 3, config.hidden_size);

    let attn_plain = build_attn(&backend, &config, false, NormMode::None);
    let attn_k_only = build_attn(&backend, &config, false, NormMode::KOnly);

    let out_plain = run_attn(&backend, &config, &attn_plain, &x);
    let out_k_only = run_attn(&backend, &config, &attn_k_only, &x);

    let diff = max_abs_diff(&out_plain, &out_k_only);
    assert!(
        diff > 1e-4,
        "K-norm alone has no observable effect on attention output \
         (max_abs_diff={diff:.3e}). The `maybe_k_norm` path is likely \
         no longer reachable from `LlamaAttention::forward`."
    );
}

#[test]
fn qwen3_both_qk_norms_wired_together() {
    // Sanity: the Qwen3 loader attaches both — verify the combined path also
    // shifts output (would catch a regression that disables the whole feature).
    let backend = CpuBackend::new();
    let config = tiny_config();
    let x = make_tensor(&backend, 0.05, 3, config.hidden_size);

    let attn_plain = build_attn(&backend, &config, false, NormMode::None);
    let attn_both = build_attn(&backend, &config, false, NormMode::Both);

    let out_plain = run_attn(&backend, &config, &attn_plain, &x);
    let out_both = run_attn(&backend, &config, &attn_both, &x);

    let diff = max_abs_diff(&out_plain, &out_both);
    assert!(
        diff > 1e-4,
        "Combined QK-norm path has no effect (diff={diff:.3e})"
    );
}

#[test]
fn qwen2_qkv_bias_is_wired_into_attention_forward() {
    let backend = CpuBackend::new();
    let config = tiny_config();
    let x = make_tensor(&backend, 0.05, 3, config.hidden_size);

    let attn_plain = build_attn(&backend, &config, false, NormMode::None);
    let attn_bias = build_attn(&backend, &config, true, NormMode::None);

    let out_plain = run_attn(&backend, &config, &attn_plain, &x);
    let out_bias = run_attn(&backend, &config, &attn_bias, &x);

    let diff = max_abs_diff(&out_plain, &out_bias);
    assert!(
        diff > 1e-4,
        "QKV bias has no observable effect on attention output \
         (max_abs_diff={diff:.3e}). The bias-add in `project_qkv` is likely \
         no longer reachable from `LlamaAttention::forward`."
    );
    assert!(
        attn_bias.has_qkv_bias(),
        "has_qkv_bias() must reflect the constructor argument"
    );
    assert!(
        !attn_plain.has_qkv_bias(),
        "has_qkv_bias() must be false when no bias was supplied"
    );
}
