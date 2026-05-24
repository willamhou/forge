//! Parity: `LlamaModel::forward_into` (persistent-buffer path) must
//! produce identical logits to `LlamaModel::forward` (alloc path) when
//! given the same model, same KV cache state, and same decode input.
//!
//! Uses the tiny F32 Llama from test_batch_forward.rs + test_decode_paged.rs.
//! PagedKvCache only — naive caches are explicitly unsupported by
//! forward_into.

use forge_backend_cpu::CpuBackend;
use forge_core::{Backend, DType, KvCache, Model, ModelConfig, ModelInput, SeqMetadata, Tensor};
use forge_kvcache::paged_cache::PagedKvCache;
use forge_model_llama::layers::{LlamaAttention, LlamaDecoderLayer, LlamaMLP, RMSNorm};
use forge_model_llama::rope::RopeFreqs;
use forge_model_llama::{LlamaModel, LlamaPersistentBuffers};

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

fn build_tiny_model(backend: &CpuBackend) -> LlamaModel<CpuBackend> {
    let config = tiny_config();
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;

    let make_weight = |rows: usize, cols: usize| -> <CpuBackend as Backend>::Tensor {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.01 + 0.1) % 1.0) - 0.5)
            .collect();
        backend.copy_from_host_f32(&data, &[rows, cols]).unwrap()
    };

    let embed_tokens = make_weight(vocab, h);
    let lm_head = make_weight(h, vocab);
    let norm = RMSNorm::new(
        backend.copy_from_host_f32(&vec![1.0; h], &[h]).unwrap(),
        config.rms_norm_eps,
    );

    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let wq = make_weight(h, q_dim);
    let wk = make_weight(h, kv_dim);
    let wv = make_weight(h, kv_dim);
    let wq_t = backend.transpose(&wq, 0, 1).unwrap();
    let wk_t = backend.transpose(&wk, 0, 1).unwrap();
    let wv_t = backend.transpose(&wv, 0, 1).unwrap();
    let cat_t = backend.cat(&[&wq_t, &wk_t, &wv_t], 0).unwrap();
    let wqkv = backend.transpose(&cat_t, 0, 1).unwrap();
    let attn = LlamaAttention::new(wqkv, make_weight(q_dim, h), &config);

    let mlp = LlamaMLP::new(
        make_weight(h, inter),
        make_weight(h, inter),
        make_weight(inter, h),
    );

    let layer_norm = RMSNorm::new(
        backend.copy_from_host_f32(&vec![1.0; h], &[h]).unwrap(),
        config.rms_norm_eps,
    );
    let post_norm = RMSNorm::new(
        backend.copy_from_host_f32(&vec![1.0; h], &[h]).unwrap(),
        config.rms_norm_eps,
    );

    let layer = LlamaDecoderLayer::new(layer_norm, attn, post_norm, mlp);
    let rope = RopeFreqs::precompute(&config, 64, backend).unwrap();

    LlamaModel::new(
        config,
        embed_tokens,
        vec![layer],
        norm,
        lm_head,
        rope,
        backend.clone(),
    )
}

fn prefill(
    model: &LlamaModel<CpuBackend>,
    kv_cache: &mut dyn KvCache<T = <CpuBackend as Backend>::Tensor>,
    seq_id: u64,
    prompt_tokens: &[u32],
) {
    kv_cache.allocate(seq_id, prompt_tokens.len()).unwrap();
    let positions: Vec<u32> = (0..prompt_tokens.len() as u32).collect();
    let input = ModelInput {
        token_ids: vec![prompt_tokens.to_vec()],
        positions: vec![positions],
        seq_metadata: vec![SeqMetadata {
            seq_id,
            prompt_len: prompt_tokens.len(),
            generated_len: 0,
            is_prefill: true,
        }],
    };
    model.forward(&input, kv_cache).unwrap();
}

fn build_decode_input() -> ModelInput {
    ModelInput {
        token_ids: vec![vec![3], vec![0]],
        positions: vec![vec![3], vec![2]],
        seq_metadata: vec![
            SeqMetadata {
                seq_id: 1,
                prompt_len: 3,
                generated_len: 0,
                is_prefill: false,
            },
            SeqMetadata {
                seq_id: 2,
                prompt_len: 2,
                generated_len: 0,
                is_prefill: false,
            },
        ],
    }
}

#[test]
fn forward_into_matches_forward() {
    let backend = CpuBackend::new();
    let model = build_tiny_model(&backend);
    let config = tiny_config();

    // ── Run 1: alloc path via forward() ──────────────────────────
    let mut kv_alloc = PagedKvCache::new(
        backend.clone(),
        /* total_blocks */ 16,
        /* block_size */ 2,
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        config.dtype,
    )
    .unwrap();
    prefill(&model, &mut kv_alloc, 1, &[0, 1, 2]);
    prefill(&model, &mut kv_alloc, 2, &[1, 3]);
    let decode_input = build_decode_input();
    let out_alloc = model.forward(&decode_input, &mut kv_alloc).unwrap();
    let logits_alloc = backend.copy_to_host_f32(&out_alloc.logits).unwrap();

    // ── Run 2: persistent-buffer path via forward_into() ─────────
    let mut kv_persistent = PagedKvCache::new(
        backend.clone(),
        16,
        2,
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        config.dtype,
    )
    .unwrap();
    prefill(&model, &mut kv_persistent, 1, &[0, 1, 2]);
    prefill(&model, &mut kv_persistent, 2, &[1, 3]);

    let batch_size = decode_input.seq_metadata.len();
    let mut buffers =
        LlamaPersistentBuffers::new_uniform(&backend, &config, batch_size).unwrap();
    model
        .forward_into(&decode_input, &mut kv_persistent, &mut buffers)
        .unwrap();
    let logits_persistent = backend.copy_to_host_f32(&buffers.logits).unwrap();

    // ── Compare ──────────────────────────────────────────────────
    assert_eq!(logits_alloc.len(), logits_persistent.len());
    let max_abs = logits_alloc.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let tol = 1e-5_f32.max(max_abs * 1e-5);

    for (i, (la, lp)) in logits_alloc.iter().zip(&logits_persistent).enumerate() {
        let diff = (la - lp).abs();
        assert!(
            diff < tol,
            "forward vs forward_into logits diverge at [{i}]: alloc={la} persistent={lp} diff={diff}"
        );
    }

    // Greedy argmax matches.
    let argmax = |chunk: &[f32]| -> usize {
        chunk
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let v = config.vocab_size;
    assert_eq!(argmax(&logits_alloc[..v]), argmax(&logits_persistent[..v]));
    assert_eq!(argmax(&logits_alloc[v..]), argmax(&logits_persistent[v..]));

    // ── Reuse buffers: a second forward_into into the SAME buffers
    //    with identical inputs must produce identical results ─────
    let mut kv_persistent2 = PagedKvCache::new(
        backend.clone(),
        16,
        2,
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        config.dtype,
    )
    .unwrap();
    prefill(&model, &mut kv_persistent2, 1, &[0, 1, 2]);
    prefill(&model, &mut kv_persistent2, 2, &[1, 3]);
    model
        .forward_into(&decode_input, &mut kv_persistent2, &mut buffers)
        .unwrap();
    let logits_reuse = backend.copy_to_host_f32(&buffers.logits).unwrap();
    assert_eq!(logits_persistent, logits_reuse);

    // Verify buffers.logits has the expected shape.
    assert_eq!(buffers.logits.shape(), &[batch_size, config.vocab_size]);
}

#[test]
fn forward_into_rejects_wrong_batch_size() {
    let backend = CpuBackend::new();
    let model = build_tiny_model(&backend);
    let config = tiny_config();

    let mut kv = PagedKvCache::new(
        backend.clone(),
        16,
        2,
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        config.dtype,
    )
    .unwrap();
    prefill(&model, &mut kv, 1, &[0, 1, 2]);
    prefill(&model, &mut kv, 2, &[1, 3]);
    let decode_input = build_decode_input(); // batch = 2

    // Buffers sized for batch=4, input has batch=2 → reject.
    let mut wrong_buffers = LlamaPersistentBuffers::new_uniform(&backend, &config, 4).unwrap();
    let _ = model
        .forward_into(&decode_input, &mut kv, &mut wrong_buffers)
        .expect_err("batch_size mismatch must error");
}

#[test]
fn forward_into_rejects_prefill() {
    let backend = CpuBackend::new();
    let model = build_tiny_model(&backend);
    let config = tiny_config();
    let mut kv = PagedKvCache::new(
        backend.clone(),
        16,
        2,
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        config.dtype,
    )
    .unwrap();
    kv.allocate(1, 2).unwrap();
    let mut buffers = LlamaPersistentBuffers::new_uniform(&backend, &config, 1).unwrap();
    let prefill_input = ModelInput {
        token_ids: vec![vec![0, 1]],
        positions: vec![vec![0, 1]],
        seq_metadata: vec![SeqMetadata {
            seq_id: 1,
            prompt_len: 2,
            generated_len: 0,
            is_prefill: true,
        }],
    };
    let _ = model
        .forward_into(&prefill_input, &mut kv, &mut buffers)
        .expect_err("prefill must be rejected");
}
