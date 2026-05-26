//! Greedy parity: batched decode through PagedKvCache (paged_attention path)
//! must produce the same logits as the same model run with NaiveKvCache
//! (batched_decode_attention fallback path). Uses the tiny F32 model from
//! test_batch_forward.rs.

use forge_backend_cpu::CpuBackend;
use forge_core::{Backend, DType, KvCache, Model, ModelConfig, ModelInput, SeqMetadata};
use forge_kvcache::naive::NaiveKvCache;
use forge_kvcache::paged_cache::PagedKvCache;
use forge_transformer::layers::{LlamaAttention, LlamaDecoderLayer, LlamaMLP, RMSNorm};
use forge_transformer::rope::RopeFreqs;
use forge_transformer::TransformerModel;

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

fn build_tiny_model(backend: &CpuBackend) -> TransformerModel<CpuBackend> {
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

    TransformerModel::new(
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
    model: &TransformerModel<CpuBackend>,
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

fn batched_decode_logits(
    model: &TransformerModel<CpuBackend>,
    backend: &CpuBackend,
    kv_cache: &mut dyn KvCache<T = <CpuBackend as Backend>::Tensor>,
) -> Vec<f32> {
    let input = ModelInput {
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
    };
    let out = model.forward(&input, kv_cache).unwrap();
    backend.copy_to_host_f32(&out.logits).unwrap()
}

#[test]
fn batched_decode_paged_matches_naive() {
    let backend = CpuBackend::new();
    let model = build_tiny_model(&backend);
    let config = tiny_config();

    // Run 1: NaiveKvCache (fallback batched_decode_attention path)
    let mut kv_naive = NaiveKvCache::new(backend.clone(), config.num_hidden_layers, 4);
    prefill(&model, &mut kv_naive, 1, &[0, 1, 2]);
    prefill(&model, &mut kv_naive, 2, &[1, 3]);
    let logits_naive = batched_decode_logits(&model, &backend, &mut kv_naive);

    // Run 2: PagedKvCache (paged_attention path).
    // num_key_value_heads = 2, head_dim = 4 → kv_dim = 8.
    // Use small block_size so multi-block paths get exercised.
    let mut kv_paged = PagedKvCache::new(
        backend.clone(),
        /* total_blocks */ 16,
        /* block_size */ 2,
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        /* dtype */ DType::F32,
    )
    .unwrap();
    prefill(&model, &mut kv_paged, 1, &[0, 1, 2]);
    prefill(&model, &mut kv_paged, 2, &[1, 3]);
    let logits_paged = batched_decode_logits(&model, &backend, &mut kv_paged);

    assert_eq!(logits_paged.len(), logits_naive.len());
    let max_abs = logits_naive.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let tol = 1e-4_f32.max(max_abs * 1e-4);

    for (i, (lp, ln)) in logits_paged.iter().zip(&logits_naive).enumerate() {
        let diff = (lp - ln).abs();
        assert!(
            diff < tol,
            "logits diverge at [{i}]: paged={lp} naive={ln} diff={diff} tol={tol}"
        );
    }

    // Sanity: greedy argmax matches.
    let argmax = |chunk: &[f32]| -> usize {
        chunk
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let v = config.vocab_size;
    assert_eq!(argmax(&logits_paged[..v]), argmax(&logits_naive[..v]));
    assert_eq!(argmax(&logits_paged[v..]), argmax(&logits_naive[v..]));
}
