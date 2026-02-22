//! Tests that batch decode forward produces the same output as sequential decode.

use forge_backend_cpu::CpuBackend;
use forge_core::{Backend, DType, KvCache, Model, ModelConfig, ModelInput, SeqMetadata};
use forge_kvcache::naive::NaiveKvCache;
use forge_model_llama::layers::{LlamaAttention, LlamaDecoderLayer, LlamaMLP, RMSNorm};
use forge_model_llama::rope::RopeFreqs;
use forge_model_llama::LlamaModel;

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

/// Build a tiny LlamaModel with deterministic random-ish weights.
fn build_tiny_model(backend: &CpuBackend) -> LlamaModel<CpuBackend> {
    let config = tiny_config();
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;

    // Simple deterministic weights (not random, but non-trivial)
    let make_weight = |rows: usize, cols: usize| -> <CpuBackend as Backend>::Tensor {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.01 + 0.1) % 1.0) - 0.5)
            .collect();
        backend.copy_from_host_f32(&data, &[rows, cols]).unwrap()
    };

    let embed_tokens = make_weight(vocab, h);
    let lm_head = make_weight(h, vocab);

    let norm = RMSNorm::new(
        backend
            .copy_from_host_f32(&vec![1.0; h], &[h])
            .unwrap(),
        config.rms_norm_eps,
    );

    // Create concatenated wqkv: [h, q_dim + 2*kv_dim]
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let attn = LlamaAttention::new(
        make_weight(h, q_dim + 2 * kv_dim), // wqkv
        make_weight(q_dim, h),              // wo
        &config,
    );

    let mlp = LlamaMLP::new(
        make_weight(h, inter), // gate
        make_weight(h, inter), // up
        make_weight(inter, h), // down
    );

    let layer_norm = RMSNorm::new(
        backend
            .copy_from_host_f32(&vec![1.0; h], &[h])
            .unwrap(),
        config.rms_norm_eps,
    );
    let post_norm = RMSNorm::new(
        backend
            .copy_from_host_f32(&vec![1.0; h], &[h])
            .unwrap(),
        config.rms_norm_eps,
    );

    let layer = LlamaDecoderLayer::new(layer_norm, attn, post_norm, mlp);
    let rope = RopeFreqs::precompute(&config, 64, backend).unwrap();

    LlamaModel::new(config, embed_tokens, vec![layer], norm, lm_head, rope, backend.clone())
}

/// Run prefill for a sequence to populate KV cache.
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

/// Run single-sequence decode and return logits.
fn decode_single(
    model: &LlamaModel<CpuBackend>,
    backend: &CpuBackend,
    kv_cache: &mut dyn KvCache<T = <CpuBackend as Backend>::Tensor>,
    seq_id: u64,
    token_id: u32,
    prompt_len: usize,
    generated_len: usize,
) -> Vec<f32> {
    let pos = (prompt_len + generated_len) as u32;
    let input = ModelInput {
        token_ids: vec![vec![token_id]],
        positions: vec![vec![pos]],
        seq_metadata: vec![SeqMetadata {
            seq_id,
            prompt_len,
            generated_len,
            is_prefill: false,
        }],
    };
    let output = model.forward(&input, kv_cache).unwrap();
    backend.copy_to_host_f32(&output.logits).unwrap()
}

#[test]
fn test_batch_decode_matches_sequential() {
    let backend = CpuBackend::new();
    let model = build_tiny_model(&backend);
    let config = tiny_config();

    // === Sequential path: two separate KV caches ===
    let mut kv_seq = NaiveKvCache::new(backend.clone(), config.num_hidden_layers, 4);

    // Prefill seq 1 with [0, 1, 2] and seq 2 with [1, 3]
    prefill(&model, &mut kv_seq, 1, &[0, 1, 2]);
    prefill(&model, &mut kv_seq, 2, &[1, 3]);

    // Decode seq 1 (token 3 at position 3) and seq 2 (token 0 at position 2)
    let logits_1 = decode_single(&model, &backend, &mut kv_seq, 1, 3, 3, 0);
    let logits_2 = decode_single(&model, &backend, &mut kv_seq, 2, 0, 2, 0);

    // === Batch path: fresh KV cache, same prefills ===
    let mut kv_batch = NaiveKvCache::new(backend.clone(), config.num_hidden_layers, 4);
    prefill(&model, &mut kv_batch, 1, &[0, 1, 2]);
    prefill(&model, &mut kv_batch, 2, &[1, 3]);

    // Batched decode: two sequences in one forward call
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
    let batch_output = model.forward(&input, &mut kv_batch).unwrap();
    let batch_logits = backend.copy_to_host_f32(&batch_output.logits).unwrap();

    // Batch output should be [2, vocab_size]
    assert_eq!(batch_logits.len(), 2 * config.vocab_size);

    let batch_logits_1 = &batch_logits[..config.vocab_size];
    let batch_logits_2 = &batch_logits[config.vocab_size..];

    // Verify batch matches sequential (within floating-point tolerance)
    for i in 0..config.vocab_size {
        assert!(
            (batch_logits_1[i] - logits_1[i]).abs() < 0.1,
            "seq1 logit[{i}]: batch={} sequential={} diff={}",
            batch_logits_1[i],
            logits_1[i],
            (batch_logits_1[i] - logits_1[i]).abs()
        );
    }
    for i in 0..config.vocab_size {
        assert!(
            (batch_logits_2[i] - logits_2[i]).abs() < 0.1,
            "seq2 logit[{i}]: batch={} sequential={} diff={}",
            batch_logits_2[i],
            logits_2[i],
            (batch_logits_2[i] - logits_2[i]).abs()
        );
    }
}

#[test]
fn test_single_seq_still_works() {
    let backend = CpuBackend::new();
    let model = build_tiny_model(&backend);
    let config = tiny_config();
    let mut kv = NaiveKvCache::new(backend.clone(), config.num_hidden_layers, 4);

    // Single sequence prefill + decode should work exactly as before
    prefill(&model, &mut kv, 1, &[0, 1]);
    let logits = decode_single(&model, &backend, &mut kv, 1, 2, 2, 0);
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_batch_rejects_prefill() {
    let backend = CpuBackend::new();
    let model = build_tiny_model(&backend);
    let config = tiny_config();
    let mut kv = NaiveKvCache::new(backend.clone(), config.num_hidden_layers, 4);
    kv.allocate(1, 2).unwrap();
    kv.allocate(2, 2).unwrap();

    // Multi-sequence with prefill should error
    let input = ModelInput {
        token_ids: vec![vec![0, 1], vec![2, 3]],
        positions: vec![vec![0, 1], vec![0, 1]],
        seq_metadata: vec![
            SeqMetadata {
                seq_id: 1,
                prompt_len: 2,
                generated_len: 0,
                is_prefill: true,
            },
            SeqMetadata {
                seq_id: 2,
                prompt_len: 2,
                generated_len: 0,
                is_prefill: true,
            },
        ],
    };
    assert!(model.forward(&input, &mut kv).is_err());
}
