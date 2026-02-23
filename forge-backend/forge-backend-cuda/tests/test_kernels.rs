use forge_backend_cuda::attention::naive_attention;
use forge_backend_cuda::flash_attention::attention_fwd;
use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, Tensor};

#[test]
fn test_rms_norm() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4])
        .unwrap();
    let w = backend
        .copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0], &[4])
        .unwrap();
    let out = backend.rms_norm(&x, &w, 1e-5).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let rms = (7.5f32).sqrt();
    let expected: Vec<f32> = vec![1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "got {a}, expected {b}, diff={}",
            (a - b).abs()
        );
    }
}

#[test]
fn test_rms_norm_multi_row() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend
        .copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], &[2, 4])
        .unwrap();
    let w = backend
        .copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0], &[4])
        .unwrap();
    let out = backend.rms_norm(&x, &w, 1e-5).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    // Row 0: all 1s, RMS = 1.0, so output ≈ 1.0
    // Row 1: all 2s, RMS = 2.0, so output ≈ 1.0
    for &v in &result[..4] {
        assert!((v - 1.0).abs() < 1e-4, "row 0: got {v}");
    }
    for &v in &result[4..] {
        assert!((v - 1.0).abs() < 1e-4, "row 1: got {v}");
    }
}

#[test]
fn test_silu() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend
        .copy_from_host_f32(&[0.0, 1.0, -1.0], &[3])
        .unwrap();
    let out = backend.silu(&x).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    // silu(x) = x * sigmoid(x)
    assert!((result[0] - 0.0).abs() < 1e-4);
    assert!((result[1] - 0.7311).abs() < 1e-3);
    assert!((result[2] - (-0.2689)).abs() < 1e-3);
}

#[test]
fn test_softmax() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0], &[1, 3])
        .unwrap();
    let out = backend.softmax(&x, -1).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    assert!(result[2] > result[1] && result[1] > result[0]);
}

#[test]
fn test_softmax_multi_row() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend
        .copy_from_host_f32(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3])
        .unwrap();
    let out = backend.softmax(&x, -1).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let sum0: f32 = result[..3].iter().sum();
    let sum1: f32 = result[3..].iter().sum();
    assert!((sum0 - 1.0).abs() < 1e-5, "row 0 sum={sum0}");
    assert!((sum1 - 1.0).abs() < 1e-5, "row 1 sum={sum1}");
}

#[test]
fn test_embedding() {
    let backend = CudaBackend::new(0).unwrap();
    // Embedding table: 4 tokens x 3 dims
    let weight = backend
        .copy_from_host_f32(
            &[
                1.0, 2.0, 3.0, // token 0
                4.0, 5.0, 6.0, // token 1
                7.0, 8.0, 9.0, // token 2
                10.0, 11.0, 12.0, // token 3
            ],
            &[4, 3],
        )
        .unwrap();
    let out = backend.embedding(&weight, &[2, 0, 3]).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    assert_eq!(result, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
}

#[test]
fn test_naive_attention_single_head() {
    let backend = CudaBackend::new(0).unwrap();
    // Q, K, V: [1, 2, 1, 4] — batch=1, seq_len=2, heads=1, head_dim=4
    let q = backend
        .copy_from_host_f32(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[1, 2, 1, 4],
        )
        .unwrap();
    let k = backend
        .copy_from_host_f32(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[1, 2, 1, 4],
        )
        .unwrap();
    let v = backend
        .copy_from_host_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 2, 1, 4],
        )
        .unwrap();

    let scale = 1.0 / (4.0f32).sqrt(); // 1/sqrt(head_dim)
    let out = naive_attention(&backend, &q, &k, &v, scale).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();

    assert_eq!(out.shape(), &[1, 2, 1, 4]);
    // Q[0] = [1,0,0,0] should attend more to K[0] = [1,0,0,0]
    // Q[1] = [0,1,0,0] should attend more to K[1] = [0,1,0,0]
    // So output[0] should be closer to V[0] and output[1] closer to V[1]
    assert!(result[0] < result[4], "first token output should be closer to V[0]");
    assert!(result.len() == 8);
}

#[test]
fn test_naive_attention_multi_head_layout() {
    let backend = CudaBackend::new(0).unwrap();
    // Q, K, V: [1, 1, 2, 2] — batch=1, seq_len=1, heads=2, head_dim=2
    // Single token, 2 heads — tests that output is [batch, seq_len, num_heads, head_dim]
    //
    // Q layout [1,1,2,2]: token0_head0=[1,0], token0_head1=[0,1]
    // K layout [1,1,2,2]: token0_head0=[1,0], token0_head1=[0,1]
    // V layout [1,1,2,2]: token0_head0=[10,20], token0_head1=[30,40]
    let q = backend
        .copy_from_host_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
        .unwrap();
    let k = backend
        .copy_from_host_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
        .unwrap();
    let v = backend
        .copy_from_host_f32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 2, 2])
        .unwrap();

    let scale = 1.0 / (2.0f32).sqrt();
    let out = naive_attention(&backend, &q, &k, &v, scale).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);

    let result = backend.copy_to_host_f32(&out).unwrap();
    // With single KV token per head, softmax is trivially 1.0, so output = V
    // Output layout should be: [token0_head0, token0_head1] = [10, 20, 30, 40]
    assert!((result[0] - 10.0).abs() < 1e-3, "got {} expected 10.0", result[0]);
    assert!((result[1] - 20.0).abs() < 1e-3, "got {} expected 20.0", result[1]);
    assert!((result[2] - 30.0).abs() < 1e-3, "got {} expected 30.0", result[2]);
    assert!((result[3] - 40.0).abs() < 1e-3, "got {} expected 40.0", result[3]);
}

#[test]
fn test_naive_attention_multi_head_multi_token() {
    let backend = CudaBackend::new(0).unwrap();
    // This is the critical test: seq_len > 1 AND num_heads > 1.
    // Q, K, V: [1, 2, 2, 2] — batch=1, seq_len=2, heads=2, head_dim=2
    //
    // Memory layout (token-major): [t0h0, t0h1, t1h0, t1h1]
    // V values chosen so each head+token combo is unique and verifiable.
    //
    // V: t0h0=[1,2], t0h1=[3,4], t1h0=[5,6], t1h1=[7,8]
    // Using identity-like Q and K so softmax is uniform (kv_len=2 per head).

    // Q: all ones — each head attends uniformly to both K tokens
    let q = backend
        .copy_from_host_f32(
            &[
                1.0, 1.0, // t0h0
                1.0, 1.0, // t0h1
                1.0, 1.0, // t1h0
                1.0, 1.0, // t1h1
            ],
            &[1, 2, 2, 2],
        )
        .unwrap();
    let k = backend
        .copy_from_host_f32(
            &[
                1.0, 1.0, // t0h0
                1.0, 1.0, // t0h1
                1.0, 1.0, // t1h0
                1.0, 1.0, // t1h1
            ],
            &[1, 2, 2, 2],
        )
        .unwrap();
    let v = backend
        .copy_from_host_f32(
            &[
                1.0, 2.0, // t0h0
                3.0, 4.0, // t0h1
                5.0, 6.0, // t1h0
                7.0, 8.0, // t1h1
            ],
            &[1, 2, 2, 2],
        )
        .unwrap();

    let scale = 1.0 / (2.0f32).sqrt();
    let out = naive_attention(&backend, &q, &k, &v, scale).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 2]);

    let result = backend.copy_to_host_f32(&out).unwrap();
    // With causal masking (seq_len > 1, kv_len == seq_len):
    //
    // t0h0 (q_pos=0): attends only to KV pos 0 → V[t0h0] = [1,2]
    // t0h1 (q_pos=0): attends only to KV pos 0 → V[t0h1] = [3,4]
    // t1h0 (q_pos=1): attends to KV pos 0,1 uniformly → mean([1,2],[5,6]) = [3,4]
    // t1h1 (q_pos=1): attends to KV pos 0,1 uniformly → mean([3,4],[7,8]) = [5,6]
    //
    // Output layout [t0h0, t0h1, t1h0, t1h1]:
    let expected = [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-2,
            "index {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_naive_attention_invalid_head_ratio() {
    let backend = CudaBackend::new(0).unwrap();
    // num_heads=3, num_kv_heads=2 — not evenly divisible, should error
    let q = backend
        .copy_from_host_f32(&[0.0; 12], &[1, 1, 3, 4])
        .unwrap();
    let k = backend
        .copy_from_host_f32(&[0.0; 8], &[1, 1, 2, 4])
        .unwrap();
    let v = backend
        .copy_from_host_f32(&[0.0; 8], &[1, 1, 2, 4])
        .unwrap();

    let result = naive_attention(&backend, &q, &k, &v, 1.0);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("multiple"), "got: {err_msg}");
}

/// Test that `attention_fwd` delegates to naive attention for F32 tensors.
/// Phase 2 will replace with PagedAttention kernel.
#[test]
fn test_attention_fwd_f32_uses_naive() {
    let backend = CudaBackend::new(0).unwrap();
    let q = backend
        .copy_from_host_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
        .unwrap();
    let k = backend
        .copy_from_host_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
        .unwrap();
    let v = backend
        .copy_from_host_f32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 2, 2])
        .unwrap();

    let scale = 1.0 / (2.0f32).sqrt();
    let fwd_out = attention_fwd(&backend, &q, &k, &v, scale, false).unwrap();
    let naive_out = naive_attention(&backend, &q, &k, &v, scale).unwrap();

    let fwd_data = backend.copy_to_host_f32(&fwd_out).unwrap();
    let naive_data = backend.copy_to_host_f32(&naive_out).unwrap();

    assert_eq!(fwd_out.shape(), naive_out.shape());
    for (i, (&a, &b)) in fwd_data.iter().zip(naive_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "attention_fwd vs naive mismatch at {i}: {a} vs {b}"
        );
    }
}

/// Test that `attention_fwd` delegates to naive attention for F16 tensors.
/// Phase 2 will replace with PagedAttention kernel.
#[test]
fn test_attention_fwd_f16_fallback() {
    let backend = CudaBackend::new(0).unwrap();
    // Create F16 tensors
    let q_f32 = backend
        .copy_from_host_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
        .unwrap();
    let k_f32 = backend
        .copy_from_host_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])
        .unwrap();
    let v_f32 = backend
        .copy_from_host_f32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 2, 2])
        .unwrap();

    use forge_core::DType;
    let q = backend.cast(&q_f32, DType::F16).unwrap();
    let k = backend.cast(&k_f32, DType::F16).unwrap();
    let v = backend.cast(&v_f32, DType::F16).unwrap();

    let scale = 1.0 / (2.0f32).sqrt();
    let fwd_out = attention_fwd(&backend, &q, &k, &v, scale, false).unwrap();

    // Verify shape is correct
    assert_eq!(fwd_out.shape(), &[1, 1, 2, 2]);

    // Compare with naive attention on the same F16 inputs
    let naive_out = naive_attention(&backend, &q, &k, &v, scale).unwrap();
    let fwd_data = backend.copy_to_host_f32(&fwd_out).unwrap();
    let naive_data = backend.copy_to_host_f32(&naive_out).unwrap();

    for (i, (&a, &b)) in fwd_data.iter().zip(naive_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.1,
            "F16 attention_fwd vs naive mismatch at {i}: {a} vs {b}"
        );
    }
}

/// Test that `multi_head_attention` Backend method routes through attention_fwd
/// and produces output matching naive attention (reshaped to 2D).
#[test]
fn test_multi_head_attention_matches_naive() {
    let backend = CudaBackend::new(0).unwrap();
    // Q: [1, 2, 2, 4], K/V: [1, 3, 2, 4]
    let q_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let q = backend
        .copy_from_host_f32(&q_data, &[1, 2, 2, 4])
        .unwrap();

    let k_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.05).collect();
    let k = backend
        .copy_from_host_f32(&k_data, &[1, 3, 2, 4])
        .unwrap();
    let v_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.02 + 0.1).collect();
    let v = backend
        .copy_from_host_f32(&v_data, &[1, 3, 2, 4])
        .unwrap();

    let scale = 1.0 / (4.0_f32).sqrt();

    // Via Backend trait method (routes through attention_fwd → naive)
    let mha_out = backend
        .multi_head_attention(&q, &k, &v, 2, 2, 4, scale, true)
        .unwrap();

    // Via direct attention_fwd + manual reshape
    let fwd_out = attention_fwd(&backend, &q, &k, &v, scale, true).unwrap();
    let fwd_flat = backend.reshape(&fwd_out, &[2, 8]).unwrap();

    assert_eq!(mha_out.shape(), &[2, 8]);
    let mha_data = backend.copy_to_host_f32(&mha_out).unwrap();
    let fwd_data = backend.copy_to_host_f32(&fwd_flat).unwrap();

    for (i, (&a, &b)) in mha_data.iter().zip(fwd_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "multi_head_attention vs attention_fwd mismatch at {i}: {a} vs {b}"
        );
    }
}

/// Test GQA config through multi_head_attention on CUDA.
#[test]
fn test_multi_head_attention_gqa_cuda() {
    let backend = CudaBackend::new(0).unwrap();
    // GQA: num_heads=4, num_kv_heads=2
    let q_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let q = backend
        .copy_from_host_f32(&q_data, &[1, 1, 4, 4])
        .unwrap();

    let k_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.05).collect();
    let k = backend
        .copy_from_host_f32(&k_data, &[1, 3, 2, 4])
        .unwrap();
    let v_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.02 + 0.1).collect();
    let v = backend
        .copy_from_host_f32(&v_data, &[1, 3, 2, 4])
        .unwrap();

    let scale = 1.0 / (4.0_f32).sqrt();
    let result = backend
        .multi_head_attention(&q, &k, &v, 4, 2, 4, scale, false)
        .unwrap();

    // Output: [1, 4*4] = [1, 16]
    assert_eq!(result.shape(), &[1, 16]);
}

/// FA2-vs-naive correctness test (only runs with flash-attn feature).
#[test]
#[cfg(feature = "flash-attn")]
fn test_flash_attn_matches_naive_f16() {
    use forge_core::DType;

    let backend = CudaBackend::new(0).unwrap();

    let seq_len = 32;
    let kv_len = 32;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|i| ((i as f32 * 0.017) % 1.0) - 0.5)
        .collect();
    let k_data: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| ((i as f32 * 0.013) % 1.0) - 0.5)
        .collect();
    let v_data: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| ((i as f32 * 0.011) % 1.0) - 0.5)
        .collect();

    let q = backend
        .copy_from_host_f32(&q_data, &[1, seq_len, num_heads, head_dim])
        .unwrap();
    let k = backend
        .copy_from_host_f32(&k_data, &[1, kv_len, num_kv_heads, head_dim])
        .unwrap();
    let v = backend
        .copy_from_host_f32(&v_data, &[1, kv_len, num_kv_heads, head_dim])
        .unwrap();

    // Naive attention (always available)
    let naive_out = naive_attention(&backend, &q, &k, &v, scale).unwrap();
    let naive_flat = backend
        .reshape(&naive_out, &[seq_len, num_heads * head_dim])
        .unwrap();
    let naive_host = backend.copy_to_host_f32(&naive_flat).unwrap();

    // FA2 via multi_head_attention (feature-gated dispatch)
    let fa2_out = backend
        .multi_head_attention(&q, &k, &v, num_heads, num_kv_heads, head_dim, scale, true)
        .unwrap();
    let fa2_host = backend.copy_to_host_f32(&fa2_out).unwrap();

    for i in 0..fa2_host.len() {
        assert!(
            (fa2_host[i] - naive_host[i]).abs() < 1e-2,
            "FA2 vs naive mismatch at {i}: fa2={} naive={} diff={}",
            fa2_host[i],
            naive_host[i],
            (fa2_host[i] - naive_host[i]).abs()
        );
    }
}
