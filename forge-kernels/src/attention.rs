//! Attention CUDA kernels: extract_head, apply_causal_mask, interleave_heads.
//!
//! These kernels enable GPU-native attention without CPU roundtrips.
//! The per-head loop remains in Rust; Phase 2 will replace with fused PagedAttention.

pub const F32_SRC: &str = r#"
extern "C" __global__ void extract_head_f32(
    float* out,
    const float* input,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int head_idx
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len * head_dim) return;
    unsigned int t = i / head_dim;
    unsigned int d = i % head_dim;
    out[i] = input[t * num_heads * head_dim + head_idx * head_dim + d];
}

extern "C" __global__ void apply_causal_mask_f32(
    float* out,
    const float* scores,
    unsigned int seq_len,
    unsigned int kv_len
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len * kv_len) return;
    unsigned int q_pos = i / kv_len;
    unsigned int k_pos = i % kv_len;
    unsigned int abs_pos = kv_len - seq_len + q_pos;
    out[i] = (k_pos > abs_pos) ? -1e30f : scores[i];
}

extern "C" __global__ void interleave_heads_f32(
    float* out,
    const float* head_data,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int head_idx
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len * head_dim) return;
    unsigned int t = i / head_dim;
    unsigned int d = i % head_dim;
    out[t * num_heads * head_dim + head_idx * head_dim + d] = head_data[i];
}
"#;

pub const F16_SRC: &str = r#"
extern "C" __global__ void extract_head_f16(
    __half* out,
    const __half* input,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int head_idx
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len * head_dim) return;
    unsigned int t = i / head_dim;
    unsigned int d = i % head_dim;
    out[i] = input[t * num_heads * head_dim + head_idx * head_dim + d];
}

extern "C" __global__ void apply_causal_mask_f16(
    __half* out,
    const __half* scores,
    unsigned int seq_len,
    unsigned int kv_len
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len * kv_len) return;
    unsigned int q_pos = i / kv_len;
    unsigned int k_pos = i % kv_len;
    unsigned int abs_pos = kv_len - seq_len + q_pos;
    out[i] = (k_pos > abs_pos) ? __float2half(-1e4f) : scores[i];
}

extern "C" __global__ void interleave_heads_f16(
    __half* out,
    const __half* head_data,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int head_idx
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len * head_dim) return;
    unsigned int t = i / head_dim;
    unsigned int d = i % head_dim;
    out[t * num_heads * head_dim + head_idx * head_dim + d] = head_data[i];
}
"#;
