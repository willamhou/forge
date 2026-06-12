//! Memory/data CUDA kernels: transpose, cast, split_qkv.

pub const F32_SRC: &str = r#"
extern "C" __global__ void transpose_f32(
    float* out, const float* in_data,
    unsigned int rows, unsigned int cols
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    out[c * rows + r] = in_data[r * cols + c];
}

extern "C" __global__ void split_qkv_f32(
    float* q_out, float* k_out, float* v_out,
    const float* qkv, unsigned int rows,
    unsigned int q_cols, unsigned int kv_cols
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;
    if (row >= rows) return;
    unsigned int total_cols = q_cols + kv_cols + kv_cols;
    const float* src = qkv + row * total_cols;
    for (unsigned int c = col; c < q_cols; c += blockDim.x)
        q_out[row * q_cols + c] = src[c];
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        k_out[row * kv_cols + c] = src[q_cols + c];
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        v_out[row * kv_cols + c] = src[q_cols + kv_cols + c];
}
"#;

pub const F16_SRC: &str = r#"
extern "C" __global__ void transpose_f16(
    __half* out, const __half* in_data,
    unsigned int rows, unsigned int cols
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    out[c * rows + r] = in_data[r * cols + c];
}

// Specialized transpose for the cuBLASLt-output layout. Input is the col-major
// matmul output (m, n) — same bytes as row-major [n, m]. Output is row-major
// [m, n]. m is small (≤ 32 for batch decode), n can be up to ~10k.
//
// Naive `transpose_f16` writes with stride n (uncoalesced): a warp of 32
// threads writes to 32 different cache lines, eating most of the gain from
// nvjet over cutlass. Tiled version reads coalesced into shared mem, syncs,
// writes coalesced.
//
// Launch: grid = (ceil(n / TILE_N),), block = (TILE_N,), shared = m * (TILE_N + 1) * 2 bytes
// TILE_N must be passed as the block dim. m and n are passed as args.
extern "C" __global__ void transpose_narrow_f16(
    __half* out,             // [m, n] row-major
    const __half* scratch,   // [n, m] row-major (= col-major (m, n))
    unsigned int m,
    unsigned int n
) {
    extern __shared__ __half tn_tile[];   // [m, TILE_N + 1] with +1 padding
    unsigned int tx = threadIdx.x;
    unsigned int tile_n = blockDim.x;
    unsigned int n_base = blockIdx.x * tile_n;
    unsigned int n_idx = n_base + tx;
    unsigned int pad = tile_n + 1;        // pad column stride to dodge 32-bank conflicts

    // Phase 1: each thread reads its column from scratch into shared mem.
    if (n_idx < n) {
        const __half* src_row = scratch + n_idx * m;
        for (unsigned int mi = 0; mi < m; ++mi) {
            tn_tile[mi * pad + tx] = src_row[mi];
        }
    }
    __syncthreads();

    // Phase 2: write coalesced — warp lanes have consecutive tx → consecutive
    // n_idx → consecutive addresses in `out[mi * n + n_idx]`.
    if (n_idx < n) {
        for (unsigned int mi = 0; mi < m; ++mi) {
            out[mi * n + n_idx] = tn_tile[mi * pad + tx];
        }
    }
}

extern "C" __global__ void split_qkv_f16(
    __half* q_out, __half* k_out, __half* v_out,
    const __half* qkv, unsigned int rows,
    unsigned int q_cols, unsigned int kv_cols
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;
    if (row >= rows) return;
    unsigned int total_cols = q_cols + kv_cols + kv_cols;
    const __half* src = qkv + row * total_cols;
    for (unsigned int c = col; c < q_cols; c += blockDim.x)
        q_out[row * q_cols + c] = src[c];
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        k_out[row * kv_cols + c] = src[q_cols + c];
    for (unsigned int c = col; c < kv_cols; c += blockDim.x)
        v_out[row * kv_cols + c] = src[q_cols + kv_cols + c];
}

extern "C" __global__ void cast_f16_to_f32(
    float* out, const __half* input, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half2float(input[i]);
    }
}

extern "C" __global__ void cast_f32_to_f16(
    __half* out, const float* input, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(input[i]);
    }
}
"#;
