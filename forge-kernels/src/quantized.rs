//! Quantized GEMV CUDA kernels.
//!
//! `gemv_q8_0_f16` computes `out = x · Wᵀ` for an f16 activation `x` (`[m, k]`,
//! row-major) against a Q8_0-quantized weight matrix `W` (`[n, k]`,
//! row-major), producing an f16 output `out` (`[m, n]`, row-major).
//!
//! Q8_0 block layout (GGUF `block_q8_0`): each 32-element block is 34 bytes —
//! a 2-byte little-endian f16 scale followed by 32 × i8 quants. Weight row `j`
//! is laid out as `k / 32` consecutive blocks. Dequant: `w = scale * q`.
//!
//! There is no F32 variant — quantized decode runs purely on the f16 path.

pub const F32_SRC: &str = "";

pub const F16_SRC: &str = r#"
// Read the f16 scale (delta) from a 34-byte Q8_0 block. The scale is stored
// little-endian in the first 2 bytes; reconstruct via __half bit pattern.
__device__ __forceinline__ float q8_0_block_scale(const unsigned char* block) {
    unsigned short bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

// Simple correctness-first kernel: one thread per output element (i, j).
// Each thread walks the k dimension in 32-element blocks, fetching the block
// scale once and accumulating x[i,l] * (scale * q) in f32.
//
//   out[i*n + j] = sum_l x[i*k + l] * dequant(W[j, l])
//
// wq is the quantized [n, k] weight: row j starts at byte offset
// j * (k/32) * 34, and within the row each block covers 32 contiguous l's.
extern "C" __global__ void gemv_q8_0_f16(
    __half* out,
    const __half* x,
    const unsigned char* wq,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = m * n;
    if (tid >= total) return;

    unsigned int i = tid / n;  // activation row
    unsigned int j = tid % n;  // weight row (output column)

    const unsigned int blocks_per_row = k / 32u;
    const unsigned char* row = wq + (size_t)j * blocks_per_row * 34u;
    const __half* xrow = x + (size_t)i * k;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row + (size_t)b * 34u;
        float scale = q8_0_block_scale(block);
        const signed char* quants = (const signed char*)(block + 2);
        unsigned int base = b * 32u;
        #pragma unroll
        for (unsigned int t = 0; t < 32u; ++t) {
            float xv = __half2float(xrow[base + t]);
            acc += xv * (scale * (float)quants[t]);
        }
    }

    out[(size_t)i * n + j] = __float2half(acc);
}
"#;
