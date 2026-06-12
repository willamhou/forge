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

// Number of warps per CUDA block in the warp-per-row GEMV. The launch side
// (matmul_quant_into) must size grid.x = ceil(m * n / WARPS_PER_BLOCK) and
// blockDim.x = WARPS_PER_BLOCK * 32 to match this kernel.
//
// This kernel is the single-stream (m=1) decode GEMV: it reads the weight once
// and is memory-bound, beating cuBLAS f16 at m=1. It does NOT batch — at m>1
// each output row re-reads the full weight (cost scales m×), so the model
// dispatches batch decode (m>1) to the f16 GEMM instead. A quantized kernel
// that beats cuBLAS f16 at batch needs int8 tensor cores (W8A8), not a
// scalar-MAC GEMV; that is deliberately out of scope here.
pub const GEMV_Q8_0_WARPS_PER_BLOCK: u32 = 8;

pub const F16_SRC: &str = r#"
// Read the f16 scale (delta) from a 34-byte Q8_0 block. The scale is stored
// little-endian in the first 2 bytes; reconstruct via __half bit pattern.
__device__ __forceinline__ float q8_0_block_scale(const unsigned char* block) {
    unsigned short bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

// Warp-per-output-row Q8_0 GEMV. One warp (32 lanes) computes one output
// element out[i, j] = sum_l x[i, l] * dequant(W[j, l]); the warp index over
// the grid selects the (i, j) pair (row-major: warp w -> i = w / n, j = w % n).
//
// The 32 lanes split the k dimension by 32-element blocks: lane `l` handles
// the weight blocks l, l+32, l+64, ... of row j. Because consecutive lanes
// touch consecutive 34-byte blocks, the per-step weight reads across the warp
// are contiguous in memory (coalesced) — the key win over the old
// one-thread-per-output kernel, which had each thread stride a full row.
//
// wq is the quantized [n, k] weight: row j starts at byte offset
// j * (k/32) * 34, and within the row each block covers 32 contiguous l's.
//
// This is the m=1 single-sequence decode GEMV: it reads each weight row once
// and is memory-bound, beating cuBLAS f16 at m=1. It does NOT amortize the
// weight across a batch — at m>1, weight row j is re-read once per activation
// row i (cost scales m×). So the model dispatches batch decode (m>1) to the
// f16 GEMM (see QuantizedLinear::matmul_decode_into); this kernel is for m=1.
extern "C" __global__ void gemv_q8_0_f16(
    __half* out,
    const __half* x,
    const unsigned char* wq,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned int total = m * n;
    if (warp_id >= total) return;

    const unsigned int i = warp_id / n;  // activation row
    const unsigned int j = warp_id % n;  // weight row (output column)

    const unsigned int blocks_per_row = k / 32u;
    const unsigned char* row = wq + (size_t)j * blocks_per_row * 34u;
    const __half* xrow = x + (size_t)i * k;

    // Each lane accumulates the blocks it owns. Consecutive lanes (and thus the
    // warp's load instruction) read consecutive blocks -> coalesced weight reads.
    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
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

    // Warp-reduce the 32 partial sums; lane 0 holds the full dot product.
    #pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0u) {
        out[(size_t)i * n + j] = __float2half(acc);
    }
}
"#;
