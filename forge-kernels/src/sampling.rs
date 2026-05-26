//! Sampling CUDA kernels: per-row argmax (greedy) and Gumbel-max multinomial.
//!
//! Both reduce one row of `[rows, cols]` logits per block.
//!
//! - `argmax_*`: index of the maximum logit. Tie-break matches the CPU
//!   `Iterator::max_by` contract — on equal values the HIGHEST index wins —
//!   so GPU greedy decode is bit-for-bit identical to the CPU sampler.
//! - `sample_gumbel_*`: sample from `softmax(logits/temperature)` via the
//!   Gumbel-max trick — `argmax_i(logits_i/T + g_i)` where `g_i = -log(-log u_i)`
//!   and `u_i` is a counter-based uniform keyed on `(seed, step, row, col)`.
//!   This is the same `softmax`-multinomial draw vLLM/SGLang do on-device,
//!   reproducible run-to-run for a fixed seed on the same hardware (it does
//!   NOT reproduce the CPU `StdRng` sequence — by design, matching those
//!   engines).

/// Counter-based RNG (splitmix64 mixing) + Gumbel transform, shared by the
/// F32 and F16 sampling kernels. A macro (not a `const`) so it expands to a
/// string literal usable inside `concat!`. Defined once per PTX bundle.
macro_rules! rng_helpers {
    () => {
        r#"
__device__ __forceinline__ float forge_uniform(
    unsigned long long seed, unsigned int step, unsigned int row, unsigned int col
) {
    // Mix the (seed, step, row, col) coordinate into a well-distributed u64.
    unsigned long long x = seed
        ^ ((unsigned long long)step * 0x9E3779B97F4A7C15ULL)
        ^ (((unsigned long long)row << 32) | (unsigned long long)col);
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    // Top 24 bits → float in (0, 1), never exactly 0 or 1.
    unsigned int hi = (unsigned int)(x >> 40);
    return ((float)hi + 0.5f) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float forge_gumbel(
    unsigned long long seed, unsigned int step, unsigned int row, unsigned int col
) {
    float u = forge_uniform(seed, step, row, col);
    // u already in (0,1); -log(-log(u)) is finite.
    return -logf(-logf(u));
}
"#
    };
}

/// Per-row argmax reduction, parameterised on the logit load expression
/// (`LOAD`) so F32 and F16 share one implementation.
macro_rules! argmax_src {
    ($name:literal, $ty:literal, $load:literal) => {
        concat!(
            "extern \"C\" __global__ void ", $name, "(\n",
            "    unsigned int* out_ids,\n",
            "    const ", $ty, "* logits,\n",
            "    unsigned int rows,\n",
            "    unsigned int cols\n",
            ") {\n",
            "    unsigned int row = blockIdx.x;\n",
            "    if (row >= rows) return;\n",
            "    const ", $ty, "* x = logits + (size_t)row * cols;\n",
            "\n",
            // Unique name: a shared `extern __shared__` array shares one symbol
            // across the whole NVRTC translation unit, so it must not collide
            // with another kernel's differently-typed `smem`.
            "    extern __shared__ unsigned char argmax_smem[];\n",
            "    float* sval = (float*)argmax_smem;\n",
            "    unsigned int* sidx = (unsigned int*)(sval + blockDim.x);\n",
            "\n",
            // NVRTC has no <math.h> INFINITY; -1e38 is below any real logit.
            "    float best = -1e38f;\n",
            "    unsigned int best_i = 0;\n",
            "    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {\n",
            "        float v = ", $load, ";\n",
            // `>=` walks to the highest index on ties (i increases), matching
            // the CPU last-maximum tie-break. NaN compares false → never wins.
            "        if (v >= best) { best = v; best_i = i; }\n",
            "    }\n",
            "    sval[threadIdx.x] = best;\n",
            "    sidx[threadIdx.x] = best_i;\n",
            "    __syncthreads();\n",
            "\n",
            "    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n",
            "        if (threadIdx.x < s) {\n",
            "            float other = sval[threadIdx.x + s];\n",
            "            unsigned int oidx = sidx[threadIdx.x + s];\n",
            "            if (other > sval[threadIdx.x] ||\n",
            "                (other == sval[threadIdx.x] && oidx > sidx[threadIdx.x])) {\n",
            "                sval[threadIdx.x] = other;\n",
            "                sidx[threadIdx.x] = oidx;\n",
            "            }\n",
            "        }\n",
            "        __syncthreads();\n",
            "    }\n",
            "    if (threadIdx.x == 0) out_ids[row] = sidx[0];\n",
            "}\n",
        )
    };
}

/// Per-row Gumbel-max multinomial sample. Same reduction as `argmax_src` but
/// the comparison key is `logits[i] * inv_temp + gumbel(seed,step,row,i)`, so
/// `argmax` over the perturbed keys draws from `softmax(logits/temperature)`.
/// Ties broken by highest index (deterministic given the RNG key).
macro_rules! sample_gumbel_src {
    ($name:literal, $ty:literal, $load:literal) => {
        concat!(
            "extern \"C\" __global__ void ", $name, "(\n",
            "    unsigned int* out_ids,\n",
            "    const ", $ty, "* logits,\n",
            "    unsigned int rows,\n",
            "    unsigned int cols,\n",
            "    float inv_temp,\n",
            "    unsigned long long seed,\n",
            "    unsigned int step\n",
            ") {\n",
            "    unsigned int row = blockIdx.x;\n",
            "    if (row >= rows) return;\n",
            "    const ", $ty, "* x = logits + (size_t)row * cols;\n",
            "\n",
            "    extern __shared__ unsigned char argmax_smem[];\n",
            "    float* sval = (float*)argmax_smem;\n",
            "    unsigned int* sidx = (unsigned int*)(sval + blockDim.x);\n",
            "\n",
            "    float best = -1e38f;\n",
            "    unsigned int best_i = 0;\n",
            "    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {\n",
            "        float key = ", $load, " * inv_temp + forge_gumbel(seed, step, row, i);\n",
            "        if (key >= best) { best = key; best_i = i; }\n",
            "    }\n",
            "    sval[threadIdx.x] = best;\n",
            "    sidx[threadIdx.x] = best_i;\n",
            "    __syncthreads();\n",
            "\n",
            "    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n",
            "        if (threadIdx.x < s) {\n",
            "            float other = sval[threadIdx.x + s];\n",
            "            unsigned int oidx = sidx[threadIdx.x + s];\n",
            "            if (other > sval[threadIdx.x] ||\n",
            "                (other == sval[threadIdx.x] && oidx > sidx[threadIdx.x])) {\n",
            "                sval[threadIdx.x] = other;\n",
            "                sidx[threadIdx.x] = oidx;\n",
            "            }\n",
            "        }\n",
            "        __syncthreads();\n",
            "    }\n",
            "    if (threadIdx.x == 0) out_ids[row] = sidx[0];\n",
            "}\n",
        )
    };
}

pub const F32_SRC: &str = concat!(
    rng_helpers!(),
    argmax_src!("argmax_f32", "float", "x[i]"),
    sample_gumbel_src!("sample_gumbel_f32", "float", "x[i]"),
);

// No `#include <cuda_fp16.h>` here — the backend prepends it once when
// assembling the F16 PTX bundle (see `CudaBackend::new`). The RNG helpers are
// dtype-agnostic and live in this (separate) PTX bundle, so no symbol clash.
pub const F16_SRC: &str = concat!(
    rng_helpers!(),
    argmax_src!("argmax_f16", "__half", "__half2float(x[i])"),
    sample_gumbel_src!("sample_gumbel_f16", "__half", "__half2float(x[i])"),
);
