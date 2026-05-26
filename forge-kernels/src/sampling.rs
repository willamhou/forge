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
    // Clamp into the open interval: at hi=2^24-1 the uniform rounds to exactly
    // 1.0 in f32, which would make -log(-log(u)) = +inf and force that token.
    // Same clamp on the CPU side (gumbel_noise in forge-core) keeps them aligned.
    u = fminf(fmaxf(u, 1.0e-7f), 1.0f - 1.0e-7f);
    return -logf(-logf(u));
}

// Block reductions over `smem` (blockDim.x floats). Each ends with a
// __syncthreads so `smem` can be reused immediately. All threads get the result.
__device__ __forceinline__ float forge_block_reduce_max(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smem[threadIdx.x + s] > smem[threadIdx.x])
            smem[threadIdx.x] = smem[threadIdx.x + s];
        __syncthreads();
    }
    float r = smem[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float forge_block_reduce_min(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smem[threadIdx.x + s] < smem[threadIdx.x])
            smem[threadIdx.x] = smem[threadIdx.x + s];
        __syncthreads();
    }
    float r = smem[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float forge_block_reduce_sum(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float r = smem[0];
    __syncthreads();
    return r;
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
            // True -inf (bit pattern) so an all-(-inf)-key row reduces the same
            // way as the CPU path's f32::NEG_INFINITY seed (NVRTC has no INFINITY).
            "    float best = __uint_as_float(0xff800000u);\n",
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
            // True -inf (bit pattern) so an all-(-inf)-key row reduces the same
            // way as the CPU path's f32::NEG_INFINITY seed (NVRTC has no INFINITY).
            "    float best = __uint_as_float(0xff800000u);\n",
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

/// Per-row sampling with per-sequence temperature / min-p / top-k / top-p.
/// A row with `temp <= 0` decodes greedily (raw argmax, filters ignored); a
/// row with `temp > 0` draws via Gumbel-max over the tokens surviving the
/// filters. One call handles a batch mixing greedy and sampled sequences.
///
/// All three filters reduce to a single scaled-logit threshold τ (prob is
/// monotonic in the logit), and we keep tokens with `z_i >= τ` where
/// `τ = max(τ_minp, τ_topk, τ_topp)`:
/// - min-p: `prob >= min_p·max_prob` ⟺ `z_i >= z_max + ln(min_p)` (closed form).
/// - top-k: τ = the k-th largest `z_i`, found by bisecting on the count of
///   `z_i >= τ` (largest τ with count >= k).
/// - top-p: τ = the nucleus cutoff, found by bisecting on the retained mass
///   `Σ_{z_i>=τ} exp(z_i - z_max)` vs `top_p · Z` (largest τ with mass >= top_p·Z).
///
/// The max-logit token always passes, so at least one candidate survives.
/// Bisection (24 iters) gives full f32 threshold precision; sampling parity
/// with the CPU path is distributional, not bit-exact (matching vLLM/SGLang).
macro_rules! sample_perrow_src {
    ($name:literal, $ty:literal, $load:literal) => {
        concat!(
            "extern \"C\" __global__ void ", $name, "(\n",
            "    unsigned int* out_ids,\n",
            "    const ", $ty, "* logits,\n",
            "    unsigned int rows,\n",
            "    unsigned int cols,\n",
            "    const float* temps,\n",
            "    const float* min_ps,\n",
            "    const unsigned int* top_ks,\n",
            "    const float* top_ps,\n",
            "    const unsigned long long* seeds,\n",
            "    const unsigned int* steps\n",
            ") {\n",
            "    unsigned int row = blockIdx.x;\n",
            "    if (row >= rows) return;\n",
            "    const ", $ty, "* x = logits + (size_t)row * cols;\n",
            "    float temp = temps[row];\n",
            "    int do_sample = (temp > 0.0f);\n",
            "    float inv_temp = do_sample ? (1.0f / temp) : 0.0f;\n",
            "    float min_p = min_ps[row];\n",
            "    unsigned int top_k = top_ks[row];\n",
            "    float top_p = top_ps[row];\n",
            "    unsigned long long seed = seeds[row];\n",
            "    unsigned int step = steps[row];\n",
            "    int do_top_k = (do_sample && top_k > 0u && top_k < cols);\n",
            "    int do_top_p = (do_sample && top_p < 1.0f);\n",
            "\n",
            "    extern __shared__ unsigned char argmax_smem[];\n",
            "    float* sval = (float*)argmax_smem;\n",
            "    unsigned int* sidx = (unsigned int*)(sval + blockDim.x);\n",
            "    float neg_inf = __uint_as_float(0xff800000u);\n",
            "\n",
            "    float thresh = neg_inf;\n",
            "    if (do_sample && (min_p > 0.0f || do_top_k || do_top_p)) {\n",
            // z_max over the scaled logits.
            "        float lmax = neg_inf;\n",
            "        for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {\n",
            "            float z = ", $load, " * inv_temp;\n",
            "            if (z > lmax) lmax = z;\n",
            "        }\n",
            "        float z_max = forge_block_reduce_max(lmax, sval);\n",
            "        if (min_p > 0.0f) thresh = z_max + logf(min_p);\n",
            "\n",
            "        if (do_top_k || do_top_p) {\n",
            // z_min for the bisection range, and Z (total mass) for top-p.
            "            float lmin = -neg_inf;\n",
            "            float lz = 0.0f;\n",
            "            for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {\n",
            "                float z = ", $load, " * inv_temp;\n",
            "                if (z < lmin) lmin = z;\n",
            "                lz += __expf(z - z_max);\n",
            "            }\n",
            "            float z_min = forge_block_reduce_min(lmin, sval);\n",
            // Floor at the f32 exp-underflow boundary so `mid` stays finite even
            // if a logit is -inf (else the bisection collapses to -inf). Tokens
            // below z_max-88 contribute 0 mass in f32 regardless.
            "            z_min = fmaxf(z_min, z_max - 88.0f);\n",
            "            float Z = forge_block_reduce_sum(lz, sval);\n",
            "\n",
            "            if (do_top_k) {\n",
            "                float lo = z_min, hi = z_max;\n",
            "                for (int it = 0; it < 24; ++it) {\n",
            "                    float mid = 0.5f * (lo + hi);\n",
            "                    float c = 0.0f;\n",
            "                    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x)\n",
            "                        if (", $load, " * inv_temp >= mid) c += 1.0f;\n",
            "                    float cnt = forge_block_reduce_sum(c, sval);\n",
            "                    if (cnt >= (float)top_k) lo = mid; else hi = mid;\n",
            "                }\n",
            "                if (lo > thresh) thresh = lo;\n",
            "            }\n",
            "            if (do_top_p) {\n",
            "                float lo = z_min, hi = z_max;\n",
            "                float target = top_p * Z;\n",
            "                for (int it = 0; it < 24; ++it) {\n",
            "                    float mid = 0.5f * (lo + hi);\n",
            "                    float m = 0.0f;\n",
            "                    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {\n",
            "                        float z = ", $load, " * inv_temp;\n",
            "                        if (z >= mid) m += __expf(z - z_max);\n",
            "                    }\n",
            "                    float mass = forge_block_reduce_sum(m, sval);\n",
            "                    if (mass >= target) lo = mid; else hi = mid;\n",
            "                }\n",
            "                if (lo > thresh) thresh = lo;\n",
            "            }\n",
            "        }\n",
            "    }\n",
            "\n",
            // Final pass: keyed argmax. Sampled rows add Gumbel noise and drop
            // tokens below the combined threshold; greedy rows take raw argmax.
            "    float best = neg_inf;\n",
            "    unsigned int best_i = 0;\n",
            "    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {\n",
            "        float lg = ", $load, ";\n",
            "        float key;\n",
            "        if (do_sample) {\n",
            "            float z = lg * inv_temp;\n",
            "            key = (z >= thresh) ? (z + forge_gumbel(seed, step, row, i)) : neg_inf;\n",
            "        } else {\n",
            "            key = lg;\n",
            "        }\n",
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
    sample_perrow_src!("sample_perrow_f32", "float", "x[i]"),
);

// No `#include <cuda_fp16.h>` here — the backend prepends it once when
// assembling the F16 PTX bundle (see `CudaBackend::new`). The RNG helpers are
// dtype-agnostic and live in this (separate) PTX bundle, so no symbol clash.
pub const F16_SRC: &str = concat!(
    rng_helpers!(),
    argmax_src!("argmax_f16", "__half", "__half2float(x[i])"),
    sample_gumbel_src!("sample_gumbel_f16", "__half", "__half2float(x[i])"),
    sample_perrow_src!("sample_perrow_f16", "__half", "__half2float(x[i])"),
);
