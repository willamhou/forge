//! Sampling CUDA kernels: per-row argmax (greedy decode).
//!
//! Each block reduces one row of `[rows, cols]` logits to the index of its
//! maximum value. Tie-break matches the CPU `Iterator::max_by` contract — on
//! equal values the HIGHEST index wins (`max_by` returns the last maximum) —
//! so GPU greedy decode is bit-for-bit identical to the CPU sampler.

/// Shared body for the per-row argmax reduction, parameterised on the logit
/// load expression (`LOAD`) so F32 and F16 share one implementation.
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

pub const F32_SRC: &str = argmax_src!("argmax_f32", "float", "x[i]");

// No `#include <cuda_fp16.h>` here — the backend prepends it once when
// assembling the F16 PTX bundle (see `CudaBackend::new`).
pub const F16_SRC: &str = argmax_src!("argmax_f16", "__half", "__half2float(x[i])");
