// This is purely so that it works with torch 2.1. For torch 2.2+ we can include ATen/cuda/PhiloxUtils.cuh

#pragma once

#ifdef FORGE_NO_PYTORCH
// Stub: dropout is disabled in the forge path, so philox state is never read.
#include <tuple>
#include <cstdint>
namespace at { namespace cuda { namespace philox {
inline std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState const& state) {
    return {state.seed_, state.offset_};
}
}}} // namespace at::cuda::philox
#else
#include <ATen/cuda/detail/UnpackRaw.cuh>
#endif
