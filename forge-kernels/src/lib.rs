//! Forge CUDA kernel source strings.
//!
//! Each module exports `F32_SRC` and `F16_SRC` constants containing CUDA C
//! source code. The backend concatenates these at runtime and compiles via NVRTC.
//!
//! **Note:** F16 sources assume `cuda_fp16.h` is already included by the caller.

pub mod attention;
pub mod elementwise;
pub mod memory;
pub mod norm;
pub mod positional;
