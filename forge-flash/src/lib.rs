//! FlashAttention v2 FFI bindings.
//!
//! Provides safe Rust wrappers around vendored FA2 C++ CUDA kernels.
//! Supports SM80 (Ampere) and SM90 (Hopper).

/// Data type for FlashAttention inputs.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum FlashDType {
    F16 = 0,
    BF16 = 1,
}
