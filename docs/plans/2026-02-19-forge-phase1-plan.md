# Forge Phase 1: MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working single-GPU LLM inference server that can load a Llama model from SafeTensors, serve OpenAI-compatible chat completions via HTTP with SSE streaming, using PagedAttention KV cache and continuous batching.

**Architecture:** Layered monolith — axum HTTP server → scheduler → model runtime → CUDA backend. All layers connected via Rust traits defined in `forge-core`. Initial implementation focuses on correctness over optimization: naive attention first, FlashAttention integration second.

**Tech Stack:** Rust (edition 2024), axum 0.8, tokio 1.x, cudarc 0.17, safetensors 0.7, tokenizers 0.21, minijinja 2.12, serde/serde_json, half (for f16/bf16 types)

**Design doc:** `docs/plans/2026-02-19-forge-design.md`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `forge-core/Cargo.toml`
- Create: `forge-core/src/lib.rs`
- Create: `forge-server/Cargo.toml`
- Create: `forge-server/src/main.rs`
- Create: `forge-backend/Cargo.toml`
- Create: `forge-backend/src/lib.rs`
- Create: `forge-backend/forge-backend-cuda/Cargo.toml`
- Create: `forge-backend/forge-backend-cuda/src/lib.rs`
- Create: `forge-runtime/Cargo.toml`
- Create: `forge-runtime/src/lib.rs`
- Create: `forge-scheduler/Cargo.toml`
- Create: `forge-scheduler/src/lib.rs`
- Create: `forge-kvcache/Cargo.toml`
- Create: `forge-kvcache/src/lib.rs`
- Create: `forge-models/forge-model-llama/Cargo.toml`
- Create: `forge-models/forge-model-llama/src/lib.rs`
- Create: `forge-loader/Cargo.toml`
- Create: `forge-loader/src/lib.rs`
- Create: `forge-kernels/Cargo.toml`
- Create: `forge-kernels/src/lib.rs`
- Create: `forge-kernels/csrc/` (empty, for CUDA C++ sources)
- Create: `forge-transport/Cargo.toml`
- Create: `forge-transport/src/lib.rs`
- Create: `forge-quantize/Cargo.toml`
- Create: `forge-quantize/src/lib.rs`
- Create: `.gitignore`
- Create: `rust-toolchain.toml`

**Step 1: Create workspace Cargo.toml**

```toml
# Cargo.toml
[workspace]
resolver = "2"
members = [
    "forge-core",
    "forge-server",
    "forge-backend",
    "forge-backend/forge-backend-cuda",
    "forge-runtime",
    "forge-scheduler",
    "forge-kvcache",
    "forge-models/forge-model-llama",
    "forge-loader",
    "forge-kernels",
    "forge-transport",
    "forge-quantize",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
repository = "https://github.com/user/forge"

[workspace.dependencies]
# Internal crates
forge-core = { path = "forge-core" }
forge-backend-cuda = { path = "forge-backend/forge-backend-cuda" }
forge-runtime = { path = "forge-runtime" }
forge-scheduler = { path = "forge-scheduler" }
forge-kvcache = { path = "forge-kvcache" }
forge-model-llama = { path = "forge-models/forge-model-llama" }
forge-loader = { path = "forge-loader" }
forge-kernels = { path = "forge-kernels" }
forge-transport = { path = "forge-transport" }
forge-quantize = { path = "forge-quantize" }

# External dependencies
tokio = { version = "1", features = ["full"] }
axum = { version = "0.8", features = ["ws"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
half = { version = "2", features = ["serde"] }
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
async-trait = "0.1"
uuid = { version = "1", features = ["v4"] }
rand = "0.8"
```

**Step 2: Create each crate's Cargo.toml with minimal deps**

Each crate:
- `forge-core`: `thiserror`, `half`, `serde`, `async-trait`
- `forge-server`: `forge-core`, `forge-runtime`, `forge-loader`, `forge-model-llama`, `forge-backend-cuda`, `forge-scheduler`, `forge-kvcache`, `forge-transport`, `axum`, `tokio`, `serde`, `serde_json`, `tracing`, `tracing-subscriber`, `uuid`
- `forge-backend`: `forge-core`
- `forge-backend-cuda`: `forge-core`, `forge-kernels`, `cudarc`
- `forge-runtime`: `forge-core`, `tokio`, `tracing`
- `forge-scheduler`: `forge-core`, `tracing`
- `forge-kvcache`: `forge-core`, `tracing`
- `forge-model-llama`: `forge-core`, `tracing`
- `forge-loader`: `forge-core`, `safetensors`, `serde`, `serde_json`, `tracing`, `memmap2`
- `forge-kernels`: `cudarc`
- `forge-transport`: `forge-core`, `async-trait`, `tokio`
- `forge-quantize`: `forge-core`

**Step 3: Create .gitignore and rust-toolchain.toml**

```
# .gitignore
/target
*.swp
*.swo
.env
*.gguf
*.safetensors
*.bin
```

```toml
# rust-toolchain.toml
[toolchain]
channel = "stable"
```

**Step 4: Create stub lib.rs/main.rs for each crate**

Each `lib.rs`:
```rust
//! Forge [crate-name]
```

`forge-server/src/main.rs`:
```rust
fn main() {
    println!("Forge LLM Inference Server");
}
```

**Step 5: Verify it builds**

Run: `cargo build --workspace`
Expected: Successful compilation with no errors.

**Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold Cargo workspace with all crates"
```

---

## Task 2: Core Types and Error Definitions

**Files:**
- Create: `forge-core/src/error.rs`
- Create: `forge-core/src/types.rs`
- Modify: `forge-core/src/lib.rs`

**Step 1: Define error types**

```rust
// forge-core/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ForgeError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    #[error("Model load error: {0}")]
    ModelLoad(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Sequence not found: {0}")]
    SeqNotFound(u64),

    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(DType),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, ForgeError>;
```

**Step 2: Define core types**

```rust
// forge-core/src/types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
    Q4K,
    Q8K,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 => 1,
            DType::I4 => 1, // packed: 2 values per byte, but min alloc is 1 byte
            DType::Q4K | DType::Q8K => 1, // block-quantized, varies
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
pub struct ModelInput {
    pub token_ids: Vec<Vec<u32>>,
    pub positions: Vec<Vec<u32>>,
    pub seq_metadata: Vec<SeqMetadata>,
}

#[derive(Debug, Clone)]
pub struct SeqMetadata {
    pub seq_id: u64,
    pub prompt_len: usize,
    pub generated_len: usize,
    pub is_prefill: bool,
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            min_p: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            max_tokens: 256,
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    MaxTokens,
    EosToken,
    StopString,
    Cancelled,
}
```

**Step 3: Wire up lib.rs**

```rust
// forge-core/src/lib.rs
pub mod error;
pub mod types;

pub use error::{ForgeError, Result};
pub use types::*;
```

**Step 4: Verify it builds**

Run: `cargo build -p forge-core`
Expected: Successful compilation.

**Step 5: Commit**

```bash
git add forge-core/
git commit -m "feat(core): add error types, DType, ModelConfig, SamplingParams"
```

---

## Task 3: Backend and Tensor Traits

**Files:**
- Create: `forge-core/src/backend.rs`
- Create: `forge-core/src/tensor.rs`
- Modify: `forge-core/src/lib.rs`

**Step 1: Define Tensor trait**

```rust
// forge-core/src/tensor.rs
use crate::{DType, Result};

pub trait Tensor: Clone + Send + Sync + std::fmt::Debug {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }
    fn size_bytes(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }
}
```

**Step 2: Define Backend trait**

```rust
// forge-core/src/backend.rs
use crate::{DType, Result};
use crate::tensor::Tensor;

pub trait Backend: Send + Sync + 'static {
    type Tensor: Tensor;

    fn name(&self) -> &str;
    fn device_count(&self) -> usize;

    // Allocation
    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;
    fn allocate_zeros(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;

    // Data transfer
    fn copy_from_host_f32(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_from_host_f16(&self, data: &[half::f16], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_from_host_bf16(&self, data: &[half::bf16], shape: &[usize]) -> Result<Self::Tensor>;
    fn copy_to_host_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>>;

    // Synchronization
    fn synchronize(&self) -> Result<()>;

    // Core ops
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn mul_scalar(&self, a: &Self::Tensor, scalar: f32) -> Result<Self::Tensor>;
    fn silu(&self, a: &Self::Tensor) -> Result<Self::Tensor>;
    fn rms_norm(&self, x: &Self::Tensor, weight: &Self::Tensor, eps: f32) -> Result<Self::Tensor>;
    fn rope(&self, x: &Self::Tensor, freqs_cos: &Self::Tensor, freqs_sin: &Self::Tensor) -> Result<Self::Tensor>;
    fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor>;
    fn embedding(&self, weight: &Self::Tensor, indices: &[u32]) -> Result<Self::Tensor>;
    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor>;
    fn transpose(&self, x: &Self::Tensor, dim0: usize, dim1: usize) -> Result<Self::Tensor>;
    fn cat(&self, tensors: &[&Self::Tensor], dim: usize) -> Result<Self::Tensor>;
}
```

**Step 3: Update lib.rs**

```rust
// forge-core/src/lib.rs
pub mod backend;
pub mod error;
pub mod tensor;
pub mod types;

pub use backend::Backend;
pub use error::{ForgeError, Result};
pub use tensor::Tensor;
pub use types::*;
```

**Step 4: Verify it builds**

Run: `cargo build -p forge-core`
Expected: Successful compilation.

**Step 5: Commit**

```bash
git add forge-core/
git commit -m "feat(core): add Backend and Tensor trait definitions"
```

---

## Task 4: Model, KvCache, Scheduler Traits

**Files:**
- Create: `forge-core/src/model.rs`
- Create: `forge-core/src/kvcache.rs`
- Create: `forge-core/src/scheduler.rs`
- Create: `forge-core/src/sampling.rs`
- Modify: `forge-core/src/lib.rs`

**Step 1: Define Model trait**

```rust
// forge-core/src/model.rs
use crate::{DType, ModelConfig, ModelInput, Result};
use crate::tensor::Tensor;

pub struct ModelOutput<T: Tensor> {
    pub logits: T, // [batch, vocab_size]
}

pub trait Model: Send + Sync {
    type T: Tensor;

    fn forward(
        &self,
        input: &ModelInput,
        kv_cache: &mut dyn KvCache<T = Self::T>,
    ) -> Result<ModelOutput<Self::T>>;

    fn config(&self) -> &ModelConfig;
}

use crate::kvcache::KvCache;
```

**Step 2: Define KvCache trait**

```rust
// forge-core/src/kvcache.rs
use crate::{Result, SeqMetadata};
use crate::tensor::Tensor;

pub struct CacheUsage {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub block_size: usize,
}

impl CacheUsage {
    pub fn free_blocks(&self) -> usize {
        self.total_blocks - self.used_blocks
    }

    pub fn usage_ratio(&self) -> f32 {
        self.used_blocks as f32 / self.total_blocks as f32
    }
}

pub trait KvCache: Send + Sync {
    type T: Tensor;

    /// Allocate cache space for a new sequence
    fn allocate(&mut self, seq_id: u64, initial_len: usize) -> Result<()>;

    /// Append new KV to cache for a specific layer
    fn append(
        &mut self,
        seq_id: u64,
        layer: usize,
        key: &Self::T,
        value: &Self::T,
    ) -> Result<()>;

    /// Get block table for a sequence (PagedAttention)
    fn get_block_table(&self, seq_id: u64) -> Result<Vec<usize>>;

    /// Get the current sequence length in cache
    fn get_seq_len(&self, seq_id: u64) -> Result<usize>;

    /// Free cache for a completed sequence
    fn free(&mut self, seq_id: u64) -> Result<()>;

    /// Current cache usage
    fn usage(&self) -> CacheUsage;

    /// Check if we can allocate for a given length
    fn can_allocate(&self, num_tokens: usize) -> bool;
}
```

**Step 3: Define Scheduler trait and InferenceRequest**

```rust
// forge-core/src/scheduler.rs
use crate::{FinishReason, Result, SamplingParams, SeqMetadata};

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
}

#[derive(Debug, Clone)]
pub struct RequestHandle {
    pub request_id: String,
    pub seq_id: u64,
}

#[derive(Debug, Default)]
pub struct ScheduleBatch {
    pub prefill_seqs: Vec<ScheduledSeq>,
    pub decode_seqs: Vec<ScheduledSeq>,
}

#[derive(Debug, Clone)]
pub struct ScheduledSeq {
    pub seq_id: u64,
    pub token_ids: Vec<u32>,
    pub position_offset: usize,
    pub sampling_params: SamplingParams,
    pub is_prefill: bool,
}

impl ScheduleBatch {
    pub fn is_empty(&self) -> bool {
        self.prefill_seqs.is_empty() && self.decode_seqs.is_empty()
    }

    pub fn total_seqs(&self) -> usize {
        self.prefill_seqs.len() + self.decode_seqs.len()
    }
}

pub trait Scheduler: Send + Sync {
    fn enqueue(&mut self, request: InferenceRequest) -> Result<RequestHandle>;
    fn cancel(&mut self, seq_id: u64) -> Result<()>;
    fn schedule(&mut self, cache_usage: &CacheUsage) -> Result<ScheduleBatch>;
    fn finish(&mut self, seq_id: u64, reason: FinishReason) -> Result<()>;
    fn append_token(&mut self, seq_id: u64, token_id: u32) -> Result<()>;
    fn get_generated_tokens(&self, seq_id: u64) -> Result<Vec<u32>>;
}

use crate::kvcache::CacheUsage;
```

**Step 4: Define sampling types**

```rust
// forge-core/src/sampling.rs
use crate::{Result, SamplingParams};
use std::collections::HashMap;

pub struct SamplingContext<'a> {
    pub generated_tokens: &'a [u32],
    pub prompt_tokens: &'a [u32],
    pub token_counts: &'a HashMap<u32, usize>,
}

pub struct SampleResult {
    pub token_id: u32,
    pub logprob: f32,
}
```

**Step 5: Update lib.rs**

```rust
// forge-core/src/lib.rs
pub mod backend;
pub mod error;
pub mod kvcache;
pub mod model;
pub mod sampling;
pub mod scheduler;
pub mod tensor;
pub mod types;

pub use backend::Backend;
pub use error::{ForgeError, Result};
pub use kvcache::{CacheUsage, KvCache};
pub use model::{Model, ModelOutput};
pub use sampling::{SampleResult, SamplingContext};
pub use scheduler::{
    InferenceRequest, RequestHandle, ScheduleBatch, ScheduledSeq, Scheduler,
};
pub use tensor::Tensor;
pub use types::*;
```

**Step 6: Verify it builds**

Run: `cargo build -p forge-core`
Expected: Successful compilation.

**Step 7: Commit**

```bash
git add forge-core/
git commit -m "feat(core): add Model, KvCache, Scheduler, Sampling traits"
```

---

## Task 5: CUDA Backend — cudarc Tensor and Basic Ops

**Files:**
- Create: `forge-backend/forge-backend-cuda/src/tensor.rs`
- Create: `forge-backend/forge-backend-cuda/src/ops.rs`
- Create: `forge-backend/forge-backend-cuda/src/backend.rs`
- Modify: `forge-backend/forge-backend-cuda/src/lib.rs`
- Create: `forge-backend/forge-backend-cuda/tests/test_ops.rs`

**Step 1: Write tests for CudaTensor basic operations**

```rust
// forge-backend/forge-backend-cuda/tests/test_ops.rs
use forge_backend_cuda::CudaBackend;
use forge_core::Backend;

#[test]
fn test_copy_roundtrip_f32() {
    let backend = CudaBackend::new(0).unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = backend.copy_from_host_f32(&data, &[2, 2]).unwrap();
    let result = backend.copy_to_host_f32(&tensor).unwrap();
    assert_eq!(data, result);
}

#[test]
fn test_tensor_shape() {
    let backend = CudaBackend::new(0).unwrap();
    let data = vec![0.0f32; 12];
    let tensor = backend.copy_from_host_f32(&data, &[3, 4]).unwrap();
    assert_eq!(tensor.shape(), &[3, 4]);
}

#[test]
fn test_add() {
    let backend = CudaBackend::new(0).unwrap();
    let a = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let b = backend.copy_from_host_f32(&[10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
    let c = backend.add(&a, &b).unwrap();
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_mul_scalar() {
    let backend = CudaBackend::new(0).unwrap();
    let a = backend.copy_from_host_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let c = backend.mul_scalar(&a, 2.0).unwrap();
    let result = backend.copy_to_host_f32(&c).unwrap();
    assert_eq!(result, vec![2.0, 4.0, 6.0]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p forge-backend-cuda`
Expected: FAIL — types and functions not yet defined.

**Step 3: Implement CudaTensor**

```rust
// forge-backend/forge-backend-cuda/src/tensor.rs
use cudarc::driver::{CudaSlice, DeviceSlice};
use forge_core::{DType, Tensor};

#[derive(Debug, Clone)]
pub struct CudaTensor {
    pub(crate) data: CudaSlice<u8>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

impl Tensor for CudaTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}
```

**Step 4: Implement CudaBackend with basic ops**

```rust
// forge-backend/forge-backend-cuda/src/backend.rs
use std::sync::Arc;
use cudarc::driver::*;
use cudarc::cublas::CudaBlas;
use forge_core::{Backend, DType, ForgeError, Result};
use crate::tensor::CudaTensor;

pub struct CudaBackend {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) blas: Arc<CudaBlas>,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal)
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| ForgeError::Cuda(e.to_string()))?;
        Ok(Self {
            device: Arc::new(device),
            blas: Arc::new(blas),
        })
    }
}
```

Implement Backend trait methods: `copy_from_host_f32`, `copy_to_host_f32`, `add`, `mul_scalar`, `matmul` (via cuBLAS), etc. Use `cudarc::driver` for memory management and `cudarc::cublas` for GEMM.

**Step 5: Wire up lib.rs**

```rust
// forge-backend/forge-backend-cuda/src/lib.rs
pub mod backend;
pub mod tensor;

pub use backend::CudaBackend;
pub use tensor::CudaTensor;
```

**Step 6: Run tests**

Run: `cargo test -p forge-backend-cuda`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add forge-backend/
git commit -m "feat(cuda): implement CudaBackend with cudarc, basic tensor ops"
```

---

## Task 6: CUDA Kernels — RMSNorm, RoPE, SiLU, Softmax, Embedding

**Files:**
- Create: `forge-kernels/csrc/kernels.cu`
- Create: `forge-kernels/csrc/kernels.h`
- Create: `forge-kernels/build.rs`
- Modify: `forge-kernels/src/lib.rs`
- Create: `forge-kernels/src/bindings.rs`
- Create: `forge-backend/forge-backend-cuda/tests/test_kernels.rs`

**Step 1: Write tests for kernel operations**

```rust
// forge-backend/forge-backend-cuda/tests/test_kernels.rs
use forge_backend_cuda::CudaBackend;
use forge_core::Backend;

#[test]
fn test_rms_norm() {
    let backend = CudaBackend::new(0).unwrap();
    // Input: [1.0, 2.0, 3.0, 4.0], weight: [1.0, 1.0, 1.0, 1.0], eps=1e-5
    // RMS = sqrt(mean([1, 4, 9, 16])) = sqrt(7.5) ≈ 2.7386
    // output = input / RMS * weight
    let x = backend.copy_from_host_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let w = backend.copy_from_host_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
    let out = backend.rms_norm(&x, &w, 1e-5).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let rms = (7.5f32).sqrt();
    let expected: Vec<f32> = vec![1.0/rms, 2.0/rms, 3.0/rms, 4.0/rms];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "got {a}, expected {b}");
    }
}

#[test]
fn test_silu() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend.copy_from_host_f32(&[0.0, 1.0, -1.0], &[3]).unwrap();
    let out = backend.silu(&x).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    // silu(x) = x * sigmoid(x)
    // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
    assert!((result[0] - 0.0).abs() < 1e-4);
    assert!((result[1] - 0.7311).abs() < 1e-3);
    assert!((result[2] - (-0.2689)).abs() < 1e-3);
}

#[test]
fn test_softmax() {
    let backend = CudaBackend::new(0).unwrap();
    let x = backend.copy_from_host_f32(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let out = backend.softmax(&x, -1).unwrap();
    let result = backend.copy_to_host_f32(&out).unwrap();
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(result[2] > result[1] && result[1] > result[0]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p forge-backend-cuda -- test_kernels`
Expected: FAIL — kernel functions not implemented.

**Step 3: Write CUDA C++ kernels**

```c
// forge-kernels/csrc/kernels.cu
#include "kernels.h"
#include <cmath>

// RMS Norm kernel
__global__ void rms_norm_kernel(
    float* output,
    const float* input,
    const float* weight,
    float eps,
    int hidden_size
) {
    int row = blockIdx.x;
    const float* x = input + row * hidden_size;
    float* o = output + row * hidden_size;

    // Compute sum of squares
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    // Block-level reduction
    __shared__ float shared[32];
    // ... warp reduction + shared memory reduction ...
    float rms = rsqrtf(ss / hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        o[i] = x[i] * rms * weight[i];
    }
}

// SiLU kernel
__global__ void silu_kernel(float* output, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// RoPE, Softmax, Embedding kernels ...

// C interface
extern "C" {
    void launch_rms_norm(float* output, const float* input, const float* weight,
                         float eps, int batch_size, int hidden_size, cudaStream_t stream);
    void launch_silu(float* output, const float* input, int n, cudaStream_t stream);
    void launch_softmax(float* output, const float* input, int rows, int cols, cudaStream_t stream);
    void launch_rope(float* output, const float* input, const float* cos_freqs,
                     const float* sin_freqs, int batch, int seq_len, int heads, int head_dim,
                     cudaStream_t stream);
    void launch_embedding(float* output, const float* weight, const int* indices,
                          int num_indices, int embedding_dim, cudaStream_t stream);
}
```

**Step 4: Create build.rs for CUDA compilation**

```rust
// forge-kernels/build.rs
fn main() {
    println!("cargo:rerun-if-changed=csrc/kernels.cu");
    println!("cargo:rerun-if-changed=csrc/kernels.h");

    cc::Build::new()
        .cuda(true)
        .file("csrc/kernels.cu")
        .flag("-O3")
        .flag("--use_fast_math")
        .compile("forge_kernels");
}
```

**Step 5: Create Rust FFI bindings**

```rust
// forge-kernels/src/bindings.rs
use std::ffi::c_void;

type CudaStream = *mut c_void;

extern "C" {
    pub fn launch_rms_norm(
        output: *mut f32, input: *const f32, weight: *const f32,
        eps: f32, batch_size: i32, hidden_size: i32, stream: CudaStream,
    );
    pub fn launch_silu(
        output: *mut f32, input: *const f32, n: i32, stream: CudaStream,
    );
    pub fn launch_softmax(
        output: *mut f32, input: *const f32, rows: i32, cols: i32, stream: CudaStream,
    );
    pub fn launch_rope(
        output: *mut f32, input: *const f32, cos_freqs: *const f32, sin_freqs: *const f32,
        batch: i32, seq_len: i32, heads: i32, head_dim: i32, stream: CudaStream,
    );
    pub fn launch_embedding(
        output: *mut f32, weight: *const f32, indices: *const i32,
        num_indices: i32, embedding_dim: i32, stream: CudaStream,
    );
}
```

**Step 6: Integrate kernels into CudaBackend ops**

Update `forge-backend-cuda/src/ops.rs` to call the FFI bindings for `rms_norm`, `silu`, `softmax`, `rope`, `embedding`.

**Step 7: Run tests**

Run: `cargo test -p forge-backend-cuda`
Expected: All tests pass.

**Step 8: Commit**

```bash
git add forge-kernels/ forge-backend/
git commit -m "feat(kernels): CUDA kernels for RMSNorm, SiLU, Softmax, RoPE, Embedding"
```

---

## Task 7: SafeTensors Weight Loader

**Files:**
- Create: `forge-loader/src/safetensors.rs`
- Create: `forge-loader/src/config.rs`
- Modify: `forge-loader/src/lib.rs`
- Create: `forge-loader/tests/test_loader.rs`

**Step 1: Write test for config loading**

```rust
// forge-loader/tests/test_loader.rs
use forge_loader::LlamaConfig;

#[test]
fn test_parse_llama_config() {
    let json = r#"{
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    }"#;
    let config: LlamaConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p forge-loader`
Expected: FAIL — `LlamaConfig` not defined.

**Step 3: Implement config parser**

```rust
// forge-loader/src/config.rs
use serde::Deserialize;
use forge_core::ModelConfig;

#[derive(Debug, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: Option<usize>,
}

fn default_kv_heads() -> usize { 32 }
fn default_rms_eps() -> f32 { 1e-5 }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_head_dim() -> Option<usize> { None }

impl LlamaConfig {
    pub fn to_model_config(&self) -> ModelConfig {
        let head_dim = self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        ModelConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            head_dim,
            rms_norm_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            dtype: forge_core::DType::F16,
        }
    }
}
```

**Step 4: Implement SafeTensors loader**

```rust
// forge-loader/src/safetensors.rs
use std::path::Path;
use safetensors::SafeTensors;
use memmap2::Mmap;
use forge_core::{Backend, DType, Result, ForgeError};

pub struct SafeTensorsLoader {
    mmaps: Vec<Mmap>,
}

impl SafeTensorsLoader {
    pub fn new(model_dir: &Path) -> Result<Self> {
        // Find all .safetensors files
        let mut files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .map(|e| e.path())
            .collect();
        files.sort();

        if files.is_empty() {
            return Err(ForgeError::ModelLoad(
                format!("No .safetensors files found in {:?}", model_dir),
            ));
        }

        let mmaps = files.iter().map(|path| {
            let file = std::fs::File::open(path)?;
            unsafe { Mmap::map(&file) }.map_err(Into::into)
        }).collect::<Result<Vec<_>>>()?;

        Ok(Self { mmaps })
    }

    /// Load a specific tensor by name
    pub fn load_tensor<B: Backend>(
        &self,
        name: &str,
        backend: &B,
    ) -> Result<B::Tensor> {
        for mmap in &self.mmaps {
            let tensors = SafeTensors::deserialize(mmap)
                .map_err(|e| ForgeError::ModelLoad(e.to_string()))?;
            if let Ok(view) = tensors.tensor(name) {
                let shape: Vec<usize> = view.shape().to_vec();
                return self.view_to_tensor(view, &shape, backend);
            }
        }
        Err(ForgeError::ModelLoad(format!("Tensor '{}' not found", name)))
    }

    /// List all tensor names across all files
    pub fn tensor_names(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for mmap in &self.mmaps {
            let tensors = SafeTensors::deserialize(mmap)
                .map_err(|e| ForgeError::ModelLoad(e.to_string()))?;
            names.extend(tensors.names().map(String::from));
        }
        Ok(names)
    }

    fn view_to_tensor<B: Backend>(
        &self,
        view: safetensors::tensor::TensorView<'_>,
        shape: &[usize],
        backend: &B,
    ) -> Result<B::Tensor> {
        let data = view.data();
        match view.dtype() {
            safetensors::Dtype::F16 => {
                let f16_data: &[half::f16] = bytemuck::cast_slice(data);
                backend.copy_from_host_f16(f16_data, shape)
            }
            safetensors::Dtype::BF16 => {
                let bf16_data: &[half::bf16] = bytemuck::cast_slice(data);
                backend.copy_from_host_bf16(bf16_data, shape)
            }
            safetensors::Dtype::F32 => {
                let f32_data: &[f32] = bytemuck::cast_slice(data);
                backend.copy_from_host_f32(f32_data, shape)
            }
            other => Err(ForgeError::ModelLoad(
                format!("Unsupported safetensors dtype: {:?}", other),
            )),
        }
    }
}
```

**Step 5: Wire up lib.rs**

```rust
// forge-loader/src/lib.rs
pub mod config;
pub mod safetensors;

pub use config::LlamaConfig;
pub use self::safetensors::SafeTensorsLoader;
```

**Step 6: Run tests**

Run: `cargo test -p forge-loader`
Expected: Config parsing test passes.

**Step 7: Commit**

```bash
git add forge-loader/
git commit -m "feat(loader): SafeTensors weight loader with mmap + config parser"
```

---

## Task 8: Tokenizer + Chat Template

**Files:**
- Create: `forge-server/src/tokenizer.rs`
- Create: `forge-server/src/chat_template.rs`
- Create: `forge-server/tests/test_tokenizer.rs`

**Step 1: Write tests**

```rust
// forge-server/tests/test_tokenizer.rs
use forge_server::chat_template::ChatTemplate;

#[test]
fn test_chatml_template() {
    let template = ChatTemplate::chatml_default().unwrap();
    let messages = vec![
        ("system", "You are helpful."),
        ("user", "Hello!"),
    ];
    let rendered = template.apply(&messages, true).unwrap();
    assert!(rendered.contains("<|im_start|>system"));
    assert!(rendered.contains("You are helpful."));
    assert!(rendered.contains("<|im_start|>user"));
    assert!(rendered.contains("Hello!"));
    assert!(rendered.contains("<|im_start|>assistant"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p forge-server`
Expected: FAIL — module not defined.

**Step 3: Implement chat template**

```rust
// forge-server/src/chat_template.rs
use minijinja::Environment;
use forge_core::{ForgeError, Result};

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new(template_str: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.add_template_owned("chat", template_str.to_string())
            .map_err(|e| ForgeError::Internal(format!("Template parse error: {e}")))?;
        Ok(Self { env })
    }

    pub fn chatml_default() -> Result<Self> {
        Self::new(CHATML_TEMPLATE)
    }

    pub fn apply(&self, messages: &[(&str, &str)], add_generation_prompt: bool) -> Result<String> {
        let tmpl = self.env.get_template("chat")
            .map_err(|e| ForgeError::Internal(e.to_string()))?;

        let msgs: Vec<minijinja::Value> = messages.iter().map(|(role, content)| {
            minijinja::context! { role => *role, content => *content }
        }).collect();

        tmpl.render(minijinja::context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
        }).map_err(|e| ForgeError::Internal(e.to_string()))
    }
}

const CHATML_TEMPLATE: &str = r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;
```

**Step 4: Implement tokenizer wrapper**

```rust
// forge-server/src/tokenizer.rs
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;
use forge_core::{ForgeError, Result};

pub struct ForgeTokenizer {
    inner: HfTokenizer,
    eos_token_id: u32,
}

impl ForgeTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| ForgeError::Tokenizer(e.to_string()))?;
        // Try to find EOS token id
        let eos_token_id = inner.token_to_id("</s>")
            .or_else(|| inner.token_to_id("<|endoftext|>"))
            .or_else(|| inner.token_to_id("<|im_end|>"))
            .unwrap_or(2); // fallback
        Ok(Self { inner, eos_token_id })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| ForgeError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, true)
            .map_err(|e| ForgeError::Tokenizer(e.to_string()))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

/// Incremental decoder for streaming
pub struct IncrementalDecoder {
    pending_ids: Vec<u32>,
    prev_text_len: usize,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self {
            pending_ids: Vec::new(),
            prev_text_len: 0,
        }
    }

    pub fn add_token(&mut self, token_id: u32, tokenizer: &ForgeTokenizer) -> Option<String> {
        self.pending_ids.push(token_id);
        let decoded = tokenizer.decode(&self.pending_ids).ok()?;

        if decoded.ends_with('\u{FFFD}') {
            return None; // incomplete UTF-8
        }

        let new_text = decoded[self.prev_text_len..].to_string();
        self.prev_text_len = decoded.len();

        if new_text.is_empty() {
            None
        } else {
            Some(new_text)
        }
    }
}
```

**Step 5: Run tests**

Run: `cargo test -p forge-server`
Expected: ChatML test passes.

**Step 6: Commit**

```bash
git add forge-server/
git commit -m "feat(server): tokenizer wrapper + chat template engine (minijinja)"
```

---

## Task 9: Naive Attention + PagedAttention KV Cache

**Files:**
- Create: `forge-kvcache/src/paged.rs`
- Modify: `forge-kvcache/src/lib.rs`
- Create: `forge-kvcache/tests/test_paged.rs`
- Create: `forge-backend/forge-backend-cuda/src/attention.rs`

**Step 1: Write tests for PagedKvCache**

```rust
// forge-kvcache/tests/test_paged.rs
// Note: These tests use a mock/CPU-based paged cache for logic testing.
// GPU integration tested separately.

use forge_kvcache::paged::{PagedKvCacheConfig, BlockManager};

#[test]
fn test_block_allocation() {
    let mut mgr = BlockManager::new(100, 16); // 100 blocks, 16 tokens each
    let blocks = mgr.allocate(3).unwrap(); // allocate 3 blocks
    assert_eq!(blocks.len(), 3);
    assert_eq!(mgr.free_count(), 97);
}

#[test]
fn test_block_free() {
    let mut mgr = BlockManager::new(100, 16);
    let blocks = mgr.allocate(5).unwrap();
    mgr.free(&blocks);
    assert_eq!(mgr.free_count(), 100);
}

#[test]
fn test_allocation_failure() {
    let mut mgr = BlockManager::new(2, 16);
    let _ = mgr.allocate(2).unwrap();
    assert!(mgr.allocate(1).is_err()); // no free blocks
}

#[test]
fn test_can_allocate() {
    let mut mgr = BlockManager::new(10, 16);
    assert!(mgr.can_allocate(160)); // 10 blocks × 16 = 160 tokens
    assert!(!mgr.can_allocate(161));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p forge-kvcache`
Expected: FAIL.

**Step 3: Implement BlockManager**

```rust
// forge-kvcache/src/paged.rs
use forge_core::{ForgeError, Result, CacheUsage};
use std::collections::HashMap;

pub struct BlockManager {
    total_blocks: usize,
    block_size: usize,
    free_blocks: Vec<usize>,
    // seq_id -> list of (block_id, fill_count)
    seq_blocks: HashMap<u64, Vec<(usize, usize)>>,
}

impl BlockManager {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            total_blocks,
            block_size,
            free_blocks: (0..total_blocks).rev().collect(),
            seq_blocks: HashMap::new(),
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Result<Vec<usize>> {
        if self.free_blocks.len() < num_blocks {
            return Err(ForgeError::OutOfMemory(
                format!("Need {} blocks, only {} free", num_blocks, self.free_blocks.len()),
            ));
        }
        let blocks: Vec<usize> = (0..num_blocks)
            .map(|_| self.free_blocks.pop().unwrap())
            .collect();
        Ok(blocks)
    }

    pub fn free(&mut self, blocks: &[usize]) {
        self.free_blocks.extend(blocks);
    }

    pub fn free_count(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        let blocks_needed = (num_tokens + self.block_size - 1) / self.block_size;
        self.free_blocks.len() >= blocks_needed
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn usage(&self) -> CacheUsage {
        CacheUsage {
            total_blocks: self.total_blocks,
            used_blocks: self.total_blocks - self.free_blocks.len(),
            block_size: self.block_size,
        }
    }

    // Sequence-level operations
    pub fn allocate_seq(&mut self, seq_id: u64, initial_tokens: usize) -> Result<()> {
        let num_blocks = (initial_tokens + self.block_size - 1) / self.block_size;
        let num_blocks = num_blocks.max(1);
        let blocks = self.allocate(num_blocks)?;
        let seq_blocks = blocks.into_iter().map(|b| (b, 0)).collect();
        self.seq_blocks.insert(seq_id, seq_blocks);
        Ok(())
    }

    pub fn free_seq(&mut self, seq_id: u64) -> Result<()> {
        let blocks = self.seq_blocks.remove(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        let block_ids: Vec<usize> = blocks.into_iter().map(|(id, _)| id).collect();
        self.free(&block_ids);
        Ok(())
    }

    pub fn get_block_table(&self, seq_id: u64) -> Result<Vec<usize>> {
        let blocks = self.seq_blocks.get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(blocks.iter().map(|(id, _)| *id).collect())
    }

    pub fn append_token(&mut self, seq_id: u64) -> Result<usize> {
        let blocks = self.seq_blocks.get_mut(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        let (_, fill) = blocks.last_mut().unwrap();
        if *fill >= self.block_size {
            // Current block full, allocate new one
            let new_block = self.allocate(1)?;
            blocks.push((new_block[0], 1));
            Ok(new_block[0])
        } else {
            *fill += 1;
            Ok(blocks.last().unwrap().0)
        }
    }

    pub fn seq_len(&self, seq_id: u64) -> Result<usize> {
        let blocks = self.seq_blocks.get(&seq_id)
            .ok_or(ForgeError::SeqNotFound(seq_id))?;
        Ok(blocks.iter().map(|(_, fill)| fill).sum())
    }
}
```

**Step 4: Wire up lib.rs**

```rust
// forge-kvcache/src/lib.rs
pub mod paged;
```

**Step 5: Run tests**

Run: `cargo test -p forge-kvcache`
Expected: All tests pass.

**Step 6: Implement naive attention in CUDA backend**

Add naive scaled dot-product attention to `forge-backend-cuda/src/attention.rs` using cuBLAS matmul + softmax kernel. This is the Level 1 attention from the design doc — correct but not optimized.

**Step 7: Commit**

```bash
git add forge-kvcache/ forge-backend/
git commit -m "feat(kvcache): PagedAttention block manager + naive attention"
```

---

## Task 10: Llama Model Implementation

**Files:**
- Create: `forge-models/forge-model-llama/src/model.rs`
- Create: `forge-models/forge-model-llama/src/layers.rs`
- Create: `forge-models/forge-model-llama/src/loader.rs`
- Modify: `forge-models/forge-model-llama/src/lib.rs`

**Step 1: Implement Llama layers**

```rust
// forge-models/forge-model-llama/src/layers.rs
// Components:
// - RMSNorm: weight + eps → rms_norm op
// - LlamaAttention: wq, wk, wv, wo weights → attention computation
//   - Supports GQA (grouped query attention)
//   - Computes Q, K, V projections
//   - Applies RoPE
//   - Computes attention via naive scaled dot-product
//   - Stores KV in PagedKvCache
// - LlamaMLP: gate_proj, up_proj, down_proj → SiLU gated MLP
// - LlamaDecoderLayer: attention + MLP with residual connections
```

Key implementation details:
- Each layer holds its weight tensors (loaded at init time)
- `forward` methods take `&B: Backend` to dispatch ops
- Attention uses `KvCache::append` and `KvCache::get_block_table`
- GQA: repeat KV heads to match Q heads count

**Step 2: Implement weight loading**

```rust
// forge-models/forge-model-llama/src/loader.rs
// Maps HuggingFace weight names to Llama layer structure:
//   model.embed_tokens.weight
//   model.layers.{i}.self_attn.q_proj.weight
//   model.layers.{i}.self_attn.k_proj.weight
//   model.layers.{i}.self_attn.v_proj.weight
//   model.layers.{i}.self_attn.o_proj.weight
//   model.layers.{i}.mlp.gate_proj.weight
//   model.layers.{i}.mlp.up_proj.weight
//   model.layers.{i}.mlp.down_proj.weight
//   model.layers.{i}.input_layernorm.weight
//   model.layers.{i}.post_attention_layernorm.weight
//   model.norm.weight
//   lm_head.weight
```

**Step 3: Implement Model trait**

```rust
// forge-models/forge-model-llama/src/model.rs
pub struct LlamaModel<B: Backend> {
    config: ModelConfig,
    embed_tokens: B::Tensor,
    layers: Vec<LlamaDecoderLayer<B>>,
    norm: RMSNorm<B>,
    lm_head: B::Tensor,
    backend: B,
}

impl<B: Backend> Model for LlamaModel<B> {
    type T = B::Tensor;

    fn forward(&self, input: &ModelInput, kv_cache: &mut dyn KvCache<T = B::Tensor>) -> Result<ModelOutput<B::Tensor>> {
        // 1. Embedding lookup
        // 2. For each layer: attention (with KV cache) + MLP
        // 3. Final RMS norm
        // 4. LM head projection → logits
    }

    fn config(&self) -> &ModelConfig { &self.config }
}
```

**Step 4: Wire up and build**

Run: `cargo build -p forge-model-llama`
Expected: Compiles (can't fully test without model weights).

**Step 5: Commit**

```bash
git add forge-models/
git commit -m "feat(llama): Llama model implementation with GQA, RoPE, SiLU MLP"
```

---

## Task 11: CPU Sampling Pipeline

**Files:**
- Create: `forge-runtime/src/sampling.rs`
- Create: `forge-runtime/tests/test_sampling.rs`

**Step 1: Write tests**

```rust
// forge-runtime/tests/test_sampling.rs
use forge_runtime::sampling::{CpuSampler, LogitProcessorPipeline};
use forge_core::SamplingParams;

#[test]
fn test_greedy_sampling() {
    let logits = vec![0.1, 0.3, 0.9, 0.2]; // token 2 has highest logit
    let params = SamplingParams { temperature: 0.0, ..Default::default() };
    let sampler = CpuSampler;
    let result = sampler.sample_single(&logits, &params, &[]).unwrap();
    assert_eq!(result.token_id, 2);
}

#[test]
fn test_temperature_scaling() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let pipeline = LogitProcessorPipeline::from_params(
        &SamplingParams { temperature: 0.5, ..Default::default() }
    );
    pipeline.apply(&mut logits, &[]);
    // After temperature 0.5: logits should be [2.0, 4.0, 6.0]
    assert!((logits[0] - 2.0).abs() < 1e-5);
    assert!((logits[1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_repetition_penalty() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    let pipeline = LogitProcessorPipeline::from_params(
        &SamplingParams { repetition_penalty: 2.0, temperature: 1.0, ..Default::default() }
    );
    // Token 1 and 3 were already generated
    pipeline.apply(&mut logits, &[1, 3]);
    // Positive logits should be divided by penalty
    assert!((logits[1] - 1.0).abs() < 1e-5); // 2.0 / 2.0
    assert!((logits[3] - 2.0).abs() < 1e-5); // 4.0 / 2.0
    // Unpenalized tokens unchanged
    assert!((logits[0] - 1.0).abs() < 1e-5);
    assert!((logits[2] - 3.0).abs() < 1e-5);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p forge-runtime`
Expected: FAIL.

**Step 3: Implement sampling pipeline**

Implement `LogitProcessorPipeline` (temperature, top-k, top-p, min-p, repetition/presence/frequency penalty) and `CpuSampler` (greedy + multinomial) as designed in section 8 of the design doc.

**Step 4: Run tests**

Run: `cargo test -p forge-runtime`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add forge-runtime/
git commit -m "feat(runtime): CPU sampling pipeline with logit processors"
```

---

## Task 12: Continuous Batching Scheduler

**Files:**
- Create: `forge-scheduler/src/continuous.rs`
- Create: `forge-scheduler/src/sequence.rs`
- Modify: `forge-scheduler/src/lib.rs`
- Create: `forge-scheduler/tests/test_scheduler.rs`

**Step 1: Write tests**

```rust
// forge-scheduler/tests/test_scheduler.rs
use forge_scheduler::ContinuousBatchingScheduler;
use forge_core::{InferenceRequest, Scheduler, SamplingParams, CacheUsage};

#[test]
fn test_enqueue_and_schedule() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache_usage = CacheUsage { total_blocks: 100, used_blocks: 0, block_size: 16 };

    let req = InferenceRequest {
        request_id: "req-1".to_string(),
        prompt_tokens: vec![1, 2, 3, 4, 5],
        sampling_params: SamplingParams::default(),
    };
    scheduler.enqueue(req).unwrap();

    let batch = scheduler.schedule(&cache_usage).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 1);
    assert_eq!(batch.decode_seqs.len(), 0);
    assert_eq!(batch.prefill_seqs[0].token_ids, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_prefill_then_decode() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache_usage = CacheUsage { total_blocks: 100, used_blocks: 0, block_size: 16 };

    let req = InferenceRequest {
        request_id: "req-1".to_string(),
        prompt_tokens: vec![1, 2, 3],
        sampling_params: SamplingParams::default(),
    };
    let handle = scheduler.enqueue(req).unwrap();

    // First schedule: prefill
    let batch = scheduler.schedule(&cache_usage).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 1);

    // Simulate token generation
    scheduler.append_token(handle.seq_id, 10).unwrap();

    // Second schedule: decode
    let batch = scheduler.schedule(&cache_usage).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 0);
    assert_eq!(batch.decode_seqs.len(), 1);
    assert_eq!(batch.decode_seqs[0].token_ids, vec![10]); // last generated token
}

#[test]
fn test_finish_removes_sequence() {
    let mut scheduler = ContinuousBatchingScheduler::new(Default::default());
    let cache_usage = CacheUsage { total_blocks: 100, used_blocks: 0, block_size: 16 };

    let req = InferenceRequest {
        request_id: "req-1".to_string(),
        prompt_tokens: vec![1, 2, 3],
        sampling_params: SamplingParams::default(),
    };
    let handle = scheduler.enqueue(req).unwrap();
    let _ = scheduler.schedule(&cache_usage).unwrap();

    scheduler.finish(handle.seq_id, forge_core::FinishReason::EosToken).unwrap();

    let batch = scheduler.schedule(&cache_usage).unwrap();
    assert!(batch.is_empty());
}

#[test]
fn test_max_batch_size() {
    let config = SchedulerConfig { max_batch_size: 2, ..Default::default() };
    let mut scheduler = ContinuousBatchingScheduler::new(config);
    let cache_usage = CacheUsage { total_blocks: 1000, used_blocks: 0, block_size: 16 };

    for i in 0..5 {
        scheduler.enqueue(InferenceRequest {
            request_id: format!("req-{i}"),
            prompt_tokens: vec![1, 2, 3],
            sampling_params: SamplingParams::default(),
        }).unwrap();
    }

    let batch = scheduler.schedule(&cache_usage).unwrap();
    assert_eq!(batch.prefill_seqs.len(), 2); // limited by max_batch_size
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p forge-scheduler`
Expected: FAIL.

**Step 3: Implement sequence state and scheduler**

```rust
// forge-scheduler/src/sequence.rs — SequenceState (Waiting/Running/Finished)
// forge-scheduler/src/continuous.rs — ContinuousBatchingScheduler
//   - Waiting queue (VecDeque)
//   - Running map (HashMap<u64, SequenceState>)
//   - schedule(): prioritize decode, then prefill new sequences within budget
```

**Step 4: Run tests**

Run: `cargo test -p forge-scheduler`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add forge-scheduler/
git commit -m "feat(scheduler): continuous batching scheduler with sequence management"
```

---

## Task 13: Engine — Runtime Loop

**Files:**
- Create: `forge-runtime/src/engine.rs`
- Modify: `forge-runtime/src/lib.rs`

**Step 1: Implement the engine**

```rust
// forge-runtime/src/engine.rs
use forge_core::*;
use tokio::sync::mpsc;

pub enum EngineEvent {
    Token { seq_id: u64, token_id: u32, text: Option<String> },
    Finish { seq_id: u64, reason: FinishReason },
    Error { seq_id: u64, error: String },
}

pub struct Engine<B: Backend, M: Model<T = B::Tensor>> {
    model: M,
    backend: B,
    scheduler: Box<dyn Scheduler>,
    kv_cache: Box<dyn KvCache<T = B::Tensor>>,
    sampler: CpuSampler,
    event_tx: mpsc::Sender<EngineEvent>,
}

impl<B: Backend, M: Model<T = B::Tensor>> Engine<B, M> {
    pub async fn run(&mut self) -> Result<()> {
        loop {
            let cache_usage = self.kv_cache.usage();
            let batch = self.scheduler.schedule(&cache_usage)?;

            if batch.is_empty() {
                tokio::time::sleep(std::time::Duration::from_micros(100)).await;
                continue;
            }

            // Build ModelInput from batch
            let input = self.build_input(&batch)?;

            // Forward pass
            let output = self.model.forward(&input, &mut *self.kv_cache)?;
            self.backend.synchronize()?;

            // Sample from logits
            let logits_host = self.backend.copy_to_host_f32(&output.logits)?;
            let all_seqs: Vec<_> = batch.prefill_seqs.iter()
                .chain(batch.decode_seqs.iter())
                .collect();

            for (i, seq) in all_seqs.iter().enumerate() {
                let vocab_size = self.model.config().vocab_size;
                let seq_logits = &logits_host[i * vocab_size..(i + 1) * vocab_size];

                let generated = self.scheduler.get_generated_tokens(seq.seq_id)?;
                let result = self.sampler.sample_single(
                    seq_logits,
                    &seq.sampling_params,
                    &generated,
                )?;

                let token_id = result.token_id;
                self.scheduler.append_token(seq.seq_id, token_id)?;

                // Check stop conditions
                if seq.sampling_params.stop_token_ids.contains(&token_id)
                    || generated.len() + 1 >= seq.sampling_params.max_tokens
                {
                    let reason = if seq.sampling_params.stop_token_ids.contains(&token_id) {
                        FinishReason::EosToken
                    } else {
                        FinishReason::MaxTokens
                    };
                    self.scheduler.finish(seq.seq_id, reason)?;
                    self.kv_cache.free(seq.seq_id)?;
                    let _ = self.event_tx.send(EngineEvent::Finish {
                        seq_id: seq.seq_id, reason,
                    }).await;
                } else {
                    let _ = self.event_tx.send(EngineEvent::Token {
                        seq_id: seq.seq_id,
                        token_id,
                        text: None, // decoded by server layer
                    }).await;
                }
            }
        }
    }
}
```

**Step 2: Verify it builds**

Run: `cargo build -p forge-runtime`
Expected: Compiles.

**Step 3: Commit**

```bash
git add forge-runtime/
git commit -m "feat(runtime): engine main loop with scheduling, forward, sampling"
```

---

## Task 14: OpenAI-Compatible HTTP API

**Files:**
- Create: `forge-server/src/api/mod.rs`
- Create: `forge-server/src/api/openai.rs`
- Create: `forge-server/src/api/types.rs`
- Modify: `forge-server/src/main.rs`

**Step 1: Define OpenAI API types**

```rust
// forge-server/src/api/types.rs
// ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk,
// ChatMessage, Choice, ChunkChoice, Delta, Usage
// All serde Serialize/Deserialize, matching OpenAI spec exactly
```

**Step 2: Implement handlers**

```rust
// forge-server/src/api/openai.rs
// POST /v1/chat/completions — non-streaming and streaming
// GET /v1/models — list loaded models
//
// Non-streaming: collect all tokens, return complete response
// Streaming: SSE with ChatCompletionChunk per token
//   - Uses IncrementalDecoder for UTF-8 safe streaming
//   - Sends [DONE] at the end
```

**Step 3: Wire up main.rs with axum router**

```rust
// forge-server/src/main.rs
use axum::{Router, routing::{get, post}};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::init();

    // Parse CLI args (model path, port, etc.)
    // Load model, create backend, scheduler, kv_cache, engine
    // Spawn engine loop
    // Start HTTP server

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/forge/v1/health", get(health));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

**Step 4: Verify it builds**

Run: `cargo build -p forge-server`
Expected: Compiles.

**Step 5: Commit**

```bash
git add forge-server/
git commit -m "feat(server): OpenAI-compatible HTTP API with SSE streaming"
```

---

## Task 15: InProcessTransport

**Files:**
- Create: `forge-transport/src/in_process.rs`
- Modify: `forge-transport/src/lib.rs`

**Step 1: Implement zero-overhead in-process transport**

```rust
// forge-transport/src/in_process.rs
// Simple wrapper: directly holds Arc<Engine>, calls methods
// This is the "distributed extension point" from the design —
// future GrpcTransport will have the same interface
```

**Step 2: Verify it builds**

Run: `cargo build -p forge-transport`
Expected: Compiles.

**Step 3: Commit**

```bash
git add forge-transport/
git commit -m "feat(transport): InProcessTransport for standalone mode"
```

---

## Task 16: FlashAttention Integration

**Files:**
- Create: `forge-kernels/csrc/flash_attn_wrapper.cu`
- Create: `forge-kernels/csrc/flash_attn_wrapper.h`
- Modify: `forge-kernels/build.rs`
- Create: `forge-backend/forge-backend-cuda/src/flash_attention.rs`

**Step 1: Create FFI wrapper for FlashAttention**

FlashAttention is built as a separate library (submodule or system install). The wrapper provides a C-compatible interface for Rust FFI:

```c
// forge-kernels/csrc/flash_attn_wrapper.h
extern "C" {
    int forge_flash_attn_fwd(
        void* q, void* k, void* v, void* out,
        int batch_size, int seqlen_q, int seqlen_k,
        int num_heads, int num_heads_k, int head_dim,
        float softmax_scale, bool is_causal,
        void* stream
    );
}
```

**Step 2: Implement Rust binding**

```rust
// forge-backend/forge-backend-cuda/src/flash_attention.rs
// Safe wrapper around flash attention FFI
// Falls back to naive attention if FlashAttention not available (compile-time feature flag)
```

**Step 3: Add feature flag**

In `forge-backend/forge-backend-cuda/Cargo.toml`:
```toml
[features]
default = ["flash-attn"]
flash-attn = []
```

**Step 4: Verify it builds**

Run: `cargo build -p forge-backend-cuda --features flash-attn`
Expected: Compiles (with FlashAttention headers available).

**Step 5: Commit**

```bash
git add forge-kernels/ forge-backend/
git commit -m "feat(kernels): FlashAttention FFI integration with fallback"
```

---

## Task 17: End-to-End Integration Test

**Files:**
- Create: `tests/integration/test_e2e.rs`
- Create: `scripts/test_server.sh`

**Step 1: Write integration test script**

```bash
#!/bin/bash
# scripts/test_server.sh
# Start server with a small model (e.g., TinyLlama-1.1B)
# Send curl request to /v1/chat/completions
# Verify response format
# Test streaming

MODEL_PATH=${1:-"/path/to/tinyllama"}

# Start server in background
cargo run --release -- serve --model "$MODEL_PATH" --port 8080 &
SERVER_PID=$!
sleep 5

# Test non-streaming
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10,
    "temperature": 0
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); assert 'choices' in r; print('Non-streaming OK')"

# Test streaming
curl -s -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10,
    "stream": true
  }' | head -5 | grep -q "data:" && echo "Streaming OK"

# Health check
curl -s http://localhost:8080/forge/v1/health | python3 -c "import sys,json; r=json.load(sys.stdin); print('Health OK')"

kill $SERVER_PID
echo "All integration tests passed!"
```

**Step 2: Run full integration test with TinyLlama or similar small model**

Run: `bash scripts/test_server.sh /path/to/tinyllama-1.1b`
Expected: All checks pass, server produces valid completions.

**Step 3: Commit**

```bash
git add tests/ scripts/
git commit -m "test: end-to-end integration test with server + API"
```

---

## Task 18: Benchmarks

**Files:**
- Create: `benches/throughput.rs`
- Create: `scripts/benchmark.sh`

**Step 1: Create benchmark harness**

Measure:
- **TTFT** (Time to First Token): latency from request to first token
- **ITL** (Inter-Token Latency): average time between consecutive tokens
- **Throughput**: total tokens/second under concurrent load
- **Memory**: peak GPU memory usage

**Step 2: Run benchmarks against TinyLlama**

```bash
# scripts/benchmark.sh
# Start server, send N concurrent requests, measure latencies
# Compare against vLLM baseline (same model, same hardware)
```

**Step 3: Commit**

```bash
git add benches/ scripts/
git commit -m "perf: add TTFT, ITL, throughput benchmarks"
```

---

## Dependency Graph

```
Task 1: Scaffolding
  └── Task 2: Core Types
       └── Task 3: Backend/Tensor Traits
            ├── Task 4: Model/KvCache/Scheduler Traits
            │    ├── Task 9: PagedAttention KV Cache
            │    ├── Task 11: CPU Sampling
            │    └── Task 12: Continuous Batching Scheduler
            ├── Task 5: CUDA Backend (cudarc)
            │    └── Task 6: CUDA Kernels
            │         └── Task 16: FlashAttention Integration
            ├── Task 7: SafeTensors Loader
            └── Task 8: Tokenizer + Chat Template
                 └── Task 10: Llama Model
                      └── Task 13: Engine Runtime Loop
                           └── Task 14: HTTP API
                                └── Task 15: Transport
                                     └── Task 17: E2E Test
                                          └── Task 18: Benchmarks
```

## Parallel Execution Opportunities

These task groups can be developed in parallel:

**Group A (Compute):** Task 5 → Task 6 → Task 16
**Group B (Logic):** Task 9, Task 11, Task 12 (all independent after Task 4)
**Group C (IO):** Task 7, Task 8 (independent after Task 2)

After groups converge at Task 10 (Llama Model), the remaining tasks (13-18) are sequential.
