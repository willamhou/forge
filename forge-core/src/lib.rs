//! Forge core types, traits, and error definitions.

pub mod backend;
pub mod error;
pub mod kvcache;
pub mod model;
pub mod sampling;
pub mod scheduler;
pub mod tensor;
pub mod types;

pub use backend::{Backend, DecodeGraphDispatch, DecodeStageInputs};
pub use error::{ForgeError, Result};
pub use kvcache::{CacheUsage, KvCache, PagedAttentionInputs};
pub use model::{Model, ModelOutput};
pub use sampling::{SampleResult, SamplingContext};
pub use scheduler::{InferenceRequest, RequestHandle, ScheduleBatch, ScheduledSeq, Scheduler};
pub use tensor::Tensor;
pub use types::*;
