//! Forge loader: SafeTensors weight loading and config parsing.

pub mod config;
pub mod safetensors;

pub use config::LlamaConfig;
pub use self::safetensors::SafeTensorsLoader;
