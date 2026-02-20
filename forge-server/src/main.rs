//! Forge LLM Inference Server — OpenAI-compatible HTTP API.

use std::path::PathBuf;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use forge_backend_cuda::CudaBackend;
use forge_kvcache::naive::NaiveKvCache;
use forge_loader::{LlamaConfig, SafeTensorsLoader};
use forge_model_llama::load_llama_model;
use forge_runtime::engine::Engine;
use forge_scheduler::{ContinuousBatchingScheduler, SchedulerConfig};
use forge_server::api::openai::{self, AppState};
use forge_server::chat_template::ChatTemplate;
use forge_server::tokenizer::ForgeTokenizer;

#[derive(Parser)]
#[command(name = "forge-server", about = "Forge LLM Inference Server")]
struct Cli {
    /// Path to the model directory (SafeTensors format)
    #[arg(long)]
    model_path: PathBuf,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Maximum batch size for continuous batching
    #[arg(long, default_value = "256")]
    max_batch_size: usize,

    /// Maximum prefill tokens per scheduling step
    #[arg(long, default_value = "4096")]
    max_prefill_tokens: usize,

    /// CUDA device ordinal
    #[arg(long, default_value = "0")]
    device: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // --- Load config ---
    let config_path = cli.model_path.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", config_path.display()))?;
    let llama_config: LlamaConfig = serde_json::from_str(&config_text)?;
    let model_config = llama_config.to_model_config();

    info!(
        "Model config: {} layers, {} heads, {} KV heads, vocab {}",
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.num_key_value_heads,
        model_config.vocab_size,
    );

    // --- Init CUDA backend ---
    let backend = CudaBackend::new(cli.device)?;
    info!("CUDA backend initialized (device {})", cli.device);

    // --- Load model weights ---
    info!("Loading model from {}...", cli.model_path.display());
    let loader = SafeTensorsLoader::new(&cli.model_path)?;
    let model = load_llama_model(&loader, model_config.clone(), &backend)?;
    info!("Model loaded successfully");

    // --- Load tokenizer ---
    let tokenizer_path = cli.model_path.join("tokenizer.json");
    let tokenizer = ForgeTokenizer::from_file(&tokenizer_path)?;
    info!(
        "Tokenizer loaded (eos_token_id={})",
        tokenizer.eos_token_id()
    );

    // --- Load chat template ---
    let chat_template = load_chat_template(&cli.model_path)?;
    info!("Chat template loaded");

    // --- Create engine components ---
    let kv_cache = NaiveKvCache::new(backend.clone(), model_config.num_hidden_layers, 64);
    let scheduler_config = SchedulerConfig {
        max_batch_size: cli.max_batch_size,
        max_prefill_tokens: cli.max_prefill_tokens,
    };
    let scheduler = ContinuousBatchingScheduler::new(scheduler_config);

    let (request_tx, request_rx) = mpsc::channel(1024);

    // --- Spawn engine ---
    let mut engine = Engine::new(
        model,
        backend,
        Box::new(scheduler),
        Box::new(kv_cache),
        request_rx,
    );
    tokio::spawn(async move {
        if let Err(e) = engine.run().await {
            error!("Engine error: {e}");
        }
    });
    info!("Engine spawned");

    // --- Determine model name from directory ---
    let model_name = cli
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // --- Start HTTP server ---
    let state = Arc::new(AppState {
        model_name: model_name.clone(),
        tokenizer,
        chat_template,
        request_tx,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
        .route("/forge/v1/health", get(openai::health))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Forge serving '{model_name}' on {addr}");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Load chat template from tokenizer_config.json or fall back to ChatML.
fn load_chat_template(model_path: &PathBuf) -> anyhow::Result<ChatTemplate> {
    let config_path = model_path.join("tokenizer_config.json");
    if config_path.exists() {
        let text = std::fs::read_to_string(&config_path)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        if let Some(tmpl) = value.get("chat_template").and_then(|v| v.as_str()) {
            return Ok(ChatTemplate::new(tmpl)?);
        }
    }
    // Fallback to ChatML
    info!("No chat_template in tokenizer_config.json, using ChatML default");
    Ok(ChatTemplate::chatml_default()?)
}
