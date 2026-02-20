//! Forge LLM Inference Server — OpenAI-compatible HTTP API.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use forge_backend_cpu::CpuBackend;
use forge_backend_cuda::CudaBackend;
use forge_core::{Backend, ModelConfig};
use forge_kvcache::naive::NaiveKvCache;
use forge_kvcache::paged_cache::PagedKvCache;
use forge_loader::{LlamaConfig, SafeTensorsLoader};
use forge_model_llama::load_llama_model;
use forge_runtime::constraints::fsm::TokenVocab;
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

    /// CUDA device ordinal (only used with --backend cuda)
    #[arg(long, default_value = "0")]
    device: usize,

    /// Backend to use: "cuda" or "cpu"
    #[arg(long, default_value = "cuda")]
    backend: String,

    /// KV cache type: "naive" or "paged"
    #[arg(long, default_value = "paged")]
    kv_cache: String,

    /// Block size for paged KV cache (tokens per block)
    #[arg(long, default_value = "16")]
    block_size: usize,

    /// Total number of KV cache blocks (paged cache only)
    #[arg(long, default_value = "2048")]
    num_blocks: usize,
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

    match cli.backend.as_str() {
        "cpu" => {
            let backend = CpuBackend::new();
            info!("CPU backend initialized");
            run_server(backend, &cli, model_config).await
        }
        "cuda" => {
            let backend = CudaBackend::new(cli.device)?;
            info!("CUDA backend initialized (device {})", cli.device);
            run_server(backend, &cli, model_config).await
        }
        other => anyhow::bail!("Unknown backend: {other}. Use 'cpu' or 'cuda'."),
    }
}

async fn run_server<B: Backend + Clone>(
    backend: B,
    cli: &Cli,
    model_config: ModelConfig,
) -> anyhow::Result<()> {
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
    let kv_cache: Box<dyn forge_core::KvCache<T = B::Tensor> + Send + Sync> =
        match cli.kv_cache.as_str() {
            "paged" => {
                let cache = PagedKvCache::new(
                    backend.clone(),
                    cli.num_blocks,
                    cli.block_size,
                    model_config.num_hidden_layers,
                    model_config.num_key_value_heads,
                    model_config.head_dim,
                );
                info!(
                    "Paged KV cache: {} blocks x {} tokens, kv_dim={}",
                    cli.num_blocks,
                    cli.block_size,
                    model_config.num_key_value_heads * model_config.head_dim,
                );
                Box::new(cache)
            }
            "naive" => {
                let cache = NaiveKvCache::new(
                    backend.clone(),
                    model_config.num_hidden_layers,
                    cli.max_batch_size,
                );
                info!("Naive KV cache (CPU-side, max {} sequences)", cli.max_batch_size);
                Box::new(cache)
            }
            other => anyhow::bail!("Unknown kv-cache type: {other}. Use 'naive' or 'paged'."),
        };
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
        kv_cache,
        request_rx,
    );
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        if let Err(e) = engine.run().await {
            error!("Engine error: {e}");
        }
        // Signal the HTTP server to shut down when the engine exits.
        let _ = shutdown_tx.send(true);
    });
    info!("Engine spawned");

    // --- Determine model name from directory ---
    let model_name = cli
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // --- Build token vocabulary for structured output ---
    let token_vocab = {
        let vocab_size = model_config.vocab_size;
        info!("Building token vocabulary for structured output ({vocab_size} tokens)...");
        let vocab = TokenVocab::from_decode_fn(vocab_size, |id| {
            tokenizer.decode(&[id]).ok()
        });
        info!("Token vocabulary built");
        Some(Arc::new(vocab))
    };

    // --- Start HTTP server ---
    let state = Arc::new(AppState {
        model_name: model_name.clone(),
        tokenizer,
        chat_template,
        request_tx,
        token_vocab,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
        .route("/forge/v1/health", get(openai::health))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Forge serving '{model_name}' on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            // Shut down when engine exits or on Ctrl-C.
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    info!("Engine exited, shutting down HTTP server");
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("Received Ctrl-C, shutting down");
                }
            }
        })
        .await?;

    Ok(())
}

/// Load chat template from tokenizer_config.json or fall back to ChatML.
fn load_chat_template(model_path: &Path) -> anyhow::Result<ChatTemplate> {
    let config_path = model_path.join("tokenizer_config.json");
    if config_path.exists() {
        let text = std::fs::read_to_string(&config_path)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        if let Some(tmpl) = value.get("chat_template").and_then(|v| v.as_str()) {
            let bos_token = extract_token_str(&value, "bos_token");
            let eos_token = extract_token_str(&value, "eos_token");
            return Ok(ChatTemplate::with_tokens(tmpl, &bos_token, &eos_token)?);
        }
    }
    // Fallback to ChatML
    info!("No chat_template in tokenizer_config.json, using ChatML default");
    Ok(ChatTemplate::chatml_default()?)
}

/// Extract a special token string from tokenizer_config.json.
/// Handles both plain string `"bos_token": "<s>"` and object form
/// `"bos_token": {"content": "<s>", ...}` used by some HuggingFace models.
fn extract_token_str(config: &serde_json::Value, key: &str) -> String {
    config
        .get(key)
        .and_then(|v| {
            v.as_str()
                .map(String::from)
                .or_else(|| v.get("content").and_then(|c| c.as_str()).map(String::from))
        })
        .unwrap_or_default()
}
