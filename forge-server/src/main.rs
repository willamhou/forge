//! Forge LLM Inference Server — OpenAI-compatible HTTP API.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tracing_subscriber::EnvFilter;

use forge_server::api::openai::{self, AppState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // TODO: Parse CLI args for model path, port, etc.
    // For now, print usage and exit — the full CLI will be wired when
    // all components are integrated in Task 17.
    eprintln!("Forge LLM Inference Server");
    eprintln!("Usage: forge-server --model <path> --port <port>");
    eprintln!();
    eprintln!("Server infrastructure is ready. Run with model path to start serving.");
    eprintln!("See Task 17 for full end-to-end integration.");

    Ok(())
}

/// Build the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
        .route("/forge/v1/health", get(openai::health))
        .with_state(state)
}
