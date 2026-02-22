//! End-to-end integration tests for the Forge HTTP API.
//!
//! Uses tower::ServiceExt::oneshot to test the axum router with mock engine
//! channels. No GPU or model weights needed.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use http_body_util::BodyExt;
use tokio::sync::mpsc;
use tower::ServiceExt;

use forge_core::FinishReason;
use forge_runtime::engine::{EngineEvent, EngineRequest};
use forge_server::api::openai::{self, AppState};
use forge_server::chat_template::ChatTemplate;
use forge_server::tokenizer::ForgeTokenizer;

// ──────────── Test Infrastructure ────────────

/// Build a minimal WordLevel tokenizer for testing (no model files needed).
fn mock_tokenizer() -> ForgeTokenizer {
    use tokenizers::Tokenizer;

    // Minimal tokenizer JSON — WordLevel with a small vocab and whitespace splitter.
    let json = r#"{
        "version": "1.0",
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
            "unk_token": "[UNK]"
        },
        "pre_tokenizer": {"type": "Whitespace"}
    }"#;
    let tokenizer: Tokenizer = json.parse().unwrap();
    ForgeTokenizer::from_inner(tokenizer, 0) // eos_token_id = 0
}

/// Build the axum router with a mock engine channel.
fn create_test_app(request_tx: mpsc::Sender<EngineRequest>) -> Router {
    let state = Arc::new(AppState {
        model_name: "test-model".to_string(),
        tokenizer: Arc::new(mock_tokenizer()),
        chat_template: ChatTemplate::chatml_default().unwrap(),
        request_tx,
        token_vocab: None,
    });

    Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
        .route("/forge/v1/health", get(openai::health))
        .with_state(state)
}

/// Mock engine: reads one request and sends predictable token events.
async fn mock_engine_simple(mut rx: mpsc::Receiver<EngineRequest>) {
    if let Some(req) = rx.recv().await {
        for (id, text) in [(1u32, "Hello"), (2, " world")] {
            let _ = req
                .event_tx
                .send(EngineEvent::Token {
                    seq_id: 0,
                    token_id: id,
                    text: Some(text.to_string()),
                })
                .await;
        }
        let _ = req
            .event_tx
            .send(EngineEvent::Finish {
                seq_id: 0,
                reason: FinishReason::EosToken,
            })
            .await;
    }
}

/// Helper to build a chat completion request body.
fn chat_request_body(stream: bool) -> String {
    serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": stream,
        "max_tokens": 10
    })
    .to_string()
}

// ──────────── Tests ────────────

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let (tx, _rx) = mpsc::channel(1);
    let app = create_test_app(tx);

    let resp = app
        .oneshot(
            Request::get("/forge/v1/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn models_endpoint_returns_list() {
    let (tx, _rx) = mpsc::channel(1);
    let app = create_test_app(tx);

    let resp = app
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "test-model");
    assert_eq!(json["data"][0]["object"], "model");
}

#[tokio::test]
async fn non_streaming_completion() {
    let (tx, rx) = mpsc::channel(16);
    let app = create_test_app(tx);

    // Spawn mock engine to handle the request
    tokio::spawn(mock_engine_simple(rx));

    let resp = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_body(false)))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "test-model");
    assert!(json["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    // Usage: 2 completion tokens sent by mock engine
    assert!(json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(json["usage"]["completion_tokens"].as_u64().unwrap(), 2);
}

#[tokio::test]
async fn streaming_completion() {
    let (tx, rx) = mpsc::channel(16);
    let app = create_test_app(tx);

    tokio::spawn(mock_engine_simple(rx));

    let resp = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_body(true)))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);

    // Parse SSE events: lines starting with "data: "
    let data_lines: Vec<&str> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();

    // Should have: initial chunk (role), token chunks, finish chunk, [DONE]
    assert!(
        data_lines.len() >= 3,
        "expected >= 3 data lines, got {}: {:?}",
        data_lines.len(),
        data_lines
    );

    // Last data line should be [DONE]
    assert_eq!(*data_lines.last().unwrap(), "[DONE]");

    // First data line should be a chunk with role="assistant"
    let first: serde_json::Value = serde_json::from_str(data_lines[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");

    // Collect streamed content from middle chunks
    let mut content = String::new();
    for line in &data_lines[1..] {
        if *line == "[DONE]" {
            break;
        }
        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(c) = chunk["choices"][0]["delta"]["content"].as_str() {
                content.push_str(c);
            }
        }
    }
    assert!(
        !content.is_empty(),
        "streamed content should not be empty"
    );
}

#[tokio::test]
async fn invalid_request_returns_error() {
    let (tx, _rx) = mpsc::channel(1);
    let app = create_test_app(tx);

    // Missing required "messages" field
    let resp = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model": "test"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    // Axum returns 422 for deserialization failures
    assert!(
        resp.status().is_client_error(),
        "expected 4xx, got {}",
        resp.status()
    );
}

#[tokio::test]
async fn engine_channel_closed_returns_503() {
    let (tx, rx) = mpsc::channel(1);
    let app = create_test_app(tx);

    // Drop the receiver — engine is "unavailable"
    drop(rx);

    let resp = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_body(false)))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["message"], "engine unavailable");
}
