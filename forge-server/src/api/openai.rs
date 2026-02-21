//! OpenAI-compatible chat completions endpoints.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::error;

use forge_core::{FinishReason, InferenceRequest, SamplingParams};
use forge_runtime::constraints::fsm::{FsmConstraint, TokenVocab};
use forge_runtime::constraints::json_schema::build_json_schema_fsm;
use forge_runtime::constraints::regex::build_regex_fsm;
use forge_runtime::engine::{EngineEvent, EngineRequest};

use crate::chat_template::ChatTemplate;
use crate::tokenizer::{ForgeTokenizer, IncrementalDecoder};

use super::types::*;

/// Shared application state passed to handlers.
pub struct AppState {
    pub model_name: String,
    pub tokenizer: Arc<ForgeTokenizer>,
    pub chat_template: ChatTemplate,
    /// Channel to submit requests to the engine.
    pub request_tx: mpsc::Sender<EngineRequest>,
    /// Token vocabulary for FSM constraint building (shared across requests).
    pub token_vocab: Option<Arc<TokenVocab>>,
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let is_stream = req.stream.unwrap_or(false);

    // Format messages using chat template
    let messages: Vec<(&str, &str)> = req
        .messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect();

    let prompt = match state.chat_template.apply(&messages, true) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": { "message": format!("template error: {e}"), "type": "invalid_request_error" }
                })),
            )
                .into_response();
        }
    };

    // Tokenize
    let prompt_tokens = match state.tokenizer.encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": { "message": format!("tokenizer error: {e}"), "type": "invalid_request_error" }
                })),
            )
                .into_response();
        }
    };

    let prompt_len = prompt_tokens.len();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    // Build sampling params
    let mut params = SamplingParams::default();
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(p) = req.top_p {
        params.top_p = p;
    }
    if let Some(k) = req.top_k {
        params.top_k = Some(k);
    }
    if let Some(m) = req.max_tokens {
        params.max_tokens = m;
    }
    if let Some(s) = req.seed {
        params.seed = Some(s);
    }
    if let Some(rp) = req.repetition_penalty {
        params.repetition_penalty = rp;
    }
    if let Some(pp) = req.presence_penalty {
        params.presence_penalty = pp;
    }
    if let Some(fp) = req.frequency_penalty {
        params.frequency_penalty = fp;
    }
    if let Some(stop_strs) = req.stop {
        params.stop_strings = stop_strs;
    }
    // Add EOS token to stop list
    params
        .stop_token_ids
        .push(state.tokenizer.eos_token_id());

    // Validate mutual exclusion of structured output fields
    if req.json_schema.is_some() && req.regex.is_some() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "cannot specify both json_schema and regex",
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response();
    }

    // Build FSM constraint if json_schema or regex is provided.
    // FSM index building is O(states * vocab_size) and runs on a blocking thread
    // to avoid starving the async runtime.
    let constraint: Option<Box<dyn FsmConstraint>> =
        if req.json_schema.is_some() || req.regex.is_some() {
            let vocab = match &state.token_vocab {
                Some(v) => Arc::clone(v),
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": {
                                "message": "structured output not available: token vocabulary not loaded",
                                "type": "invalid_request_error"
                            }
                        })),
                    )
                        .into_response();
                }
            };

            let schema_str = req.json_schema.as_ref().map(|s| serde_json::to_string(s).unwrap_or_default());
            let regex_str = req.regex.clone();

            let build_result = tokio::task::spawn_blocking(move || {
                if let Some(schema) = schema_str {
                    build_json_schema_fsm(&schema, &vocab)
                } else if let Some(pattern) = regex_str {
                    build_regex_fsm(&pattern, &vocab)
                } else {
                    unreachable!()
                }
            })
            .await;

            match build_result {
                Ok(Ok(fsm)) => Some(Box::new(fsm) as Box<dyn FsmConstraint>),
                Ok(Err(e)) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": { "message": format!("constraint error: {e}"), "type": "invalid_request_error" }
                        })),
                    )
                        .into_response();
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": { "message": format!("constraint build failed: {e}"), "type": "server_error" }
                        })),
                    )
                        .into_response();
                }
            }
        } else {
            None
        };

    let inference_req = InferenceRequest {
        request_id: request_id.clone(),
        prompt_tokens,
        sampling_params: params,
    };

    // Create per-request event channel
    let (event_tx, event_rx) = mpsc::channel(256);

    let engine_req = EngineRequest {
        inference_req,
        event_tx,
        constraint,
    };

    if state.request_tx.send(engine_req).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": { "message": "engine unavailable", "type": "server_error" }
            })),
        )
            .into_response();
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if is_stream {
        handle_streaming(event_rx, state.clone(), request_id, now).into_response()
    } else {
        handle_non_streaming(
            event_rx,
            &state.tokenizer,
            request_id,
            state.model_name.clone(),
            prompt_len,
            now,
        )
        .await
        .into_response()
    }
}

/// Collect all tokens and return a single response.
async fn handle_non_streaming(
    mut event_rx: mpsc::Receiver<EngineEvent>,
    tokenizer: &ForgeTokenizer,
    request_id: String,
    model: String,
    prompt_len: usize,
    created: u64,
) -> Response {
    let mut token_ids = Vec::new();
    let mut finish_reason = None;
    let mut engine_error = None;

    while let Some(event) = event_rx.recv().await {
        match event {
            EngineEvent::Token { token_id, .. } => {
                token_ids.push(token_id);
            }
            EngineEvent::Finish { reason, .. } => {
                finish_reason = Some(reason);
                break;
            }
            EngineEvent::Error { error, .. } => {
                error!("engine error: {error}");
                engine_error = Some(error);
                break;
            }
        }
    }

    if let Some(err) = engine_error {
        // Distinguish client errors (bad request) from server errors.
        let is_client_error = err.contains("exceeds max_prefill_tokens")
            || err.contains("empty prompt")
            || err.contains("blocks but cache only has")
            || err.contains("failed to enqueue")
            || err.contains("constraint violated");
        let (status, error_type) = if is_client_error {
            (StatusCode::BAD_REQUEST, "invalid_request_error")
        } else {
            (StatusCode::INTERNAL_SERVER_ERROR, "server_error")
        };
        return (
            status,
            Json(serde_json::json!({
                "error": { "message": err, "type": error_type }
            })),
        )
            .into_response();
    }

    // If the engine channel closed without sending Finish or Error, the engine
    // crashed or was shut down unexpectedly. Return 500 instead of a partial 200.
    if finish_reason.is_none() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": { "message": "engine terminated unexpectedly", "type": "server_error" }
            })),
        )
            .into_response();
    }

    let text = tokenizer.decode(&token_ids).unwrap_or_default();
    let completion_tokens = token_ids.len();

    let reason_str = finish_reason.map(|r| match r {
        FinishReason::EosToken => "stop".to_string(),
        FinishReason::MaxTokens => "length".to_string(),
        FinishReason::StopString => "stop".to_string(),
        FinishReason::Cancelled => "stop".to_string(),
    });

    Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: reason_str,
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    })
    .into_response()
}

/// Stream tokens as SSE events.
fn handle_streaming(
    mut event_rx: mpsc::Receiver<EngineEvent>,
    state: Arc<AppState>,
    request_id: String,
    created: u64,
) -> Sse<ReceiverStream<std::result::Result<Event, std::convert::Infallible>>> {
    let (sse_tx, sse_rx) =
        mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(256);

    tokio::spawn(async move {
        let mut decoder = IncrementalDecoder::new();
        let model = state.model_name.clone();

        let initial_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };

        if let Ok(data) = serde_json::to_string(&initial_chunk) {
            let _ = sse_tx.send(Ok(Event::default().data(data))).await;
        }

        let mut terminated = false;
        while let Some(event) = event_rx.recv().await {
            match event {
                EngineEvent::Token { token_id, text, .. } => {
                    let content = text.or_else(|| decoder.add_token(token_id, &state.tokenizer));
                    let content = match content {
                        Some(c) if !c.is_empty() => c,
                        _ => continue,
                    };
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(content),
                            },
                            finish_reason: None,
                        }],
                    };
                    if let Ok(data) = serde_json::to_string(&chunk) {
                        if sse_tx.send(Ok(Event::default().data(data))).await.is_err() {
                            break;
                        }
                    }
                }
                EngineEvent::Finish { reason, .. } => {
                    // Flush any remaining decoded text from multi-byte sequences
                    if let Some(remaining) = decoder.flush(&state.tokenizer) {
                        let flush_chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: Some(remaining),
                                },
                                finish_reason: None,
                            }],
                        };
                        if let Ok(data) = serde_json::to_string(&flush_chunk) {
                            let _ = sse_tx.send(Ok(Event::default().data(data))).await;
                        }
                    }

                    let reason_str = match reason {
                        FinishReason::EosToken | FinishReason::StopString => "stop",
                        FinishReason::MaxTokens => "length",
                        FinishReason::Cancelled => "stop",
                    };
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(reason_str.to_string()),
                        }],
                    };
                    if let Ok(data) = serde_json::to_string(&chunk) {
                        let _ = sse_tx.send(Ok(Event::default().data(data))).await;
                    }
                    let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
                    terminated = true;
                    break;
                }
                EngineEvent::Error { error, .. } => {
                    // Send an SSE error event so streaming clients see the failure
                    // instead of interpreting a bare [DONE] as successful completion.
                    let err_payload = serde_json::json!({
                        "error": { "message": error, "type": "server_error" }
                    });
                    if let Ok(data) = serde_json::to_string(&err_payload) {
                        let _ = sse_tx.send(Ok(Event::default().event("error").data(data))).await;
                    }
                    let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
                    terminated = true;
                    break;
                }
            }
        }

        // If the loop exited because the channel closed (no Finish/Error),
        // emit a terminal error event so clients see a clean end-of-stream.
        if !terminated {
            let err_payload = serde_json::json!({
                "error": { "message": "stream interrupted", "type": "server_error" }
            });
            if let Ok(data) = serde_json::to_string(&err_payload) {
                let _ = sse_tx.send(Ok(Event::default().event("error").data(data))).await;
            }
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
        }
    });

    Sse::new(ReceiverStream::new(sse_rx))
}

/// GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "forge",
        }]
    }))
}

/// GET /forge/v1/health
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok"
    }))
}
