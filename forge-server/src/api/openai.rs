//! OpenAI-compatible chat completions endpoints.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::error;

use forge_core::{FinishReason, InferenceRequest, SamplingParams};
use forge_runtime::engine::EngineEvent;

use crate::chat_template::ChatTemplate;
use crate::tokenizer::ForgeTokenizer;

use super::types::*;

/// Shared application state passed to handlers.
pub struct AppState {
    pub model_name: String,
    pub tokenizer: ForgeTokenizer,
    pub chat_template: ChatTemplate,
    /// Channel to submit requests to the engine.
    pub request_tx: mpsc::Sender<SubmitRequest>,
}

/// A request submitted to the engine, with a channel for sending events back.
pub struct SubmitRequest {
    pub inference_req: InferenceRequest,
    /// Engine sends events into this channel; the HTTP handler holds the receiver.
    pub event_tx: mpsc::Sender<EngineEvent>,
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
            return Json(serde_json::json!({
                "error": { "message": format!("template error: {e}"), "type": "invalid_request_error" }
            }))
            .into_response();
        }
    };

    // Tokenize
    let prompt_tokens = match state.tokenizer.encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return Json(serde_json::json!({
                "error": { "message": format!("tokenizer error: {e}"), "type": "invalid_request_error" }
            }))
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
    // Add EOS token to stop list
    params
        .stop_token_ids
        .push(state.tokenizer.eos_token_id());

    let inference_req = InferenceRequest {
        request_id: request_id.clone(),
        prompt_tokens,
        sampling_params: params,
    };

    // Create per-request event channel
    let (event_tx, event_rx) = mpsc::channel(256);

    let submit = SubmitRequest {
        inference_req,
        event_tx,
    };

    if state.request_tx.send(submit).await.is_err() {
        return Json(serde_json::json!({
            "error": { "message": "engine unavailable", "type": "server_error" }
        }))
        .into_response();
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        handle_streaming(event_rx, request_id, state.model_name.clone(), now).into_response()
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
) -> Json<ChatCompletionResponse> {
    let mut token_ids = Vec::new();
    let mut finish_reason = None;

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
                // Return what we have with an error finish
                error!("engine error: {error}");
                break;
            }
        }
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
}

/// Stream tokens as SSE events.
fn handle_streaming(
    mut event_rx: mpsc::Receiver<EngineEvent>,
    request_id: String,
    model: String,
    created: u64,
) -> Sse<ReceiverStream<std::result::Result<Event, std::convert::Infallible>>> {
    let (sse_tx, sse_rx) =
        mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(256);

    tokio::spawn(async move {
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

        let data = serde_json::to_string(&initial_chunk).unwrap();
        let _ = sse_tx.send(Ok(Event::default().data(data))).await;

        while let Some(event) = event_rx.recv().await {
            match event {
                EngineEvent::Token { text, .. } => {
                    let content = text.unwrap_or_default();
                    if content.is_empty() {
                        continue;
                    }
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
                    let data = serde_json::to_string(&chunk).unwrap();
                    if sse_tx.send(Ok(Event::default().data(data))).await.is_err() {
                        break;
                    }
                }
                EngineEvent::Finish { reason, .. } => {
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
                    let data = serde_json::to_string(&chunk).unwrap();
                    let _ = sse_tx.send(Ok(Event::default().data(data))).await;
                    // Send [DONE]
                    let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
                    break;
                }
                EngineEvent::Error { .. } => {
                    let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
                    break;
                }
            }
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
