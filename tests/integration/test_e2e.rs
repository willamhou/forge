//! End-to-end integration tests for the Forge HTTP API.
//!
//! These tests verify the API contract by exercising the axum router with
//! mock engine channels. They do NOT require a GPU or model weights.
//!
//! For live testing with a real model, use: `bash scripts/test_server.sh /path/to/model`

// NOTE: This file requires `forge-server` to be added as a dev-dependency
// in a top-level test crate, or it can be run as part of forge-server's tests.
// Implementation requires `axum::test::TestClient` or `tower::ServiceExt`
// to send requests to the router without starting a real HTTP server.

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires forge-server as dev-dependency"]
    fn health_endpoint_returns_ok() {
        // TODO: Wire up axum TestClient with mock engine channel
    }

    #[test]
    #[ignore = "requires forge-server as dev-dependency"]
    fn models_endpoint_returns_list() {
        // TODO: Wire up axum TestClient with mock engine channel
    }

    #[test]
    #[ignore = "requires forge-server as dev-dependency"]
    fn non_streaming_chat_completion() {
        // TODO: Send non-streaming request, verify ChatCompletionResponse
    }

    #[test]
    #[ignore = "requires forge-server as dev-dependency"]
    fn streaming_chat_completion() {
        // TODO: Send streaming request, verify SSE events ending with [DONE]
    }

    #[test]
    #[ignore = "requires forge-server as dev-dependency"]
    fn invalid_request_returns_400() {
        // TODO: Send request with missing fields, verify HTTP 400
    }

    #[test]
    #[ignore = "requires forge-server as dev-dependency"]
    fn engine_unavailable_returns_503() {
        // TODO: Close engine channel, verify HTTP 503
    }
}
