//! End-to-end integration tests for the Forge HTTP API.
//!
//! These tests verify the API contract by exercising the axum router with
//! mock engine channels. They do NOT require a GPU or model weights.

// NOTE: This file requires `forge-server` to be added as a dev-dependency
// in a top-level test crate, or it can be run as part of forge-server's tests.
// For now, this serves as documentation of the test plan.
//
// The actual tests would be:
//
// 1. Health endpoint returns {"status": "ok"}
// 2. Models endpoint returns a valid list
// 3. Non-streaming chat completions returns valid ChatCompletionResponse
// 4. Streaming chat completions returns valid SSE events ending with [DONE]
// 5. Invalid request returns HTTP 400 with error JSON
// 6. Engine unavailable returns HTTP 503
//
// Implementation requires `axum::test::TestClient` or `tower::ServiceExt`
// to send requests to the router without starting a real HTTP server.

fn main() {
    println!("Integration test placeholder — run scripts/test_server.sh with a model for E2E.");
}
