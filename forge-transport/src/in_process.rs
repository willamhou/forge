//! In-process transport: zero-overhead bridge between API and engine.
//!
//! Directly routes requests to the engine via channels. Future transports
//! (e.g., gRPC) will implement the same interface for distributed deployments.

use tokio::sync::mpsc;

use forge_core::{InferenceRequest, Result};
use forge_runtime::engine::EngineEvent;

/// Routes requests to a local engine via channels.
pub struct InProcessTransport {
    /// Channel for submitting inference requests to the engine.
    request_tx: mpsc::Sender<InProcessRequest>,
}

/// A request routed through the in-process transport.
pub struct InProcessRequest {
    pub inference_req: InferenceRequest,
    /// Channel where the engine will send events for this request.
    pub event_tx: mpsc::Sender<EngineEvent>,
}

impl InProcessTransport {
    pub fn new(request_tx: mpsc::Sender<InProcessRequest>) -> Self {
        Self { request_tx }
    }

    /// Submit an inference request and get a channel for receiving events.
    pub async fn submit(
        &self,
        req: InferenceRequest,
    ) -> Result<mpsc::Receiver<EngineEvent>> {
        let (event_tx, event_rx) = mpsc::channel(256);
        let transport_req = InProcessRequest {
            inference_req: req,
            event_tx,
        };
        self.request_tx
            .send(transport_req)
            .await
            .map_err(|_| forge_core::ForgeError::Internal("engine channel closed".into()))?;
        Ok(event_rx)
    }
}
