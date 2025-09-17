use std::sync::Arc;

use axum::{routing::{get, post}, Json, Router, extract::State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

pub mod engine;
use engine::{InferenceEngine, DummyEngine};

#[derive(Debug, Serialize)]
struct Healthz {
    status: &'static str,
}

#[derive(Debug, Deserialize)]
struct InferRequest {
    prompt: String,
}

#[derive(Debug, Serialize)]
struct InferResponse {
    output: String,
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn InferenceEngine>,
}

pub fn create_app() -> Router {
    let state = AppState { engine: Arc::new(DummyEngine) };
    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_origin(Any)
        .allow_headers(Any);

    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/infer", post(infer))
        .with_state(state)
        .layer(cors)
}

async fn healthz() -> (StatusCode, Json<Healthz>) {
    (StatusCode::OK, Json(Healthz { status: "ok" }))
}

async fn infer(State(state): State<AppState>, Json(req): Json<InferRequest>) -> (StatusCode, Json<InferResponse>) {
    let output = state.engine.infer(&req.prompt).await.unwrap_or_else(|e| format!("error: {}", e));
    (StatusCode::OK, Json(InferResponse { output }))
}

async fn readyz() -> (StatusCode, Json<Healthz>) {
    // 简单返回OK，未来可加入引擎、下游依赖探针
    (StatusCode::OK, Json(Healthz { status: "ok" }))
}


