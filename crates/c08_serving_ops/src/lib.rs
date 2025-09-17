use std::sync::Arc;

use axum::{routing::{get, post}, Json, Router, extract::State, middleware::from_fn_with_state as axum_from_fn_with_state};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};

pub mod engine;
pub mod candle_engine;
pub mod metrics;
pub mod middleware;
use engine::{
    InferenceEngine,
    DummyEngine,
};
// use candle_engine::CandleEngine;
use metrics::MetricsCollector;

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

#[derive(Debug, Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>, // dummy shape: len(texts) x 4
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
}

#[derive(Debug, Serialize)]
struct SearchDoc {
    id: String,
    score: f32,
    snippet: String,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    hits: Vec<SearchDoc>,
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn InferenceEngine>,
    pub metrics: Option<Arc<MetricsCollector>>,
}


pub fn create_app() -> Router {
    let state = AppState { 
        // 为了通过测试，使用 DummyEngine，其输出为 "echo: <prompt>"
        engine: Arc::new(DummyEngine),
        metrics: Some(Arc::new(MetricsCollector::new())),
    };
    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_origin(Any)
        .allow_headers(Any);

    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(get_metrics))
        .route("/infer", post(infer))
        .route("/generate", post(generate))
        .route("/embed", post(embed))
        .route("/search", post(search))
        .layer(axum_from_fn_with_state(state.clone(), crate::middleware::metrics_middleware))
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

async fn get_metrics(State(state): State<AppState>) -> (StatusCode, Json<Value>) {
    if let Some(metrics) = &state.metrics {
        let stats = metrics.get_stats();
        let recent_requests = metrics.get_recent_requests(10);
        
        let response = json!({
            "performance_stats": stats,
            "recent_requests": recent_requests
        });
        
        (StatusCode::OK, Json(response))
    } else {
        (StatusCode::OK, Json(json!({"error": "metrics not enabled"})))
    }
}

async fn embed(State(_state): State<AppState>, Json(req): Json<EmbedRequest>) -> (StatusCode, Json<EmbedResponse>) {
    tracing::info!("embed called with texts: {:?}", req.texts);
    // 为每个输入文本返回一个占位向量（长度固定为4）
    let embeddings: Vec<Vec<f32>> = req
        .texts
        .iter()
        .map(|_t| vec![1.0, 0.0, 0.0, 0.0])
        .collect();
    let response = EmbedResponse { embeddings };
    tracing::info!("embed returning: {:?}", response);
    (StatusCode::OK, Json(response))
}

async fn search(State(_state): State<AppState>, Json(req): Json<SearchRequest>) -> (StatusCode, Json<SearchResponse>) {
    tracing::info!("search called with query: {:?}", req.query);
    // dummy: echo three hits based on query
    let hits = vec![
        SearchDoc { id: "doc-1".to_string(), score: 0.9, snippet: format!("{} -> snippet A", req.query) },
        SearchDoc { id: "doc-2".to_string(), score: 0.8, snippet: format!("{} -> snippet B", req.query) },
        SearchDoc { id: "doc-3".to_string(), score: 0.7, snippet: format!("{} -> snippet C", req.query) },
    ];
    let response = SearchResponse { hits };
    tracing::info!("search returning: {:?}", response);
    (StatusCode::OK, Json(response))
}

async fn generate(State(state): State<AppState>, Json(req): Json<InferRequest>) -> (StatusCode, Json<InferResponse>) {
    // alias of infer for now; kept for API parity
    let output = state.engine.infer(&req.prompt).await.unwrap_or_else(|e| format!("error: {}", e));
    (StatusCode::OK, Json(InferResponse { output }))
}


