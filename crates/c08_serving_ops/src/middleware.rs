use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::time::Instant;
use super::AppState;

pub async fn metrics_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    
    let response = next.run(request).await;
    
    let duration = start.elapsed();
    let status_code = response.status().as_u16();
    
    if let Some(metrics) = &state.metrics {
        let endpoint = format!("{} {}", method, path);
        metrics.record_request(endpoint, duration, status_code);
    }
    
    response
}

// 错误处理中间件
pub async fn error_handler_middleware(
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    
    // 如果是错误响应，记录日志
    if response.status().is_client_error() || response.status().is_server_error() {
        tracing::error!(
            "Request failed with status: {} for path: {}",
            response.status(),
            path
        );
    }
    
    response
}
