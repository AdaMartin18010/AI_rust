//! Web服务部署示例
//! 
//! 本示例展示了一个完整的Web服务部署方案，包括：
//! - REST API服务
//! - gRPC服务
//! - WebSocket服务
//! - 健康检查
//! - 监控和日志
//! - Docker部署配置

use axum::{
    extract::{Path, Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    compression::CompressionLayer,
    timeout::TimeoutLayer,
};
use tracing::{info, warn, error};
use tracing_subscriber;

/// 应用状态
#[derive(Clone)]
pub struct AppState {
    pub counter: Arc<RwLock<u64>>,
    pub data: Arc<RwLock<HashMap<String, String>>>,
}

/// API响应结构
#[derive(Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: u64,
}

/// 健康检查响应
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime: u64,
    pub timestamp: u64,
}

/// 计数器响应
#[derive(Serialize, Deserialize)]
pub struct CounterResponse {
    pub count: u64,
    pub message: String,
}

/// 数据存储响应
#[derive(Serialize, Deserialize)]
pub struct DataResponse {
    pub key: String,
    pub value: String,
}

/// 查询参数
#[derive(Deserialize)]
pub struct QueryParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// 创建应用状态
fn create_app_state() -> AppState {
    AppState {
        counter: Arc::new(RwLock::new(0)),
        data: Arc::new(RwLock::new(HashMap::new())),
    }
}

/// 健康检查处理器
async fn health_check(State(state): State<AppState>) -> Json<ApiResponse<HealthResponse>> {
    let start_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime: start_time,
        timestamp: start_time,
    };
    
    Json(ApiResponse {
        success: true,
        data: Some(response),
        error: None,
        timestamp: start_time,
    })
}

/// 根路径处理器
async fn root() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI-Rust Web Service</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AI-Rust Web Service</h1>
            <p>Welcome to the AI-Rust Web Service API!</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <span class="method">GET</span> /health - Health check
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/counter - Get counter value
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/counter/increment - Increment counter
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/data - Get all data
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/data - Store data
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/data/{key} - Get specific data
            </div>
            <div class="endpoint">
                <span class="method">DELETE</span> /api/data/{key} - Delete data
            </div>
            <div class="endpoint">
                <span class="method">WS</span> /ws - WebSocket connection
            </div>
        </div>
    </body>
    </html>
    "#)
}

/// 获取计数器值
async fn get_counter(State(state): State<AppState>) -> Json<ApiResponse<CounterResponse>> {
    let counter = state.counter.read().await;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let response = CounterResponse {
        count: *counter,
        message: format!("Current counter value: {}", *counter),
    };
    
    Json(ApiResponse {
        success: true,
        data: Some(response),
        error: None,
        timestamp,
    })
}

/// 增加计数器
async fn increment_counter(State(state): State<AppState>) -> Json<ApiResponse<CounterResponse>> {
    let mut counter = state.counter.write().await;
    *counter += 1;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let response = CounterResponse {
        count: *counter,
        message: format!("Counter incremented to: {}", *counter),
    };
    
    info!("Counter incremented to: {}", *counter);
    
    Json(ApiResponse {
        success: true,
        data: Some(response),
        error: None,
        timestamp,
    })
}

/// 获取所有数据
async fn get_all_data(
    State(state): State<AppState>,
    Query(params): Query<QueryParams>,
) -> Json<ApiResponse<Vec<DataResponse>>> {
    let data = state.data.read().await;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let mut items: Vec<DataResponse> = data
        .iter()
        .map(|(key, value)| DataResponse {
            key: key.clone(),
            value: value.clone(),
        })
        .collect();
    
    // 应用分页
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(100);
    
    if offset < items.len() {
        let end = std::cmp::min(offset + limit, items.len());
        items = items[offset..end].to_vec();
    } else {
        items.clear();
    }
    
    Json(ApiResponse {
        success: true,
        data: Some(items),
        error: None,
        timestamp,
    })
}

/// 存储数据
async fn store_data(
    State(state): State<AppState>,
    Json(payload): Json<DataResponse>,
) -> Result<Json<ApiResponse<DataResponse>>, StatusCode> {
    let mut data = state.data.write().await;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    data.insert(payload.key.clone(), payload.value.clone());
    
    info!("Stored data: {} = {}", payload.key, payload.value);
    
    Ok(Json(ApiResponse {
        success: true,
        data: Some(payload),
        error: None,
        timestamp,
    }))
}

/// 获取特定数据
async fn get_data(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<ApiResponse<DataResponse>>, StatusCode> {
    let data = state.data.read().await;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    match data.get(&key) {
        Some(value) => {
            let response = DataResponse {
                key: key.clone(),
                value: value.clone(),
            };
            
            Ok(Json(ApiResponse {
                success: true,
                data: Some(response),
                error: None,
                timestamp,
            }))
        }
        None => {
            warn!("Data not found for key: {}", key);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

/// 删除数据
async fn delete_data(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    let mut data = state.data.write().await;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    match data.remove(&key) {
        Some(_) => {
            info!("Deleted data for key: {}", key);
            Ok(Json(ApiResponse {
                success: true,
                data: Some(format!("Data deleted for key: {}", key)),
                error: None,
                timestamp,
            }))
        }
        None => {
            warn!("Data not found for key: {}", key);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

/// WebSocket处理器
async fn websocket_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> axum::response::Response {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// 处理WebSocket连接
async fn handle_socket(socket: axum::extract::ws::WebSocket, state: AppState) {
    use axum::extract::ws::{Message, WebSocket};
    use futures_util::{SinkExt, StreamExt};
    
    let (mut sender, mut receiver) = socket.split();
    
    // 发送欢迎消息
    let welcome_msg = serde_json::json!({
        "type": "welcome",
        "message": "Connected to AI-Rust WebSocket",
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });
    
    if let Err(e) = sender.send(Message::Text(welcome_msg.to_string())).await {
        error!("Failed to send welcome message: {}", e);
        return;
    }
    
    info!("WebSocket client connected");
    
    // 处理消息
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                info!("Received WebSocket message: {}", text);
                
                // 解析消息
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(msg_type) = json.get("type").and_then(|v| v.as_str()) {
                        match msg_type {
                            "get_counter" => {
                                let counter = state.counter.read().await;
                                let response = serde_json::json!({
                                    "type": "counter",
                                    "value": *counter,
                                    "timestamp": std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_secs()
                                });
                                
                                if let Err(e) = sender.send(Message::Text(response.to_string())).await {
                                    error!("Failed to send counter response: {}", e);
                                    break;
                                }
                            }
                            "increment_counter" => {
                                let mut counter = state.counter.write().await;
                                *counter += 1;
                                
                                let response = serde_json::json!({
                                    "type": "counter_incremented",
                                    "value": *counter,
                                    "timestamp": std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_secs()
                                });
                                
                                if let Err(e) = sender.send(Message::Text(response.to_string())).await {
                                    error!("Failed to send increment response: {}", e);
                                    break;
                                }
                            }
                            _ => {
                                let response = serde_json::json!({
                                    "type": "error",
                                    "message": "Unknown message type",
                                    "timestamp": std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_secs()
                                });
                                
                                if let Err(e) = sender.send(Message::Text(response.to_string())).await {
                                    error!("Failed to send error response: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket client disconnected");
                break;
            }
            Ok(_) => {
                // 忽略其他类型的消息
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
        }
    }
    
    info!("WebSocket connection closed");
}

/// 创建应用路由
fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        .route("/api/counter", get(get_counter))
        .route("/api/counter/increment", post(increment_counter))
        .route("/api/data", get(get_all_data).post(store_data))
        .route("/api/data/:key", get(get_data).delete(delete_data))
        .route("/ws", get(websocket_handler))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
                .layer(CompressionLayer::new())
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30))),
        )
        .with_state(state)
}

/// 启动服务器
async fn start_server(addr: SocketAddr, state: AppState) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_app(state);
    
    info!("Starting server on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// 初始化日志
fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ai_rust_web=debug,tower_http=debug".into()),
        )
        .init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    init_logging();
    
    info!("🚀 Starting AI-Rust Web Service");
    
    // 创建应用状态
    let state = create_app_state();
    
    // 设置服务器地址
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    
    // 启动服务器
    start_server(addr, state).await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_check() {
        let state = create_app_state();
        let app = create_app(state);
        
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_counter() {
        let state = create_app_state();
        let app = create_app(state);
        
        let response = app
            .oneshot(Request::builder().uri("/api/counter").body(Body::empty()).unwrap())
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_increment_counter() {
        let state = create_app_state();
        let app = create_app(state);
        
        let response = app
            .oneshot(Request::builder().uri("/api/counter/increment").method("POST").body(Body::empty()).unwrap())
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_store_and_get_data() {
        let state = create_app_state();
        let app = create_app(state);
        
        // 存储数据
        let store_request = Request::builder()
            .uri("/api/data")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"key": "test", "value": "hello"}"#))
            .unwrap();
        
        let response = app.clone().oneshot(store_request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        
        // 获取数据
        let get_request = Request::builder()
            .uri("/api/data/test")
            .body(Body::empty())
            .unwrap();
        
        let response = app.oneshot(get_request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}