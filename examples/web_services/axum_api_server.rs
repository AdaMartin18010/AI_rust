//! Axum Web API服务框架实现
//! 
//! 本示例展示了一个完整的Web API服务框架，包括：
//! - RESTful API设计
//! - 中间件系统
//! - 错误处理
//! - 请求验证
//! - 响应序列化
//! - 健康检查
//! - 指标监控

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware,
    response::{Html, Json},
    routing::{get, post, put, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
    compression::CompressionLayer,
};
use tracing::{info, warn, error};
use uuid::Uuid;

/// API响应结构
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub message: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            message: None,
            timestamp: chrono::Utc::now(),
        }
    }
    
    pub fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            message: None,
            timestamp: chrono::Utc::now(),
        }
    }
    
    pub fn message(message: String) -> Self {
        Self {
            success: true,
            data: None,
            error: None,
            message: Some(message),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// 用户数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub name: String,
    pub email: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// 创建用户请求
#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub name: String,
    #[serde(rename = "email")]
    pub email: String,
}

/// 更新用户请求
#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    pub name: Option<String>,
    pub email: Option<String>,
}

/// 查询参数
#[derive(Debug, Deserialize)]
pub struct UserQuery {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub search: Option<String>,
}

/// 应用状态
#[derive(Debug, Clone)]
pub struct AppState {
    pub users: Arc<RwLock<HashMap<Uuid, User>>>,
    pub metrics: Arc<RwLock<ApiMetrics>>,
}

/// API指标
#[derive(Debug, Clone, Default)]
pub struct ApiMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time: f64,
    pub active_connections: u64,
}

/// 健康检查响应
#[derive(Debug, Serialize)]
pub struct HealthCheck {
    pub status: String,
    pub version: String,
    pub uptime: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metrics: ApiMetrics,
}

/// 服务器信息
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub endpoints: Vec<String>,
    pub features: Vec<String>,
}

/// 错误类型
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("User not found: {id}")]
    UserNotFound { id: Uuid },
    
    #[error("Invalid email format: {email}")]
    InvalidEmail { email: String },
    
    #[error("User already exists: {email}")]
    UserExists { email: String },
    
    #[error("Validation error: {message}")]
    Validation { message: String },
    
    #[error("Internal server error: {message}")]
    Internal { message: String },
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_message) = match self {
            ApiError::UserNotFound { .. } => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::InvalidEmail { .. } => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::UserExists { .. } => (StatusCode::CONFLICT, self.to_string()),
            ApiError::Validation { .. } => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::Internal { .. } => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        
        let response = ApiResponse::<()>::error(error_message);
        (status, Json(response)).into_response()
    }
}

/// 验证邮箱格式
fn validate_email(email: &str) -> Result<(), ApiError> {
    if !email.contains('@') || !email.contains('.') {
        return Err(ApiError::InvalidEmail {
            email: email.to_string(),
        });
    }
    Ok(())
}

/// 验证用户数据
fn validate_user_request(req: &CreateUserRequest) -> Result<(), ApiError> {
    if req.name.trim().is_empty() {
        return Err(ApiError::Validation {
            message: "Name cannot be empty".to_string(),
        });
    }
    
    if req.email.trim().is_empty() {
        return Err(ApiError::Validation {
            message: "Email cannot be empty".to_string(),
        });
    }
    
    validate_email(&req.email)?;
    Ok(())
}

/// 创建用户
async fn create_user(
    State(state): State<AppState>,
    Json(request): Json<CreateUserRequest>,
) -> Result<Json<ApiResponse<User>>, ApiError> {
    info!("Creating user: {}", request.email);
    
    // 验证请求数据
    validate_user_request(&request)?;
    
    // 检查用户是否已存在
    let users = state.users.read().await;
    if users.values().any(|user| user.email == request.email) {
        return Err(ApiError::UserExists {
            email: request.email,
        });
    }
    drop(users);
    
    // 创建新用户
    let user = User {
        id: Uuid::new_v4(),
        name: request.name,
        email: request.email,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    
    // 保存用户
    let mut users = state.users.write().await;
    users.insert(user.id, user.clone());
    
    // 更新指标
    let mut metrics = state.metrics.write().await;
    metrics.total_requests += 1;
    metrics.successful_requests += 1;
    
    info!("User created successfully: {}", user.id);
    Ok(Json(ApiResponse::success(user)))
}

/// 获取用户列表
async fn get_users(
    State(state): State<AppState>,
    Query(query): Query<UserQuery>,
) -> Result<Json<ApiResponse<Vec<User>>>, ApiError> {
    info!("Getting users with query: {:?}", query);
    
    let users = state.users.read().await;
    let mut user_list: Vec<User> = users.values().cloned().collect();
    
    // 搜索过滤
    if let Some(search) = &query.search {
        user_list.retain(|user| {
            user.name.to_lowercase().contains(&search.to_lowercase())
                || user.email.to_lowercase().contains(&search.to_lowercase())
        });
    }
    
    // 分页
    let page = query.page.unwrap_or(1);
    let limit = query.limit.unwrap_or(10);
    let start = ((page - 1) * limit) as usize;
    let end = (start + limit as usize).min(user_list.len());
    
    let paginated_users = if start < user_list.len() {
        user_list[start..end].to_vec()
    } else {
        Vec::new()
    };
    
    // 更新指标
    let mut metrics = state.metrics.write().await;
    metrics.total_requests += 1;
    metrics.successful_requests += 1;
    
    Ok(Json(ApiResponse::success(paginated_users)))
}

/// 获取单个用户
async fn get_user(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ApiResponse<User>>, ApiError> {
    info!("Getting user: {}", id);
    
    let users = state.users.read().await;
    let user = users.get(&id).cloned()
        .ok_or_else(|| ApiError::UserNotFound { id })?;
    
    // 更新指标
    let mut metrics = state.metrics.write().await;
    metrics.total_requests += 1;
    metrics.successful_requests += 1;
    
    Ok(Json(ApiResponse::success(user)))
}

/// 更新用户
async fn update_user(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(request): Json<UpdateUserRequest>,
) -> Result<Json<ApiResponse<User>>, ApiError> {
    info!("Updating user: {}", id);
    
    let mut users = state.users.write().await;
    let user = users.get_mut(&id)
        .ok_or_else(|| ApiError::UserNotFound { id })?;
    
    // 更新字段
    if let Some(name) = request.name {
        if name.trim().is_empty() {
            return Err(ApiError::Validation {
                message: "Name cannot be empty".to_string(),
            });
        }
        user.name = name;
    }
    
    if let Some(email) = request.email {
        if email.trim().is_empty() {
            return Err(ApiError::Validation {
                message: "Email cannot be empty".to_string(),
            });
        }
        validate_email(&email)?;
        
        // 检查邮箱是否已被其他用户使用
        if users.values().any(|u| u.id != id && u.email == email) {
            return Err(ApiError::UserExists { email });
        }
        user.email = email;
    }
    
    user.updated_at = chrono::Utc::now();
    let updated_user = user.clone();
    
    // 更新指标
    let mut metrics = state.metrics.write().await;
    metrics.total_requests += 1;
    metrics.successful_requests += 1;
    
    info!("User updated successfully: {}", id);
    Ok(Json(ApiResponse::success(updated_user)))
}

/// 删除用户
async fn delete_user(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ApiResponse<()>>, ApiError> {
    info!("Deleting user: {}", id);
    
    let mut users = state.users.write().await;
    users.remove(&id)
        .ok_or_else(|| ApiError::UserNotFound { id })?;
    
    // 更新指标
    let mut metrics = state.metrics.write().await;
    metrics.total_requests += 1;
    metrics.successful_requests += 1;
    
    info!("User deleted successfully: {}", id);
    Ok(Json(ApiResponse::message("User deleted successfully".to_string())))
}

/// 健康检查
async fn health_check(State(state): State<AppState>) -> Json<ApiResponse<HealthCheck>> {
    let metrics = state.metrics.read().await;
    let health = HealthCheck {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        timestamp: chrono::Utc::now(),
        metrics: metrics.clone(),
    };
    
    Json(ApiResponse::success(health))
}

/// 服务器信息
async fn server_info() -> Json<ApiResponse<ServerInfo>> {
    let info = ServerInfo {
        name: "AI-Rust API Server".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "High-performance AI API server built with Rust and Axum".to_string(),
        endpoints: vec![
            "GET /api/v1/users".to_string(),
            "POST /api/v1/users".to_string(),
            "GET /api/v1/users/:id".to_string(),
            "PUT /api/v1/users/:id".to_string(),
            "DELETE /api/v1/users/:id".to_string(),
            "GET /health".to_string(),
            "GET /info".to_string(),
        ],
        features: vec![
            "RESTful API".to_string(),
            "Request validation".to_string(),
            "Error handling".to_string(),
            "Metrics collection".to_string(),
            "Health checks".to_string(),
            "CORS support".to_string(),
            "Request compression".to_string(),
            "Request timeout".to_string(),
        ],
    };
    
    Json(ApiResponse::success(info))
}

/// 根路径
async fn root() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI-Rust API Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AI-Rust API Server</h1>
            <p>Welcome to the AI-Rust API Server! This is a high-performance API server built with Rust and Axum.</p>
            
            <h2>📋 Available Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/users - Get all users
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/v1/users - Create a new user
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/users/:id - Get user by ID
            </div>
            <div class="endpoint">
                <span class="method">PUT</span> /api/v1/users/:id - Update user
            </div>
            <div class="endpoint">
                <span class="method">DELETE</span> /api/v1/users/:id - Delete user
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /health - Health check
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /info - Server information
            </div>
            
            <h2>🔧 Features</h2>
            <ul>
                <li>RESTful API design</li>
                <li>Request validation</li>
                <li>Error handling</li>
                <li>Metrics collection</li>
                <li>Health checks</li>
                <li>CORS support</li>
                <li>Request compression</li>
                <li>Request timeout</li>
            </ul>
        </div>
    </body>
    </html>
    "#)
}

/// 请求日志中间件
async fn request_logger(
    headers: HeaderMap,
    request: axum::http::Request<axum::body::Body>,
    next: middleware::Next,
) -> axum::response::Response {
    let start = std::time::Instant::now();
    let user_agent = headers.get("user-agent")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");
    
    info!("Request: {} {} - User-Agent: {}", request.method(), request.uri(), user_agent);
    
    let response = next.run(request).await;
    let duration = start.elapsed();
    
    info!("Response: {} - Duration: {:?}", response.status(), duration);
    response
}

/// 创建应用路由
pub fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        .route("/info", get(server_info))
        .route("/api/v1/users", get(get_users).post(create_user))
        .route("/api/v1/users/:id", get(get_user).put(update_user).delete(delete_user))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
                .layer(middleware::from_fn(request_logger))
        )
        .with_state(state)
}

/// 启动服务器
pub async fn start_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    // 初始化状态
    let state = AppState {
        users: Arc::new(RwLock::new(HashMap::new())),
        metrics: Arc::new(RwLock::new(ApiMetrics::default())),
    };
    
    // 创建应用
    let app = create_app(state);
    
    // 创建监听器
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    
    info!("🚀 AI-Rust API Server starting on port {}", port);
    info!("📋 Available endpoints:");
    info!("  GET  / - Root page");
    info!("  GET  /health - Health check");
    info!("  GET  /info - Server information");
    info!("  GET  /api/v1/users - Get all users");
    info!("  POST /api/v1/users - Create user");
    info!("  GET  /api/v1/users/:id - Get user by ID");
    info!("  PUT  /api/v1/users/:id - Update user");
    info!("  DELETE /api/v1/users/:id - Delete user");
    
    // 启动服务器
    axum::serve(listener, app).await?;
    
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

    async fn create_test_state() -> AppState {
        AppState {
            users: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ApiMetrics::default())),
        }
    }

    #[tokio::test]
    async fn test_create_user() {
        let state = create_test_state().await;
        let request = CreateUserRequest {
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };
        
        let response = create_user(State(state), Json(request)).await;
        assert!(response.is_ok());
        
        let api_response = response.unwrap().0;
        assert!(api_response.success);
        assert!(api_response.data.is_some());
        
        let user = api_response.data.unwrap();
        assert_eq!(user.name, "Test User");
        assert_eq!(user.email, "test@example.com");
    }

    #[tokio::test]
    async fn test_create_user_invalid_email() {
        let state = create_test_state().await;
        let request = CreateUserRequest {
            name: "Test User".to_string(),
            email: "invalid-email".to_string(),
        };
        
        let response = create_user(State(state), Json(request)).await;
        assert!(response.is_err());
        
        let error = response.unwrap_err();
        match error {
            ApiError::InvalidEmail { .. } => {},
            _ => panic!("Expected InvalidEmail error"),
        }
    }

    #[tokio::test]
    async fn test_get_users() {
        let state = create_test_state().await;
        
        // 创建测试用户
        let user = User {
            id: Uuid::new_v4(),
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        state.users.write().await.insert(user.id, user);
        
        let query = UserQuery {
            page: None,
            limit: None,
            search: None,
        };
        
        let response = get_users(State(state), Query(query)).await;
        assert!(response.is_ok());
        
        let api_response = response.unwrap().0;
        assert!(api_response.success);
        assert!(api_response.data.is_some());
        
        let users = api_response.data.unwrap();
        assert_eq!(users.len(), 1);
    }

    #[tokio::test]
    async fn test_get_user_not_found() {
        let state = create_test_state().await;
        let id = Uuid::new_v4();
        
        let response = get_user(State(state), Path(id)).await;
        assert!(response.is_err());
        
        let error = response.unwrap_err();
        match error {
            ApiError::UserNotFound { .. } => {},
            _ => panic!("Expected UserNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_health_check() {
        let state = create_test_state().await;
        
        let response = health_check(State(state)).await;
        let api_response = response.0;
        
        assert!(api_response.success);
        assert!(api_response.data.is_some());
        
        let health = api_response.data.unwrap();
        assert_eq!(health.status, "healthy");
    }

    #[tokio::test]
    async fn test_server_info() {
        let response = server_info().await;
        let api_response = response.0;
        
        assert!(api_response.success);
        assert!(api_response.data.is_some());
        
        let info = api_response.data.unwrap();
        assert_eq!(info.name, "AI-Rust API Server");
        assert!(!info.endpoints.is_empty());
        assert!(!info.features.is_empty());
    }
}

/// 性能基准测试
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn benchmark_create_user() {
        let state = create_test_state().await;
        let request = CreateUserRequest {
            name: "Benchmark User".to_string(),
            email: "benchmark@example.com".to_string(),
        };
        
        let start = Instant::now();
        for i in 0..1000 {
            let mut req = request.clone();
            req.email = format!("benchmark{}@example.com", i);
            let _ = create_user(State(state.clone()), Json(req)).await;
        }
        let duration = start.elapsed();
        
        println!("Created 1000 users in {:?}", duration);
        println!("Average time per user: {:?}", duration / 1000);
    }

    #[tokio::test]
    async fn benchmark_get_users() {
        let state = create_test_state().await;
        
        // 创建测试数据
        let mut users = state.users.write().await;
        for i in 0..1000 {
            let user = User {
                id: Uuid::new_v4(),
                name: format!("User {}", i),
                email: format!("user{}@example.com", i),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };
            users.insert(user.id, user);
        }
        drop(users);
        
        let query = UserQuery {
            page: Some(1),
            limit: Some(10),
            search: None,
        };
        
        let start = Instant::now();
        for _ in 0..100 {
            let _ = get_users(State(state.clone()), Query(query.clone())).await;
        }
        let duration = start.elapsed();
        
        println!("Performed 100 get_users requests in {:?}", duration);
        println!("Average time per request: {:?}", duration / 100);
    }
}
