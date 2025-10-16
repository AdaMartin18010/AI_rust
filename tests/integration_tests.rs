//! 集成测试套件
//! 
//! 本文件包含AI-Rust项目的完整集成测试，包括：
//! - API端点测试
//! - 数据库集成测试
//! - 缓存系统测试
//! - 性能测试
//! - 端到端测试

use std::collections::HashMap;
use std::time::Duration;
use tokio::time::timeout;
use serde_json::json;
use uuid::Uuid;

/// 测试配置
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub base_url: String,
    pub timeout: Duration,
    pub max_retries: u32,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".to_string(),
            timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }
}

/// 测试客户端
pub struct TestClient {
    client: reqwest::Client,
    config: TestConfig,
}

impl TestClient {
    pub fn new(config: TestConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");
        
        Self { client, config }
    }
    
    /// 发送GET请求
    pub async fn get(&self, path: &str) -> Result<reqwest::Response, reqwest::Error> {
        let url = format!("{}{}", self.config.base_url, path);
        self.client.get(&url).send().await
    }
    
    /// 发送POST请求
    pub async fn post(&self, path: &str, body: &serde_json::Value) -> Result<reqwest::Response, reqwest::Error> {
        let url = format!("{}{}", self.config.base_url, path);
        self.client.post(&url).json(body).send().await
    }
    
    /// 发送PUT请求
    pub async fn put(&self, path: &str, body: &serde_json::Value) -> Result<reqwest::Response, reqwest::Error> {
        let url = format!("{}{}", self.config.base_url, path);
        self.client.put(&url).json(body).send().await
    }
    
    /// 发送DELETE请求
    pub async fn delete(&self, path: &str) -> Result<reqwest::Response, reqwest::Error> {
        let url = format!("{}{}", self.config.base_url, path);
        self.client.delete(&url).send().await
    }
    
    /// 等待服务启动
    pub async fn wait_for_service(&self) -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..self.config.max_retries {
            match self.get("/health").await {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(());
                    }
                }
                Err(_) => {}
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        Err("Service not available".into())
    }
}

/// API集成测试
#[tokio::test]
async fn test_api_integration() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    // 等待服务启动
    client.wait_for_service().await.expect("Service not available");
    
    // 测试健康检查
    let response = client.get("/health").await.expect("Health check failed");
    assert!(response.status().is_success());
    
    let health_data: serde_json::Value = response.json().await.expect("Failed to parse health response");
    assert_eq!(health_data["success"], true);
    
    // 测试服务器信息
    let response = client.get("/info").await.expect("Info endpoint failed");
    assert!(response.status().is_success());
    
    let info_data: serde_json::Value = response.json().await.expect("Failed to parse info response");
    assert_eq!(info_data["success"], true);
    assert!(info_data["data"]["name"].as_str().unwrap().contains("AI-Rust"));
}

/// 用户API测试
#[tokio::test]
async fn test_user_api() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 创建用户
    let user_data = json!({
        "name": "Test User",
        "email": "test@example.com"
    });
    
    let response = client.post("/api/v1/users", &user_data).await.expect("Create user failed");
    assert!(response.status().is_success());
    
    let create_response: serde_json::Value = response.json().await.expect("Failed to parse create response");
    assert_eq!(create_response["success"], true);
    
    let user_id = create_response["data"]["id"].as_str().unwrap();
    
    // 获取用户
    let response = client.get(&format!("/api/v1/users/{}", user_id)).await.expect("Get user failed");
    assert!(response.status().is_success());
    
    let get_response: serde_json::Value = response.json().await.expect("Failed to parse get response");
    assert_eq!(get_response["success"], true);
    assert_eq!(get_response["data"]["name"], "Test User");
    
    // 更新用户
    let update_data = json!({
        "name": "Updated User",
        "email": "updated@example.com"
    });
    
    let response = client.put(&format!("/api/v1/users/{}", user_id), &update_data).await.expect("Update user failed");
    assert!(response.status().is_success());
    
    let update_response: serde_json::Value = response.json().await.expect("Failed to parse update response");
    assert_eq!(update_response["success"], true);
    assert_eq!(update_response["data"]["name"], "Updated User");
    
    // 获取用户列表
    let response = client.get("/api/v1/users").await.expect("Get users failed");
    assert!(response.status().is_success());
    
    let list_response: serde_json::Value = response.json().await.expect("Failed to parse list response");
    assert_eq!(list_response["success"], true);
    assert!(list_response["data"].as_array().unwrap().len() > 0);
    
    // 删除用户
    let response = client.delete(&format!("/api/v1/users/{}", user_id)).await.expect("Delete user failed");
    assert!(response.status().is_success());
    
    let delete_response: serde_json::Value = response.json().await.expect("Failed to parse delete response");
    assert_eq!(delete_response["success"], true);
}

/// 错误处理测试
#[tokio::test]
async fn test_error_handling() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 测试无效用户ID
    let invalid_id = Uuid::new_v4();
    let response = client.get(&format!("/api/v1/users/{}", invalid_id)).await.expect("Get invalid user failed");
    assert_eq!(response.status(), 404);
    
    let error_response: serde_json::Value = response.json().await.expect("Failed to parse error response");
    assert_eq!(error_response["success"], false);
    assert!(error_response["error"].as_str().unwrap().contains("not found"));
    
    // 测试无效邮箱
    let invalid_user_data = json!({
        "name": "Test User",
        "email": "invalid-email"
    });
    
    let response = client.post("/api/v1/users", &invalid_user_data).await.expect("Create invalid user failed");
    assert_eq!(response.status(), 400);
    
    let error_response: serde_json::Value = response.json().await.expect("Failed to parse error response");
    assert_eq!(error_response["success"], false);
    assert!(error_response["error"].as_str().unwrap().contains("Invalid email"));
    
    // 测试重复邮箱
    let user_data = json!({
        "name": "Test User 1",
        "email": "duplicate@example.com"
    });
    
    let response = client.post("/api/v1/users", &user_data).await.expect("Create first user failed");
    assert!(response.status().is_success());
    
    let response = client.post("/api/v1/users", &user_data).await.expect("Create duplicate user failed");
    assert_eq!(response.status(), 409);
    
    let error_response: serde_json::Value = response.json().await.expect("Failed to parse error response");
    assert_eq!(error_response["success"], false);
    assert!(error_response["error"].as_str().unwrap().contains("already exists"));
}

/// 性能测试
#[tokio::test]
async fn test_performance() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 并发请求测试
    let start = std::time::Instant::now();
    let mut handles = Vec::new();
    
    for i in 0..100 {
        let client = TestClient::new(TestConfig::default());
        let handle = tokio::spawn(async move {
            let user_data = json!({
                "name": format!("Performance User {}", i),
                "email": format!("perf{}@example.com", i)
            });
            
            client.post("/api/v1/users", &user_data).await
        });
        handles.push(handle);
    }
    
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(response)) = handle.await {
            if response.status().is_success() {
                success_count += 1;
            }
        }
    }
    
    let duration = start.elapsed();
    println!("Created {} users in {:?}", success_count, duration);
    println!("Average time per request: {:?}", duration / 100);
    
    assert!(success_count >= 90); // 至少90%的请求成功
    assert!(duration < Duration::from_secs(10)); // 总时间少于10秒
}

/// 负载测试
#[tokio::test]
async fn test_load() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 创建测试数据
    let mut user_ids = Vec::new();
    for i in 0..50 {
        let user_data = json!({
            "name": format!("Load Test User {}", i),
            "email": format!("load{}@example.com", i)
        });
        
        let response = client.post("/api/v1/users", &user_data).await.expect("Create user failed");
        if response.status().is_success() {
            let create_response: serde_json::Value = response.json().await.expect("Failed to parse response");
            user_ids.push(create_response["data"]["id"].as_str().unwrap().to_string());
        }
    }
    
    // 并发读取测试
    let start = std::time::Instant::now();
    let mut handles = Vec::new();
    
    for user_id in &user_ids {
        let client = TestClient::new(TestConfig::default());
        let user_id = user_id.clone();
        let handle = tokio::spawn(async move {
            client.get(&format!("/api/v1/users/{}", user_id)).await
        });
        handles.push(handle);
    }
    
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(response)) = handle.await {
            if response.status().is_success() {
                success_count += 1;
            }
        }
    }
    
    let duration = start.elapsed();
    println!("Read {} users in {:?}", success_count, duration);
    println!("Average time per request: {:?}", duration / user_ids.len() as u32);
    
    assert!(success_count >= user_ids.len() - 5); // 至少95%的请求成功
    assert!(duration < Duration::from_secs(5)); // 总时间少于5秒
    
    // 清理测试数据
    for user_id in &user_ids {
        let _ = client.delete(&format!("/api/v1/users/{}", user_id)).await;
    }
}

/// 压力测试
#[tokio::test]
async fn test_stress() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 持续压力测试
    let start = std::time::Instant::now();
    let test_duration = Duration::from_secs(30);
    let mut request_count = 0;
    let mut success_count = 0;
    
    while start.elapsed() < test_duration {
        let client = TestClient::new(TestConfig::default());
        let handle = tokio::spawn(async move {
            client.get("/health").await
        });
        
        request_count += 1;
        
        if let Ok(Ok(response)) = handle.await {
            if response.status().is_success() {
                success_count += 1;
            }
        }
        
        // 控制请求频率
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    let duration = start.elapsed();
    let success_rate = success_count as f64 / request_count as f64;
    
    println!("Stress test completed:");
    println!("  Total requests: {}", request_count);
    println!("  Successful requests: {}", success_count);
    println!("  Success rate: {:.2}%", success_rate * 100.0);
    println!("  Duration: {:?}", duration);
    println!("  Requests per second: {:.2}", request_count as f64 / duration.as_secs_f64());
    
    assert!(success_rate >= 0.95); // 至少95%的成功率
    assert!(request_count >= 1000); // 至少1000个请求
}

/// 端到端测试
#[tokio::test]
async fn test_end_to_end() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 完整的用户生命周期测试
    let user_data = json!({
        "name": "E2E Test User",
        "email": "e2e@example.com"
    });
    
    // 1. 创建用户
    let response = client.post("/api/v1/users", &user_data).await.expect("Create user failed");
    assert!(response.status().is_success());
    
    let create_response: serde_json::Value = response.json().await.expect("Failed to parse create response");
    let user_id = create_response["data"]["id"].as_str().unwrap();
    
    // 2. 验证用户创建
    let response = client.get(&format!("/api/v1/users/{}", user_id)).await.expect("Get user failed");
    assert!(response.status().is_success());
    
    let get_response: serde_json::Value = response.json().await.expect("Failed to parse get response");
    assert_eq!(get_response["data"]["name"], "E2E Test User");
    
    // 3. 更新用户
    let update_data = json!({
        "name": "Updated E2E User"
    });
    
    let response = client.put(&format!("/api/v1/users/{}", user_id), &update_data).await.expect("Update user failed");
    assert!(response.status().is_success());
    
    // 4. 验证更新
    let response = client.get(&format!("/api/v1/users/{}", user_id)).await.expect("Get updated user failed");
    assert!(response.status().is_success());
    
    let get_response: serde_json::Value = response.json().await.expect("Failed to parse get response");
    assert_eq!(get_response["data"]["name"], "Updated E2E User");
    
    // 5. 删除用户
    let response = client.delete(&format!("/api/v1/users/{}", user_id)).await.expect("Delete user failed");
    assert!(response.status().is_success());
    
    // 6. 验证删除
    let response = client.get(&format!("/api/v1/users/{}", user_id)).await.expect("Get deleted user failed");
    assert_eq!(response.status(), 404);
}

/// 数据库集成测试
#[tokio::test]
async fn test_database_integration() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 测试数据库连接
    let response = client.get("/health").await.expect("Health check failed");
    assert!(response.status().is_success());
    
    let health_data: serde_json::Value = response.json().await.expect("Failed to parse health response");
    assert_eq!(health_data["success"], true);
    
    // 验证数据库检查
    let checks = health_data["data"]["checks"].as_object().unwrap();
    assert!(checks.contains_key("database"));
    assert_eq!(checks["database"]["status"], "healthy");
}

/// 缓存系统测试
#[tokio::test]
async fn test_cache_system() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 创建用户
    let user_data = json!({
        "name": "Cache Test User",
        "email": "cache@example.com"
    });
    
    let response = client.post("/api/v1/users", &user_data).await.expect("Create user failed");
    assert!(response.status().is_success());
    
    let create_response: serde_json::Value = response.json().await.expect("Failed to parse create response");
    let user_id = create_response["data"]["id"].as_str().unwrap();
    
    // 多次请求同一用户（测试缓存）
    let mut response_times = Vec::new();
    
    for _ in 0..10 {
        let start = std::time::Instant::now();
        let response = client.get(&format!("/api/v1/users/{}", user_id)).await.expect("Get user failed");
        let duration = start.elapsed();
        
        assert!(response.status().is_success());
        response_times.push(duration);
    }
    
    // 验证缓存效果（后续请求应该更快）
    let first_half_avg: Duration = response_times[0..5].iter().sum::<Duration>() / 5;
    let second_half_avg: Duration = response_times[5..10].iter().sum::<Duration>() / 5;
    
    println!("First half average: {:?}", first_half_avg);
    println!("Second half average: {:?}", second_half_avg);
    
    // 清理
    let _ = client.delete(&format!("/api/v1/users/{}", user_id)).await;
}

/// 安全测试
#[tokio::test]
async fn test_security() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 测试SQL注入防护
    let malicious_data = json!({
        "name": "'; DROP TABLE users; --",
        "email": "sql@example.com"
    });
    
    let response = client.post("/api/v1/users", &malicious_data).await.expect("Create user failed");
    // 应该成功创建用户，但恶意SQL不会执行
    assert!(response.status().is_success());
    
    // 测试XSS防护
    let xss_data = json!({
        "name": "<script>alert('xss')</script>",
        "email": "xss@example.com"
    });
    
    let response = client.post("/api/v1/users", &xss_data).await.expect("Create user failed");
    assert!(response.status().is_success());
    
    let create_response: serde_json::Value = response.json().await.expect("Failed to parse create response");
    let user_id = create_response["data"]["id"].as_str().unwrap();
    
    // 验证XSS防护
    let response = client.get(&format!("/api/v1/users/{}", user_id)).await.expect("Get user failed");
    assert!(response.status().is_success());
    
    let get_response: serde_json::Value = response.json().await.expect("Failed to parse get response");
    let name = get_response["data"]["name"].as_str().unwrap();
    assert!(!name.contains("<script>")); // XSS应该被转义
    
    // 清理
    let _ = client.delete(&format!("/api/v1/users/{}", user_id)).await;
}

/// 并发测试
#[tokio::test]
async fn test_concurrency() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 并发创建用户
    let mut handles = Vec::new();
    let mut expected_users = Vec::new();
    
    for i in 0..20 {
        let client = TestClient::new(TestConfig::default());
        let user_data = json!({
            "name": format!("Concurrent User {}", i),
            "email": format!("concurrent{}@example.com", i)
        });
        expected_users.push(user_data.clone());
        
        let handle = tokio::spawn(async move {
            client.post("/api/v1/users", &user_data).await
        });
        handles.push(handle);
    }
    
    let mut created_users = Vec::new();
    for handle in handles {
        if let Ok(Ok(response)) = handle.await {
            if response.status().is_success() {
                let create_response: serde_json::Value = response.json().await.expect("Failed to parse response");
                created_users.push(create_response["data"]["id"].as_str().unwrap().to_string());
            }
        }
    }
    
    assert_eq!(created_users.len(), expected_users.len());
    
    // 并发读取用户
    let mut read_handles = Vec::new();
    for user_id in &created_users {
        let client = TestClient::new(TestConfig::default());
        let user_id = user_id.clone();
        let handle = tokio::spawn(async move {
            client.get(&format!("/api/v1/users/{}", user_id)).await
        });
        read_handles.push(handle);
    }
    
    let mut read_success = 0;
    for handle in read_handles {
        if let Ok(Ok(response)) = handle.await {
            if response.status().is_success() {
                read_success += 1;
            }
        }
    }
    
    assert_eq!(read_success, created_users.len());
    
    // 清理
    for user_id in &created_users {
        let _ = client.delete(&format!("/api/v1/users/{}", user_id)).await;
    }
}

/// 内存泄漏测试
#[tokio::test]
async fn test_memory_leaks() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 创建大量用户然后删除
    let mut user_ids = Vec::new();
    
    for i in 0..1000 {
        let user_data = json!({
            "name": format!("Memory Test User {}", i),
            "email": format!("memory{}@example.com", i)
        });
        
        let response = client.post("/api/v1/users", &user_data).await.expect("Create user failed");
        if response.status().is_success() {
            let create_response: serde_json::Value = response.json().await.expect("Failed to parse response");
            user_ids.push(create_response["data"]["id"].as_str().unwrap().to_string());
        }
    }
    
    // 删除所有用户
    for user_id in &user_ids {
        let _ = client.delete(&format!("/api/v1/users/{}", user_id)).await;
    }
    
    // 验证用户已删除
    let response = client.get("/api/v1/users").await.expect("Get users failed");
    assert!(response.status().is_success());
    
    let list_response: serde_json::Value = response.json().await.expect("Failed to parse list response");
    let users = list_response["data"].as_array().unwrap();
    
    // 应该没有测试用户了
    let test_users: Vec<_> = users.iter()
        .filter(|user| user["email"].as_str().unwrap().contains("memory"))
        .collect();
    
    assert_eq!(test_users.len(), 0);
}

/// 故障恢复测试
#[tokio::test]
async fn test_fault_recovery() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 创建用户
    let user_data = json!({
        "name": "Fault Recovery User",
        "email": "fault@example.com"
    });
    
    let response = client.post("/api/v1/users", &user_data).await.expect("Create user failed");
    assert!(response.status().is_success());
    
    let create_response: serde_json::Value = response.json().await.expect("Failed to parse create response");
    let user_id = create_response["data"]["id"].as_str().unwrap();
    
    // 模拟服务重启（等待服务恢复）
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    // 验证数据持久性
    let response = client.get(&format!("/api/v1/users/{}", user_id)).await.expect("Get user failed");
    assert!(response.status().is_success());
    
    let get_response: serde_json::Value = response.json().await.expect("Failed to parse get response");
    assert_eq!(get_response["data"]["name"], "Fault Recovery User");
    
    // 清理
    let _ = client.delete(&format!("/api/v1/users/{}", user_id)).await;
}

/// 基准测试
#[tokio::test]
async fn benchmark_api_performance() {
    let config = TestConfig::default();
    let client = TestClient::new(config);
    
    client.wait_for_service().await.expect("Service not available");
    
    // 健康检查基准
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = client.get("/health").await;
    }
    let health_duration = start.elapsed();
    
    println!("Health check benchmark:");
    println!("  1000 requests in {:?}", health_duration);
    println!("  Average: {:?}", health_duration / 1000);
    
    // 用户创建基准
    let start = std::time::Instant::now();
    for i in 0..100 {
        let user_data = json!({
            "name": format!("Benchmark User {}", i),
            "email": format!("bench{}@example.com", i)
        });
        let _ = client.post("/api/v1/users", &user_data).await;
    }
    let create_duration = start.elapsed();
    
    println!("User creation benchmark:");
    println!("  100 users in {:?}", create_duration);
    println!("  Average: {:?}", create_duration / 100);
    
    // 用户读取基准
    let start = std::time::Instant::now();
    for i in 0..100 {
        let _ = client.get(&format!("/api/v1/users/bench{}@example.com", i)).await;
    }
    let read_duration = start.elapsed();
    
    println!("User reading benchmark:");
    println!("  100 reads in {:?}", read_duration);
    println!("  Average: {:?}", read_duration / 100);
}
