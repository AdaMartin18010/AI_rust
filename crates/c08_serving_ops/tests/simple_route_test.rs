#[cfg(test)]
mod tests {
    use axum::{routing::get, Router, Json};
    use axum::http::StatusCode;
    use serde_json::Value;
    use tokio::net::TcpListener;
    use std::net::SocketAddr;

    async fn simple_handler() -> (StatusCode, Json<Value>) {
        (StatusCode::OK, Json(serde_json::json!({"message": "simple test"})))
    }

    fn create_simple_app() -> Router {
        Router::new()
            .route("/test", get(simple_handler))
    }

    async fn spawn_simple_app() -> (String, tokio::task::JoinHandle<()>) {
        let app = create_simple_app();
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}:{}", addr.ip(), addr.port());
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (base_url, handle)
    }

    #[tokio::test]
    async fn test_simple_route() {
        let (base, handle) = spawn_simple_app().await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        
        let resp = client
            .get(format!("{}/test", base))
            .send()
            .await
            .expect("request failed");
            
        let status = resp.status();
        let body: serde_json::Value = resp.json().await.expect("json parse");
        
        println!("Simple route test - Status: {:?}, Body: {}", status, body);
        
        assert!(status.is_success());
        assert_eq!(body["message"], "simple test");
        
        handle.abort();
    }
}
