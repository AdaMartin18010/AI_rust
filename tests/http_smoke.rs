#[cfg(test)]
mod tests {
    use reqwest::Client;
    use tokio::net::TcpListener;
    use ai_rust_svc::create_app;
    use std::net::SocketAddr;

    async fn spawn_app() -> (String, tokio::task::JoinHandle<()>) {
        let app = create_app();
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}:{}", addr.ip(), addr.port());
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        // give a tick for accept loop
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (base_url, handle)
    }

    #[tokio::test]
    async fn healthz_should_ok() {
        let (base, handle) = spawn_app().await;
        let client = Client::builder().no_proxy().build().unwrap();
        let resp = client
            .get(format!("{}/healthz", base))
            .send()
            .await
            .expect("request failed");
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        eprintln!("healthz status={:?} body={}", status, text);
        assert!(status.is_success());
        handle.abort();
    }

    #[tokio::test]
    async fn infer_should_echo() {
        let (base, handle) = spawn_app().await;
        let client = Client::builder().no_proxy().build().unwrap();
        let resp = client
            .post(format!("{}/infer", base))
            .json(&serde_json::json!({"prompt":"hello"}))
            .send()
            .await
            .expect("request failed");
        let status = resp.status();
        let body: serde_json::Value = resp.json().await.expect("json parse");
        eprintln!("infer status={:?} body={}", status, body);
        assert!(status.is_success());
        assert_eq!(body["output"], "echo: hello");
        handle.abort();
    }
}
