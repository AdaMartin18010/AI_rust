use async_trait::async_trait;
use anyhow::Result;

#[async_trait]
pub trait InferenceEngine: Send + Sync {
    async fn infer(&self, prompt: &str) -> Result<String>;
}

pub struct DummyEngine;

#[async_trait]
impl InferenceEngine for DummyEngine {
    async fn infer(&self, prompt: &str) -> Result<String> {
        Ok(format!("echo: {}", prompt))
    }
}
