use anyhow::Result;
use std::future::Future;
use std::pin::Pin;

pub trait InferenceEngine: Send + Sync {
    fn infer<'a>(&'a self, prompt: &'a str) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

pub struct DummyEngine;

impl InferenceEngine for DummyEngine {
    fn infer<'a>(&'a self, prompt: &'a str) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move { Ok(format!("echo: {}", prompt)) })
    }
}