# Rust × AI 生态全景（2025-09）

## 推理与训练

- `candle`：轻量高性能推理/训练，Hugging Face 维护
- `burn`：模块化深度学习框架，后端可插拔（WGPU/Accelerate）
- `tch-rs`：PyTorch C++ 后端绑定
- `onnxruntime`：跨平台推理
- `llama.cpp`/`ggml`：本地化量化推理绑定

## 科学计算与数据

- `ndarray`、`nalgebra`、`linfa`、`polars`
- `safetensors`、`tokenizers`

## 系统与工程

- Web/服务：`axum`、`actix-web`、`tower`
- 异步与并发：`tokio`、`rayon`
- 观测：`tracing`、`opentelemetry`

## RAG/检索

- 向量与全文：`qdrant`/`milvus` 客户端、`tantivy`
- 数据处理：`serde`、`reqwest`、`sqlx`

## 部署与集成

- `wasm`、`pyo3`、`maturin`
- GPU 加速：`cuda`/`wgpu` 生态

> 实操建议：先以 `candle + axum + qdrant` 搭建最小 RAG 服务，再迭代多代理与评测。
