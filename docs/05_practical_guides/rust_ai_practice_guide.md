# Rust AI 实践指南（精简版）

## 目录

- [Rust AI 实践指南（精简版）](#rust-ai-实践指南精简版)
  - [目录](#目录)
  - [1. Rust AI 生态概览](#1-rust-ai-生态概览)
    - [1.1 推理与训练](#11-推理与训练)
    - [1.2 科学计算](#12-科学计算)
    - [1.3 系统与工程](#13-系统与工程)
  - [2. 核心库选择](#2-核心库选择)
    - [2.1 深度学习](#21-深度学习)
    - [2.2 数据处理](#22-数据处理)
    - [2.3 Web服务](#23-web服务)
  - [3. 快速实践项目](#3-快速实践项目)
    - [3.1 线性回归实现](#31-线性回归实现)
    - [3.2 简单神经网络](#32-简单神经网络)
    - [3.3 API服务](#33-api服务)
  - [4. 性能优化技巧](#4-性能优化技巧)
    - [4.1 并行计算](#41-并行计算)
    - [4.2 SIMD优化](#42-simd优化)
    - [4.3 内存优化](#43-内存优化)
  - [5. 部署与集成](#5-部署与集成)
    - [5.1 Docker部署](#51-docker部署)
    - [5.2 Kubernetes部署](#52-kubernetes部署)
    - [5.2 Python集成](#52-python集成)
    - [5.3 WebAssembly](#53-webassembly)
  - [总结](#总结)
  - [附录：最小闭环与验收标准](#附录最小闭环与验收标准)
    - [最小闭环（RAG服务）](#最小闭环rag服务)
    - [按 crate 练习与验收标准](#按-crate-练习与验收标准)

## 1. Rust AI 生态概览

### 1.1 推理与训练

- **candle**：轻量高性能推理/训练，Hugging Face 维护
- **burn**：模块化深度学习框架，后端可插拔
- **tch-rs**：PyTorch C++ 后端绑定
- **onnxruntime**：跨平台推理

### 1.2 科学计算

- **ndarray**：多维数组计算
- **nalgebra**：线性代数库
- **linfa**：机器学习工具包
- **polars**：高性能数据处理

### 1.3 系统与工程

- **axum**：现代Web框架
- **tokio**：异步运行时
- **tracing**：结构化日志
- **serde**：序列化框架

## 2. 核心库选择

### 2.1 深度学习

```rust
// 推荐组合
use candle_core::{Device, Tensor, Result};
use candle_nn::{linear, Linear, VarBuilder, Module};
use candle_optimisers::{AdamW, ParamsAdamW};
```

### 2.2 数据处理

```rust
use ndarray::{Array2, Array3, Axis};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
```

### 2.3 Web服务

```rust
use axum::{routing::post, Router, Json};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
```

## 3. 快速实践项目

### 3.1 线性回归实现

```rust
use ndarray::{Array2, Array1};
use linfa::prelude::*;
use linfa_linear::LinearRegression;

pub fn train_linear_regression() -> Result<(), Box<dyn std::error::Error>> {
    // 生成示例数据
    let x = Array2::random((100, 2), Uniform::new(0.0, 1.0));
    let y = Array1::random(100, Uniform::new(0.0, 1.0));
    
    // 训练模型
    let dataset = Dataset::new(x, y);
    let model = LinearRegression::default().fit(&dataset)?;
    
    // 预测
    let predictions = model.predict(&dataset);
    println!("Predictions: {:?}", predictions);
    
    Ok(())
}
```

### 3.2 简单神经网络

```rust
use candle_core::{Device, Tensor, Result};
use candle_nn::{linear, Linear, VarBuilder, Module};

pub struct MLP {
    layers: Vec<Linear>,
}

impl MLP {
    pub fn new(vs: VarBuilder, input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Result<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;
        
        for &hidden_dim in &hidden_dims {
            layers.push(linear(prev_dim, hidden_dim, vs.pp("layer"))?);
            prev_dim = hidden_dim;
        }
        layers.push(linear(prev_dim, output_dim, vs.pp("output"))?);
        
        Ok(Self { layers })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            if i < self.layers.len() - 1 {
                xs = candle_nn::ops::relu(&xs)?;
            }
        }
        Ok(xs)
    }
}
```

### 3.3 API服务

```rust
use axum::{routing::post, Router, Json};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

#[derive(Deserialize)]
struct PredictRequest {
    input: Vec<f32>,
}

#[derive(Serialize)]
struct PredictResponse {
    prediction: f32,
    confidence: f32,
}

async fn predict(Json(payload): Json<PredictRequest>) -> Json<PredictResponse> {
    // 模型推理逻辑
    let prediction = payload.input.iter().sum::<f32>() / payload.input.len() as f32;
    
    Json(PredictResponse {
        prediction,
        confidence: 0.95,
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/predict", post(predict));
    
    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

## 4. 性能优化技巧

### 4.1 并行计算

```rust
use rayon::prelude::*;

// 并行矩阵运算
pub fn parallel_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| x * y)
        .sum()
}
```

### 4.2 SIMD优化

```rust
use std::simd::*;

pub fn simd_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let chunks = a.chunks_exact(4);
    let b_chunks = b.chunks_exact(4);
    
    let mut result = Vec::new();
    
    for (a_chunk, b_chunk) in chunks.zip(b_chunks) {
        let a_simd = f32x4::from_slice(a_chunk);
        let b_simd = f32x4::from_slice(b_chunk);
        let sum = a_simd + b_simd;
        result.extend_from_slice(&sum.to_array());
    }
    
    result
}
```

### 4.3 内存优化

```rust
use std::alloc::{GlobalAlloc, Layout, System};

// 自定义内存分配器
pub struct OptimizedAllocator;

unsafe impl GlobalAlloc for OptimizedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}
```

## 5. 部署与集成

### 5.1 Docker部署

```dockerfile
# 多阶段构建优化
FROM rust:1.75 as builder
WORKDIR /app

# 缓存依赖层
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# 构建应用
COPY src ./src
COPY Cargo.toml ./
RUN touch src/main.rs && cargo build --release

# 运行时镜像
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/ai-service /usr/local/bin/
EXPOSE 3000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["ai-service"]
```

### 5.2 Kubernetes部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-service
  template:
    metadata:
      labels:
        app: ai-service
    spec:
      containers:
      - name: ai-service
        image: ai-service:latest
        ports:
        - containerPort: 3000
        env:
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-service
spec:
  selector:
    app: ai-service
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

### 5.2 Python集成

```rust
use pyo3::prelude::*;

#[pyfunction]
fn rust_ml_function(input: Vec<f32>) -> PyResult<f32> {
    // Rust ML 计算
    Ok(input.iter().sum::<f32>() / input.len() as f32)
}

#[pymodule]
fn rust_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_ml_function, m)?)?;
    Ok(())
}
```

### 5.3 WebAssembly

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn wasm_ml_predict(input: &[f32]) -> f32 {
    // WASM ML 推理
    input.iter().sum::<f32>() / input.len() as f32
}
```

## 总结

这个精简版Rust AI实践指南提供了：

1. **核心生态概览**：重点介绍AI相关的Rust库
2. **快速上手**：提供实用的代码示例
3. **性能优化**：展示Rust的性能优势
4. **部署集成**：支持多种部署方式

重点突出Rust在AI系统中的工程价值，而非语言特性本身。适合有AI基础、需要高性能系统的开发者快速上手。

## 附录：最小闭环与验收标准

### 最小闭环（RAG服务）

目标：完成 `candle + axum + qdrant` 的最小RAG服务，具备检索、生成、评测与观测。

- 必做：
  - `/embed`：文本向量化接口（批处理/限流/错误处理）
  - `/search`：混合检索（`tantivy` + 向量DB），缓存与重排序
  - `/generate`：带 KV 缓存的生成接口（Top-k/Top-p），支持流式
  - 观测：`tracing` 指标 + 结构化日志 + 请求链路追踪
  - 评测：召回/精排/生成质量 + 延迟/QPS 基准

交付物：Docker 镜像、基准报告、演示视频（或 GIF）。

### 按 crate 练习与验收标准

- `crates/c01_base`
  - 练习：Rust 基础 + 并发入门（`tokio` + `rayon`）
  - 验收：并发安全的矩阵乘法（CPU 并行版）+ 单测覆盖 ≥ 90%
- `crates/c02_data`
  - 练习：`ndarray/nalgebra/polars` 数据处理
  - 验收：实现 PCA（从零/库）并比较数值误差与性能
- `crates/c03_ml_basics`
  - 练习：线性/逻辑回归、决策树（从零/库）
  - 验收：在公开数据集上达到指定指标，提供基准与误差分析
- `crates/c04_dl_fundamentals`
  - 练习：`candle` 实现 MLP/CNN；训练循环与优化器使用
  - 验收：分类任务达到基准精度；GPU/CPU 对比与Profiling
- `crates/c05_nlp_transformers`
  - 练习：分词、位置编码、注意力推理；LoRA 微调脚本
  - 验收：指令微调小模型的可复现结果与推理加速（INT8）
- `crates/c06_retrieval_tools`
  - 练习：`tantivy` + 向量DB（Qdrant/Milvus）
  - 验收：混合检索 + 重排序，在评测集中达到既定指标
- `crates/c07_agents_systems`
  - 练习：多代理框架（规划/执行/回顾），消息通信
  - 验收：端到端任务成功率与成本统计，附可视化日志
- `crates/c08_serving_ops`
  - 练习：`axum` 服务、限流/重试、批处理、健康检查
  - 验收：通过 `tests/http_smoke.rs`，并新增性能与稳定性基准
