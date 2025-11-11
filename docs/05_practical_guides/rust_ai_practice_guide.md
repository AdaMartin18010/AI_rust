# Rust AI 实践指南（精简版）

## 目录

- [Rust AI 实践指南（精简版）](#rust-ai-实践指南精简版)
  - [目录](#目录)
  - [1. Rust AI 生态概览](#1-rust-ai-生态概览)
  - [0. 概念与口径对齐（DAR/Metrics/Mapping）](#0-概念与口径对齐darmetricsmapping)
    - [0.1 DAR 最小卡片](#01-dar-最小卡片)
    - [0.2 指标与采样口径](#02-指标与采样口径)
    - [0.3 从实践到实现的映射](#03-从实践到实现的映射)
    - [0.4 案例桥接（最小证据包）](#04-案例桥接最小证据包)
    - [0.5 交叉引用](#05-交叉引用)
    - [1.1 推理与训练](#11-推理与训练)
    - [2025年11月更新：推理优化优先](#2025年11月更新推理优化优先)
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

Rust在AI领域的生态系统正在快速发展，2025年已经形成了完整的工具链和框架体系。

**核心优势**：

- **性能**：接近C/C++的性能，零成本抽象
- **安全**：内存安全保证，减少运行时错误
- **并发**：优秀的并发模型，适合高并发AI服务
- **生态**：丰富的AI库和工具，持续增长

**2025年最新发展**：

- **推理优化**：从"重训练"转向"重推理"，量化、缓存、路由成为重点
- **边缘部署**：WebAssembly AI推理成熟，边缘部署率提升
- **生态成熟**：CCF开源大会、蓝河OS等产业应用验证Rust成熟度

**参考**：详见 `03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md` §"2025年11月最新趋势更新"

## 0. 概念与口径对齐（DAR/Metrics/Mapping）

- 与《2025 Rust AI 实战指南》§0、《综合知识框架》附录Y、《技术趋势》附录Z一致：统一术语、属性口径与映射。

### 0.1 DAR 最小卡片

- 模板：Definition｜Attributes（单位/口径）｜Relations（类型+强度）｜Evidence（等级/来源）。
- 示例：KV缓存｜Impl｜P95、命中率、成本/请求｜Optimizes(解码延迟)｜A。

### 0.2 指标与采样口径

- 分位统计：P50/P95/P99 指定窗口与算法（t-digest）。
- 吞吐：稳态/峰值 QPS，注明批量与并发。
- 能效：tokens/J 排除冷启动；
- 经济：$/1k tok 含检索分摊，TCO 口径明确。

### 0.3 从实践到实现的映射

- 指标→架构：缓存/并发/背压/熔断/埋点；
- 架构→代码：`axum/tokio`、`tracing`、`candle/onnxruntime`；
- 代码→运维：金丝雀/回滚、基线、预算护栏。

### 0.4 案例桥接（最小证据包）

- 案例A：INT4 量化+小模型回退路由的成本-延迟 Pareto；
  - 指标：P95/P99、QPS、tokens/J、$/1k tok；
  - 证据：对照/消融；脚本与 trace 列表。
- 案例B：混合检索+部分重排提升引用率并控制端到端成本；
  - 指标：recall、NDCG、citation_rate、成本/查询；
  - 证据：K/K' 消融；索引参数与版本固定。

### 0.5 交叉引用

- 实战：`docs/05_practical_guides/2025_rust_ai_practical_guide.md` §0；
- 知识：`docs/02_knowledge_structures/2025_ai_rust_comprehensive_knowledge_framework.md` 附录Y；
- 趋势：`docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md` 附录Z。

### 1.1 推理与训练

- **candle**：轻量高性能推理/训练，Hugging Face 维护
  - 特点：易用、轻量、支持多模态
  - 2025年更新：多模态支持增强，WebAssembly集成，INT8/INT4量化支持
  - 适用：快速原型、推理服务、边缘部署
  - 性能：推理延迟低，内存占用小

- **burn**：模块化深度学习框架，后端可插拔
  - 特点：类型安全、模块化、多后端
  - 2025年更新：分布式训练支持，性能优化显著
  - 适用：研究、自定义架构、分布式训练
  - 优势：编译时类型检查，减少运行时错误

- **tch-rs**：PyTorch C++ 后端绑定
  - 特点：PyTorch兼容、功能完整
  - 2025年更新：性能优化显著，支持最新PyTorch特性
  - 适用：PyTorch模型迁移、训练+推理
  - 优势：可以直接使用PyTorch预训练模型

- **onnxruntime**：跨平台推理
  - 特点：跨平台、硬件加速、模型格式统一
  - 2025年更新：新硬件支持（NPU等），性能优化
  - 适用：生产环境、多平台部署、硬件加速
  - 优势：模型格式标准化，硬件加速支持好

### 2025年11月更新：推理优化优先

根据2025年11月趋势，推理框架重点：

- 量化支持：INT8/INT4量化成为标配
- 边缘部署：WebAssembly推理成熟
- 成本优化：推理成本降低30-50%

### 1.2 科学计算

- **ndarray**：多维数组计算
  - 特点：高性能、SIMD优化、多维数组操作
  - 2025年更新：SIMD优化增强，性能提升20-30%
  - 适用：数值计算、矩阵运算、科学计算
  - 性能：接近NumPy性能，内存效率更高

- **nalgebra**：线性代数库
  - 特点：类型安全、几何计算、优化算法
  - 适用：3D图形、物理模拟、优化问题
  - 优势：编译时维度检查，减少运行时错误

- **linfa**：机器学习工具包
  - 特点：Rust的scikit-learn等价物
  - 功能：分类、回归、聚类、降维
  - 适用：传统机器学习任务
  - 优势：类型安全、性能优异

- **polars**：高性能数据处理
  - 特点：列式存储、懒计算、多线程
  - 2025年更新：性能接近Apache Spark
  - 适用：大数据处理、数据分析、ETL
  - 性能：比Pandas快10-100倍

### 1.3 系统与工程

- **axum**：现代Web框架
  - 特点：异步、类型安全、中间件丰富
  - 2025年更新：异步性能提升，生态完善
  - 适用：Web服务、API服务、微服务
  - 性能：QPS高，延迟低，资源占用少

- **tokio**：异步运行时
  - 特点：高性能、生态丰富、生产就绪
  - 适用：异步IO、并发处理、网络服务
  - 性能：并发性能优异，资源利用率高

- **tracing**：结构化日志
  - 特点：结构化、可观测、性能优异
  - 2025年更新：结构化日志增强，OpenTelemetry集成
  - 适用：日志记录、性能追踪、调试
  - 优势：零成本抽象，生产环境性能好

- **serde**：序列化框架
  - 特点：零成本、类型安全、格式丰富
  - 适用：数据序列化、API通信、配置管理
  - 性能：序列化/反序列化速度快，内存效率高

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

**原理**：利用多核CPU并行处理数据，提升计算性能。

**实现要点**：

- 使用`rayon`进行数据并行处理
- 自动任务窃取，负载均衡
- 零成本抽象，性能接近手写并行代码

**性能提升**：

- 矩阵运算：4核CPU提升3-4倍
- 数据处理：大规模数据提升5-10倍
- 适用场景：CPU密集型任务、大规模数据处理

**注意事项**：

- 避免在并行循环中使用共享可变状态
- 小数据集可能因线程开销反而变慢
- 使用`par_bridge()`适配迭代器

```rust
use rayon::prelude::*;

// 并行矩阵运算
pub fn parallel_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| x * y)
        .sum()
}

// 并行批量处理
pub fn parallel_batch_process(data: &[Vec<f32>]) -> Vec<f32> {
    data.par_iter()
        .map(|vec| vec.iter().sum())
        .collect()
}
```

### 4.2 SIMD优化

**原理**：使用单指令多数据（SIMD）指令，一次处理多个数据元素。

**实现要点**：

- Rust 1.90+ 内置SIMD支持（`std::simd`）
- 自动向量化优化
- 手动SIMD用于关键路径

**性能提升**：

- 向量运算：提升4-8倍（取决于SIMD宽度）
- 矩阵运算：提升2-4倍
- 适用场景：数值计算、图像处理、信号处理

**注意事项**：

- 数据对齐要求（通常16字节对齐）
- 需要处理剩余元素（非SIMD宽度倍数）
- 不同CPU的SIMD指令集不同

```rust
use std::simd::*;

pub fn simd_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let chunks = a.chunks_exact(4);
    let b_chunks = b.chunks_exact(4);
    
    let mut result = Vec::new();
    
    // SIMD处理对齐部分
    for (a_chunk, b_chunk) in chunks.zip(b_chunks) {
        let a_simd = f32x4::from_slice(a_chunk);
        let b_simd = f32x4::from_slice(b_chunk);
        let sum = a_simd + b_simd;
        result.extend_from_slice(&sum.to_array());
    }
    
    // 处理剩余元素
    let remainder = a.len() % 4;
    if remainder > 0 {
        for i in (a.len() - remainder)..a.len() {
            result.push(a[i] + b[i]);
        }
    }
    
    result
}
```

### 4.3 内存优化

**原理**：减少内存分配、复用内存缓冲区、使用零拷贝技术。

**实现要点**：

- 使用内存池预分配内存
- 复用缓冲区，避免频繁分配
- 使用`bytes`和`zerocopy`减少内存复制
- 使用`Box`和`Arc`优化大对象管理

**性能提升**：

- 内存分配：减少90%+的分配次数
- 延迟降低：减少15-30%的内存相关延迟
- 适用场景：高并发服务、大规模数据处理

**注意事项**：

- 内存池大小需要根据实际负载调整
- 避免内存泄漏（使用`Arc`时注意循环引用）
- 监控内存使用，避免OOM

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use bytes::{Bytes, BytesMut};

// 内存池示例
pub struct MemoryPool {
    buffers: Vec<BytesMut>,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            buffers: Vec::with_capacity(100),
        }
    }
    
    pub fn get_buffer(&mut self, size: usize) -> BytesMut {
        self.buffers.pop()
            .filter(|b| b.capacity() >= size)
            .unwrap_or_else(|| BytesMut::with_capacity(size))
    }
    
    pub fn return_buffer(&mut self, mut buffer: BytesMut) {
        buffer.clear();
        if self.buffers.len() < 100 {
            self.buffers.push(buffer);
        }
    }
}

// 零拷贝示例
pub fn zero_copy_process(data: &[u8]) -> Bytes {
    // 直接返回Bytes，不复制数据
    Bytes::copy_from_slice(data)
}

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

**原理**：使用Docker容器化部署，实现环境一致性和快速部署。

**实现要点**：

- 多阶段构建：减少最终镜像大小
- 依赖缓存：优化构建速度
- 最小化镜像：使用Alpine或distroless基础镜像
- 安全配置：非root用户运行、最小权限

**性能优化**：

- 镜像大小：多阶段构建可减少80%+镜像大小
- 构建速度：依赖缓存可提升50-70%构建速度
- 启动时间：优化后容器启动时间<5s

**最佳实践**：

- 使用`.dockerignore`排除不必要文件
- 设置健康检查（`HEALTHCHECK`）
- 配置资源限制（CPU、内存）
- 使用Docker Compose管理多容器应用

```dockerfile
# 多阶段构建优化
FROM rust:1.90 as builder
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
WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# 复制二进制文件
COPY --from=builder /app/target/release/myapp /app/myapp

# 非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000
CMD ["/app/myapp"]
```

### 5.2 Kubernetes部署

**原理**：使用Kubernetes进行容器编排，实现自动扩缩容、负载均衡、服务发现。

**实现要点**：

- Deployment：管理Pod副本
- Service：服务发现和负载均衡
- ConfigMap/Secret：配置和密钥管理
- HPA：水平自动扩缩容
- 资源限制：CPU、内存配额

**性能优化**：

- 自动扩缩容：根据CPU/内存/QPS自动调整副本数
- 负载均衡：Service自动分发请求
- 资源利用率：合理设置requests和limits

**最佳实践**：

- 设置合理的资源requests和limits
- 配置健康检查和就绪探针
- 使用ConfigMap管理配置
- 实现优雅关闭（graceful shutdown）

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
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 5.2 Python集成

**原理**：使用PyO3将Rust代码编译为Python扩展模块，在Python中调用高性能Rust代码。

**实现要点**：

- 使用`pyo3`创建Python绑定
- 类型转换：Rust类型 ↔ Python类型
- 异步支持：Rust异步函数暴露给Python
- 错误处理：Rust错误转换为Python异常

**性能优势**：

- 计算密集型任务：比纯Python快10-100倍
- 内存效率：Rust的内存管理更高效
- 并发性能：Rust的并发模型更安全高效

**最佳实践**：

- 使用`#[pyclass]`和`#[pymethods]`定义Python类
- 使用`#[pyfunction]`定义Python函数
- 使用`maturin`构建和发布Python包
- 提供Python风格的文档字符串

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

**原理**：将Rust代码编译为WebAssembly，在浏览器或Node.js中运行高性能AI推理。

**实现要点**：

- 使用`wasm-bindgen`创建JavaScript绑定
- 使用`wasm-pack`构建和发布WASM包
- 模型量化：INT8/INT4量化减少模型大小
- 内存管理：使用`SharedArrayBuffer`共享内存

**性能优势**：

- 接近原生性能：执行速度是JavaScript的80-90%
- 边缘推理：在浏览器中直接运行，低延迟
- 隐私保护：数据不离开客户端

**最佳实践**：

- 使用`wasm-pack`构建WASM包
- 优化WASM文件大小（使用`wasm-opt`）
- 实现流式推理，提升用户体验
- 处理异步操作（使用`wasm-bindgen-futures`）

**2025年11月更新：推理优化优先**:

根据2025年11月趋势，WASM AI推理重点：

- 边缘部署：边缘部署率从10%提升至40%
- 成本优化：边缘推理成本比云端降低60%
- 实时性能：P95延迟从200ms降至50ms

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmMLModel {
    weights: Vec<f32>,
}

#[wasm_bindgen]
impl WasmMLModel {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Vec<f32>) -> WasmMLModel {
        WasmMLModel { weights }
    }
    
    #[wasm_bindgen]
    pub fn predict(&self, input: &[f32]) -> f32 {
        // WASM ML 推理
        input.iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f32>()
    }
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
