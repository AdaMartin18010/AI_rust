# 📖 AI-Rust API 参考文档

## 概述

本文档提供AI-Rust项目的完整API参考，包括核心模块、接口定义、使用示例和最佳实践。

**版本**: 0.1.0
**更新日期**: 2025年12月3日
**适用Rust版本**: 1.90+

---

## 📚 目录

- [核心模块](#核心模块)
  - [推理引擎](#推理引擎)
  - [模型管理](#模型管理)
  - [数据处理](#数据处理)
- [AI功能模块](#ai功能模块)
  - [RAG系统](#rag系统)
  - [多模态处理](#多模态处理)
  - [Agent系统](#agent系统)
- [工具和辅助](#工具和辅助)
  - [监控和日志](#监控和日志)
  - [性能优化](#性能优化)
- [HTTP API](#http-api)

---

## 核心模块

### 推理引擎

#### `InferenceEngine`

通用推理引擎trait，支持同步和异步推理。

```rust
pub trait InferenceEngine<Input, Output> {
    /// 执行推理
    fn infer(&self, input: Input) -> Result<Output, InferenceError>;

    /// 批量推理
    fn batch_infer(&self, inputs: Vec<Input>) -> Result<Vec<Output>, InferenceError>;
}
```

##### 示例

```rust
use ai_rust::inference::InferenceEngine;

// 创建推理引擎
let engine = MyInferenceEngine::new(model_path)?;

// 单次推理
let input = create_input();
let output = engine.infer(input)?;

// 批量推理
let inputs = vec![input1, input2, input3];
let outputs = engine.batch_infer(inputs)?;
```

##### 实现指南

```rust
use ai_rust::inference::{InferenceEngine, InferenceError};

pub struct MyInferenceEngine {
    model: Model,
}

impl InferenceEngine<Tensor, Tensor> for MyInferenceEngine {
    fn infer(&self, input: Tensor) -> Result<Tensor, InferenceError> {
        // 预处理
        let processed = self.preprocess(input)?;

        // 推理
        let output = self.model.forward(processed)?;

        // 后处理
        Ok(self.postprocess(output)?)
    }

    fn batch_infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, InferenceError> {
        // 批量处理优化
        let batch = stack_tensors(inputs)?;
        let outputs = self.model.forward(batch)?;
        Ok(split_tensors(outputs)?)
    }
}
```

---

#### `AsyncInferenceEngine`

异步推理引擎trait，适用于高并发场景。

```rust
#[async_trait]
pub trait AsyncInferenceEngine<Input, Output> {
    /// 异步推理
    async fn infer(&self, input: Input) -> Result<Output, InferenceError>;

    /// 异步批量推理
    async fn batch_infer(&self, inputs: Vec<Input>) -> Result<Vec<Output>, InferenceError>;
}
```

##### 示例

```rust
use ai_rust::inference::AsyncInferenceEngine;

#[tokio::main]
async fn main() -> Result<()> {
    let engine = MyAsyncEngine::new(model_path).await?;

    // 并发推理
    let handles: Vec<_> = inputs.into_iter()
        .map(|input| {
            let engine = engine.clone();
            tokio::spawn(async move {
                engine.infer(input).await
            })
        })
        .collect();

    let results = futures::future::try_join_all(handles).await?;
    Ok(())
}
```

---

### 模型管理

#### `ModelLoader`

模型加载器，支持多种模型格式。

```rust
pub struct ModelLoader {
    cache_dir: PathBuf,
    config: LoaderConfig,
}

impl ModelLoader {
    /// 创建新的模型加载器
    pub fn new(cache_dir: PathBuf) -> Self;

    /// 从文件加载模型
    pub fn load_from_file(&self, path: &Path) -> Result<Model, LoadError>;

    /// 从HuggingFace加载模型
    pub async fn load_from_hub(&self, repo: &str) -> Result<Model, LoadError>;

    /// 加载量化模型
    pub fn load_quantized(&self, path: &Path, bits: u8) -> Result<Model, LoadError>;
}
```

##### 示例

```rust
use ai_rust::model::ModelLoader;

// 创建加载器
let loader = ModelLoader::new(PathBuf::from("./models"));

// 从本地加载
let model = loader.load_from_file(Path::new("model.safetensors"))?;

// 从HuggingFace加载
let model = loader.load_from_hub("sentence-transformers/all-MiniLM-L6-v2").await?;

// 加载量化模型
let quantized_model = loader.load_quantized(
    Path::new("model.safetensors"),
    8  // INT8量化
)?;
```

---

#### `ModelRegistry`

模型注册表，管理多个模型实例。

```rust
pub struct ModelRegistry {
    models: HashMap<String, Arc<dyn Model>>,
}

impl ModelRegistry {
    /// 创建新的注册表
    pub fn new() -> Self;

    /// 注册模型
    pub fn register(&mut self, name: String, model: Arc<dyn Model>);

    /// 获取模型
    pub fn get(&self, name: &str) -> Option<Arc<dyn Model>>;

    /// 列出所有模型
    pub fn list(&self) -> Vec<String>;

    /// 移除模型
    pub fn remove(&mut self, name: &str) -> Option<Arc<dyn Model>>;
}
```

##### 示例

```rust
use ai_rust::model::ModelRegistry;
use std::sync::Arc;

// 创建注册表
let mut registry = ModelRegistry::new();

// 注册模型
registry.register("embedding".to_string(), Arc::new(embedding_model));
registry.register("llm".to_string(), Arc::new(llm_model));

// 使用模型
if let Some(model) = registry.get("embedding") {
    let output = model.infer(input)?;
}

// 列出所有模型
let models = registry.list();
println!("Available models: {:?}", models);
```

---

### 数据处理

#### `Preprocessor`

数据预处理trait。

```rust
pub trait Preprocessor<Input, Output> {
    /// 预处理数据
    fn preprocess(&self, input: Input) -> Result<Output, PreprocessError>;

    /// 批量预处理
    fn preprocess_batch(&self, inputs: Vec<Input>) -> Result<Vec<Output>, PreprocessError>;
}
```

##### 示例

```rust
use ai_rust::preprocessing::Preprocessor;

pub struct TextPreprocessor {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl Preprocessor<String, Tensor> for TextPreprocessor {
    fn preprocess(&self, input: String) -> Result<Tensor, PreprocessError> {
        // Tokenization
        let tokens = self.tokenizer.encode(&input)?;

        // Padding/Truncation
        let padded = pad_or_truncate(tokens, self.max_length);

        // Convert to tensor
        Ok(Tensor::from_vec(padded))
    }

    fn preprocess_batch(&self, inputs: Vec<String>) -> Result<Vec<Tensor>, PreprocessError> {
        inputs.into_iter()
            .map(|input| self.preprocess(input))
            .collect()
    }
}
```

---

## AI功能模块

### RAG系统

#### `RAGSystem`

检索增强生成系统。

```rust
pub struct RAGSystem {
    retriever: Arc<dyn Retriever>,
    generator: Arc<dyn Generator>,
    config: RAGConfig,
}

impl RAGSystem {
    /// 创建新的RAG系统
    pub fn new(
        retriever: Arc<dyn Retriever>,
        generator: Arc<dyn Generator>,
        config: RAGConfig,
    ) -> Self;

    /// 查询RAG系统
    pub async fn query(&self, query: &str) -> Result<String, RAGError>;

    /// 添加文档
    pub async fn add_document(&mut self, doc: Document) -> Result<(), RAGError>;

    /// 搜索相关文档
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<Document>, RAGError>;
}
```

##### 完整示例

```rust
use ai_rust::rag::{RAGSystem, RAGConfig, VectorRetriever, LLMGenerator};

#[tokio::main]
async fn main() -> Result<()> {
    // 创建检索器
    let retriever = Arc::new(VectorRetriever::new(
        embedding_model,
        vector_store,
    ));

    // 创建生成器
    let generator = Arc::new(LLMGenerator::new(llm_model));

    // 创建RAG系统
    let config = RAGConfig {
        top_k: 5,
        min_relevance: 0.7,
        ..Default::default()
    };

    let mut rag = RAGSystem::new(retriever, generator, config);

    // 添加文档
    let docs = vec![
        Document::new("Rust is a systems programming language."),
        Document::new("AI models can be deployed in Rust."),
    ];

    for doc in docs {
        rag.add_document(doc).await?;
    }

    // 查询
    let answer = rag.query("What is Rust?").await?;
    println!("Answer: {}", answer);

    Ok(())
}
```

---

#### `VectorStore`

向量存储接口。

```rust
#[async_trait]
pub trait VectorStore {
    /// 添加向量
    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<(), StoreError>;

    /// 批量添加
    async fn add_batch(&mut self, items: Vec<(String, Vec<f32>)>) -> Result<(), StoreError>;

    /// 搜索最相似的向量
    async fn search(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>, StoreError>;

    /// 删除向量
    async fn remove(&mut self, id: &str) -> Result<(), StoreError>;
}
```

##### 内存向量存储实现

```rust
use ai_rust::rag::VectorStore;

pub struct InMemoryVectorStore {
    vectors: HashMap<String, Vec<f32>>,
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<(), StoreError> {
        self.vectors.insert(id, vector);
        Ok(())
    }

    async fn search(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>, StoreError> {
        let mut results: Vec<_> = self.vectors
            .iter()
            .map(|(id, vec)| {
                let score = cosine_similarity(&query, vec);
                SearchResult {
                    id: id.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);

        Ok(results)
    }

    async fn remove(&mut self, id: &str) -> Result<(), StoreError> {
        self.vectors.remove(id);
        Ok(())
    }

    async fn add_batch(&mut self, items: Vec<(String, Vec<f32>)>) -> Result<(), StoreError> {
        for (id, vector) in items {
            self.add(id, vector).await?;
        }
        Ok(())
    }
}
```

---

### 多模态处理

#### `MultimodalProcessor`

多模态数据处理器。

```rust
pub struct MultimodalProcessor {
    text_processor: Arc<dyn Processor<String>>,
    image_processor: Arc<dyn Processor<Image>>,
    audio_processor: Arc<dyn Processor<Audio>>,
}

impl MultimodalProcessor {
    /// 创建多模态处理器
    pub fn new(
        text_processor: Arc<dyn Processor<String>>,
        image_processor: Arc<dyn Processor<Image>>,
        audio_processor: Arc<dyn Processor<Audio>>,
    ) -> Self;

    /// 处理文本
    pub fn process_text(&self, text: String) -> Result<Embedding, ProcessError>;

    /// 处理图像
    pub fn process_image(&self, image: Image) -> Result<Embedding, ProcessError>;

    /// 处理音频
    pub fn process_audio(&self, audio: Audio) -> Result<Embedding, ProcessError>;

    /// 融合多模态特征
    pub fn fuse(&self, embeddings: Vec<Embedding>) -> Result<Embedding, ProcessError>;
}
```

##### 示例

```rust
use ai_rust::multimodal::{MultimodalProcessor, Embedding};

// 创建处理器
let processor = MultimodalProcessor::new(
    text_processor,
    image_processor,
    audio_processor,
);

// 处理不同模态的数据
let text_emb = processor.process_text("A cat sitting on a mat".to_string())?;
let image_emb = processor.process_image(cat_image)?;

// 融合特征
let fused = processor.fuse(vec![text_emb, image_emb])?;
```

---

### Agent系统

#### `Agent`

智能Agent基础trait。

```rust
#[async_trait]
pub trait Agent {
    /// 处理用户消息
    async fn process(&mut self, message: &str) -> Result<String, AgentError>;

    /// 重置Agent状态
    fn reset(&mut self);

    /// 获取Agent状态
    fn state(&self) -> &AgentState;
}
```

##### 实现示例

```rust
use ai_rust::agent::{Agent, AgentState, Tool};

pub struct MyAgent {
    llm: Arc<dyn LLM>,
    tools: Vec<Box<dyn Tool>>,
    state: AgentState,
}

#[async_trait]
impl Agent for MyAgent {
    async fn process(&mut self, message: &str) -> Result<String, AgentError> {
        // 1. 理解用户意图
        let intent = self.llm.analyze_intent(message).await?;

        // 2. 选择合适的工具
        let tool = self.select_tool(&intent)?;

        // 3. 执行工具
        let result = tool.execute(&intent.parameters).await?;

        // 4. 生成响应
        let response = self.llm.generate_response(&result).await?;

        // 5. 更新状态
        self.state.update(message, &response);

        Ok(response)
    }

    fn reset(&mut self) {
        self.state = AgentState::default();
    }

    fn state(&self) -> &AgentState {
        &self.state
    }
}
```

---

#### `Tool`

Agent工具trait。

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    /// 工具名称
    fn name(&self) -> &str;

    /// 工具描述
    fn description(&self) -> &str;

    /// 执行工具
    async fn execute(&self, params: &str) -> Result<String, ToolError>;
}
```

##### 工具实现示例

```rust
use ai_rust::agent::Tool;

pub struct WebSearchTool {
    client: reqwest::Client,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for information"
    }

    async fn execute(&self, params: &str) -> Result<String, ToolError> {
        // 解析参数
        let query: SearchQuery = serde_json::from_str(params)?;

        // 执行搜索
        let results = self.search(&query.query).await?;

        // 格式化结果
        Ok(format_search_results(results))
    }
}
```

---

## 工具和辅助

### 监控和日志

#### `MetricsCollector`

性能指标收集器。

```rust
use prometheus::{Counter, Histogram, Registry};

pub struct MetricsCollector {
    registry: Registry,
    inference_counter: Counter,
    inference_duration: Histogram,
    error_counter: Counter,
}

impl MetricsCollector {
    /// 创建新的指标收集器
    pub fn new() -> Self;

    /// 记录推理
    pub fn record_inference(&self, duration: Duration, success: bool);

    /// 记录错误
    pub fn record_error(&self, error_type: &str);

    /// 导出指标
    pub fn export(&self) -> String;
}
```

##### 使用示例

```rust
use ai_rust::monitoring::MetricsCollector;

// 创建收集器
let metrics = MetricsCollector::new();

// 记录推理
let start = Instant::now();
let result = model.infer(input);
let duration = start.elapsed();

metrics.record_inference(duration, result.is_ok());

// 导出指标
let metrics_text = metrics.export();
```

---

### 性能优化

#### `CacheManager`

缓存管理器。

```rust
pub struct CacheManager<K, V> {
    cache: Arc<Mutex<LruCache<K, V>>>,
    max_size: usize,
}

impl<K, V> CacheManager<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// 创建缓存管理器
    pub fn new(max_size: usize) -> Self;

    /// 获取缓存值
    pub fn get(&self, key: &K) -> Option<V>;

    /// 设置缓存值
    pub fn put(&self, key: K, value: V);

    /// 清除缓存
    pub fn clear(&self);

    /// 获取缓存统计
    pub fn stats(&self) -> CacheStats;
}
```

---

## HTTP API

### 推理端点

#### POST `/api/v1/infer`

执行模型推理。

**请求体**:

```json
{
  "model": "embedding",
  "input": "Your input text here",
  "options": {
    "temperature": 0.7,
    "max_length": 512
  }
}
```

**响应**:

```json
{
  "output": [...],
  "metadata": {
    "duration_ms": 45,
    "model_version": "1.0.0"
  }
}
```

**Rust客户端示例**:

```rust
use reqwest::Client;
use serde_json::json;

async fn infer(client: &Client, input: &str) -> Result<Vec<f32>> {
    let response = client
        .post("http://localhost:8080/api/v1/infer")
        .json(&json!({
            "model": "embedding",
            "input": input
        }))
        .send()
        .await?;

    let result: InferenceResponse = response.json().await?;
    Ok(result.output)
}
```

---

### RAG端点

#### POST `/api/v1/rag/query`

查询RAG系统。

**请求体**:

```json
{
  "query": "What is Rust?",
  "top_k": 5,
  "min_relevance": 0.7
}
```

**响应**:

```json
{
  "answer": "Rust is a systems programming language...",
  "sources": [
    {
      "id": "doc_1",
      "relevance": 0.95,
      "text": "..."
    }
  ],
  "metadata": {
    "duration_ms": 150,
    "num_docs_retrieved": 5
  }
}
```

---

### 健康检查

#### GET `/health`

检查服务健康状态。

**响应**:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600
}
```

---

## 错误处理

### 错误类型

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AIError {
    #[error("推理错误: {0}")]
    InferenceError(String),

    #[error("模型加载错误: {0}")]
    ModelLoadError(String),

    #[error("预处理错误: {0}")]
    PreprocessError(String),

    #[error("IO错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("序列化错误: {0}")]
    SerdeError(#[from] serde_json::Error),
}
```

### 错误处理最佳实践

```rust
use ai_rust::{AIError, Result};

pub async fn safe_inference(
    model: &Model,
    input: Input,
) -> Result<Output> {
    // 输入验证
    validate_input(&input)?;

    // 推理执行
    let output = model.infer(input).await
        .map_err(|e| AIError::InferenceError(e.to_string()))?;

    // 输出验证
    validate_output(&output)?;

    Ok(output)
}
```

---

## 配置

### 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `RUST_LOG` | 日志级别 | `info` |
| `MODEL_CACHE_DIR` | 模型缓存目录 | `./models` |
| `MAX_BATCH_SIZE` | 最大批量大小 | `32` |
| `SERVER_PORT` | 服务器端口 | `8080` |

### 配置文件

```toml
# config.toml

[server]
host = "0.0.0.0"
port = 8080
workers = 4

[model]
cache_dir = "./models"
default_model = "embedding"

[inference]
max_batch_size = 32
timeout_seconds = 30

[logging]
level = "info"
format = "json"
```

---

## 更多信息

- [快速开始指南](QUICK_START.md)
- [最佳实践](docs/05_practical_guides/rust_ai_best_practices.md)
- [示例代码](examples/)
- [贡献指南](CONTRIBUTING.md)

---

*最后更新: 2025年12月3日*
*维护者: AI-Rust项目团队*
