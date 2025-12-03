# 📖 AI-Rust 用户手册

## 欢迎使用 AI-Rust！

本手册将引导您从零开始，快速掌握AI-Rust项目的使用方法，包括环境配置、基本操作、高级功能和常见问题解答。

**版本**: 0.1.0
**更新日期**: 2025年12月3日
**适合人群**: 初学者到高级用户

---

## 📚 目录

1. [新手入门](#新手入门)
2. [基础操作](#基础操作)
3. [核心功能](#核心功能)
4. [高级应用](#高级应用)
5. [常见问题](#常见问题)
6. [故障排查](#故障排查)

---

## 🚀 新手入门

### 第一步：环境准备

#### 安装Rust

```bash
# 安装Rust (1.90+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 验证安装
rustc --version  # 应显示 1.90.0 或更高版本

# 更新Rust
rustup update
```

#### 安装系统依赖

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev git
```

**macOS**:
```bash
brew install pkg-config openssl git
```

**Windows**:
- 下载并安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- 安装 [Git for Windows](https://git-scm.com/download/win)

---

### 第二步：获取项目

```bash
# 克隆项目
git clone https://github.com/your-org/ai-rust.git
cd ai-rust

# 查看项目结构
ls -la
```

**项目结构说明**:
```
ai-rust/
├── src/          # 核心源代码
├── crates/       # 子项目模块
├── examples/     # 示例代码
├── docs/         # 文档
├── tests/        # 测试代码
└── Cargo.toml    # 项目配置
```

---

### 第三步：构建项目

```bash
# 开发构建 (快速，用于调试)
cargo build

# 发布构建 (优化，用于生产)
cargo build --release

# 运行测试
cargo test

# 生成文档
cargo doc --open
```

**构建时间参考**:
- 开发构建: 约5分钟 (首次)
- 发布构建: 约10分钟 (首次)
- 增量构建: 约30秒

---

### 第四步：运行第一个示例

```bash
# 运行GAT特性展示
cargo run --example gat_ai_inference

# 预期输出：
# Rust 1.90 GAT特性展示:
# ✅ 线性模型推理: 15.0
# ✅ 神经网络推理: [0.88]
# ✅ 多模态融合: 2.45
```

**🎉 恭喜！您已经成功运行了第一个AI示例！**

---

## 📖 基础操作

### 运行示例代码

AI-Rust提供了丰富的示例代码，涵盖各种应用场景。

#### 1. RAG系统示例

```bash
# 运行RAG系统
cargo run --example rag_system

# 功能说明：
# - 文档嵌入和存储
# - 语义检索
# - 增强生成
```

**示例输出**:
```
RAG系统演示:
1. 添加文档...
   ✅ 已添加 3 个文档
2. 执行检索...
   ✅ 找到 2 个相关文档
3. 生成答案...
   ✅ 答案: Rust是一种系统编程语言...
```

---

#### 2. 多模态处理示例

```bash
# 运行多模态处理
cargo run --example multimodal_processing

# 功能说明：
# - 文本处理
# - 图像处理
# - 特征融合
```

---

#### 3. Agent系统示例

```bash
# 运行Agent系统
cargo run --example agent_system_framework

# 功能说明：
# - 意图理解
# - 工具调用
# - 任务执行
```

---

### 使用Web API

#### 启动服务

```bash
# 启动Web服务
cargo run --release --bin ai_service

# 服务将在 http://localhost:8080 启动
```

#### 基本API调用

**健康检查**:
```bash
curl http://localhost:8080/health

# 响应：
# {"status":"healthy","version":"0.1.0"}
```

**推理请求**:
```bash
curl -X POST http://localhost:8080/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embedding",
    "input": "Hello, AI-Rust!",
    "options": {
      "max_length": 512
    }
  }'

# 响应：
# {
#   "output": [0.123, 0.456, ...],
#   "metadata": {
#     "duration_ms": 45,
#     "model_version": "1.0.0"
#   }
# }
```

**RAG查询**:
```bash
curl -X POST http://localhost:8080/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Rust?",
    "top_k": 5
  }'

# 响应：
# {
#   "answer": "Rust is a systems programming language...",
#   "sources": [...]
# }
```

---

## 🎯 核心功能

### 1. 模型推理

#### 加载模型

```rust
use ai_rust::model::ModelLoader;

// 创建模型加载器
let loader = ModelLoader::new("./models".into());

// 从本地文件加载
let model = loader.load_from_file("model.safetensors")?;

// 从HuggingFace加载
let model = loader.load_from_hub("sentence-transformers/all-MiniLM-L6-v2").await?;
```

#### 执行推理

```rust
use ai_rust::inference::InferenceEngine;

// 单次推理
let input = create_input("Hello, world!");
let output = model.infer(input)?;

// 批量推理
let inputs = vec![input1, input2, input3];
let outputs = model.batch_infer(inputs)?;

// 异步推理
let output = model.infer_async(input).await?;
```

---

### 2. RAG系统

#### 创建RAG系统

```rust
use ai_rust::rag::{RAGSystem, RAGConfig};

// 配置RAG系统
let config = RAGConfig {
    top_k: 5,
    min_relevance: 0.7,
    ..Default::default()
};

// 创建RAG系统
let mut rag = RAGSystem::new(retriever, generator, config);
```

#### 添加文档

```rust
// 添加单个文档
let doc = Document::new("Rust is a systems programming language.");
rag.add_document(doc).await?;

// 批量添加文档
let docs = vec![
    Document::new("Rust ensures memory safety."),
    Document::new("Rust has zero-cost abstractions."),
];

for doc in docs {
    rag.add_document(doc).await?;
}
```

#### 查询RAG

```rust
// 执行查询
let answer = rag.query("What is Rust?").await?;
println!("Answer: {}", answer);

// 搜索相关文档
let docs = rag.search("memory safety", 5).await?;
for doc in docs {
    println!("- {}", doc.text);
}
```

---

### 3. 多模态处理

#### 处理不同模态

```rust
use ai_rust::multimodal::MultimodalProcessor;

// 创建处理器
let processor = MultimodalProcessor::new(
    text_processor,
    image_processor,
    audio_processor,
);

// 处理文本
let text_emb = processor.process_text("A cat")?;

// 处理图像
let image_emb = processor.process_image(cat_image)?;

// 融合特征
let fused = processor.fuse(vec![text_emb, image_emb])?;
```

---

### 4. Agent系统

#### 创建Agent

```rust
use ai_rust::agent::{Agent, Tool};

// 创建Agent
let mut agent = MyAgent::new(llm, tools);

// 处理用户消息
let response = agent.process("Search for Rust tutorials").await?;
println!("Agent: {}", response);

// 重置Agent状态
agent.reset();
```

#### 添加工具

```rust
// 定义工具
struct WebSearchTool;

impl Tool for WebSearchTool {
    fn name(&self) -> &str { "web_search" }

    fn description(&self) -> &str {
        "Search the web for information"
    }

    async fn execute(&self, params: &str) -> Result<String> {
        // 执行搜索
        Ok(search_results)
    }
}

// 添加到Agent
agent.add_tool(Box::new(WebSearchTool));
```

---

## 🔥 高级应用

### 性能优化

#### 启用批量推理

```rust
use ai_rust::inference::BatchedInferenceEngine;

// 创建批量推理引擎
let engine = BatchedInferenceEngine::new(
    model,
    batch_size: 32,
    timeout: Duration::from_millis(100),
);

// 自动批处理推理
let output = engine.infer(input).await?;
```

**性能提升**: 吞吐量提升5-10倍

---

#### 启用缓存

```rust
use ai_rust::cache::CacheManager;

// 创建缓存管理器
let cache = CacheManager::new(max_size: 1000);

// 使用缓存
if let Some(cached) = cache.get(&key) {
    return Ok(cached);
}

let result = expensive_computation()?;
cache.put(key, result.clone());
```

**性能提升**: 缓存命中时延迟降低99%

---

#### 并行处理

```rust
use rayon::prelude::*;

// 并行批量推理
let outputs: Vec<_> = inputs
    .par_iter()
    .map(|input| model.infer(input).unwrap())
    .collect();
```

**性能提升**: 多核利用率提升至90%

---

### 自定义扩展

#### 实现自定义模型

```rust
use ai_rust::inference::InferenceEngine;

pub struct MyModel {
    weights: Tensor,
}

impl InferenceEngine<Tensor, Tensor> for MyModel {
    fn infer(&self, input: Tensor) -> Result<Tensor> {
        // 自定义推理逻辑
        let output = self.forward(input)?;
        Ok(output)
    }
}
```

---

#### 实现自定义工具

```rust
use ai_rust::agent::Tool;

pub struct CustomTool;

#[async_trait]
impl Tool for CustomTool {
    fn name(&self) -> &str {
        "custom_tool"
    }

    fn description(&self) -> &str {
        "A custom tool for specific tasks"
    }

    async fn execute(&self, params: &str) -> Result<String> {
        // 自定义工具逻辑
        Ok(result)
    }
}
```

---

### Docker部署

#### 构建镜像

```bash
# 构建Docker镜像
docker build -t ai-rust:latest .

# 查看镜像
docker images | grep ai-rust
```

#### 运行容器

```bash
# 运行容器
docker run -d \
  --name ai-rust \
  -p 8080:8080 \
  -v $(pwd)/models:/data/models \
  -e RUST_LOG=info \
  ai-rust:latest

# 查看日志
docker logs -f ai-rust

# 停止容器
docker stop ai-rust
```

---

## ❓ 常见问题

### Q1: 如何指定Rust版本？

**A**: 使用rustup指定版本：

```bash
# 安装特定版本
rustup install 1.90.0

# 设置默认版本
rustup default 1.90.0

# 在项目中使用特定版本
echo "1.90.0" > rust-toolchain
```

---

### Q2: 编译时内存不足怎么办？

**A**: 减少并行编译单元：

```toml
# 在Cargo.toml中添加
[profile.dev]
codegen-units = 1

[profile.release]
codegen-units = 1
```

或者使用：
```bash
cargo build -j 2  # 限制为2个并行任务
```

---

### Q3: 如何加快编译速度？

**A**: 使用sccache缓存编译结果：

```bash
# 安装sccache
cargo install sccache

# 配置环境变量
export RUSTC_WRAPPER=sccache

# 查看缓存统计
sccache --show-stats
```

---

### Q4: 模型文件放在哪里？

**A**: 默认模型目录：

```bash
# 创建模型目录
mkdir -p ./models

# 下载模型
wget https://huggingface.co/.../model.safetensors -O ./models/model.safetensors

# 或使用环境变量指定
export MODEL_CACHE_DIR=/path/to/models
```

---

### Q5: 如何启用详细日志？

**A**: 使用RUST_LOG环境变量：

```bash
# 启用调试日志
RUST_LOG=debug cargo run

# 指定模块日志级别
RUST_LOG=ai_rust=debug,info cargo run

# 日志级别: trace, debug, info, warn, error
```

---

### Q6: API返回错误怎么办？

**A**: 检查错误响应：

```json
{
  "error": "InferenceError",
  "message": "Model not found",
  "details": {
    "model_name": "unknown_model"
  }
}
```

常见错误：
- `ModelNotFound`: 模型文件不存在
- `InvalidInput`: 输入格式错误
- `Timeout`: 推理超时
- `OutOfMemory`: 内存不足

---

## 🔧 故障排查

### 编译问题

#### 问题: `error: linker 'cc' not found`

**解决方案**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

---

#### 问题: `error: failed to load source for dependency`

**解决方案**:
```bash
# 清理缓存
cargo clean

# 更新依赖
cargo update

# 重新构建
cargo build
```

---

### 运行时问题

#### 问题: 内存使用过高

**解决方案**:
1. 减少批处理大小
2. 启用对象池
3. 清理缓存
4. 使用量化模型

```rust
// 限制批处理大小
let config = InferenceConfig {
    batch_size: 16,  // 从32减少到16
    ..Default::default()
};
```

---

#### 问题: 推理速度慢

**解决方案**:
1. 启用发布模式构建
2. 使用批量推理
3. 启用缓存
4. 优化CPU特性

```bash
# 使用发布模式
cargo run --release

# 启用CPU优化
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

---

#### 问题: 服务无响应

**解决方案**:

```bash
# 检查服务状态
curl http://localhost:8080/health

# 查看进程
ps aux | grep ai_service

# 查看日志
tail -f logs/app.log

# 重启服务
pkill ai_service
./target/release/ai_service
```

---

## 📚 学习路径

### 初学者路径 (1-2周)

1. **第一天**: 环境配置和Hello World
2. **第二天**: 运行基础示例
3. **第三天**: 理解核心概念
4. **第四天**: 尝试API调用
5. **第二周**: 实现简单应用

---

### 进阶路径 (2-4周)

1. **第一周**: 深入RAG系统
2. **第二周**: 学习多模态处理
3. **第三周**: 掌握Agent系统
4. **第四周**: 性能优化实践

---

### 专家路径 (1-2个月)

1. **架构设计**: 大规模系统架构
2. **性能调优**: 极致性能优化
3. **生产部署**: Kubernetes部署
4. **开源贡献**: 参与项目开发

---

## 🔗 相关资源

### 官方文档
- [快速开始](QUICK_START.md)
- [API参考](API_REFERENCE.md)
- [最佳实践](docs/05_practical_guides/rust_ai_best_practices.md)
- [性能优化](docs/05_practical_guides/performance_optimization_guide.md)
- [部署指南](DEPLOYMENT_GUIDE.md)

### 示例代码
- [examples/rust_190_demo](examples/rust_190_demo/) - Rust 1.90特性展示
- [examples/practical_systems](examples/practical_systems/) - 实用AI系统
- [examples/web_services](examples/web_services/) - Web服务示例

### 社区资源
- [GitHub Issues](https://github.com/your-org/ai-rust/issues) - 问题反馈
- [讨论区](https://github.com/your-org/ai-rust/discussions) - 交流讨论
- [贡献指南](CONTRIBUTING.md) - 参与贡献

---

## 💬 获取帮助

如果您遇到问题，可以通过以下方式获取帮助：

1. **查看文档**: 首先查看相关文档
2. **搜索Issues**: 在GitHub Issues中搜索类似问题
3. **提问讨论**: 在讨论区提问
4. **提交Issue**: 如果是Bug，请提交Issue

**提问模板**:
```
## 问题描述
简要描述您遇到的问题

## 环境信息
- OS: [e.g. Ubuntu 22.04]
- Rust版本: [e.g. 1.90.0]
- 项目版本: [e.g. 0.1.0]

## 复现步骤
1. 第一步
2. 第二步
3. ...

## 期望行为
描述您期望的结果

## 实际行为
描述实际发生的情况

## 相关日志
```
粘贴相关日志
```
```

---

## 🎉 祝您使用愉快！

感谢您选择AI-Rust！如果您觉得这个项目有帮助，请给我们一个Star ⭐️

---

*最后更新: 2025年12月3日*
*维护者: AI-Rust项目团队*
*版本: 0.1.0*
