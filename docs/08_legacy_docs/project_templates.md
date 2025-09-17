# 项目模板与脚手架（2025版）

> 快速跳转：
>
> - 学习总览：`docs/ai_learning_overview.md`
> - 评估量表：`docs/assessment_rubrics.md`
> - 实践指南：`docs/rust_ai_practice_guide.md`

## 目录

- [快速开始](#快速开始)
- [模板分类](#模板分类)
  - [基础模板](#基础模板)
  - [AI应用模板](#ai应用模板)
  - [系统服务模板](#系统服务模板)
- [脚手架工具](#脚手架工具)
- [最佳实践](#最佳实践)
- [PRD/技术方案/实验日志模板](#prd技术方案实验日志模板)
- [API 合同与错误码规范](#api-合同与错误码规范)
- [环境与配置矩阵](#环境与配置矩阵)
- [合规与安全清单](#合规与安全清单)
- [专项模板：RAG / LLM微调 / 多代理](#专项模板rag--llm微调--多代理)

## 快速开始

```bash
# 使用脚手架创建项目
cargo install cargo-generate
cargo generate --git https://github.com/your-org/rust-ai-templates

# 或直接克隆模板
git clone https://github.com/your-org/rust-ai-templates.git
cd rust-ai-templates
```

## 模板分类

### 基础模板

#### 1. 机器学习基础项目

```toml
# Cargo.toml
[package]
name = "ml-basics"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.15"
nalgebra = "0.32"
linfa = "0.7"
linfa-linear = "0.7"
linfa-trees = "0.7"
polars = "0.40"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.0", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
criterion = "0.5"
approx = "0.5"
```

**项目结构**：

```text
ml-basics/
├── src/
│   ├── lib.rs
│   ├── algorithms/
│   │   ├── linear_regression.rs
│   │   ├── logistic_regression.rs
│   │   └── decision_tree.rs
│   ├── data/
│   │   ├── dataset.rs
│   │   └── preprocessing.rs
│   └── utils/
│       ├── metrics.rs
│       └── visualization.rs
├── examples/
│   ├── linear_regression_demo.rs
│   └── classification_demo.rs
├── benches/
│   └── performance.rs
└── tests/
    └── integration_tests.rs
```

#### 2. 深度学习项目

```toml
# Cargo.toml
[package]
name = "dl-project"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.3"
candle-nn = "0.3"
candle-datasets = "0.3"
candle-transformers = "0.3"
ndarray = "0.15"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
tracing = "0.1"
```

**项目结构**：

```text
dl-project/
├── src/
│   ├── lib.rs
│   ├── models/
│   │   ├── mlp.rs
│   │   ├── cnn.rs
│   │   └── transformer.rs
│   ├── training/
│   │   ├── trainer.rs
│   │   ├── optimizer.rs
│   │   └── scheduler.rs
│   └── data/
│       ├── dataloader.rs
│       └── transforms.rs
├── examples/
│   ├── train_mlp.rs
│   └── inference_demo.rs
└── models/
    └── pretrained/
```

### AI应用模板

#### 3. RAG系统模板

```toml
# Cargo.toml
[package]
name = "rag-system"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.3"
candle-nn = "0.3"
candle-transformers = "0.3"
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
qdrant-client = "1.7"
tantivy = "0.21"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1.0"
```

**项目结构**：

```text
rag-system/
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── api/
│   │   ├── routes.rs
│   │   ├── handlers.rs
│   │   └── middleware.rs
│   ├── embedding/
│   │   ├── model.rs
│   │   └── service.rs
│   ├── retrieval/
│   │   ├── vector_db.rs
│   │   ├── text_search.rs
│   │   └── reranker.rs
│   ├── generation/
│   │   ├── llm.rs
│   │   └── prompt.rs
│   └── config/
│       └── settings.rs
├── migrations/
├── tests/
│   ├── integration/
│   └── load_tests/
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

#### 4. 多代理系统模板

```toml
# Cargo.toml
[package]
name = "multi-agent-system"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
axum = "0.7"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres", "chrono", "uuid"] }
redis = { version = "0.24", features = ["tokio-comp"] }
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1.0"
thiserror = "1.0"
```

**项目结构**：

```text
multi-agent-system/
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── agents/
│   │   ├── base.rs
│   │   ├── planner.rs
│   │   ├── executor.rs
│   │   └── reviewer.rs
│   ├── communication/
│   │   ├── message.rs
│   │   ├── protocol.rs
│   │   └── broker.rs
│   ├── memory/
│   │   ├── short_term.rs
│   │   ├── long_term.rs
│   │   └── working.rs
│   ├── tools/
│   │   ├── calculator.rs
│   │   ├── web_search.rs
│   │   └── code_executor.rs
│   └── orchestration/
│       ├── coordinator.rs
│       └── scheduler.rs
├── migrations/
└── tests/
    └── agent_tests/
```

### 系统服务模板

#### 5. 推理服务模板

```toml
# Cargo.toml
[package]
name = "inference-service"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.3"
candle-nn = "0.3"
candle-transformers = "0.3"
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace", "compression-gzip"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
tracing-opentelemetry = "0.21"
opentelemetry = "0.21"
opentelemetry-jaeger = "0.20"
metrics = "0.22"
metrics-exporter-prometheus = "0.13"
anyhow = "1.0"
```

**项目结构**：

```text
inference-service/
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── api/
│   │   ├── routes.rs
│   │   ├── handlers.rs
│   │   └── middleware.rs
│   ├── models/
│   │   ├── registry.rs
│   │   ├── loader.rs
│   │   └── cache.rs
│   ├── inference/
│   │   ├── engine.rs
│   │   ├── batch.rs
│   │   └── streaming.rs
│   ├── monitoring/
│   │   ├── metrics.rs
│   │   ├── tracing.rs
│   │   └── health.rs
│   └── config/
│       └── settings.rs
├── models/
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
└── docker/
    └── Dockerfile
```

## 脚手架工具

### 1. 项目生成器

```rust
// tools/project-generator/src/main.rs
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rust-ai-generator")]
#[command(about = "Generate Rust AI project templates")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a new project
    New {
        /// Project name
        name: String,
        /// Template type
        #[arg(short, long)]
        template: String,
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// List available templates
    List,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::New { name, template, output } => {
            generate_project(&name, &template, output).await?;
        }
        Commands::List => {
            list_templates().await?;
        }
    }
    
    Ok(())
}

async fn generate_project(
    name: &str,
    template: &str,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output.unwrap_or_else(|| PathBuf::from(name));
    
    match template {
        "ml-basics" => generate_ml_basics(&output_dir).await?,
        "dl-project" => generate_dl_project(&output_dir).await?,
        "rag-system" => generate_rag_system(&output_dir).await?,
        "multi-agent" => generate_multi_agent(&output_dir).await?,
        "inference-service" => generate_inference_service(&output_dir).await?,
        _ => return Err("Unknown template".into()),
    }
    
    println!("✅ Project '{}' generated successfully!", name);
    Ok(())
}
```

### 2. 代码生成器

```rust
// tools/code-generator/src/main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rust-ai-codegen")]
#[command(about = "Generate Rust AI code templates")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate model implementation
    Model {
        /// Model name
        name: String,
        /// Model type (mlp, cnn, transformer)
        #[arg(short, long)]
        model_type: String,
    },
    /// Generate API handler
    Handler {
        /// Handler name
        name: String,
        /// HTTP method
        #[arg(short, long)]
        method: String,
    },
    /// Generate test template
    Test {
        /// Test name
        name: String,
        /// Test type (unit, integration, benchmark)
        #[arg(short, long)]
        test_type: String,
    },
}
```

## 最佳实践

### 1. 项目结构规范

```text
project-name/
├── src/
│   ├── lib.rs              # 库入口
│   ├── main.rs             # 二进制入口
│   ├── algorithms/         # 算法实现
│   ├── models/             # 模型定义
│   ├── data/               # 数据处理
│   ├── api/                # API接口
│   ├── utils/              # 工具函数
│   └── config/             # 配置管理
├── examples/               # 示例代码
├── tests/                  # 测试代码
├── benches/                # 基准测试
├── docs/                   # 文档
├── migrations/             # 数据库迁移
├── docker/                 # Docker配置
├── k8s/                    # Kubernetes配置
├── Cargo.toml              # 项目配置
├── README.md               # 项目说明
├── .gitignore              # Git忽略文件
├── .github/                # GitHub配置
│   └── workflows/          # CI/CD
└── Makefile                # 构建脚本
```

### 2. 依赖管理

```toml
# Cargo.toml 最佳实践
[package]
name = "your-project"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "Your project description"
license = "MIT"
repository = "https://github.com/your-org/your-project"
keywords = ["ai", "machine-learning", "rust"]
categories = ["science::machine-learning"]

[dependencies]
# 核心依赖
anyhow = "1.0"              # 错误处理
thiserror = "1.0"           # 自定义错误
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"             # 日志
tracing-subscriber = "0.3"

# AI/ML依赖
candle-core = "0.3"
candle-nn = "0.3"
ndarray = "0.15"
linfa = "0.7"

# Web服务依赖
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
tower = "0.4"

[dev-dependencies]
criterion = "0.5"           # 基准测试
approx = "0.5"              # 数值比较
tempfile = "3.0"            # 临时文件

[profile.release]
lto = true                  # 链接时优化
codegen-units = 1           # 减少代码生成单元
panic = "abort"             # 减少二进制大小
```

### 3. 测试策略

```rust
// tests/integration_tests.rs
use your_project::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_end_to_end_pipeline() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::from_file("tests/fixtures/config.toml").unwrap();
    
    // 测试完整流程
    let result = run_pipeline(&config, &temp_dir.path()).await;
    assert!(result.is_ok());
}

// benches/performance.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use your_project::*;

fn benchmark_inference(c: &mut Criterion) {
    let model = load_model("models/test.bin").unwrap();
    let input = generate_test_input();
    
    c.bench_function("inference", |b| {
        b.iter(|| {
            model.predict(black_box(&input))
        })
    });
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
```

### 4. CI/CD配置

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        components: rustfmt, clippy
    
    - name: Cache cargo registry
      uses: actions/cache@v3
      with:
        path: ~/.cargo/registry
        key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Cache cargo index
      uses: actions/cache@v3
      with:
        path: ~/.cargo/git
        key: ${{ runner.os }}-cargo-index-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Cache cargo build
      uses: actions/cache@v3
      with:
        path: target
        key: ${{ runner.os }}-cargo-build-target-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Run tests
      run: cargo test --verbose
    
    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings
    
    - name: Run fmt
      run: cargo fmt --all -- --check
    
    - name: Run benchmarks
      run: cargo bench --no-run
```

## PRD/技术方案/实验日志模板

### 1) PRD（产品需求文档）

```text
背景/目标：
用户画像/场景：
核心问题与成功指标：
功能列表（MoSCoW）：
验收标准（ACs）：
非功能需求（性能/安全/合规）：
里程碑与风险：
```

### 2) 技术方案（Tech Spec）

```text
问题定义与约束：
整体架构图：
数据流/控制流：
技术选型对比（成本/性能/风险）：
接口设计（API/Schema/错误码）：
可观测性与SLO：
落地计划（阶段/人力/预估）：
```

### 3) 实验日志（Experiment Log）

```text
实验编号/日期：
目标与假设：
数据与预处理：
模型/超参/代码版本：
评测指标与结果：
分析与结论：
后续动作：
复现脚本路径：
```

这个项目模板文档提供了完整的项目脚手架，包括不同场景的模板、代码生成工具和最佳实践，帮助快速启动Rust AI项目。

## API 合同与错误码规范

### 1) 统一响应结构

```json
{
  "code": 0,
  "message": "ok",
  "data": {}
}
```

- code：0 表示成功；非0为错误（参见错误码表）
- message：人类可读信息
- data：业务负载

### 2) 错误码表（示例）

| 范围 | 含义 | 示例 |
|---|---|---|
| 0 | 成功 | 0 |
| 1000-1999 | 客户端错误 | 1001 参数缺失；1002 参数非法 |
| 2000-2999 | 服务错误 | 2001 下游超时；2002 资源不足 |
| 3000-3999 | 安全/合规 | 3001 鉴权失败；3002 频控触发 |

### 3) RAG 接口草案

- POST `/embed`：输入文本列表，返回向量
- POST `/search`：输入查询，返回文档候选（混检+重排）
- POST `/generate`：输入上下文+提示，返回生成结果（支持流式）

## 环境与配置矩阵

| 维度 | 开发 | 测试 | 预发 | 生产 |
|---|---|---|---|---|
| 日志级别 | debug | info | info | warn |
| 指标上报 | 本地Prom | Prom | Prom/OTel | Prom/OTel |
| 模型缓存 | 本地 | 本地/Redis | Redis | Redis |
| 向量库 | 本地Qdrant | 测试Qdrant | 预发Qdrant | 托管Qdrant |
| 文本检索 | tantivy本地 | tantivy | tantivy集群 | 专用集群 |
| 限流 | 关闭 | 低阈值 | 中阈值 | 严格阈值 |

## 合规与安全清单

- 鉴权与授权：API Key/JWT；最小权限；密钥轮换
- 数据治理：敏感字段脱敏；访问审计；数据保留策略
- 模型安全：提示注入防护；输出过滤（PII）
- 供应链：Cargo.lock 固定；许可证扫描；依赖告警
- 运行安全：速率限制；WAF；重放保护；签名校验

## 专项模板：RAG / LLM微调 / 多代理

### 1) RAG 项目模板补充

- PRD 关键项：
  - 使用场景与检索对象边界；延迟与召回质量KPI（如 P95<300ms，nDCG@10≥0.45）
  - 安全与审计：查询/响应审计、敏感信息屏蔽
- Tech Spec 关键项：
  - 索引方案（倒排/向量/Hybrid），路由策略（BM25→Dense→Rerank）
  - 上下文构造策略（窗口/重写/压缩）与提示模版版本化
  - 评测集构建与对齐（问题-证据-答案三元组）
- 压测与评测脚本骨架：
  - `scripts/benchmark_rag.rs`：并发QPS/延迟、缓存命中、错误率
  - `scripts/eval_rag.rs`：召回/精排指标（Recall@k、MRR、nDCG）与生成质量（BLEU/ROUGE/参考评分）

### 2) LLM 微调模板补充

- PRD 关键项：
  - 目标任务与域内分布、输入输出长度、成本与时限约束
  - 指标目标：SFT 任务指标、对齐后拒答率/有害率、离线与在线指标对齐
- Tech Spec 关键项：
  - 数据治理（去重/过滤/红线词）、采样与配比；分词器与max_len策略
  - 参数高效微调（LoRA/QLoRA）与混合精度策略；检查点与恢复
  - 推理优化：INT8/INT4 量化、KV缓存、批处理
- 评测与对齐脚本骨架：
  - `scripts/train_sft.py/.rs`、`scripts/train_dpo.py/.rs`
  - `scripts/eval_gen.py/.rs`：长度归一化、重复惩罚、对比搜索
  - `scripts/safety_audit.md`：对齐与安全用例集

### 3) 多代理系统模板补充

- PRD 关键项：
  - 任务分解类型（规划/工具/回顾）、时限/成本预算、失败兜底策略
  - KPI：任务成功率、平均步数、成本/时长、人工介入率
- Tech Spec 关键项：
  - 角色协议（消息格式/心跳/重试）、记忆（短/长/工作）与检索策略
  - 工具执行沙盒与安全隔离；回顾与自检机制
- 评测与记录：
  - `scripts/eval_agents.rs`：任务集回放、成功率分布、代价曲线
  - 结构化日志约定：每步“思考/行动/观察”与错误码
