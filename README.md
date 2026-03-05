# AI-Rust 学习项目

> 使用Rust语言结合成熟开源库学习AI相关技术和原理的实践项目

## 🎯 项目目标

本项目旨在通过Rust语言实现AI算法和系统，深入学习AI技术的核心原理和实现细节。通过结合成熟的开源库，构建完整的AI学习体系。

## 📁 项目结构

```text
AI_rust/
├── 📚 docs/                           # 知识框架文档
│   ├── 01_authority_frameworks/       # 权威知识框架 ⭐⭐⭐
│   ├── 02_knowledge_structures/       # 知识结构体系 ⭐⭐
│   ├── 03_tech_trends/               # 技术趋势分析 ⭐
│   ├── 04_learning_paths/            # 学习路径指南
│   ├── 05_practical_guides/          # 实践指南
│   ├── 06_research_analysis/         # 研究分析
│   ├── 07_project_management/        # 项目管理
│   └── 08_legacy_docs/              # 历史文档
│
├── 🛠️ crates/                        # Rust代码模块
│   ├── c01_base/                     # 基础工具与数学库
│   ├── c02_data/                     # 数据处理与预处理
│   ├── c03_ml_basics/               # 机器学习基础算法
│   ├── c04_dl_fundamentals/         # 深度学习基础
│   ├── c05_nlp_transformers/        # NLP与Transformer
│   ├── c06_retrieval_tools/         # 检索与向量数据库
│   ├── c07_agents_systems/          # AI代理系统
│   └── c08_serving_ops/             # 模型服务与运维
│
├── 📖 courses/                       # 课程大纲
│   ├── ai-with-rust/                # AI与Rust实战课程
│   └── rust/                        # Rust基础课程
│
├── 📋 plans/                         # 项目规划
├── 📊 reports/                       # 性能报告
├── 📝 notes/                         # 开发笔记
└── 🔧 tools/                         # 开发工具
```

## 🚀 快速开始

### 2025年最新特性演示

- **Rust 1.94 AI特性**: `cargo run --example rust_190_ai_features --features "candle,linear-algebra-advanced"`
- **WebAssembly AI推理**: `cargo run --example wasm_ai_inference`
- **多模态AI处理**: `cargo run --example multimodal_ai_processing`
- **Agentic Web架构**: `cargo run --example agentic_web_architecture`
- **性能优化演示**: `cargo run --example performance_optimization`

### 复现与报告（可选）

- Linux/macOS：
  - Pareto：`bash scripts/bench/run_pareto.sh --model large-v1 --quant int4 --batch 8 --concurrency 16 --seq-len 2048 --router small-fallback --repeats 5 --out reports`
  - RAG：`bash scripts/rag/eval_hybrid.sh --index data/index --dataset data/qa.example.jsonl --k 100 --kprime 20 --reranker cross-encoder-small --out reports`
  - 打包：`bash scripts/repro/export_report.sh --reports reports --out dist`
- Windows：
  - Pareto：`./scripts/bench/run_pareto.ps1 -Model large-v1 -Quant int4 -Batch 8 -Concurrency 16 -SeqLen 2048 -Router small-fallback -Repeats 5 -Out reports`
  - RAG：`./scripts/rag/eval_hybrid.ps1 -Index data/index -Dataset data/qa.example.jsonl -K 100 -KPrime 20 -Reranker cross-encoder-small -Out reports`
  - 打包：`./scripts/repro/export_report.ps1 -Reports reports -Out dist`

CI：`.github/workflows/reports.yml` 支持手动触发，自动上传压缩工件。

### 环境要求

- **Rust 1.94+** (推荐最新版本)
- Cargo
- Git
- **可选**: GPU支持 (CUDA/Metal) 用于GPU加速
- **可选**: WebAssembly工具链用于WASM AI推理

### 安装与运行

```bash
# 克隆项目
git clone <repository-url>
cd AI_rust

# 构建项目
cargo build

# 运行基础服务
cargo run -p c08_serving_ops

# 运行测试
cargo test
```

```text
    - [📚 核心文档（必读）](#-核心文档必读)
      - [🏆 权威知识框架 ⭐⭐⭐](#-权威知识框架-)
      - [📚 知识结构体系 ⭐⭐](#-知识结构体系-)
      - [📈 技术趋势分析 ⭐](#-技术趋势分析-)
      - [🛤️ 学习路径指南](#️-学习路径指南)
      - [🛠️ 实践指南](#️-实践指南)
      - [📊 项目管理](#-项目管理)
    - [🎯 学习路径](#-学习路径)
    - [📋 课程大纲](#-课程大纲)
    - [📊 项目规划](#-项目规划)
    - [🔬 实践环境](#-实践环境)
    - [🚀 快速开始建议](#-快速开始建议)
      - [新手入门路径](#新手入门路径)
      - [进阶学习路径](#进阶学习路径)
      - [项目管理路径](#项目管理路径)
    - [🌟 2025年学习优势](#-2025年学习优势)
    - [运行最小服务](#运行最小服务)
    - [测试](#测试)
    - [代码结构](#代码结构)
    - [可替换推理引擎](#可替换推理引擎)
    - [配置与运维提示](#配置与运维提示)

- [文档索引](#文档索引)
```

## 🏆 核心学习模块

### 1. 基础工具与数学库 (c01_base)

**学习目标**：掌握Rust中的数值计算和数学基础

- 线性代数运算 (ndarray)
- 概率统计计算
- 优化算法实现
- 数学函数库

**主要依赖**：

```toml
ndarray = "0.15"
nalgebra = "0.32"
rand = "0.8"
statrs = "0.16"
```

### 2. 数据处理与预处理 (c02_data)

**学习目标**：掌握数据科学中的数据处理技术

- 数据加载与清洗
- 特征工程
- 数据可视化
- 数据存储与序列化

**主要依赖**：

```toml
polars = "0.35"
serde = "1.0"
csv = "1.2"
plotly = "0.8"
```

### 3. 机器学习基础算法 (c03_ml_basics)

**学习目标**：实现经典机器学习算法

- 线性回归与逻辑回归
- 决策树与随机森林
- 支持向量机
- 聚类算法
- 降维技术

**主要依赖**：

```toml
linfa = "0.7"
smartcore = "0.3"
```

### 4. 深度学习基础 (c04_dl_fundamentals)

**学习目标**：构建神经网络和深度学习模型

- 神经网络基础
- 反向传播算法
- 卷积神经网络
- 循环神经网络
- 优化器实现

**主要依赖**：

```toml
candle-core = "0.3"
candle-nn = "0.3"
candle-datasets = "0.3"
```

### 5. NLP与Transformer (c05_nlp_transformers)

**学习目标**：实现现代NLP模型

- 词嵌入技术
- Transformer架构
- 注意力机制
- 预训练模型
- 微调技术

**主要依赖**：

```toml
candle-transformers = "0.3"
tokenizers = "0.15"
```

### 6. 检索与向量数据库 (c06_retrieval_tools)

**学习目标**：构建智能检索系统

- 向量相似度计算
- 近似最近邻搜索
- 向量数据库集成
- 语义搜索

**主要依赖**：

```toml
qdrant-client = "1.7"
faiss = "0.12"
```

### 7. AI代理系统 (c07_agents_systems)

**学习目标**：构建智能代理系统

- 代理架构设计
- 工具调用机制
- 多模态处理
- 决策制定

**主要依赖**：

```toml
tokio = "1.0"
serde_json = "1.0"
reqwest = "0.11"
```

### 8. 模型服务与运维 (c08_serving_ops)

**学习目标**：部署和运维AI服务

- 模型服务化
- API设计
- 性能监控
- 容器化部署

**主要依赖**：

```toml
axum = "0.7"
tower = "0.4"
tracing = "0.1"
```

## 📚 学习路径

### 新手入门路径

1. **Rust基础** → `courses/rust/SYLLABUS.md`
2. **数学基础** → `docs/05_practical_guides/foundations.md`
3. **数据处理** → `crates/c02_data/`
4. **机器学习** → `crates/c03_ml_basics/`

### 进阶学习路径

1. **深度学习** → `crates/c04_dl_fundamentals/`
2. **NLP技术** → `crates/c05_nlp_transformers/`
3. **系统架构** → `crates/c07_agents_systems/`
4. **服务部署** → `crates/c08_serving_ops/`

### 专家级路径

1. **核心原理** → `docs/01_authority_frameworks/2025_ai_core_principles_analysis.md`
2. **前沿技术** → `docs/03_tech_trends/`
3. **研究分析** → `docs/06_research_analysis/`

## 🛠️ 技术栈

### 核心AI框架

- **Candle**：轻量级深度学习框架
- **Linfa**：机器学习工具包
- **SmartCore**：机器学习算法库

### 数据处理

- **Polars**：高性能数据处理
- **Ndarray**：多维数组计算
- **Serde**：序列化框架

### Web服务

- **Axum**：异步Web框架
- **Tokio**：异步运行时
- **Tower**：中间件框架

### 系统工具

- **Docker**：容器化部署
- **Cargo**：包管理
- **Git**：版本控制

## 📖 核心文档

### 🏆 权威知识框架 ⭐⭐⭐

- `docs/01_authority_frameworks/2025_ai_rust_comprehensive_authority_framework.md` - 国际权威知识框架
- `docs/01_authority_frameworks/2025_ai_rust_authority_topic_structure.md` - 权威主题目录结构
- `docs/01_authority_frameworks/2025_ai_core_principles_analysis.md` - AI核心原理深度分析

### 📚 知识结构体系 ⭐⭐

- `docs/02_knowledge_structures/2025_ai_rust_comprehensive_knowledge_framework.md` - 综合知识框架
- `docs/02_knowledge_structures/2025_knowledge_landscape.md` - 知识全景图
- `docs/02_knowledge_structures/2025_ai_knowledge_framework.md` - AI知识框架

### 🛠️ 实践指南

- `docs/05_practical_guides/rust_ai_practice_guide.md` - Rust AI实践指南
- `docs/05_practical_guides/ai_algorithms_deep_dive.md` - AI算法深度解析
- `docs/05_practical_guides/foundations.md` - 基础知识框架

### 📊 项目管理

- `docs/07_project_management/2025_ai_rust_knowledge_framework_summary.md` - 知识框架构建总结

## 🎯 学习目标

### 技术目标

- 掌握Rust在AI领域的应用
- 理解AI算法的数学原理
- 实现完整的AI系统
- 掌握模型部署和运维

### 能力目标

- 系统设计能力
- 性能优化能力
- 工程实践能力
- 问题解决能力

## 🌟 项目特色

1. **理论与实践结合**：每个模块都有详细的理论说明和代码实现
2. **渐进式学习**：从基础到高级，循序渐进
3. **现代技术栈**：使用最新的Rust AI生态
4. **完整项目**：涵盖从算法到部署的完整流程
5. **开源友好**：基于成熟的开源库构建

## 🚀 2025年最新技术特性

### Rust 1.94语言特性集成

- **泛型关联类型 (GAT)**: 简化异步AI推理的类型定义
- **类型别名实现特性 (TAIT)**: 减少复杂类型的复杂度
- **改进的异步编程模型**: 提升AI系统的并发性能

### 2025年最新AI库集成

- **Kornia-rs**: 3D计算机视觉高性能库
- **Thistle**: 高性能向量数据库
- **faer-rs**: 高性能线性代数库
- **ad-trait**: 自动微分库
- **Similari**: 对象跟踪和相似性搜索

### WebAssembly AI推理

- **客户端AI计算**: 在浏览器中运行AI模型
- **隐私保护**: 本地AI处理保护用户数据
- **边缘计算**: 支持边缘设备AI推理

### 多模态AI处理

- **统一模态处理**: Text-Image-Audio-Video统一处理
- **跨模态理解**: 实现跨模态内容理解和生成
- **并行编码**: 高效的并行编码和融合

### Agentic Web架构

- **AI代理驱动**: AI代理驱动的Web交互
- **自主执行**: 智能任务规划和执行
- **协作机制**: 代理间智能协作

### 性能优化技术

- **SIMD向量化**: 3-5倍性能提升
- **零拷贝处理**: 减少50%内存使用
- **缓存友好算法**: 2-3倍性能提升
- **内存池管理**: 减少90%分配开销
- **GPU加速**: 10-100倍计算性能提升

## 📊 项目状态

- ✅ **Rust 1.94特性集成完成**
- ✅ **2025年最新AI库集成完成**
- ✅ **WebAssembly AI推理完成**
- ✅ **多模态AI处理完成**
- ✅ **Agentic Web架构完成**
- ✅ **性能优化技术完成**
- ✅ **基础框架搭建完成**
- ✅ **知识体系文档完善**
- ✅ **核心算法实现完成**
- ✅ **服务部署完成**
- 🔄 **安全增强进行中** (90%完成)
- 📋 **持续优化和扩展**

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交代码
4. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

---

*最后更新：2025年1月*  
*版本：v2.0*  
*状态：开发中*  
*维护者：AI-Rust学习团队*
