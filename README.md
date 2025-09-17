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

### 环境要求

- Rust 1.70+
- Cargo
- Git

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

- `docs/01_authority_frameworks/2025_ai_rust_comprehensive_authority_framework.md`
- `docs/01_authority_frameworks/2025_ai_core_principles_analysis.md`

### 📚 知识结构体系 ⭐⭐

- `docs/02_knowledge_structures/2025_knowledge_landscape.md`
- `docs/02_knowledge_structures/2025_ai_knowledge_framework.md`

### 🛠️ 实践指南

- `docs/05_practical_guides/rust_ai_practice_guide.md`
- `docs/05_practical_guides/ai_algorithms_deep_dive.md`

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

## 📊 项目状态

- ✅ 基础框架搭建完成
- ✅ 知识体系文档完善
- 🔄 核心算法实现中
- 📋 服务部署待完善

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
