# AI-Rust 项目总结

> 对齐声明：术语统一见 `docs/02_knowledge_structures/2025_ai_知识术语表_GLOSSARY.md`；指标与报告口径见 `docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md` §Z.7。

## 🎯 项目概述

**AI-Rust** 是一个使用Rust语言结合成熟开源库学习AI相关技术和原理的实践项目。
项目旨在通过实际的代码实现来深入理解AI算法的数学原理和工程实践。

## 📁 项目结构

### 🛠️ 核心代码模块 (crates/)

- **c01_base** - 基础工具与数学库
- **c02_data** - 数据处理与预处理
- **c03_ml_basics** - 机器学习基础算法
- **c04_dl_fundamentals** - 深度学习基础
- **c05_nlp_transformers** - NLP与Transformer
- **c06_retrieval_tools** - 检索与向量数据库
- **c07_agents_systems** - AI代理系统
- **c08_serving_ops** - 模型服务与运维

### 📚 知识框架文档 (docs/)

- **01_authority_frameworks** - 权威知识框架 ⭐⭐⭐
- **02_knowledge_structures** - 知识结构体系 ⭐⭐
- **03_tech_trends** - 技术趋势分析 ⭐
- **04_learning_paths** - 学习路径指南
- **05_practical_guides** - 实践指南
- **06_research_analysis** - 研究分析
- **07_project_management** - 项目管理
- **08_legacy_docs** - 历史文档

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

## 🎯 学习目标

### 技术目标

- 掌握Rust在AI领域的应用
- 理解AI算法的数学原理
- 实现完整的AI系统

## 📦 可复现与报告

- 文档闭环：理念→分类→层次→映射→口径→论证→案例 已在实践/知识/权威/趋势/算法文档中对齐。
- 脚本与数据：
  - Linux/macOS：`scripts/bench/run_pareto.sh`、`scripts/rag/eval_hybrid.sh`、`scripts/repro/export_report.sh`
  - Windows：`scripts/bench/run_pareto.ps1`、`scripts/rag/eval_hybrid.ps1`、`scripts/repro/export_report.ps1`
  - 数据模板：`data/qa.example.jsonl`、`data/index/README.md`
- 报告与口径：`reports/README.md` 指定 CSV 列与指标统一口径；追踪样本支持审计与回放。
- CI：`.github/workflows/reports.yml` 手动触发生成与上传工件。
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

- ✅ **基础框架搭建完成**：项目结构、文档体系
- ✅ **知识体系文档完善**：权威框架、学习路径
- 🔄 **核心算法实现中**：各模块的代码实现
- 📋 **服务部署待完善**：生产环境部署

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
