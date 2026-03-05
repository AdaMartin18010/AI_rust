# AI-Rust 项目文档结构图

> 使用Rust语言结合成熟开源库学习AI相关技术和原理的实践项目

## 📁 文档组织结构

```text
docs/
├── 📖 README.md                           # 文档结构说明和使用指南
├── 📊 DOCUMENT_STRUCTURE.md               # 本文档结构图
│
├── 🏆 01_authority_frameworks/            # 权威知识框架 ⭐⭐⭐
│   ├── 2025_ai_rust_comprehensive_authority_framework.md
│   ├── 2025_ai_rust_authority_topic_structure.md
│   ├── 2025_ai_core_principles_analysis.md
│   └── 2025_international_authority_topic_structure.md
│
├── 📚 02_knowledge_structures/            # 知识结构体系 ⭐⭐
│   ├── 2025_ai_rust_comprehensive_knowledge_framework.md
│   ├── 2025_ai_knowledge_framework.md
│   ├── 2025_knowledge_landscape.md
│   └── 2025_knowledge_index.md
│
├── 📈 03_tech_trends/                     # 技术趋势分析 ⭐
│   ├── 2025_tech_trends_update.md
│   ├── 2025_ai_rust_technology_trends_comprehensive_report.md
│   ├── 2025_ai_rust_tech_landscape.md
│   └── 2025_tech_architecture_diagram.md
│
├── 🛤️ 04_learning_paths/                  # 学习路径指南
│   ├── 2025_ai_rust_learning_path.md
│   ├── 2025_learning_path_best_practices.md
│   └── ai_learning_path.md
│
├── 🛠️ 05_practical_guides/                # 实践指南
│   ├── rust_ai_practice_guide.md
│   ├── 2025_rust_ai_practical_guide.md
│   ├── ai_algorithms_deep_dive.md
│   └── foundations.md
│
├── 🔬 06_research_analysis/               # 研究分析
│   ├── 2025_ai_research_papers_analysis.md
│   ├── research_watch.md
│   └── reading_list.md
│
├── 📊 07_project_management/              # 项目管理
│   ├── 2025_ai_rust_knowledge_framework_summary.md
│   ├── 2025_knowledge_consolidation_summary.md
│   ├── 2025_multi_task_progress_summary.md
│   └── multi_task_progress_summary.md
│
└── 📦 08_legacy_docs/                     # 历史文档
    ├── 2025_web_knowledge_guide.md
    ├── ai_rust_landscape.md
    ├── curricula_2025.md
    ├── taxonomy_2025.md
    └── ... (其他历史文档)
```

## 🎯 学习路径映射

### 新手入门路径

```text
01_authority_frameworks/
├── 2025_ai_rust_comprehensive_authority_framework.md  # 起点
└── 2025_ai_rust_authority_topic_structure.md

02_knowledge_structures/
└── 2025_knowledge_landscape.md

04_learning_paths/
└── 2025_ai_rust_learning_path.md

05_practical_guides/
└── rust_ai_practice_guide.md
```

### 进阶学习路径

```text
01_authority_frameworks/
└── 2025_ai_core_principles_analysis.md

03_tech_trends/
└── 2025_tech_trends_update.md

05_practical_guides/
└── ai_algorithms_deep_dive.md

06_research_analysis/
└── 2025_ai_research_papers_analysis.md
```

### 项目管理路径

```text
07_project_management/
├── 2025_ai_rust_knowledge_framework_summary.md
└── 2025_multi_task_progress_summary.md
```

## 📋 文档分类标准

### 优先级分类

- **⭐⭐⭐ 最高优先级**：权威知识框架，对标国际标准
- **⭐⭐ 高优先级**：知识结构体系，系统性组织
- **⭐ 重要参考**：技术趋势分析，最新发展动态
- **无标记**：实用指南，具体操作指导

### 内容分类

- **理论性**：权威框架、知识结构、核心原理
- **实践性**：学习路径、实践指南、算法实现
- **分析性**：技术趋势、研究分析、项目管理
- **历史性**：legacy文档，已整合或过时内容

## 🔄 维护流程

### 新文档添加

1. 根据内容类型选择合适文件夹
2. 按照命名规范命名文件
3. 更新README.md中的文档列表
4. 更新本文档结构图

### 文档升级

1. 重要文档可升级到更高优先级文件夹
2. 更新文档间的交叉引用
3. 通知相关维护人员

### 文档归档

1. 过时文档移动到08_legacy_docs
2. 保留历史版本供参考
3. 更新文档索引

## 🎯 项目定位

**AI-Rust** 是一个使用Rust语言结合成熟开源库学习AI相关技术和原理的实践项目。项目旨在通过实际的代码实现来深入理解AI算法的数学原理和工程实践。

## 📊 统计信息

- **总文档数**：约40个文档
- **权威框架**：4个核心文档
- **知识结构**：4个体系文档
- **技术趋势**：4个分析文档
- **学习路径**：3个指导文档
- **实践指南**：3个实用文档
- **研究分析**：3个研究文档
- **项目管理**：4个管理文档
- **历史文档**：约15个legacy文档

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

---

*最后更新：2025年1月*
*版本：v2.0*
*状态：已重新组织*
*维护者：AI-Rust项目团队*
