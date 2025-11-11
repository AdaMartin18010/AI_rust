# AI/ML 主题文件夹总览

## 概述

本文件夹包含Rust 1.90版本中AI/ML领域的完整技术栈，按主题组织，便于学习和使用。

## 文件夹结构

### 01. 深度学习框架 (Deep Learning Frameworks)

- **Candle**: Hugging Face的极简ML框架
- **Burn**: 纯Rust深度学习框架
- **Tch**: PyTorch Rust绑定
- **DFDx**: 类型安全的深度学习框架

### 02. 机器学习库 (Machine Learning Libraries)

- **Linfa**: Rust的scikit-learn等价物
- **SmartCore**: 纯Rust机器学习库
- **ndarray**: 多维数组库
- **nalgebra**: 线性代数库

### 03. 自然语言处理 (Natural Language Processing)

- **llm-chain**: 大语言模型链式处理框架
- **rust-bert**: BERT模型Rust实现
- **tokenizers**: 高性能分词器
- **whatlang**: 语言检测

### 04. 计算机视觉 (Computer Vision)

- **Kornia-rs**: 高性能3D计算机视觉库
- **OpenCV**: 成熟的计算机视觉库
- **image**: 纯Rust图像处理库
- **imageproc**: 图像处理算法库

### 05. 数据处理 (Data Processing)

- **Polars**: 高性能DataFrame库
- **DataFusion**: 查询执行引擎
- **Arrow**: 列式内存格式
- **Serde**: 序列化和反序列化

### 06. 向量搜索 (Vector Search)

- **Qdrant**: 向量数据库客户端
- **Tantivy**: 全文搜索引擎
- **HNSW-rs**: 分层导航小世界图算法
- **Faiss-rs**: Facebook AI相似性搜索

### 07. 新兴技术 (Emerging Technologies)

- **多模态AI**: 融合多种模态的AI技术
- **联邦学习**: 分布式隐私保护学习
- **边缘AI**: 边缘设备AI部署
- **量子机器学习**: 量子计算与ML结合

### 08. 性能优化 (Performance Optimization)

- **编译优化**: LTO、CPU指令集优化
- **内存优化**: 零拷贝、内存池
- **并行优化**: 多线程、SIMD、GPU加速
- **算法优化**: 高效算法和数据结构

### 09. 安全与隐私 (Security & Privacy)

- **模型安全**: 对抗攻击防护
- **数据安全**: 加密、访问控制
- **隐私保护**: 差分隐私、联邦学习
- **合规性**: GDPR、CCPA等法规遵循

### 10. 生产部署 (Production Deployment)

- **容器化**: Docker、Kubernetes
- **云原生**: AWS、Azure、GCP
- **监控**: 指标收集、日志管理
- **部署策略**: 蓝绿部署、滚动更新

## 技术栈推荐

### 生产环境

```text
深度学习: Candle + Burn
传统ML: Linfa + SmartCore
NLP: llm-chain + rust-bert
计算机视觉: Kornia-rs + OpenCV
数据处理: Polars + DataFusion
向量搜索: Qdrant + HNSW-rs
```

### 研究环境

```text
深度学习: Burn + DFDx
传统ML: SmartCore
NLP: tokenizers + rust-bert
计算机视觉: Kornia-rs
数据处理: Polars
向量搜索: HNSW-rs + Faiss-rs
```

### 学习环境

```text
深度学习: Tch (文档丰富)
传统ML: SmartCore (纯Rust)
NLP: tokenizers + whatlang
计算机视觉: image + imageproc
数据处理: Polars
向量搜索: HNSW-rs (简单)
```

## 快速开始

### 1. 选择技术栈

根据你的需求选择合适的技术栈：

- **生产应用**: 选择成熟稳定的库
- **研究项目**: 选择灵活的研究库
- **学习目的**: 选择文档完善的库

### 2. 安装依赖

在`Cargo.toml`中添加所需依赖：

```toml
[dependencies]
# 深度学习
candle-core = "0.9.1"
candle-nn = "0.9.1"

# 机器学习
linfa = "0.7.1"
smartcore = "0.4.2"

# NLP
llm-chain = "0.13.0"
rust-bert = "0.23.0"

# 计算机视觉
opencv = "0.95.1"
image = "0.25.8"

# 数据处理
polars = "0.50.0"
datafusion = "49.0.2"

# 向量搜索
qdrant-client = "1.15.0"
tantivy = "0.25.0"
```

### 3. 运行示例

每个主题文件夹都包含详细的示例代码：

```bash
# 深度学习示例
cargo run --example deep_learning

# NLP示例
cargo run --example nlp_pipeline

# 计算机视觉示例
cargo run --example computer_vision

# 向量搜索示例
cargo run --example vector_search
```

## 学习路径

### 初学者（0-6个月）

**阶段1：基础（1-2个月）**：

1. **机器学习基础**：02_机器学习库（Linfa、SmartCore）
   - 学习线性回归、逻辑回归、决策树
   - 掌握数据预处理和特征工程
   - 完成基础分类和回归项目
   - **验收标准**：在公开数据集上达到基准准确率

2. **数据处理**：05_数据处理（Polars、DataFusion）
   - 学习DataFrame操作和查询
   - 掌握数据清洗和转换
   - 完成数据分析项目
   - **验收标准**：处理1GB+数据集，性能优于Pandas

**阶段2：进阶（3-4个月）**：

3. **深度学习**：01_深度学习框架（Candle、Burn）
   - 学习神经网络基础
   - 掌握CNN、RNN、Transformer
   - 完成图像分类和文本生成项目
   - **验收标准**：训练CNN模型，准确率>90%

4. **NLP应用**：03_自然语言处理（llm-chain、rust-bert）
   - 学习文本分类、NER、问答系统
   - 掌握LLM应用开发
   - 完成NLP管道项目
   - **验收标准**：构建RAG系统，引用率>80%

**阶段3：应用（5-6个月）**：

5. **计算机视觉**：04_计算机视觉（OpenCV、image）
   - 学习图像处理和目标检测
   - 完成视觉应用项目
   - **验收标准**：实现目标检测，mAP>0.7

6. **向量搜索**：06_向量搜索（Qdrant、Tantivy）
   - 学习语义搜索和RAG系统
   - 完成检索增强生成项目
   - **验收标准**：检索延迟<50ms，召回率>85%

### 中级开发者（6-12个月）

**阶段4：深度（7-8个月）**：

1. **深度学习框架**：01_深度学习框架（深入）
   - 掌握分布式训练
   - 学习模型优化和量化
   - 完成大规模训练项目
   - **验收标准**：分布式训练，线性扩展到10+节点

2. **新兴技术**：07_新兴技术（多模态、联邦学习）
   - 学习多模态AI技术
   - 掌握边缘AI部署
   - 完成创新应用项目
   - **验收标准**：边缘推理延迟<300ms

**阶段5：优化（9-10个月）**：

3. **性能优化**：08_性能优化
   - 学习编译优化、内存优化、并行优化
   - 掌握性能分析和调优
   - 完成性能优化项目
   - **验收标准**：性能提升30%+，建立性能基线

4. **生产部署**：10_生产部署
   - 学习容器化、Kubernetes部署
   - 掌握监控和运维
   - 完成生产级项目
   - **验收标准**：部署到K8s，SLO达标率>99%

**阶段6：安全（11-12个月）**：

5. **安全与隐私**：09_安全与隐私
   - 学习模型安全、数据安全
   - 掌握隐私保护技术
   - 完成安全合规项目
   - **验收标准**：通过安全审计，符合GDPR要求

### 高级开发者（12个月以上）

**阶段7：研究（13-15个月）**：

1. **新兴技术研究**：07_新兴技术（深入）
   - 研究前沿AI技术
   - 探索创新应用场景
   - 发表研究成果
   - **验收标准**：发表论文或开源项目获得100+ stars

2. **深度学习框架**：01_深度学习框架（研究级）
   - 研究新架构和算法
   - 优化框架性能
   - 贡献开源项目
   - **验收标准**：贡献被主流框架采纳

**阶段8：架构（16-18个月）**：

3. **生产部署架构**：10_生产部署（架构设计）
   - 设计大规模AI系统架构
   - 优化系统性能和可扩展性
   - 领导技术团队
   - **验收标准**：设计支持1000+ QPS的系统架构

4. **性能优化架构**：08_性能优化（系统级）
   - 设计性能优化方案
   - 建立性能优化体系
   - 指导团队优化实践
   - **验收标准**：建立性能优化知识库，指导团队

**阶段9：创新（19个月以上）**：

5. **技术创新**：07_新兴技术 + 09_安全与隐私
   - 探索新技术方向
   - 创新应用场景
   - 推动行业发展
   - **验收标准**：推动行业标准或最佳实践

**2025年11月更新：推理优化优先**

根据2025年11月趋势，学习路径重点调整：

- **推理优化**：增加推理优化课程，量化、缓存、路由成为重点
- **边缘部署**：增加边缘AI部署实践，边缘部署率目标40%
- **成本优化**：增加成本优化课程，推理成本降低30-50%

**参考**：详见 `docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md` §"2025年11月最新趋势更新"

## 贡献指南

### 添加新库

1. 在相应主题文件夹中添加库信息
2. 提供示例代码和文档
3. 更新本README文件

### 改进现有内容

1. 更新库版本信息
2. 添加新的示例代码
3. 改进文档和说明

### 报告问题

1. 在GitHub上创建Issue
2. 提供详细的错误信息
3. 包含复现步骤

## 相关资源

- [Rust AI生态](https://github.com/rust-ai/awesome-rust-ai)
- [机器学习最佳实践](https://github.com/rust-ai/ml-best-practices)
- [性能优化指南](https://github.com/rust-ai/performance-guide)
- [安全最佳实践](https://github.com/rust-ai/security-guide)

## 更新日志

### 2025年1月

- 初始版本发布
- 包含10个主要主题
- 支持Rust 1.90
- 集成最新AI/ML库

### 2025年11月

- **推理优化优先**：补充推理优化相关内容，量化、缓存、路由成为重点
- **边缘部署**：增加边缘AI部署实践和案例
- **成本优化**：补充成本优化策略和实践
- **学习路径更新**：调整学习路径，增加推理优化和边缘部署内容，包含详细阶段划分和验收标准

**详细更新**：详见 `docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md` §"2025年11月最新趋势更新"

---

**注意**: 本文件夹内容会持续更新，请定期查看最新版本。如有问题或建议，欢迎提交Issue或Pull Request。
