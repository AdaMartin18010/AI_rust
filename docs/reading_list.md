# 分级阅读清单与论文优先级（2025）

## 目录

- [入门（能跑起来）](#入门能跑起来)
- [进阶（会调优）](#进阶会调优)
- [专业（能复现/能改造）](#专业能复现能改造)
- [论文快线（90天）](#论文快线90天)

## 入门（能跑起来）

### 必读书籍（按优先级）

1. **《深度学习》Goodfellow**
   - 第1章：引言（1天）
   - 第2章：线性代数（3天，配合代码实现）
   - 第3章：概率与信息论（3天）
   - 第4章：数值计算（2天）
   - 第5章：机器学习基础（4天）
   - 第6章：深度前馈网络（5天）
   - **总计：18天，每天2-3小时**

2. **《统计学习方法》李航**
   - 第1章：统计学习及监督学习概论（2天）
   - 第2章：感知机（1天，实现算法）
   - 第3章：k近邻法（1天）
   - 第4章：朴素贝叶斯法（2天）
   - 第5章：决策树（2天，实现ID3/C4.5）
   - 第6章：逻辑斯谛回归与最大熵模型（3天）
   - **总计：11天，每天2小时**

### 在线课程（精选章节）

1. **CS229 机器学习**
   - 第1-2讲：线性回归、逻辑回归（3天）
   - 第3-4讲：广义线性模型、生成学习算法（3天）
   - 第5-6讲：支持向量机、核方法（4天）
   - **总计：10天，每天1.5小时**

2. **CS231n 计算机视觉**
   - 第1-2讲：图像分类、线性分类器（2天）
   - 第3-4讲：神经网络、反向传播（4天）
   - 第5-6讲：卷积神经网络（4天）
   - **总计：10天，每天2小时**

### 实践项目

- **Week 1-2**：实现线性回归、逻辑回归（从零开始）
- **Week 3-4**：实现简单神经网络（2层MLP）
- **Week 5-6**：实现CNN进行图像分类（CIFAR-10）

## 进阶（会调优）

- 书籍：
  - 《Pattern Recognition and Machine Learning》（选读）
  - 《Convex Optimization》Boyd（选读）
- 教程/课程：
  - CS224n（NLP）关键讲座
  - Berkeley CS285（DRL）选讲

## 专业（能复现/能改造）

- 书籍/专著：
  - 《Graph Representation Learning》
  - 《Information Theory, Inference, and Learning Algorithms》
- 实践：
  - LLM 微调手册（QLoRA/LoRA）
  - 推理优化与量化工程指南

## 论文快线（90天）

### 第1-30天：基础架构

**Week 1-2：Transformer基础**:

- [ ] "Attention Is All You Need" (Vaswani et al., 2017)
  - 动机：解决RNN序列建模的并行化问题
  - 核心：自注意力机制、多头注意力、位置编码
  - 实现：从零实现Transformer编码器
  - 验收：在简单序列任务上达到预期性能

**Week 3-4：预训练语言模型**:

- [ ] "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- [ ] "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et al., 2019)
  - 对比：双向vs单向、掩码vs自回归
  - 实现：简化版BERT/GPT训练脚本
  - 验收：在小数据集上复现预训练效果

### 第31-60天：微调与对齐

**Week 5-6：参数高效微调**:

- [ ] "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- [ ] "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
  - 核心：低秩分解、量化感知训练
  - 实现：LoRA微调脚本，支持多种任务
  - 验收：在指令跟随任务上达到全参数微调80%性能

**Week 7-8：人类反馈对齐**:

- [ ] "Training language models to follow instructions with human feedback" (InstructGPT, Ouyang et al., 2022)
- [ ] "Direct Preference Optimization" (DPO, Rafailov et al., 2023)
  - 对比：RLHF vs DPO的优缺点
  - 实现：DPO训练pipeline
  - 验收：在偏好数据集上提升对齐质量

### 第61-90天：应用与优化

**Week 9-10：检索增强生成**:

- [ ] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG, Lewis et al., 2020)
- [ ] "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
  - 核心：密集检索、生成器-检索器联合训练
  - 实现：端到端RAG系统
  - 验收：在知识问答任务上显著提升准确性

**Week 11-12：多模态理解**:

- [ ] "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, Radford et al., 2021)
- [ ] "BLIP: Bootstrapping Language-Image Pre-training" (Li et al., 2022)
  - 核心：对比学习、跨模态对齐
  - 实现：简化版CLIP训练
  - 验收：在图像-文本匹配任务上达到合理性能

### 执行法：每篇配"动机→方法→公式→伪码→实验→可复现脚本"

**论文阅读模板**：

1. **摘要理解**（15分钟）：核心贡献、方法概述
2. **动机分析**（20分钟）：问题定义、现有方法局限
3. **方法推导**（45分钟）：数学公式、算法伪码
4. **实验分析**（30分钟）：数据集、指标、结果解读
5. **代码实现**（2-4小时）：核心算法、关键组件
6. **实验验证**（1-2小时）：复现关键结果、性能对比
