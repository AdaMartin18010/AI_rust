# 2025 年 AI 知识分类与架构（对标 2025-09）

本文件对齐 2025 年 9 月产业与学术共识，聚焦四层视角：能力层、系统层、工程层、治理与评测层，并提供论证与形式化提示，供持续更新。

## 1. 能力层（Models & Capabilities）

- 基座模型（Foundation Models）：LLM、VLM、音频/语音、多模态统一模型（Text-Image-Audio-Video-Action）。
- 代理式系统（Agentic Systems）：检索增强、工具使用、代码执行、规划/反思、自主任务分解。
- 推理与数学（Reasoning & Math）：链式思维、图式思维、程序合成、证明辅助（Lean/Isabelle 接口）。
- 个性化与小型化：轻量模型（数十亿参数级）、蒸馏、量化、SFT/DPO/ORPO/GRPO。

形式化提示：将模型行为表述为策略函数 π(s) → a，证据通过可重复基准（如 MATH、GSM8K、MMMU、AgentBench）给出统计显著差异。

## 2. 系统层（Systems & Architectures）

- RAG 架构：索引（向量/图/混合）、检索（BM25+ANN）、重排、结构化上下文、缓存与路由。
- 工具与执行：程序执行沙盒、语言/图数据库、工作流编排（DAG/状态机）。
- 多代理协作：角色分工、协同协议、冲突解决、黑板/市场机制。
- 在线学习与反馈：偏好优化回路、人类与合成反馈、对齐与安全控制。

论证要点：以端到端任务成功率/时延/成本三指标进行 Pareto 比较；对关键组件做消融实验。

## 3. 工程层（Engineering & Ops）

- 数据工程：数据治理、合规与可追溯、合成数据管线、去重与毒性过滤。
- 训练与微调：高效优化器、混合并行、LoRA/QLoRA、对齐微调。
- 部署与服务：CPU/GPU/NPU 异构、ONNX/TensorRT、低时延推理（KV Cache/Speculative）。
- 可观察性：日志/分布追踪/漂移监测/评测即服务（EaaS）。

形式证明提示：将部署 SLA 定义为 (p99 延迟, 可用性, 误差界)，用排队论与资源模型估计容量上界。

## 4. 治理与评测（Safety, Governance, Evaluation）

- 安全与对齐：越狱防护、红队测试、能力防护栏、隐私与合规（GDPR/CCPA）。
- 评测：静态基准 + 动态任务（合成与真实）、过程评测（过程正确性与引用完整性）。
- 负责 AI：透明度、可解释性、来源标注（Provenance）、版权与许可。

## 5. 对应 Rust 实践（简表）

- 推理：`candle`、`burn`、`onnxruntime`、`tch-rs`、`llama.cpp` 绑定
- RAG：`tantivy`/`qdrant`/`milvus` 客户端、`polars`、`serde`、`reqwest`
- 服务：`axum`/`actix-web`、`tokio`、`tracing`、`opentelemetry`
- 工具：`tokenizers`、`safetensors`、`pyo3`、`wasm-pack`

更新节奏：每月一次快照；出现重要论文或生态更新时即时补充。
