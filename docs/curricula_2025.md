# 名校课程对齐（2024-2025）与能力地图

本文件将 Stanford/MIT/Berkeley/CMU 等课程主题映射到能力项，并提供 Rust 实践锚点。

## 目录

- [1. 课程主题 → 能力项](#1-课程主题--能力项)
- [2. Rust 实践锚点](#2-rust-实践锚点)
- [3. 推荐学习顺序（可与计划联动）](#3-推荐学习顺序可与计划联动)

## 1. 课程主题 → 能力项

- Transformers/LLM 前沿（如 Stanford CS25）：模型与推理、微调与对齐、多模态进展
- 深度学习系统（Berkeley、CMU 系统方向）：分布训练、存算优化、编译与加速
- 负责任的 AI（多校开设）：评测、安全与合规、红队与越狱防护
- 数据系统与工程（Berkeley Data/Systems）：数据治理、流批一体、检索系统

## 2. Rust 实践锚点

- 模型推理：`candle`/`onnxruntime` 搭建推理服务（`axum`）
- RAG 系统：`tantivy` + `qdrant` 客户端 + `reqwest` 工具调用
- 可观察性：`tracing` + `opentelemetry` 采集指标
- 工程化：`tokio` 异步、`anyhow` 错误、`thiserror` 自定义错误

## 3. 推荐学习顺序（可与计划联动）

1) 基础数学与概率 → 深度学习导论
2) Transformers/LLM → RAG 与工具调用 → 多代理
3) 系统与工程 → 部署与可观察性 → 安全与评测
