# Rust × AI 生态全景（2025-09）

## 目录

- [Rust × AI 生态全景（2025-09）](#rust--ai-生态全景2025-09)
  - [目录](#目录)
  - [1. 推理与训练](#1-推理与训练)
    - [1.1 框架对比](#11-框架对比)
    - [1.2 选型指南](#12-选型指南)
  - [2. 科学计算与数据](#2-科学计算与数据)
    - [2.1 数据处理栈](#21-数据处理栈)
    - [2.2 模型格式与工具](#22-模型格式与工具)
    - [2.3 性能优化策略](#23-性能优化策略)
  - [3. 系统与工程](#3-系统与工程)
  - [4. RAG/检索](#4-rag检索)
  - [5. 部署与集成](#5-部署与集成)
  - [6. 实操建议](#6-实操建议)
  - [7. 岗位技能矩阵](#7-岗位技能矩阵)
  - [8. 行业案例速览](#8-行业案例速览)
  - [9. 最小技能闭环（从零到上线）](#9-最小技能闭环从零到上线)

## 1. 推理与训练

### 1.1 框架对比

| 框架 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| `candle` | 轻量、HuggingFace生态、易用 | 功能相对简单 | 快速原型、推理服务 |
| `burn` | 模块化、多后端、类型安全 | 学习曲线陡峭 | 研究、自定义架构 |
| `tch-rs` | PyTorch兼容、功能完整 | 依赖PyTorch C++ | 模型迁移、研究 |
| `onnxruntime` | 跨平台、优化推理 | 训练支持有限 | 生产部署 |
| `llama.cpp` | 极致优化、量化支持 | 仅推理 | 边缘设备、本地部署 |

### 1.2 选型指南

**推理优先**：

- 生产环境：`onnxruntime` + 模型转换
- 本地部署：`candle` + 量化模型
- 边缘设备：`llama.cpp` + INT4量化

**训练优先**：

- 快速实验：`candle` + 简单模型
- 复杂架构：`burn` + 自定义后端
- PyTorch迁移：`tch-rs` + 现有代码

**混合场景**：

- 研究到生产：`burn`训练 → `onnxruntime`推理
- 全栈开发：`candle`统一训练推理

## 2. 科学计算与数据

### 2.1 数据处理栈

| 库 | 功能 | 性能特点 | 使用场景 |
|----|------|----------|----------|
| `ndarray` | 多维数组、广播 | 内存高效、SIMD优化 | 数值计算、机器学习 |
| `nalgebra` | 线性代数、几何 | 类型安全、编译时优化 | 3D图形、机器人学 |
| `polars` | 数据分析、查询 | 列式存储、并行处理 | 大数据分析、ETL |
| `linfa` | 机器学习工具包 | 模块化、可扩展 | 传统ML算法 |

### 2.2 模型格式与工具

- `safetensors`：安全张量格式，支持零拷贝加载
- `tokenizers`：高性能分词器，支持多种算法
- `candle-datasets`：数据集加载与预处理
- `huggingface-hub`：模型下载与管理

### 2.3 性能优化策略

```rust
// 并行数据处理
use rayon::prelude::*;

let processed_data: Vec<f64> = raw_data
    .par_iter()
    .map(|x| expensive_computation(x))
    .collect();

// 内存映射大文件
use memmap2::Mmap;
let file = File::open("large_dataset.bin")?;
let mmap = unsafe { Mmap::map(&file)? };
let data = &mmap[..];

// 零拷贝张量操作
use candle_core::{Tensor, Device};
let tensor = Tensor::from_slice(&data, (rows, cols), &Device::Cpu)?;
```

## 3. 系统与工程

- Web/服务：`axum`、`actix-web`、`tower`
- 异步与并发：`tokio`、`rayon`
- 观测：`tracing`、`opentelemetry`

## 4. RAG/检索

- 向量与全文：`qdrant`/`milvus` 客户端、`tantivy`
- 数据处理：`serde`、`reqwest`、`sqlx`

## 5. 部署与集成

- `wasm`、`pyo3`、`maturin`
- GPU 加速：`cuda`/`wgpu` 生态

## 6. 实操建议

> 先以 `candle + axum + qdrant` 搭建最小 RAG 服务，再迭代多代理与评测。

## 7. 岗位技能矩阵

| 岗位 | 必备 | 加分 | 产出指标 |
|---|---|---|---|
| 算法工程师 | ML/DL 基础、LLM 微调、评测指标 | 蒸馏/量化、对齐（DPO/RLHF） | 精度、鲁棒性、数据效率 |
| 系统工程师 | `axum`/`tokio`、缓存/批处理、Tracing | GPU 调度、FSDP/ZeRO、Wasm | 延迟(P50/P95)、QPS、稳定性 |
| 检索/RAG 工程师 | `tantivy`/向量DB、重排序、提示工程 | 混合检索、查询重写、路由 | 召回/精排指标、命中率 |
| 多代理工程师 | 规划/执行/回顾、消息中间件 | 记忆/工具调用、评测基准 | 任务成功率、成本 |

## 8. 行业案例速览

- 内容生成：本地化 GPTQ/INT4 推理 + 缓存层，千并发短文本生成
- 企业知识助手：RAG（混合检索+重排序）+ 私有化部署 + 审计追踪
- 工业质检：多模态推理（视觉+文本）+ 边缘部署（`wgpu`）
- 科研助理：论文检索/归纳 + 代理式工作流 + 可解释报告

## 9. 最小技能闭环（从零到上线）

1. 模型推理：`candle` 加载量化权重、Top-k/Top-p 采样
2. 服务化：`axum` 暴露 `/generate` 与 `/embed`，加入批处理与限流
3. 检索增强：`qdrant`/`tantivy` 混合检索，重排序与缓存命中
4. 观测性：`tracing` 指标、`opentelemetry` 上报、错误预算
5. 部署：Docker 镜像 + 健康检查 + 灰度发布 + 基准与告警
