# Rust 课程大纲（基础 → 进阶）

> 快速跳转：
>
> - 文档导航：`docs/ai_learning_overview.md`
> - 学习路径：`docs/ai_learning_path.md`
> - 数学基础：`docs/ai_mathematical_foundations.md`
> - 算法解析：`docs/ai_algorithms_deep_dive.md`
> - 实践指南：`docs/rust_ai_practice_guide.md`
> - 阅读清单：`docs/reading_list.md`

## 基础

- 所有权与借用、生命周期
- 模式匹配、`Result`/`Option` 错误边界
- 模块、`trait`、泛型

## 并发与异步

- `Send`/`Sync` 与安全并发
- `tokio` 基础、异步 I/O、超时与重试

## 工程

- 组织结构、`cargo` 工作区、测试/基准
- 观测：`tracing`、日志结构化

## 数据与计算

- `ndarray`/`nalgebra`、`polars`
- FFI：`pyo3` 与 `wasm` 简介

## 实战里程碑

- CLI 工具、HTTP 服务、序列化（`serde`）
- 小结与综合练习
