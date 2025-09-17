# AI_rust

## 1. GPT模型

[RustGPT]<https://github.com/tekaratzas/RustGPT>
项目复刻了整个LLM的功能链，从预训练、指令微调，一直到你能跟它聊天的交互模式，全给办了。架构非常简单：

整个项目设计得跟乐高积木一样，模块化特别清晰：src/main.rs：总指挥部，负责整个训练流程、数据怎么处理、以及最后怎么跟你聊天。

src/llm.rs：核心引擎，大模型的前向传播、反向传播、训练逻辑全在这。
src/transformer.rs：Transformer的核心部件，把注意力和前馈网络这两个左膀右臂组装起来。
src/self_attention.rs：大名鼎鼎的多头自注意力机制，模型“理解”上下文关系就靠它。
src/feed_forward.rs：位置感知前馈网络，让模型处理信息更有层次。
src/embeddings.rs：词嵌入层，把文字转化成模型能懂的数学语言。
src/output_projection.rs：输出投影层，把模型计算出的数学结果再翻译成人类语言。
src/vocab.rs：字典和分词器，模型的“新华字典”。
src/layer_norm.rs：层归一化，防止模型训练时“飘了”的稳定器。
src/adam.rs：Adam优化器，指导模型如何“学习”得又快又好。

模型的配置参数非常“迷你”，但五脏俱全，能解剖一个完整的大模型。
它的成长之路也跟主流模型一样，分两步走：预训练阶段先给它“喂”一堆事实性陈述，让它学习世界的基本规律。
比如告诉它“太阳东升西落”、“水往低处流”这类知识。
指令微调阶段教它学会“对话”。
用一问一答的语料库来训练，让它明白人类是如何交流的。
比如“用户：山是怎么形成的？
助手：山是地壳板块运动或火山活动形成的……”等这两步都走完，这个纯Rust打造的模型就能跟你进行简单的常识问答了.

## 2. 推理引擎

推理引擎
[mistral.rs](https://github.com/EricLBuehler/mistral.rs)

Hugging Face的
[candle框架](https://github.com/huggingface/candle)

## 3. RUST的使用框架

rustassistant-using-llms-to-fix-compilation-errors-in-rust-code
<https://www.microsoft.com/en-us/research/publication/rustassistant-using-llms-to-fix-compilation-errors-in-rust-code>
<https://www.amazon.science/publications/verifying-dynamic-trait-objects-in-rust>
<https://cjwebb.com/aws-bedrock-with-rust>
<https://dev.to/hamzakhan/vs-rust-vs-python-the-ultimate-showdown-of-speed-and-simplicity-for-2024-2afi>
<https://www.pullrequest.com/blog/rust-safety-writing-secure-concurrency-without-fear>

## 4. RUST的AI框架扩展

学习使用的分类和路径
rust 1.89版本 2025年最新的特性
以及最成熟稳定的开源库

---

## 快速导航与学习路径（2025-09 对齐）

- 文档总览：
  - `docs/taxonomy_2025.md`：2025 年 AI 分类与架构（含论证/形式化提示）
  - `docs/curricula_2025.md`：名校课程对齐与能力地图
  - `docs/foundations.md`：数学/计算机/科学基础知识框架
  - `docs/ai_rust_landscape.md`：Rust × AI 生态全景与落地建议
  - `docs/research_watch.md`：递归迭代的检索与更新机制

- 计划与推进：
  - `plans/MASTER_PLAN.md`：主计划与里程碑
  - `plans/SPRINT_01.md`：第一阶段子计划（两周）

- 课程与实践：
  - `courses/rust/SYLLABUS.md`：Rust 基础→进阶大纲
  - `courses/ai-with-rust/SYLLABUS.md`：AI × Rust 实战大纲

- 目录占位：
  - `src/`：实战源码
  - `reports/`：基准与测试报告
  - `notes/`：经验记录与问题清单

> 建议从 `docs/taxonomy_2025.md` 开始阅读，然后按 `plans/SPRINT_01.md` 执行，完成功能的同时在 `docs/research_watch.md` 流程下持续补充资料。

### 运行最小服务

```bash
# Linux/macOS
cargo run
# 另一个终端
curl http://127.0.0.1:8080/healthz
curl -X POST http://127.0.0.1:8080/infer -H "Content-Type: application/json" -d '{"prompt":"hello"}'
```

- Windows PowerShell 提示：不要用 `&&` 链接命令，改为分行执行。

### 测试

```bash
cargo test
```

- 集成测试会在测试进程内启动一个临时端口的 axum 服务（见 `tests/http_smoke.rs`），无需本地先行运行可执行文件。
- 若本机有系统代理/HTTP 代理（如 Privoxy、公司代理），可能导致本地环回请求被劫持。我们的测试使用 `reqwest::Client::builder().no_proxy()` 显式禁用代理。

### 代码结构

- `src/lib.rs`：导出 `create_app()`，封装路由与状态
- `src/engine.rs`：推理引擎 trait `InferenceEngine` 与默认实现 `DummyEngine`
- `src/main.rs`：仅负责引导（日志、监听、`axum::serve`）
- `tests/http_smoke.rs`：在进程内启动服务进行 HTTP 冒烟测试

### 可替换推理引擎

- 引擎抽象：`src/engine.rs` 中定义 `InferenceEngine`
- 默认实现：`DummyEngine`（回显）
- 替换方式：实现该 trait 并在 `create_app()` 中注入自定义引擎（或暴露构造函数以供 main 注入）

### 配置与运维提示

- 监听端口：支持环境变量 `PORT`，默认 `8080`
- CORS：默认放开 `origin/method/headers`，如需收紧可在 `src/lib.rs` 的 `CorsLayer` 中修改
- 就绪/存活：新增 `GET /readyz` 与 `GET /healthz`
- 优雅关停：支持 `Ctrl+C`（Windows）/ `SIGINT`、`SIGTERM`（Unix）
