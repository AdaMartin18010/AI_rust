# 2025 AI-Rust 全局术语表（GLOSSARY）

> 说明：本术语表为跨文档统一引用来源，包含定义、属性与关系提示。采用分层（Meta/Domain/Application/Implementation）标注与证据等级（A/B/C）。

## 术语结构模板

- 名称（英文/中文）
- 抽象层级：Meta | Domain | Application | Implementation
- 定义：精准、可操作
- 关键属性：可度量或可验证
- 常见关系：IsA/Uses/DependsOn/Enables/Optimizes 等
- 证据等级：A（强）/B（中）/C（弱）

- 别名（Aliases）：常见同义词/缩写，指定主词条
- 计量口径（Metrics Scope）：涉及指标的单位/采样窗口/统计方法
- DAR 卡片最小化：Definition｜Attributes（单位/口径）｜Relations（类型+强度）
- 交叉引用：指向权威框架/知识框架/实践指南/趋势附录Z的锚点
- 复现脚本（如适用）：关联 `scripts/*` 与示例参数

---

## 核心术语

### 多模态AI（Multimodal AI）

- 抽象层级：Domain
- 定义：联合处理文本/图像/音频/视频并在统一空间进行表示与推理的系统。
- 关键属性：表示对齐度、模态覆盖率、融合策略、鲁棒性、延迟、能效。
- 常见关系：Uses(Transformer)、DependsOn(跨模态注意力)、Enables(Agentic Web)。
- 证据等级：A

- 别名：MMAI、多模态
- 计量口径：
  - 延迟：P50/P95/P99（t-digest，窗口≥5min）
  - 能效：tokens/J（排除冷启动）
  - 覆盖：模态数与任务覆盖度
- DAR：Definition｜Attributes{对齐度、覆盖、延迟、能效}｜Relations{Uses,DependsOn,Enables}
- 交叉引用：实践§0；知识附录Y；趋势§Z.6/§Z.7

反例与边界：

- 反例：模态间标签不一致导致对齐失败；
- 边界：端侧功耗与带宽限制，跨模态长序列导致P99激增。

### 代理式系统（Agentic Systems）

- 抽象层级：Application
- 定义：具备感知-推理-规划-行动-记忆闭环，能自主分解与执行任务的AI系统。
- 关键属性：目标导向性、工具使用能力、反思/纠错、协作协议、可追踪性。
- 常见关系：Uses(RAG/工具执行)、DependsOn(知识库/环境接口)、Enables(自动化运营)。
- 证据等级：B

- 别名：Agentic、代理系统
- 计量口径：任务完成率、越权率、预算违规率、P95延迟
- DAR：Definition｜Attributes{权限、预算、可解释轨迹}｜Relations{Uses,DependsOn,Enables}
- 复现脚本：Pareto 路由示例 `scripts/bench/run_pareto.sh`（参数见README）
- 交叉引用：实践§5；趋势§Z.8

反例与边界：

- 反例：工具未设权限导致越权；
- 边界：预算护栏触发中断，复杂回路导致可解释性下降。

### 知识增强生成（KAG, Knowledge Augmented Generation）

- 抽象层级：Application
- 定义：将检索/知识库与生成式模型融合以提升真实性、时效性与可解释性的方法族。
- 关键属性：检索质量、上下文构建、来源追踪、验证环、事实一致性。
- 常见关系：Uses(Retrieval/Ranking)、Optimizes(事实准确率)。
- 证据等级：A

- 别名：RAG（检索增强生成）
- 计量口径：recall、NDCG、citation_rate、coverage、端到端P95/成本/查询
- DAR：Definition｜Attributes{引用率、覆盖、延迟、成本}｜Relations{Uses,Optimizes}
- 复现脚本：`scripts/rag/eval_hybrid.sh`/`.ps1`
- 交叉引用：实践§4；知识附录Y；趋势§Z.7/§Z.11

反例与边界：

- 反例：低质量索引导致引用率下降；
- 边界：K/K' 过大引发延迟与成本不可接受。

### 稀疏专家模型（MoE）

- 抽象层级：Implementation
- 定义：通过门控机制在推理时仅激活部分专家子网络的可扩展架构。
- 关键属性：激活稀疏度、专家容量、路由稳定性、吞吐/延迟。
- 常见关系：Optimizes(吞吐量)、DependsOn(高效路由/并行通信)。
- 证据等级：A

- 别名：Mixture-of-Experts
- 计量口径：吞吐（QPS/TPM）、延迟P95、AllToAll带宽占比
- DAR：Definition｜Attributes{稀疏度、容量、稳定性}｜Relations{Optimizes,DependsOn}
- 交叉引用：趋势§Z.6；实践§2.1

反例与边界：

- 反例：路由不稳定导致退化；
- 边界：AllToAll 通信瓶颈与显存碎片。

### Agentic Web（代理化Web）

- 抽象层级：Application
- 定义：由自主AI代理驱动的Web交互生态，代理可在浏览器/服务端安全执行复杂任务。
- 关键属性：权限边界、可解释性、任务可恢复性、跨模态交互。
- 常见关系：Enables(自动化工作流)、DependsOn(WebAPI/安全策略)。
- 证据等级：B

### 边缘AI（Edge AI）

- 抽象层级：Application
- 定义：在靠近数据源的设备侧进行AI推理与部分学习，以降低延迟与成本、提升隐私。
- 关键属性：模型大小、量化精度、能耗、冷启动时延、联邦能力。
- 常见关系：Uses(WebAssembly/NPU)、Optimizes(延迟/带宽)。
- 证据等级：A

### 可观测性（Observability for AI）

- 抽象层级：Implementation
- 定义：以指标/日志/追踪三层协同对AI系统行为进行端到端可视化与可验证。
- 关键属性：端到端追踪比例、指标粒度、异常检测召回率、告警时效。
- 常见关系：Requires(Tracing/OTel)、Optimizes(可靠性/MTTR)。
- 证据等级：A

---

引用规范：各文档首次出现术语以粗体标注，并在末尾“附录F/术语与交叉引用”处链接此表。

## 策略与治理相关术语（新增）

### 路由策略（Routing Strategy）

- 抽象层级：Application
- 定义：基于任务类型、上下文长度、预算/SLO 将请求分配给不同模型或推理路径的策略（如阈值、多臂赌博、MoE门控）。
- 关键属性：分流规则、探索率、稳定性、质量/成本弹性。
- 常见关系：Optimizes(成本/延迟/质量权衡)、DependsOn(指标与预算信号)。
- 证据等级：A

- 别名：模型路由、能力路由
- 计量口径：P95、$/1k tok、一致性率、失败切换率
- DAR：Definition｜Attributes{分流、探索、弹性}｜Relations{Optimizes,DependsOn}
- 复现脚本：`scripts/bench/run_pareto.*`
- 交叉引用：实践§2.1；趋势§Z.8

### 一致性筛选（Consistency Filtering）

- 抽象层级：Implementation
- 定义：对并行生成的多个候选进行一致性评估并保留高置信输出的方法（如自洽/交叉一致）。
- 关键属性：候选数n、一致性阈值、代价、质量提升Δ。
- 常见关系：Optimizes(一致性/事实性)、Increases(成本/延迟)。
- 证据等级：B

- 别名：一致性选择、自洽
- 计量口径：一致率、事实性、端到端P95/成本
- DAR：Definition｜Attributes{n、阈值、Δ}｜Relations{Optimizes,Increases}
- 交叉引用：实践§2.1/§0.10

### 预算护栏（Budget Guardrail）

- 抽象层级：Application
- 定义：在会话/租户/全局层面对时间/金钱/能耗/令牌设定上限并触发限流/降级/中断的策略集合。
- 关键属性：配额、预扣/结算、告警阈值、违规处理。
- 常见关系：Prevents(预算超支)、Requires(观测与策略引擎)。
- 证据等级：A

- 别名：成本护栏、预算限额
- 计量口径：budget_used、budget_violation、$/1k tok、tokens/J
- DAR：Definition｜Attributes{配额、阈值、降级}｜Relations{Prevents,Requires}
- 交叉引用：实践§3.2/§7.1/§7.4

### 追踪ID传播（Trace ID Propagation）

- 抽象层级：Implementation
- 定义：在分布式系统中将 trace-id/tracestate 跨服务/线程/异步边界透传以实现端到端追踪与审计。
- 关键属性：覆盖率、采样决策、跨边界丢失率、隐私与合规。
- 常见关系：Requires(OTel/Tracing)、Enables(审计/回放)。
- 证据等级：A

- 别名：traceparent 传播、分布式追踪
- 计量口径：覆盖率%、丢失率%、关键路径标注率
- DAR：Definition｜Attributes{覆盖、采样、丢失}｜Relations{Requires,Enables}
- 交叉引用：实践§7.3/§0.10；趋势§Z.11

## 索引（按层级）

- Meta：—
- Domain：多模态AI
- Application：Agentic、KAG、边缘AI、路由策略、金丝雀发布、预算护栏
- Implementation：MoE、可观测性、KV缓存、量化、重排序、一致性筛选、追踪ID传播

### KV缓存（KV Cache）

- 抽象层级：Implementation
- 定义：在自回归解码中缓存注意力键/值以减少重复计算、降低延迟和成本的机制。
- 关键属性：命中率、失效策略、跨请求复用、持久化、一致性风险。
- 常见关系：Optimizes(解码延迟/成本)、DependsOn(内存/存储带宽)。
- 证据等级：A

- 别名：Key-Value Cache、解码缓存
- 计量口径：
  - 延迟：P50/P95/P99（分解码阶段）
  - 命中率：按token或step统计
  - 成本：$/1k tok（含存储/序列化分摊）
- DAR：Definition｜Attributes{命中率、复用、持久化、风险}｜Relations{Optimizes,DependsOn}
- 交叉引用：实践§2.1/§3；趋势§Z.7；术语：量化、路由

反例与边界：

- 反例：跨请求复用导致陈旧上下文污染输出；
- 边界：一致性与隐私约束下的持久化策略。

### 量化（Quantization）

- 抽象层级：Implementation
- 定义：将模型权重/激活从高精度（FP16/FP32）映射到低比特（INT8/INT4）以提升吞吐、降低成本的方法族。
- 关键属性：位宽、对称/非对称、分组/通道级、校准集、精度回退Δ。
- 常见关系：Optimizes(延迟/能效/显存)、Affects(精度)。
- 证据等级：A

- 别名：INT8、INT4、量化感知/后训练量化（QAT/PTQ）
- 计量口径：
  - 能效：tokens/J（排除冷启动）
  - 精度回退：任务指标Δ（含置信区间）
  - 显存峰值：峰值/平均
- DAR：Definition｜Attributes{位宽、方案、校准、Δ}｜Relations{Optimizes,Affects}
- 复现脚本：`scripts/bench/run_pareto.*`
- 交叉引用：实践§2.2/§0.10；趋势§Z.7/§Z.8

反例与边界：

- 反例：极端任务上精度回退不可接受；
- 边界：位宽/对称性/分组策略错配导致数稳问题。

### 重排序（Re-ranking）

- 抽象层级：Implementation
- 定义：对初始检索候选进行更精细的相关性打分以提升排序质量的过程（如cross-encoder）。
- 关键属性：K/K'、模型类型、延迟成本、质量收益（NDCG/引用率）。
- 常见关系：Optimizes(检索质量)、Increases(延迟/成本)。
- 证据等级：A

- 别名：Rerank、Cross-Encoder 重排
- 计量口径：NDCG、recall、citation_rate、端到端P95/成本/查询
- DAR：Definition｜Attributes{K/K'、模型、延迟、成本}｜Relations{Optimizes,Increases}
- 复现脚本：`scripts/rag/eval_hybrid.*`
- 交叉引用：实践§4.2/§4.3；知识附录Y；趋势§Z.11

反例与边界：

- 反例：训练分布与线上分布漂移导致收益不显著；
- 边界：延迟预算内的K' 上限。

### 金丝雀发布（Canary Release）

- 抽象层级：Application
- 定义：将新版本以小流量逐步放量验证质量与稳定性，在异常时自动回滚的发布策略。
- 关键属性：流量比例、监控指标阈值、回滚条件、审计留痕。
- 常见关系：Prevents(大规模故障)、Requires(可观测/回滚机制)。
- 证据等级：A

- 别名：金丝雀、灰度发布（含差异）
- 计量口径：
  - 指标：P95/P99、错误率、预算违规率、质量指标（引用率等）
  - 触发：超阈即回滚，记录trace与配置哈希
- DAR：Definition｜Attributes{流量、阈值、回滚、审计}｜Relations{Prevents,Requires}
- 交叉引用：实践§7.4/§0.10；趋势§Z.8；顶层README“复现与报告”

反例与边界：

- 反例：监控覆盖不足导致坏版本放量；
- 边界：小流量下统计功效不足，误判风险升高。
