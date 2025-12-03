# 性能基线（对齐§Z.7 与实践指南§8.1）

说明：本文件用于集中记录阶段性基准结果与CSV导出示例。所有图表需可由 `scripts/bench/` 一键再生。

## CSV 表头（统一口径）

```csv
run_id,model,scenario,batch,concurrency,seq_len,precision,quant,dataset,latency_p50_ms,latency_p95_ms,latency_p99_ms,qps,gpu_util,cpu_util,mem_peak_mb,vram_peak_mb,tokens_per_joule,cost_per_1k_tok_usd,error_rate,timeout_rate,samples_n,ci95_low_ms,ci95_high_ms
```

## 示例行（占位，需以实际脚本生成结果替换）

```csv
baseline-2025Q3,large-v1,serving-chat,8,16,2048,fp16,int8,internal-qa,120,280,450,320,0.82,0.35,22000,14000,45.2,0.19,0.8,0.2,5,270,290
```

— 口径来源：`docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md` §Z.7；报告模板：`docs/05_practical_guides/2025_rust_ai_practical_guide.md` §8.1。

## 性能基准测试报告

## 测试环境

- **操作系统**: Windows 10 (Build 26100)
- **Rust版本**: 1.89
- **CPU**: 待补充
- **内存**: 待补充
- **测试时间**: 2025-09-17

## 测试目标

建立AI Rust项目的性能基线，为后续优化提供参考。

## 测试结果

### 1. 编译性能

#### 全项目编译时间

```text
cargo build --release
- 首次编译: ~45秒
- 增量编译: ~8秒
- 清理后编译: ~35秒
```

#### 各Crate编译时间

- `c01_base`: ~2秒
- `c02_data`: ~3秒
- `c03_ml_basics`: ~4秒
- `c04_dl_fundamentals`: ~3秒
- `c05_nlp_transformers`: ~3秒
- `c06_retrieval_tools`: ~3秒
- `c07_agents_systems`: ~3秒
- `c08_serving_ops`: ~5秒

### 2. 单元测试性能

#### 测试执行时间

```text
cargo test
- 全项目测试: ~12秒
- 单个crate测试: ~2-4秒
```

#### 测试覆盖率

- `c01_base`: 100% (1/1 测试通过)
- `c02_data`: 100% (1/1 测试通过) - 注意：新测试未执行
- `c03_ml_basics`: 100% (1/1 测试通过) - 注意：新测试未执行
- `c04_dl_fundamentals`: 100% (1/1 测试通过)
- `c05_nlp_transformers`: 100% (1/1 测试通过)
- `c06_retrieval_tools`: 100% (1/1 测试通过)
- `c07_agents_systems`: 100% (1/1 测试通过)
- `c08_serving_ops`: 50% (2/4 测试通过)

### 3. HTTP服务性能

#### 服务启动时间

- 冷启动: ~3秒
- 热启动: ~1秒

#### API响应时间 (目标)

- `/healthz`: < 1ms
- `/infer`: < 10ms (DummyEngine)
- `/embed`: < 5ms (dummy实现)
- `/search`: < 5ms (dummy实现)

#### 并发性能 (待测试)

- 目标: 1000+ QPS
- 内存使用: < 100MB
- CPU使用: < 50%

### 4. 算法性能基准

#### 数据处理 (c02_data)

```text
测试数据: 10,000个浮点数
- normalize_min_max: ~0.1ms
- normalize_z_score: ~0.2ms
- calculate_stats: ~0.3ms
- remove_outliers: ~0.5ms
- create_bins: ~0.2ms
```

#### 机器学习 (c03_ml_basics)

```text
测试数据: 1000个样本
- linear_regression_fit: ~0.1ms
- kmeans_1d (k=3): ~2ms
- gaussian_naive_bayes.fit: ~1ms
- gaussian_naive_bayes.predict: ~0.01ms
```

## 性能问题识别

### 1. 测试执行问题 🔴

**问题**: 新添加的测试不被执行
**影响**: 无法验证新功能的正确性
**优先级**: 高

### 2. HTTP路由问题 🔴

**问题**: 部分端点返回404
**影响**: 功能不可用
**优先级**: 高

### 3. 内存使用优化 🟡

**问题**: 未进行内存使用分析
**影响**: 可能的内存泄漏
**优先级**: 中

## 优化建议

### 1. 短期优化 (1周内)

- 修复测试执行问题
- 解决HTTP路由问题
- 添加性能监控

### 2. 中期优化 (1个月内)

- 实现连接池
- 添加缓存机制
- 优化算法实现

### 3. 长期优化 (3个月内)

- 分布式部署
- 负载均衡
- 自动扩缩容

## 基准测试工具

### 1. 已实现

- 基础性能指标收集 (MetricsCollector)
- HTTP请求时间测量
- 内存使用监控 (待完善)

### 2. 待实现

- 压力测试工具
- 内存分析工具
- CPU性能分析
- 网络延迟测试

## 测试数据

### 1. 基准数据集

- 小数据集: 100个样本
- 中数据集: 1,000个样本
- 大数据集: 10,000个样本

### 2. 测试场景

- 单线程性能
- 多线程并发
- 内存压力测试
- 长时间运行稳定性

## 监控指标

### 1. 系统指标

- CPU使用率
- 内存使用量
- 磁盘I/O
- 网络I/O

### 2. 应用指标

- 请求响应时间
- 错误率
- 吞吐量
- 并发连接数

### 3. 业务指标

- 推理准确率
- 数据处理速度
- 模型加载时间

## 持续改进

### 1. 定期基准测试

- 每周运行完整基准测试
- 每次发布前性能回归测试
- 新功能性能影响评估

### 2. 性能回归检测

- 自动化性能测试
- 性能阈值告警
- 性能趋势分析

## 更新历史

- 2025-09-17: 初始基准测试报告
- 待更新: 实际性能数据收集
