# Bench 脚本使用说明（对齐§Z.7 与实践§8.1）

## 运行示例

```bash
bash scripts/bench/run_pareto.sh \
  --model large-v1 \
  --quant int4 \
  --batch 8 \
  --concurrency 16 \
  --seq-len 2048 \
  --router small-fallback \
  --repeats 5 \
  --out reports
```

## 参数口径

- model：模型ID（需在服务端可用）
- quant：量化位宽（fp16/int8/int4）
- batch：批量大小
- concurrency：并发度
- seq-len：序列长度
- router：路由策略（small-fallback 等）
- repeats：重复次数（n≥5）
- out：结果输出目录（CSV/图表）

## 输出

- `reports/<date>/metrics.csv` 表头参见 `docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md` §Z.7
- 图表可由脚本一键再生并嵌入报告
