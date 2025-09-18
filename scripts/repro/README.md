# 报告导出与图表再生（repro）

本目录用于将 `reports/` 下的 CSV 指标按统一表头（§Z.7）导出汇总，并再生图表嵌入到报告中。

## 一键导出（示例）

```bash
bash scripts/repro/export_report.sh \
  --input reports/2025-09-18/metrics.csv \
  --spec docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md \
  --out reports/2025-09-18/
```

Windows PowerShell:

```powershell
scripts/repro/export_report.ps1 \
  -Input reports/2025-09-18/metrics.csv \
  -Spec docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md \
  -Out reports/2025-09-18/
```

## 输入口径

- 必须包含 `run_id,model,scenario,batch,concurrency,seq_len,precision,quant,dataset,latency_p50_ms,latency_p95_ms,latency_p99_ms,qps,gpu_util,cpu_util,mem_peak_mb,vram_peak_mb,tokens_per_joule,cost_per_1k_tok_usd,error_rate,timeout_rate,samples_n,ci95_low_ms,ci95_high_ms`
- 参考：`docs/03_tech_trends/...` §Z.7 与实践指南 §8.1

## 产出

- 聚合表 `reports/<date>/summary.md`
- 图表（延迟CDF、QPS-并发曲线、能效-量化位宽、Pareto）
- 可复制到 `docs/05_practical_guides/2025_rust_ai_practical_guide.md` §8 或 `reports/performance_baseline.md`
