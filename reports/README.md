# Reports Directory

This directory stores reproducible evaluation artifacts.

Files

- pareto_results.csv: Model/routing comparison for Pareto analysis
- rag_eval.csv: Hybrid retrieval evaluation
- trace_samples.csv: Trace-id samples for audit and replay
- plots/: Optional generated charts

Schemas

- pareto_results.csv
  - model, quant, batch, concurrency, seq_len, router,
  - p50_ms, p95_ms, p99_ms, qps, tokens_per_j, cost_per_1k_tok
- rag_eval.csv
  - dataset, index, k, kprime, reranker,
  - recall, ndcg, citation_rate, coverage,
  - p50_ms, p95_ms, p99_ms, cost_per_query

Metric definitions

- P50/P95/P99: latency percentiles; specify window and algorithm (e.g., t-digest)
- QPS: steady-state and peak to be reported separately in papers; include batch and concurrency
- tokens/J: energy efficiency; exclude cold-start; document sampling frequency
- cost/1k tok, cost/query: include model fees and retrieval allocation; document accounting window
- recall/ndcg/citation_rate/coverage: follow task-specific definitions; citation has page/segment ids

Provenance

- Include hardware, drivers, dataset versions, seeds, and configuration hash in the parent report.

Packaging

- Use scripts/repro/export_report.sh to create a tar.gz bundle.

Windows

- PowerShell equivalents are provided:
  - Pareto: `./scripts/bench/run_pareto.ps1 -Model large-v1 -Quant int4 -Batch 8 -Concurrency 16 -SeqLen 2048 -Router small-fallback -Repeats 5 -Out reports`
  - RAG Eval: `./scripts/rag/eval_hybrid.ps1 -Index data/index -Dataset data/qa.jsonl -K 100 -KPrime 20 -Reranker cross-encoder-small -Out reports`
  - Export: `./scripts/repro/export_report.ps1 -Reports reports -Out dist`
