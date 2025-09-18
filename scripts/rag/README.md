# RAG 评测脚本使用说明（对齐§Z.7/§Z.11 与实践§4.3）

## 运行示例

```bash
bash scripts/rag/eval_hybrid.sh \
  --index data/index \
  --dataset data/qa.jsonl \
  --k 100 \
  --kprime 20 \
  --reranker cross-encoder-small \
  --out reports
```

## 参数口径

- index：索引目录（含参数快照）
- dataset：评测集（含版本与校验和）
- k：初检候选数
- kprime：重排候选数
- reranker：重排模型（版本）
- out：结果输出目录（CSV/图表）

## 输出

- `reports/<date>/metrics.csv`：需包含 recall@k、ndcg、citation_rate、端到端 {p50/p95/qps/cost}
- 附 `trace-id` 样本与配置快照，便于回放与审计
