#!/usr/bin/env bash
set -euo pipefail

# run_pareto.sh
# Purpose: Compare FP16/INT8/INT4 and routing strategies to trace a Pareto frontier (latency/cost/quality).
# Outputs:
# - reports/pareto_results.csv
# - reports/metrics.csv (standard header per §Z.7; created if missing)
# - reports/trace_samples.csv (trace-id list)
# - reports/plots/ (generated charts)
#
# Usage:
#   bash scripts/bench/run_pareto.sh \
#     --model large-v1 --quant {fp16,int8,int4} \
#     --batch 8 --concurrency 16 --seq-len 2048 \
#     --router {none,small-fallback,consistency} \
#     --repeats 5 --out reports
#
# Example:
#   bash scripts/bench/run_pareto.sh --model large-v1 --quant int4 --batch 8 --concurrency 16 --seq-len 2048 --router small-fallback --repeats 5 --out reports

MODEL="large-v1"
QUANT="int4"
BATCH=8
CONCURRENCY=16
SEQ_LEN=2048
ROUTER="small-fallback"
REPEATS=5
OUT_DIR="reports"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --quant) QUANT="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --router) ROUTER="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR/plots"

# TODO: wire to real benchmark binary or Python/Rust harness.
echo "model,quant,batch,concurrency,seq_len,router,p50_ms,p95_ms,p99_ms,qps,tokens_per_j,cost_per_1k_tok" > "$OUT_DIR/pareto_results.csv"

# Ensure standard metrics header exists (aligned with §Z.7). Downstream tools may append rows.
STD_METRICS="$OUT_DIR/metrics.csv"
if [[ ! -f "$STD_METRICS" ]]; then
  echo "run_id,model,scenario,batch,concurrency,seq_len,precision,quant,dataset,latency_p50_ms,latency_p95_ms,latency_p99_ms,qps,gpu_util,cpu_util,mem_peak_mb,vram_peak_mb,tokens_per_joule,cost_per_1k_tok_usd,error_rate,timeout_rate,samples_n,ci95_low_ms,ci95_high_ms" > "$STD_METRICS"
fi

for i in $(seq 1 "$REPEATS"); do
  # Placeholder measurements. Replace with real invocations.
  P50=$((200 + RANDOM % 50))
  P95=$((300 + RANDOM % 60))
  P99=$((380 + RANDOM % 80))
  QPS=$((50 + RANDOM % 10))
  TOKJ=$((200 + RANDOM % 40))
  COST=$(printf "%.2f" "$(awk -v a=$RANDOM 'BEGIN{print (a%50)/10}')")
  echo "$MODEL,$QUANT,$BATCH,$CONCURRENCY,$SEQ_LEN,$ROUTER,$P50,$P95,$P99,$QPS,$TOKJ,$COST" >> "$OUT_DIR/pareto_results.csv"
  echo "trace-id-$(date +%s)-$i" >> "$OUT_DIR/trace_samples.csv"
done

echo "Done. Results at $OUT_DIR/pareto_results.csv"
