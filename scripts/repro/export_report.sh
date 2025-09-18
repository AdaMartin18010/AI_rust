#!/usr/bin/env bash
set -euo pipefail

# export_report.sh
# Purpose: Bundle CSVs, plots, logs, and trace samples into a single reproducible artifact.
# Inputs: --reports reports --out dist
# Outputs: dist/report_YYYYmmdd_HHMMSS.tar.gz

REPORTS_DIR="reports"
OUT_DIR="dist"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reports) REPORTS_DIR="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"

ARCHIVE="$OUT_DIR/report_${STAMP}.tar.gz"

# Optional header check for standard metrics CSV (aligned with §Z.7)
STD="$REPORTS_DIR/metrics.csv"
HDR="run_id,model,scenario,batch,concurrency,seq_len,precision,quant,dataset,latency_p50_ms,latency_p95_ms,latency_p99_ms,qps,gpu_util,cpu_util,mem_peak_mb,vram_peak_mb,tokens_per_joule,cost_per_1k_tok_usd,error_rate,timeout_rate,samples_n,ci95_low_ms,ci95_high_ms"
if [[ -f "$STD" ]]; then
  FIRST=$(head -n 1 "$STD" | tr -d '\r')
  if [[ "$FIRST" != "$HDR" ]]; then
    echo "[ERROR] $STD header mismatch. Expected:\n$HDR\nGot:\n$FIRST" >&2
    exit 2
  fi
else
  echo "[WARN] $STD not found. You can copy reports/samples/metrics.example.csv as a starting point." >&2
fi

tar -czf "$ARCHIVE" \
  "$REPORTS_DIR/pareto_results.csv" \
  "$REPORTS_DIR/rag_eval.csv" \
  "$REPORTS_DIR/metrics.csv" \
  "$REPORTS_DIR/trace_samples.csv" \
  "$REPORTS_DIR/plots" 2>/dev/null || true

echo "Created $ARCHIVE"
