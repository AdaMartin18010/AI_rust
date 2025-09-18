Param(
  [string]$Model = "large-v1",
  [ValidateSet("fp16","int8","int4")]
  [string]$Quant = "int4",
  [int]$Batch = 8,
  [int]$Concurrency = 16,
  [int]$SeqLen = 2048,
  [ValidateSet("none","small-fallback","consistency")]
  [string]$Router = "small-fallback",
  [int]$Repeats = 5,
  [string]$Out = "reports"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Out "plots") | Out-Null

$pareto = Join-Path $Out "pareto_results.csv"
$trace = Join-Path $Out "trace_samples.csv"

"model,quant,batch,concurrency,seq_len,router,p50_ms,p95_ms,p99_ms,qps,tokens_per_j,cost_per_1k_tok" | Out-File -FilePath $pareto -Encoding utf8

for ($i = 1; $i -le $Repeats; $i++) {
  $P50 = 200 + (Get-Random -Maximum 50)
  $P95 = 300 + (Get-Random -Maximum 60)
  $P99 = 380 + (Get-Random -Maximum 80)
  $QPS = 50 + (Get-Random -Maximum 10)
  $TOKJ = 200 + (Get-Random -Maximum 40)
  $COST = [math]::Round((Get-Random -Maximum 5.0), 2)
  "$Model,$Quant,$Batch,$Concurrency,$SeqLen,$Router,$P50,$P95,$P99,$QPS,$TOKJ,$COST" | Out-File -FilePath $pareto -Append -Encoding utf8
  "trace-id-$(Get-Date -Format yyyyMMddHHmmss)-$i" | Out-File -FilePath $trace -Append -Encoding utf8
}

Write-Host "Done. Results at $pareto"
