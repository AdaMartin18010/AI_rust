Param(
  [string]$Reports = "reports",
  [string]$Out = "dist"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $Out | Out-Null

$stamp = Get-Date -Format yyyyMMdd_HHmmss
$archive = Join-Path $Out "report_${stamp}.zip"

$items = @()
$possible = @(
  (Join-Path $Reports "pareto_results.csv"),
  (Join-Path $Reports "rag_eval.csv"),
  (Join-Path $Reports "trace_samples.csv"),
  (Join-Path $Reports "plots")
)
foreach ($p in $possible) {
  if (Test-Path $p) { $items += $p }
}

Compress-Archive -Path $items -DestinationPath $archive -Force

Write-Host "Created $archive"
