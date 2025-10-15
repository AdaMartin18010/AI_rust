# Rust 1.90 特性演示脚本 (Windows PowerShell)
# 展示GAT、TAIT等新特性在AI场景下的应用

Write-Host "🚀 Rust 1.90 AI特性演示" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green

# 检查Rust版本
Write-Host "📋 检查Rust版本..." -ForegroundColor Yellow
rustc --version
Write-Host ""

# 编译项目
Write-Host "🔨 编译项目..." -ForegroundColor Yellow
cargo build --release
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 编译失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 编译成功" -ForegroundColor Green
Write-Host ""

# 运行测试
Write-Host "🧪 运行测试..." -ForegroundColor Yellow
cargo test --workspace
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 测试失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 所有测试通过" -ForegroundColor Green
Write-Host ""

# 运行GAT特性演示
Write-Host "📊 运行GAT特性演示..." -ForegroundColor Yellow
cargo run --example gat_ai_inference
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ GAT演示失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ GAT演示完成" -ForegroundColor Green
Write-Host ""

# 运行TAIT特性演示
Write-Host "🔧 运行TAIT特性演示..." -ForegroundColor Yellow
cargo run --example tait_advanced_types
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ TAIT演示失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ TAIT演示完成" -ForegroundColor Green
Write-Host ""

# 运行性能对比测试
Write-Host "⚡ 运行性能对比测试..." -ForegroundColor Yellow
cargo run --example performance_comparison
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 性能对比测试失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 性能对比测试完成" -ForegroundColor Green
Write-Host ""

# 运行综合演示
Write-Host "🎯 运行综合演示..." -ForegroundColor Yellow
cargo run --example main --manifest-path examples/rust_190_demo/Cargo.toml
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 综合演示失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 综合演示完成" -ForegroundColor Green
Write-Host ""

# 运行基准测试
Write-Host "📈 运行基准测试..." -ForegroundColor Yellow
cargo bench --bench gat_benchmarks
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ 基准测试失败（可能因为缺少benchmark依赖）" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "🎉 所有演示完成！" -ForegroundColor Green
Write-Host ""
Write-Host "📈 Rust 1.90在AI场景下的优势总结：" -ForegroundColor Cyan
Write-Host "   • 更好的类型安全性" -ForegroundColor White
Write-Host "   • 更灵活的生命周期管理" -ForegroundColor White
Write-Host "   • 更简洁的代码结构" -ForegroundColor White
Write-Host "   • 更高的运行时性能" -ForegroundColor White
Write-Host "   • 更好的开发体验" -ForegroundColor White
Write-Host ""
Write-Host "🔗 相关文档：" -ForegroundColor Cyan
Write-Host "   • 快速开始指南: docs/QUICK_START.md" -ForegroundColor White
Write-Host "   • 改进计划: PROJECT_IMPROVEMENT_PLAN_2025.md" -ForegroundColor White
Write-Host "   • 进度跟踪: improvement_tracking/weekly_progress.md" -ForegroundColor White
