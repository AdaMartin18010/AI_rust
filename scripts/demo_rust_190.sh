#!/bin/bash

# Rust 1.90 特性演示脚本
# 展示GAT、TAIT等新特性在AI场景下的应用

echo "🚀 Rust 1.90 AI特性演示"
echo "========================"

# 检查Rust版本
echo "📋 检查Rust版本..."
rustc --version
echo ""

# 编译项目
echo "🔨 编译项目..."
cargo build --release
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi
echo "✅ 编译成功"
echo ""

# 运行测试
echo "🧪 运行测试..."
cargo test --workspace
if [ $? -ne 0 ]; then
    echo "❌ 测试失败"
    exit 1
fi
echo "✅ 所有测试通过"
echo ""

# 运行GAT特性演示
echo "📊 运行GAT特性演示..."
cargo run --example gat_ai_inference
if [ $? -ne 0 ]; then
    echo "❌ GAT演示失败"
    exit 1
fi
echo "✅ GAT演示完成"
echo ""

# 运行TAIT特性演示
echo "🔧 运行TAIT特性演示..."
cargo run --example tait_advanced_types
if [ $? -ne 0 ]; then
    echo "❌ TAIT演示失败"
    exit 1
fi
echo "✅ TAIT演示完成"
echo ""

# 运行性能对比测试
echo "⚡ 运行性能对比测试..."
cargo run --example performance_comparison
if [ $? -ne 0 ]; then
    echo "❌ 性能对比测试失败"
    exit 1
fi
echo "✅ 性能对比测试完成"
echo ""

# 运行综合演示
echo "🎯 运行综合演示..."
cargo run --example main --manifest-path examples/rust_190_demo/Cargo.toml
if [ $? -ne 0 ]; then
    echo "❌ 综合演示失败"
    exit 1
fi
echo "✅ 综合演示完成"
echo ""

# 运行基准测试
echo "📈 运行基准测试..."
cargo bench --bench gat_benchmarks
if [ $? -ne 0 ]; then
    echo "⚠️ 基准测试失败（可能因为缺少benchmark依赖）"
fi
echo ""

echo "🎉 所有演示完成！"
echo ""
echo "📈 Rust 1.90在AI场景下的优势总结："
echo "   • 更好的类型安全性"
echo "   • 更灵活的生命周期管理"
echo "   • 更简洁的代码结构"
echo "   • 更高的运行时性能"
echo "   • 更好的开发体验"
echo ""
echo "🔗 相关文档："
echo "   • 快速开始指南: docs/QUICK_START.md"
echo "   • 改进计划: PROJECT_IMPROVEMENT_PLAN_2025.md"
echo "   • 进度跟踪: improvement_tracking/weekly_progress.md"
