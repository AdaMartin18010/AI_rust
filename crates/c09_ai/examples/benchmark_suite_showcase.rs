//! 基准测试套件展示 - 性能对比和压力测试
//! 
//! 本示例展示了AI-Rust项目的基准测试功能，包括：
//! - 性能基准测试
//! - 压力测试
//! - 性能对比分析
//! - 性能报告生成

use c19_ai::*;
use std::time::{Duration};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志系统
    tracing_subscriber::fmt::init();
    
    println!("🏃 AI-Rust 基准测试套件展示");
    println!("🎯 目标：展示性能对比和压力测试功能");
    println!("{}", "=".repeat(60));

    // 创建AI引擎
    let mut engine = create_benchmark_engine().await?;
    
    // 展示性能基准测试
    demonstrate_performance_benchmarks(&mut engine).await?;
    
    // 展示压力测试
    demonstrate_stress_testing(&mut engine).await?;
    
    // 展示性能对比分析
    demonstrate_performance_comparison(&mut engine).await?;
    
    // 展示性能报告生成
    demonstrate_performance_reporting(&mut engine).await?;
    
    println!("\n🎉 基准测试套件展示完成！");
    println!("📊 性能测试总结：");
    print_benchmark_summary(&engine).await;
    
    // 清理资源
    engine.cleanup()?;
    println!("✅ 资源清理完成");
    
    Ok(())
}

/// 创建基准测试引擎
async fn create_benchmark_engine() -> Result<AIEngine, Error> {
    println!("\n🔧 创建基准测试引擎...");
    
    let mut config = EngineConfig::default();
    config.enable_gpu = true;
    config.enable_monitoring = true;
    config.max_models = 20;
    config.cache_size = 10000;
    
    let mut engine = AIEngine::with_config(config);
    
    // 设置基准测试相关状态
    engine.set_state("benchmark_mode", "performance")?;
    engine.set_state("test_environment", "production")?;
    engine.set_state("gpu_acceleration", "enabled")?;
    
    println!("✅ 基准测试引擎创建完成");
    Ok(engine)
}

/// 展示性能基准测试
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_performance_benchmarks(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n⚡ 展示性能基准测试...");
    
    // 创建基准测试套件
    let mut suite = engine.get_benchmark_suite();
    
    // 测试1: 模型加载性能
    println!("🔄 测试1: 模型加载性能");
    let load_result = suite.run_benchmark("model_loading", 50, || async {
        // 模拟模型加载
        sleep(Duration::from_millis(100)).await;
        Ok(())
    }).await;
    
    // 测试2: 推理性能
    println!("🔄 测试2: 推理性能");
    let inference_result = suite.run_benchmark("inference", 1000, || async {
        // 模拟推理操作
        sleep(Duration::from_millis(10)).await;
        Ok(())
    }).await;
    
    // 测试3: 批处理性能
    println!("🔄 测试3: 批处理性能");
    let batch_result = suite.run_benchmark("batch_processing", 500, || async {
        // 模拟批处理操作
        sleep(Duration::from_millis(20)).await;
        Ok(())
    }).await;
    
    // 测试4: GPU计算性能
    println!("🔄 测试4: GPU计算性能");
    let gpu_result = suite.run_benchmark("gpu_computation", 2000, || async {
        // 模拟GPU计算
        sleep(Duration::from_millis(5)).await;
        Ok(())
    }).await;
    
    // 测试5: 数据预处理性能
    println!("🔄 测试5: 数据预处理性能");
    let preprocessing_result = suite.run_benchmark("data_preprocessing", 800, || async {
        // 模拟数据预处理
        sleep(Duration::from_millis(15)).await;
        Ok(())
    }).await;
    
    // 显示基准测试结果
    println!("📊 基准测试结果汇总:");
    let results = suite.get_results();
    for result in results {
        println!("   • {}: {:.2} ops/sec, {:.2}ms 平均耗时, {:.2}% 错误率", 
                result.name, result.operations_per_second, 
                result.duration.as_millis() as f64 / result.operations as f64,
                result.error_rate);
    }
    
    Ok(())
}

/// 展示压力测试
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_stress_testing(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n🔥 展示压力测试...");
    
    // 创建基准测试套件
    let mut suite = engine.get_benchmark_suite();
    
    // 压力测试1: 轻负载
    println!("🔄 压力测试1: 轻负载 (10并发用户)");
    let light_load_config = benchmarks::StressTestConfig {
        duration: Duration::from_secs(10),
        concurrent_users: 10,
        requests_per_second: 50,
        ramp_up_duration: Duration::from_secs(2),
        ramp_down_duration: Duration::from_secs(2),
        target_error_rate: 1.0,
        max_response_time: Duration::from_millis(100),
    };
    
    let light_load_result = suite.run_stress_test(light_load_config, || {
        // 模拟轻负载请求
        Ok(())
    }).await;
    
    // 压力测试2: 中等负载
    println!("🔄 压力测试2: 中等负载 (50并发用户)");
    let medium_load_config = benchmarks::StressTestConfig {
        duration: Duration::from_secs(15),
        concurrent_users: 50,
        requests_per_second: 200,
        ramp_up_duration: Duration::from_secs(3),
        ramp_down_duration: Duration::from_secs(3),
        target_error_rate: 2.0,
        max_response_time: Duration::from_millis(150),
    };
    
    let medium_load_result = suite.run_stress_test(medium_load_config, || {
        // 模拟中等负载请求
        Ok(())
    }).await;
    
    // 压力测试3: 重负载
    println!("🔄 压力测试3: 重负载 (100并发用户)");
    let heavy_load_config = benchmarks::StressTestConfig {
        duration: Duration::from_secs(20),
        concurrent_users: 100,
        requests_per_second: 500,
        ramp_up_duration: Duration::from_secs(5),
        ramp_down_duration: Duration::from_secs(5),
        target_error_rate: 5.0,
        max_response_time: Duration::from_millis(200),
    };
    
    let heavy_load_result = suite.run_stress_test(heavy_load_config, || {
        // 模拟重负载请求
        Ok(())
    }).await;
    
    // 压力测试4: 峰值负载
    println!("🔄 压力测试4: 峰值负载 (200并发用户)");
    let peak_load_config = benchmarks::StressTestConfig {
        duration: Duration::from_secs(25),
        concurrent_users: 200,
        requests_per_second: 1000,
        ramp_up_duration: Duration::from_secs(8),
        ramp_down_duration: Duration::from_secs(8),
        target_error_rate: 10.0,
        max_response_time: Duration::from_millis(300),
    };
    
    let peak_load_result = suite.run_stress_test(peak_load_config, || {
        // 模拟峰值负载请求
        Ok(())
    }).await;
    
    // 显示压力测试结果
    println!("📊 压力测试结果汇总:");
    let stress_results = suite.get_stress_test_results();
    for result in stress_results {
        println!("   • {}并发用户: {} 总请求, {:.2} req/s, {:.2}ms 平均响应, {:.2}% 错误率, {}",
                result.config.concurrent_users,
                result.total_requests,
                result.requests_per_second,
                result.average_response_time.as_millis(),
                result.error_rate,
                if result.passed { "✅ 通过" } else { "❌ 失败" });
    }
    
    Ok(())
}

/// 展示性能对比分析
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_performance_comparison(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n📊 展示性能对比分析...");
    
    // 创建基准测试套件
    let mut suite = engine.get_benchmark_suite();
    
    // 运行不同配置的相同测试
    println!("🔄 CPU vs GPU 性能对比");
    
    // CPU版本测试
    let cpu_result = suite.run_benchmark("cpu_inference", 500, || async {
        // 模拟CPU推理
        sleep(Duration::from_millis(25)).await;
        Ok(())
    }).await;
    
    // GPU版本测试
    let gpu_result = suite.run_benchmark("gpu_inference", 500, || async {
        // 模拟GPU推理
        sleep(Duration::from_millis(8)).await;
        Ok(())
    }).await;
    
    // 混合精度测试
    let mixed_precision_result = suite.run_benchmark("mixed_precision_inference", 500, || async {
        // 模拟混合精度推理
        sleep(Duration::from_millis(12)).await;
        Ok(())
    }).await;
    
    // 性能对比
    let comparison = suite.compare_performance(
        "推理性能对比",
        vec!["cpu_inference", "gpu_inference", "mixed_precision_inference"]
    );
    
    println!("📈 性能对比结果:");
    println!("   {}", comparison.summary);
    
    if let Some(winner) = &comparison.winner {
        println!("   🏆 最佳性能: {}", winner);
    }
    
    if let Some(improvement) = comparison.improvement_percentage {
        println!("   📈 性能提升: {:.2}%", improvement);
    }
    
    // 内存使用对比
    println!("\n🔄 内存使用对比");
    
    let low_memory_result = suite.run_benchmark("low_memory_config", 200, || async {
        // 模拟低内存配置
        sleep(Duration::from_millis(30)).await;
        Ok(())
    }).await;
    
    let high_memory_result = suite.run_benchmark("high_memory_config", 200, || async {
        // 模拟高内存配置
        sleep(Duration::from_millis(15)).await;
        Ok(())
    }).await;
    
    let memory_comparison = suite.compare_performance(
        "内存配置对比",
        vec!["low_memory_config", "high_memory_config"]
    );
    
    println!("📈 内存配置对比结果:");
    println!("   {}", memory_comparison.summary);
    
    Ok(())
}

/// 展示性能报告生成
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_performance_reporting(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n📋 展示性能报告生成...");
    
    // 生成性能报告
    let report = engine.generate_performance_report();
    
    println!("📊 性能报告预览:");
    println!("{}", "=".repeat(60));
    
    // 显示报告的前几行
    let lines: Vec<&str> = report.lines().take(20).collect();
    for line in lines {
        println!("{}", line);
    }
    
    if report.lines().count() > 20 {
        println!("... (报告包含更多内容)");
    }
    
    println!("{}", "=".repeat(60));
    
    // 保存报告到文件（在实际应用中）
    println!("💾 报告已生成，包含以下内容:");
    println!("   • 基准测试结果详情");
    println!("   • 压力测试统计信息");
    println!("   • 性能对比分析");
    println!("   • 系统资源使用情况");
    println!("   • 性能优化建议");
    
    Ok(())
}

// 辅助函数

/// 打印基准测试摘要
#[allow(dead_code)]
#[allow(unused_variables)]
async fn print_benchmark_summary(engine: &AIEngine) {
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│                   基准测试摘要                        │");
    println!("├─────────────────────────────────────────────────────────┤");
    
    // 获取引擎统计信息
    let stats = engine.get_stats();
    
    println!("│ 引擎版本: {:<42} │", stats.get("version").unwrap_or(&"未知".to_string()));
    println!("│ 运行时间: {:<42} │", format!("{}秒", stats.get("uptime_seconds").unwrap_or(&"0".to_string())));
    println!("│ 模块数量: {:<42} │", stats.get("modules_count").unwrap_or(&"0".to_string()));
    println!("│ 模型数量: {:<42} │", stats.get("models_count").unwrap_or(&"0".to_string()));
    println!("│ 当前设备: {:<42} │", stats.get("current_device").unwrap_or(&"未知".to_string()));
    
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│ 测试功能特点:                                        │");
    println!("│ • ✅ 性能基准测试                                      │");
    println!("│ • ✅ 压力测试和负载测试                                │");
    println!("│ • ✅ 性能对比分析                                      │");
    println!("│ • ✅ 详细性能报告                                      │");
    println!("│ • ✅ 系统资源监控                                      │");
    println!("│ • ✅ 错误率和稳定性测试                                │");
    println!("│ • ✅ 响应时间分析 (P95, P99)                           │");
    println!("│ • ✅ 吞吐量测试                                        │");
    println!("└─────────────────────────────────────────────────────────┘");
    
    println!("\n🎯 基准测试能力:");
    println!("   • 🔥 支持多种负载模式 (轻、中、重、峰值)");
    println!("   • 📊 详细的性能指标收集和分析");
    println!("   • 🏆 自动性能对比和排名");
    println!("   • 📈 性能趋势分析和报告生成");
    println!("   • 🎮 GPU vs CPU 性能对比");
    println!("   • 💾 内存使用优化分析");
    println!("   • ⚡ 响应时间和吞吐量测试");
    println!("   • 🛡️ 错误率和稳定性验证");
}
