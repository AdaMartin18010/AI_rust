//! 监控仪表板展示 - 实时性能监控和资源使用情况
//! 
//! 本示例展示了AI-Rust项目的监控仪表板功能，包括：
//! - 实时性能监控：跟踪AI模型推理、训练等操作的性能指标
//! - 系统资源使用情况：监控CPU、内存、GPU等硬件资源使用率
//! - 应用性能指标：记录请求处理时间、成功率、错误率等业务指标
//! - 监控数据可视化：生成时间序列图表和性能热力图
//! - 监控告警系统：基于阈值触发告警和自动恢复
//! 
//! ## 功能特点
//! 
//! ### 1. 实时性能监控
//! - 操作耗时统计：模型加载、推理、训练等操作的执行时间
//! - 设备利用率：GPU、CPU等计算设备的负载情况
//! - 请求处理统计：成功/失败请求数量、平均响应时间
//! 
//! ### 2. 系统资源监控
//! - CPU使用率：实时监控处理器负载
//! - 内存使用率：跟踪内存占用和可用空间
//! - GPU使用率：监控显卡计算资源使用情况
//! - 网络IO：记录网络数据传输量
//! - 磁盘使用：监控存储空间使用情况
//! 
//! ### 3. 应用性能指标
//! - 请求速率：每秒处理的请求数量
//! - 响应时间分布：P95、P99等百分位数统计
//! - 错误率统计：失败请求占总请求的比例
//! - 吞吐量监控：系统整体处理能力指标
//! 
//! ### 4. 数据可视化
//! - 时间序列图表：展示指标随时间的变化趋势
//! - 性能热力图：显示不同时段的系统负载分布
//! - 实时仪表板：提供直观的监控数据展示
//! 
//! ### 5. 告警系统
//! - 阈值告警：基于预设阈值触发告警
//! - 告警恢复：自动检测指标恢复正常
//! - 告警分级：支持不同严重程度的告警级别
//! 
//! ## 使用方法
//! 
//! ```bash
//! # 运行监控仪表板展示
//! cargo run --example monitoring_dashboard_showcase --features monitoring
//! 
//! # 启用完整功能（包括GPU支持）
//! cargo run --example monitoring_dashboard_showcase --features full
//! ```
//! 
//! ## 技术实现
//! 
//! - 使用 `tracing` 进行结构化日志记录
//! - 基于 `tokio` 异步运行时实现高性能监控
//! - 采用 `HashMap` 存储监控指标，支持标签化查询
//! - 实现时间窗口滑动统计，提供实时性能分析
//! - 支持多种监控数据格式，便于集成外部监控系统

use c19_ai::*;
use std::time::{Duration};
use std::collections::HashMap;
use tokio::time::sleep;

#[tokio::main]
#[allow(dead_code)]
#[allow(unused_variables)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志系统
    tracing_subscriber::fmt::init();
    
    println!("📊 AI-Rust 监控仪表板展示");
    println!("🎯 目标：展示实时性能监控和资源使用情况");
    println!("{}", "=".repeat(60));

    // 创建AI引擎并启动监控
    let mut engine = create_monitoring_engine().await?;
    
    // 启动监控系统
    engine.start_monitoring().await?;
    println!("✅ 监控系统已启动");
    
    // 展示实时性能监控
    demonstrate_real_time_monitoring(&mut engine).await?;
    
    // 展示系统资源监控
    demonstrate_system_resource_monitoring(&mut engine).await?;
    
    // 展示应用性能指标
    demonstrate_application_metrics(&mut engine).await?;
    
    // 展示监控数据可视化
    demonstrate_monitoring_visualization(&mut engine).await?;
    
    // 展示监控告警
    demonstrate_monitoring_alerts(&mut engine).await?;
    
    println!("\n🎉 监控仪表板展示完成！");
    println!("📊 监控数据统计：");
    print_monitoring_summary(&engine).await;
    
    // 停止监控并清理资源
    engine.stop_monitoring();
    engine.cleanup()?;
    println!("✅ 监控系统已停止，资源清理完成");
    
    Ok(())
}

/// 创建监控引擎
/// 
/// 初始化AI引擎并配置监控相关参数，包括：
/// - 启用GPU加速支持
/// - 启用实时监控功能
/// - 设置模型缓存大小
/// - 配置监控数据保留时间
/// - 启用告警系统
/// 
/// # 返回值
/// 
/// 返回配置完成的AI引擎实例，如果配置失败则返回错误
/// 
/// # 错误处理
/// 
/// 如果状态设置失败，会返回相应的错误信息
async fn create_monitoring_engine() -> Result<AIEngine, Error> {
    println!("\n🔧 创建监控引擎...");
    
    // 创建引擎配置，启用GPU和监控功能
    let mut config = EngineConfig::default();
    config.enable_gpu = true;           // 启用GPU加速
    config.enable_monitoring = true;    // 启用监控功能
    config.max_models = 10;             // 最大模型数量
    config.cache_size = 5000;           // 缓存大小
    
    let mut engine = AIEngine::with_config(config);
    
    // 设置监控相关状态参数
    engine.set_state("monitoring_mode", "real_time")?;    // 实时监控模式
    engine.set_state("data_retention", "1h")?;            // 数据保留1小时
    engine.set_state("alert_enabled", "true")?;           // 启用告警
    engine.set_state("dashboard_refresh", "5s")?;         // 仪表板5秒刷新
    
    println!("✅ 监控引擎创建完成");
    Ok(engine)
}

/// 展示实时性能监控
/// 
/// 模拟不同类型的AI操作并记录相应的性能指标，包括：
/// - 模型加载时间统计
/// - 推理操作耗时记录
/// - 训练过程性能监控
/// - 数据处理时间统计
/// - GPU计算资源使用情况
/// 
/// # 参数
/// 
/// * `engine` - AI引擎实例的可变引用
/// 
/// # 返回值
/// 
/// 如果监控记录成功则返回Ok(())，否则返回错误
/// 
/// # 监控指标
/// 
/// - `operation_duration_*`: 各种操作的执行时间
/// - `cpu_usage`: CPU使用率
/// - `memory_usage`: 内存使用率  
/// - `gpu_usage`: GPU使用率
/// - 请求成功/失败统计
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_real_time_monitoring(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n📈 展示实时性能监控...");
    
    // 定义不同类型的AI操作及其典型耗时（毫秒）
    let operations = vec![
        ("model_loading", 150.0),      // 模型加载：150ms
        ("inference", 45.0),           // 推理操作：45ms
        ("training", 2500.0),          // 模型训练：2.5秒
        ("data_processing", 120.0),    // 数据处理：120ms
        ("gpu_computation", 80.0),     // GPU计算：80ms
    ];
    
    // 遍历各种操作，记录性能指标
    for (operation, duration) in operations {
        println!("🔄 执行操作: {} (耗时: {:.1}ms)", operation, duration);
        
        // 创建操作标签，用于指标分类和查询
        let mut labels = HashMap::new();
        labels.insert("operation".to_string(), operation.to_string());
        labels.insert("device".to_string(), "gpu".to_string());
        
        // 记录操作耗时指标（带标签）
        engine.record_monitoring_metric(
            &format!("operation_duration_{}", operation),
            duration,
            Some(labels)
        );
        
        // 记录请求处理结果（成功/失败）
        let success = duration < 200.0; // 模拟：耗时超过200ms视为失败
        engine.record_monitoring_request(success, duration);
        
        // 记录系统资源使用情况（基于操作耗时动态计算）
        engine.record_metric("cpu_usage", 45.0 + (duration / 10.0));
        engine.record_metric("memory_usage", 60.0 + (duration / 20.0));
        engine.record_metric("gpu_usage", 70.0 + (duration / 15.0));
        
        // 模拟操作间隔
        sleep(Duration::from_millis(100)).await;
    }
    
    // 获取并显示实时监控数据摘要
    let dashboard_data = engine.get_monitoring_data().await;
    println!("📊 实时指标统计:");
    println!("   • 系统运行时间: {}秒", dashboard_data.uptime);
    println!("   • 应用指标数量: {}", dashboard_data.application_metrics.len());
    println!("   • 系统指标数量: {}", dashboard_data.system_metrics.len());
    
    Ok(())
}

/// 展示系统资源监控
/// 
/// 模拟不同负载场景下的系统资源使用情况，监控包括：
/// - CPU使用率变化
/// - 内存使用情况
/// - GPU计算资源占用
/// - 网络IO数据传输
/// - 磁盘存储使用情况
/// 
/// # 参数
/// 
/// * `engine` - AI引擎实例的可变引用
/// 
/// # 返回值
/// 
/// 如果监控记录成功则返回Ok(())，否则返回错误
/// 
/// # 监控场景
/// 
/// 1. **低负载场景**: CPU 25%, 内存 40%, GPU 30%
/// 2. **中等负载场景**: CPU 55%, 内存 65%, GPU 60%
/// 3. **高负载场景**: CPU 85%, 内存 90%, GPU 88%
/// 4. **峰值负载场景**: CPU 95%, 内存 98%, GPU 95%
/// 
/// # 监控指标
/// 
/// - `system_cpu_usage`: 系统CPU使用率
/// - `system_memory_usage`: 系统内存使用率
/// - `system_gpu_usage`: GPU计算使用率
/// - `system_gpu_memory_usage`: GPU显存使用率
/// - `network_bytes_sent`: 网络发送字节数
/// - `network_bytes_received`: 网络接收字节数
/// - `disk_usage_percent`: 磁盘使用百分比
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_system_resource_monitoring(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n💻 展示系统资源监控...");
    
    // 定义不同负载场景的资源使用情况 (CPU%, 内存%, GPU%)
    let resource_scenarios = vec![
        ("低负载", 25.0, 40.0, 30.0),      // 系统空闲状态
        ("中等负载", 55.0, 65.0, 60.0),    // 正常业务负载
        ("高负载", 85.0, 90.0, 88.0),      // 高并发处理
        ("峰值负载", 95.0, 98.0, 95.0),    // 系统极限状态
    ];
    
    // 遍历各种负载场景，记录资源使用指标
    for (scenario, cpu, memory, gpu) in resource_scenarios {
        println!("🎯 模拟场景: {}", scenario);
        
        // 记录核心系统资源指标
        engine.record_metric("system_cpu_usage", cpu);
        engine.record_metric("system_memory_usage", memory);
        engine.record_metric("system_gpu_usage", gpu);
        engine.record_metric("system_gpu_memory_usage", memory * 0.8); // GPU显存通常为内存的80%
        
        // 记录网络IO指标（基于CPU和内存使用率计算）
        engine.record_metric("network_bytes_sent", 1024.0 * 1024.0 * cpu);      // 发送数据量
        engine.record_metric("network_bytes_received", 1024.0 * 1024.0 * memory); // 接收数据量
        
        // 记录磁盘使用情况（通常与内存使用相关）
        engine.record_metric("disk_usage_percent", memory * 0.6); // 磁盘使用率通常为内存的60%
        
        // 显示当前场景的资源使用情况
        println!("   📊 CPU使用率: {:.1}%", cpu);
        println!("   🧠 内存使用率: {:.1}%", memory);
        println!("   🎮 GPU使用率: {:.1}%", gpu);
        
        // 模拟场景持续时间
        sleep(Duration::from_millis(200)).await;
    }
    
    // 统计资源监控指标总数
    let metrics = engine.get_metrics();
    let resource_metrics = metrics.iter()
        .filter(|(name, _)| name.contains("system_") || name.contains("network_") || name.contains("disk_"))
        .count();
    
    println!("📈 资源监控指标总数: {}", resource_metrics);
    
    Ok(())
}

/// 展示应用性能指标
/// 
/// 模拟不同负载场景下的应用性能表现，监控包括：
/// - 请求处理统计（总数、成功、失败）
/// - 响应时间分布（平均、P95、P99）
/// - 请求处理速率（QPS）
/// - 系统吞吐量指标
/// 
/// # 参数
/// 
/// * `engine` - AI引擎实例的可变引用
/// 
/// # 返回值
/// 
/// 如果监控记录成功则返回Ok(())，否则返回错误
/// 
/// # 性能场景
/// 
/// 1. **轻负载**: 50请求/分钟, 95%成功率, 25ms平均响应时间
/// 2. **正常负载**: 200请求/分钟, 90%成功率, 45ms平均响应时间
/// 3. **重负载**: 500请求/分钟, 85%成功率, 85ms平均响应时间
/// 4. **超载**: 800请求/分钟, 70%成功率, 150ms平均响应时间
/// 
/// # 监控指标
/// 
/// - `requests_per_second`: 每秒请求数（QPS）
/// - `response_time_p95`: 95%百分位响应时间
/// - `response_time_p99`: 99%百分位响应时间
/// - 请求成功/失败统计
/// - 平均响应时间
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_application_metrics(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n🚀 展示应用性能指标...");
    
    // 定义不同负载场景的性能参数 (场景名, 请求数, 成功率%, 失败数, 平均响应时间ms)
    let performance_scenarios = vec![
        ("轻负载", 50, 95, 5, 25.0),      // 低并发，高成功率
        ("正常负载", 200, 90, 20, 45.0),  // 中等并发，良好性能
        ("重负载", 500, 85, 75, 85.0),    // 高并发，性能下降
        ("超载", 800, 70, 240, 150.0),    // 极限负载，性能显著下降
    ];
    
    // 遍历各种性能场景，模拟请求处理
    for (scenario, requests, success_rate, failed_requests, avg_response_time) in performance_scenarios {
        println!("🎯 性能场景: {}", scenario);
        
        // 模拟批量请求处理
        for i in 0..requests {
            // 根据成功率决定请求是否成功
            let success = i < (requests * success_rate / 100);
            // 模拟响应时间变化（基础时间 + 微小波动）
            let response_time = avg_response_time + (i as f64 * 0.1);
            
            // 记录请求处理结果
            engine.record_monitoring_request(success, response_time);
            
            // 记录性能指标（每50个请求记录一次，避免过于频繁）
            if i % 50 == 0 {
                engine.record_metric("requests_per_second", requests as f64 / 60.0);
                engine.record_metric("response_time_p95", avg_response_time * 1.5);  // P95通常是平均值的1.5倍
                engine.record_metric("response_time_p99", avg_response_time * 2.0);  // P99通常是平均值的2倍
            }
            
            // 每50个请求暂停一下，模拟真实处理间隔
            if i % 50 == 0 {
                sleep(Duration::from_millis(10)).await;
            }
        }
        
        // 显示当前场景的性能统计
        println!("   📊 总请求数: {}", requests);
        println!("   ✅ 成功率: {}%", success_rate);
        println!("   ❌ 失败请求: {}", failed_requests);
        println!("   ⏱️  平均响应时间: {:.1}ms", avg_response_time);
        println!("   🚀 请求速率: {:.1} req/s", requests as f64 / 60.0);
        
        // 场景间暂停
        sleep(Duration::from_millis(300)).await;
    }
    
    // 获取并显示最新的应用指标数据
    let app_metrics = engine.get_monitoring_dashboard().get_application_metrics().await;
    if let Some(latest_metrics) = app_metrics.last() {
        println!("📈 最新应用指标:");
        println!("   • 总请求数: {}", latest_metrics.total_requests);
        println!("   • 成功请求数: {}", latest_metrics.successful_requests);
        println!("   • 失败请求数: {}", latest_metrics.failed_requests);
        println!("   • 平均响应时间: {:.2}ms", latest_metrics.average_response_time);
        println!("   • 请求速率: {:.2} req/s", latest_metrics.requests_per_second);
    }
    
    Ok(())
}

/// 展示监控数据可视化
/// 
/// 生成和展示各种监控数据的可视化形式，包括：
/// - 时间序列图表数据
/// - 性能热力图数据
/// - 实时数据更新展示
/// - 可视化数据统计
/// 
/// # 参数
/// 
/// * `engine` - AI引擎实例的可变引用
/// 
/// # 返回值
/// 
/// 如果可视化数据生成成功则返回Ok(())，否则返回错误
/// 
/// # 可视化类型
/// 
/// ## 1. 时间序列数据
/// - 模拟20个时间点的系统指标变化
/// - 包含CPU、内存、GPU使用率和请求数量
/// - 使用正弦/余弦函数模拟真实波动
/// 
/// ## 2. 性能热力图
/// - 生成24小时×12个5分钟间隔的热力图数据
/// - 模拟一天中的负载模式（夜间低、工作时间高）
/// - 识别高负载时段（强度>80%）
/// 
/// # 监控指标
/// 
/// - `timeseries_cpu`: 时间序列CPU使用率
/// - `timeseries_memory`: 时间序列内存使用率
/// - `timeseries_gpu`: 时间序列GPU使用率
/// - `timeseries_requests`: 时间序列请求数量
/// - `heatmap_HH_MM`: 热力图数据点（小时:分钟格式）
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_monitoring_visualization(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n📊 展示监控数据可视化...");
    
    // 生成时间序列数据（20个时间点）
    let time_series_data = generate_time_series_data().await;
    
    // 记录时间序列指标
    for (timestamp, cpu, memory, gpu, requests) in time_series_data {
        // 记录各种系统指标的时间序列数据
        engine.record_metric("timeseries_cpu", cpu);
        engine.record_metric("timeseries_memory", memory);
        engine.record_metric("timeseries_gpu", gpu);
        engine.record_metric("timeseries_requests", requests);
        
        // 每5个时间点显示一次数据，模拟实时仪表板更新
        if timestamp % 5 == 0 {
            println!("📈 时间点 {}: CPU={:.1}%, 内存={:.1}%, GPU={:.1}%, 请求={}", 
                    timestamp, cpu, memory, gpu, requests as u32);
        }
    }
    
    // 生成性能热力图数据（24小时×12个5分钟间隔）
    let heatmap_data = generate_performance_heatmap().await;
    
    // 记录热力图数据并识别高负载时段
    for (hour, minute, intensity) in heatmap_data {
        let metric_name = format!("heatmap_{:02}_{:02}", hour, minute);
        engine.record_metric(&metric_name, intensity);
        
        // 识别并显示高负载时段（强度>80%）
        if intensity > 80.0 {
            println!("🔥 高负载时段: {}:{:02} (强度: {:.1}%)", hour, minute, intensity);
        }
    }
    
    // 统计可视化数据
    let all_metrics = engine.get_metrics();
    let timeseries_count = all_metrics.iter()
        .filter(|(name, _)| name.starts_with("timeseries_"))
        .count();
    let heatmap_count = all_metrics.iter()
        .filter(|(name, _)| name.starts_with("heatmap_"))
        .count();
    
    println!("📊 可视化数据统计:");
    println!("   • 时间序列指标: {}", timeseries_count);
    println!("   • 热力图数据点: {}", heatmap_count);
    
    Ok(())
}

/// 展示监控告警
/// 
/// 模拟各种告警场景的触发和恢复过程，包括：
/// - 系统资源告警（CPU、内存、GPU使用率过高）
/// - 性能指标告警（响应时间过长、错误率过高）
/// - 告警标签和分级管理
/// - 告警自动恢复检测
/// 
/// # 参数
/// 
/// * `engine` - AI引擎实例的可变引用
/// 
/// # 返回值
/// 
/// 如果告警演示成功则返回Ok(())，否则返回错误
/// 
/// # 告警场景
/// 
/// 1. **CPU使用率过高**: 95% > 阈值
/// 2. **内存使用率过高**: 90% > 阈值
/// 3. **GPU使用率过高**: 98% > 阈值
/// 4. **响应时间过长**: P95响应时间 > 200ms
/// 5. **错误率过高**: 错误率 > 15%
/// 
/// # 告警恢复
/// 
/// 模拟系统自动恢复，将指标值降至正常范围：
/// - CPU使用率: 95% → 60%
/// - 内存使用率: 90% → 70%
/// - GPU使用率: 98% → 75%
/// - 响应时间: 200ms → 120ms
/// - 错误率: 15% → 2%
/// 
/// # 监控指标
/// 
/// - `alert_*`: 告警相关指标（带标签）
/// - 告警标签: `severity=warning`, `alert_type=告警名称`
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_monitoring_alerts(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n🚨 展示监控告警...");
    
    // 定义各种告警场景 (告警名称, 指标名称, 触发阈值)
    let alert_scenarios = vec![
        ("CPU使用率过高", "system_cpu_usage", 95.0),      // CPU使用率超过95%
        ("内存使用率过高", "system_memory_usage", 90.0),  // 内存使用率超过90%
        ("GPU使用率过高", "system_gpu_usage", 98.0),      // GPU使用率超过98%
        ("响应时间过长", "response_time_p95", 200.0),     // P95响应时间超过200ms
        ("错误率过高", "error_rate", 15.0),               // 错误率超过15%
    ];
    
    // 模拟告警触发过程
    for (alert_name, metric_name, threshold_value) in alert_scenarios {
        println!("⚠️  触发告警: {}", alert_name);
        
        // 记录触发告警的指标值
        engine.record_metric(metric_name, threshold_value);
        
        // 创建告警标签，用于告警分类和管理
        let mut labels = HashMap::new();
        labels.insert("severity".to_string(), "warning".to_string());
        labels.insert("alert_type".to_string(), alert_name.to_string());
        
        // 记录告警指标（带标签）
        engine.record_monitoring_metric(
            &format!("alert_{}", metric_name),
            threshold_value,
            Some(labels)
        );
        
        // 显示告警详情
        println!("   📊 指标: {} = {:.1}", metric_name, threshold_value);
        println!("   🚨 告警级别: WARNING");
        
        // 告警间隔
        sleep(Duration::from_millis(150)).await;
    }
    
    // 模拟告警恢复过程
    println!("✅ 模拟告警恢复...");
    let recovery_metrics = vec![
        ("system_cpu_usage", 60.0),      // CPU使用率恢复正常
        ("system_memory_usage", 70.0),   // 内存使用率恢复正常
        ("system_gpu_usage", 75.0),      // GPU使用率恢复正常
        ("response_time_p95", 120.0),    // 响应时间恢复正常
        ("error_rate", 2.0),             // 错误率恢复正常
    ];
    
    // 记录恢复后的正常指标值
    for (metric_name, normal_value) in recovery_metrics {
        engine.record_metric(metric_name, normal_value);
        println!("   📈 {} 恢复正常: {:.1}", metric_name, normal_value);
    }
    
    println!("🎉 告警演示完成");
    
    Ok(())
}

// 辅助函数

/// 生成时间序列数据
/// 
/// 生成模拟的时间序列监控数据，用于演示数据可视化功能。
/// 数据包含20个时间点的系统指标变化，使用数学函数模拟真实的波动模式。
/// 
/// # 返回值
/// 
/// 返回包含时间序列数据的向量，每个元素为：
/// `(时间戳, CPU使用率%, 内存使用率%, GPU使用率%, 请求数量)`
/// 
/// # 数据特征
/// 
/// - **时间戳**: 0-19的连续时间点
/// - **CPU使用率**: 基础40% + 线性增长 + 正弦波动
/// - **内存使用率**: 基础50% + 线性增长 + 余弦波动  
/// - **GPU使用率**: 基础60% + 线性增长 + 正弦波动
/// - **请求数量**: 基础100 + 线性增长 + 余弦波动
/// 
/// # 数学公式
/// 
/// - CPU = 40 + 2*i + 10*sin(0.5*i)
/// - Memory = 50 + 1.5*i + 8*cos(0.3*i)
/// - GPU = 60 + 1.8*i + 12*sin(0.4*i)
/// - Requests = 100 + 5*i + 20*cos(0.2*i)
/// 
/// 所有百分比指标限制在0-100%范围内。
#[allow(dead_code)]
#[allow(unused_variables)]
async fn generate_time_series_data() -> Vec<(i32, f64, f64, f64, f64)> {
    let mut data = Vec::new();
    
    // 生成20个时间点的数据
    for i in 0..20 {
        let timestamp = i;
        
        // 使用数学函数生成具有真实波动特征的指标数据
        let cpu = 40.0 + (i as f64 * 2.0) + (i as f64 * 0.5).sin() * 10.0;        // CPU: 基础40% + 增长 + 波动
        let memory = 50.0 + (i as f64 * 1.5) + (i as f64 * 0.3).cos() * 8.0;     // 内存: 基础50% + 增长 + 波动
        let gpu = 60.0 + (i as f64 * 1.8) + (i as f64 * 0.4).sin() * 12.0;       // GPU: 基础60% + 增长 + 波动
        let requests = 100.0 + (i as f64 * 5.0) + (i as f64 * 0.2).cos() * 20.0; // 请求: 基础100 + 增长 + 波动
        
        // 确保百分比指标在合理范围内
        data.push((timestamp, cpu.min(100.0), memory.min(100.0), gpu.min(100.0), requests));
    }
    
    data
}

/// 生成性能热力图数据
/// 
/// 生成24小时×12个5分钟间隔的性能热力图数据，模拟一天中的系统负载模式。
/// 数据反映了典型的业务负载分布：夜间低负载、工作时间高负载、早晚高峰等。
/// 
/// # 返回值
/// 
/// 返回包含热力图数据的向量，每个元素为：
/// `(小时, 分钟, 负载强度%)`
/// 
/// # 负载模式
/// 
/// - **夜间时段 (0-6点)**: 基础强度20%，系统空闲
/// - **早高峰 (7-9点)**: 基础强度60%，用户开始活跃
/// - **工作时间 (10-17点)**: 基础强度80%，业务高峰期
/// - **晚高峰 (18-21点)**: 基础强度70%，用户活跃
/// - **夜间 (22-23点)**: 基础强度30%，用户减少
/// 
/// # 数据特征
/// 
/// - 时间间隔：每5分钟一个数据点
/// - 总数据点：24小时 × 12个间隔 = 288个数据点
/// - 强度范围：0-100%
/// - 波动模拟：基于分钟数的正弦函数变化
/// 
/// # 数学公式
/// 
/// 强度 = 基础强度 + 10*sin(0.1*分钟)
/// 最终强度限制在0-100%范围内
#[allow(dead_code)]
#[allow(unused_variables)]
async fn generate_performance_heatmap() -> Vec<(u8, u8, f64)> {
    let mut data = Vec::new();
    
    // 遍历24小时
    for hour in 0..24 {
        // 每5分钟生成一个数据点
        for minute in (0..60).step_by(5) {
            // 根据小时确定基础负载强度，模拟真实的业务负载模式
            let base_intensity = match hour {
                0..=6 => 20.0,    // 夜间低负载：系统维护和批处理时间
                7..=9 => 60.0,    // 早高峰：用户开始工作，系统负载上升
                10..=17 => 80.0,  // 工作时间：业务高峰期，系统高负载
                18..=21 => 70.0,  // 晚高峰：用户活跃，但负载略低于工作时间
                22..=23 => 30.0,  // 夜间：用户减少，系统负载下降
                _ => 50.0,        // 默认中等负载
            };
            
            // 添加基于分钟数的微小波动，模拟真实负载变化
            let variation = (minute as f64 * 0.1).sin() * 10.0;
            let intensity = (base_intensity + variation).max(0.0).min(100.0);
            
            data.push((hour, minute, intensity));
        }
    }
    
    data
}

/// 打印监控摘要
/// 
/// 生成并显示完整的监控数据摘要报告，包括：
/// - 系统运行状态统计
/// - 监控指标数量统计
/// - 应用性能指标摘要
/// - 指标分类统计
/// - 监控功能特点总结
/// 
/// # 参数
/// 
/// * `engine` - AI引擎实例的不可变引用
/// 
/// # 显示内容
/// 
/// ## 1. 系统状态
/// - 系统运行时间
/// - 总指标数量
/// - 系统指标数量
/// - 应用指标数量
/// 
/// ## 2. 应用性能摘要
/// - 总请求数
/// - 成功/失败请求数
/// - 平均响应时间
/// - 请求处理速率
/// 
/// ## 3. 指标分类统计
/// - 系统指标：包含"system_"的指标
/// - 应用指标：包含"operation_"或"response_"的指标
/// - 时间序列：以"timeseries_"开头的指标
/// 
/// ## 4. 功能特点
/// - 实时性能监控
/// - 系统资源监控
/// - 应用性能指标
/// - 时间序列数据收集
/// - 性能热力图生成
/// - 监控告警系统
/// - 数据可视化支持
#[allow(dead_code)]
#[allow(unused_variables)]
async fn print_monitoring_summary(engine: &AIEngine) {
    // 获取监控数据和指标
    let dashboard_data = engine.get_monitoring_data().await;
    let all_metrics = engine.get_metrics();
    
    // 打印监控摘要标题和边框
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│                   监控数据摘要                        │");
    println!("├─────────────────────────────────────────────────────────┤");
    
    // 显示系统运行状态
    println!("│ 系统运行时间: {:<42} │", format!("{}秒", dashboard_data.uptime));
    println!("│ 总指标数量: {:<44} │", all_metrics.len());
    println!("│ 系统指标数量: {:<42} │", dashboard_data.system_metrics.len());
    println!("│ 应用指标数量: {:<42} │", dashboard_data.application_metrics.len());
    
    // 显示应用性能指标摘要
    if let Some(latest_app) = dashboard_data.application_metrics.last() {
        println!("│ 总请求数: {:<46} │", latest_app.total_requests);
        println!("│ 成功请求数: {:<44} │", latest_app.successful_requests);
        println!("│ 失败请求数: {:<44} │", latest_app.failed_requests);
        println!("│ 平均响应时间: {:<40} │", format!("{:.2}ms", latest_app.average_response_time));
        println!("│ 请求速率: {:<46} │", format!("{:.2} req/s", latest_app.requests_per_second));
    }
    
    // 统计不同类型的指标数量
    let system_metrics_count = all_metrics.iter()
        .filter(|(name, _)| name.contains("system_"))
        .count();
    let application_metrics_count = all_metrics.iter()
        .filter(|(name, _)| name.contains("operation_") || name.contains("response_"))
        .count();
    let timeseries_metrics_count = all_metrics.iter()
        .filter(|(name, _)| name.starts_with("timeseries_"))
        .count();
    
    // 显示指标分类统计
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│ 指标分类统计:                                        │");
    println!("│ • 系统指标: {:<42} │", system_metrics_count);
    println!("│ • 应用指标: {:<42} │", application_metrics_count);
    println!("│ • 时间序列: {:<42} │", timeseries_metrics_count);
    println!("└─────────────────────────────────────────────────────────┘");
    
    // 显示监控功能特点总结
    println!("\n🎯 监控功能特点:");
    println!("   • ✅ 实时性能监控");
    println!("   • ✅ 系统资源使用情况");
    println!("   • ✅ 应用性能指标");
    println!("   • ✅ 时间序列数据收集");
    println!("   • ✅ 性能热力图生成");
    println!("   • ✅ 监控告警系统");
    println!("   • ✅ 数据可视化支持");
}
