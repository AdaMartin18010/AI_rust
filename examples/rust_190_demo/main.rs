//! Rust 1.90 特性展示主程序
//! 
//! 本程序展示了Rust 1.90中GAT和TAIT特性在AI场景下的实际应用，
//! 包括性能对比、类型安全性和代码可读性的提升。

use std::time::Instant;

mod gat_ai_inference;
mod tait_advanced_types;
mod performance_comparison;

use gat_ai_inference::*;
use tait_advanced_types::*;
use performance_comparison::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Rust 1.90 AI特性展示程序");
    println!("================================");
    
    // 1. 展示GAT特性
    println!("\n📊 1. GAT (Generic Associated Types) 特性展示");
    await demonstrate_gat_features().await?;
    
    // 2. 展示TAIT特性
    println!("\n🔧 2. TAIT (Type Alias Impl Trait) 特性展示");
    await demonstrate_tait_features().await?;
    
    // 3. 性能对比测试
    println!("\n⚡ 3. 性能对比测试");
    await demonstrate_performance_comparison().await?;
    
    // 4. 综合应用示例
    println!("\n🎯 4. 综合应用示例");
    await demonstrate_comprehensive_usage().await?;
    
    println!("\n✅ 所有演示完成！");
    println!("\n📈 Rust 1.90在AI场景下的优势总结：");
    println!("   • 更好的类型安全性");
    println!("   • 更灵活的生命周期管理");
    println!("   • 更简洁的代码结构");
    println!("   • 更高的运行时性能");
    println!("   • 更好的开发体验");
    
    Ok(())
}

/// 演示GAT特性
async fn demonstrate_gat_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("创建不同类型的AI模型...");
    
    // 创建线性模型
    let linear_model = LinearModel::new(vec![1.0, 2.0, 3.0], 1.0);
    println!("✅ 线性模型创建完成");
    
    // 创建神经网络模型
    let neural_model = NeuralNetwork::new(
        vec![
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            vec![vec![0.5, 0.6]]
        ],
        vec![NeuralNetwork::relu, NeuralNetwork::sigmoid]
    );
    println!("✅ 神经网络模型创建完成");
    
    // 创建多模态模型
    let multimodal_model = MultimodalModel::new();
    println!("✅ 多模态模型创建完成");
    
    // 测试推理
    println!("\n执行推理测试...");
    
    let input = [1.0, 2.0, 3.0];
    let linear_result = linear_model.infer(&input).await;
    println!("线性模型推理结果: {:?}", linear_result);
    
    let neural_result = neural_model.infer(&input).await;
    println!("神经网络推理结果: {:?}", neural_result);
    
    let multimodal_input = MultimodalInput {
        text: "hello world",
        image: &[1.0, 2.0],
    };
    let multimodal_result = multimodal_model.infer(multimodal_input).await;
    println!("多模态模型推理结果: {:?}", multimodal_result);
    
    // 测试批量处理
    println!("\n测试批量处理...");
    let batch_processor = BatchProcessor::new(linear_model, 2);
    let batch_input = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];
    let batch_result = batch_processor.infer(&batch_input).await;
    println!("批量处理结果: {:?}", batch_result);
    
    Ok(())
}

/// 演示TAIT特性
async fn demonstrate_tait_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("创建不同类型的AI模型...");
    
    // 创建线性回归模型
    let mut linear_model = ModelFactory::create_linear_regression(3, 0.01);
    println!("✅ 线性回归模型创建完成");
    
    // 创建神经网络模型
    let mut neural_model = ModelFactory::create_neural_network(&[3, 4, 1], 0.01);
    println!("✅ 神经网络模型创建完成");
    
    // 准备训练数据
    let training_data = vec![
        (vec![1.0, 2.0, 3.0], 6.0),
        (vec![2.0, 3.0, 4.0], 9.0),
        (vec![3.0, 4.0, 5.0], 12.0),
    ];
    println!("✅ 训练数据准备完成");
    
    // 训练模型
    println!("\n开始模型训练...");
    let linear_metrics = ModelFactory::train_model(&mut linear_model, &training_data, 3).await?;
    println!("线性回归训练完成，最终损失: {:.4}", linear_metrics.last().unwrap().loss);
    
    let neural_metrics = ModelFactory::train_model(&mut neural_model, &training_data, 3).await?;
    println!("神经网络训练完成，最终损失: {:.4}", neural_metrics.last().unwrap().loss);
    
    // 模型评估
    println!("\n开始模型评估...");
    let test_data = vec![
        (vec![1.5, 2.5, 3.5], 7.5),
        (vec![2.5, 3.5, 4.5], 10.5),
    ];
    
    let linear_evaluation = ModelEvaluator::evaluate_model(&linear_model, &test_data).await?;
    println!("线性回归评估结果 - MSE: {:.4}, 准确率: {:.2}%", 
             linear_evaluation.mse, linear_evaluation.accuracy * 100.0);
    
    let neural_evaluation = ModelEvaluator::evaluate_model(&neural_model, &test_data).await?;
    println!("神经网络评估结果 - MSE: {:.4}, 准确率: {:.2}%", 
             neural_evaluation.mse, neural_evaluation.accuracy * 100.0);
    
    Ok(())
}

/// 演示性能对比
async fn demonstrate_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("开始性能对比测试...");
    
    let start = Instant::now();
    let results = PerformanceTester::run_comprehensive_benchmark().await;
    let test_duration = start.elapsed();
    
    println!("性能测试完成，耗时: {:?}", test_duration);
    
    // 生成性能报告
    let report = PerformanceTester::generate_performance_report(&results);
    println!("\n{}", report);
    
    // 内存使用分析
    let memory_usage = MemoryAnalyzer::analyze_memory_usage();
    let memory_report = MemoryAnalyzer::generate_memory_report(&memory_usage);
    println!("{}", memory_report);
    
    Ok(())
}

/// 演示综合应用
async fn demonstrate_comprehensive_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("创建综合AI系统...");
    
    // 创建不同类型的模型
    let mut linear_model = ModelFactory::create_linear_regression(10, 0.01);
    let mut neural_model = ModelFactory::create_neural_network(&[10, 20, 1], 0.01);
    
    // 创建GAT模型用于对比
    let gat_model = GATLinearModel::new(10);
    let tait_model = TAITLinearModel::new(10);
    
    // 准备大规模训练数据
    let training_data: Vec<(Vec<f64>, f64)> = (0..1000)
        .map(|i| {
            let input = (0..10).map(|j| (i + j) as f64).collect();
            let target = i as f64 * 2.0;
            (input, target)
        })
        .collect();
    
    println!("✅ 训练数据准备完成 (1000个样本)");
    
    // 训练模型
    println!("\n开始模型训练...");
    let training_start = Instant::now();
    
    let linear_metrics = ModelFactory::train_model(&mut linear_model, &training_data, 5).await?;
    let neural_metrics = ModelFactory::train_model(&mut neural_model, &training_data, 5).await?;
    
    let training_duration = training_start.elapsed();
    println!("训练完成，耗时: {:?}", training_duration);
    
    // 性能测试
    println!("\n开始性能测试...");
    let test_inputs: Vec<Vec<f64>> = (0..100)
        .map(|i| (0..10).map(|j| (i + j) as f64).collect())
        .collect();
    
    // 测试不同方法的性能
    let perf_start = Instant::now();
    
    // 传统方法
    let traditional_model = TraditionalLinearModel::new(10);
    let traditional_result = PerformanceTester::test_traditional_sync(&traditional_model, &test_inputs, 10);
    
    // GAT方法
    let gat_result = PerformanceTester::test_gat(&gat_model, &test_inputs, 10).await;
    
    // TAIT方法
    let tait_result = PerformanceTester::test_tait(&tait_model, &test_inputs, 10).await;
    
    let perf_duration = perf_start.elapsed();
    println!("性能测试完成，耗时: {:?}", perf_duration);
    
    // 显示结果
    println!("\n=== 性能对比结果 ===");
    println!("传统方法: {:.2} ops/s", traditional_result.throughput);
    println!("GAT方法: {:.2} ops/s", gat_result.throughput);
    println!("TAIT方法: {:.2} ops/s", tait_result.throughput);
    
    // 模型评估
    println!("\n开始模型评估...");
    let test_data: Vec<(Vec<f64>, f64)> = (1000..1100)
        .map(|i| {
            let input = (0..10).map(|j| (i + j) as f64).collect();
            let target = i as f64 * 2.0;
            (input, target)
        })
        .collect();
    
    let linear_eval = ModelEvaluator::evaluate_model(&linear_model, &test_data).await?;
    let neural_eval = ModelEvaluator::evaluate_model(&neural_model, &test_data).await?;
    
    println!("线性回归 - MSE: {:.4}, 准确率: {:.2}%", 
             linear_eval.mse, linear_eval.accuracy * 100.0);
    println!("神经网络 - MSE: {:.4}, 准确率: {:.2}%", 
             neural_eval.mse, neural_eval.accuracy * 100.0);
    
    // 展示Rust 1.90特性优势
    println!("\n=== Rust 1.90特性优势总结 ===");
    println!("🎯 类型安全性: GAT和TAIT提供了更好的类型检查");
    println!("⚡ 性能提升: 减少了运行时开销，提高了执行效率");
    println!("🔧 代码简洁: 简化了复杂的类型定义，提高了可读性");
    println!("🚀 开发体验: 更好的IDE支持和错误提示");
    println!("🛡️ 内存安全: 保持了Rust的内存安全保证");
    
    Ok(())
}

/// 辅助函数：格式化时间
fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{}ms", duration.as_millis())
    } else {
        format!("{}μs", duration.as_micros())
    }
}

/// 辅助函数：格式化吞吐量
fn format_throughput(throughput: f64) -> String {
    if throughput >= 1_000_000.0 {
        format!("{:.2}M ops/s", throughput / 1_000_000.0)
    } else if throughput >= 1_000.0 {
        format!("{:.2}K ops/s", throughput / 1_000.0)
    } else {
        format!("{:.2} ops/s", throughput)
    }
}
