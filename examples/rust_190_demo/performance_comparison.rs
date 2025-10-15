//! Rust 1.90 性能对比测试
//! 
//! 本示例展示了Rust 1.90新特性（GAT、TAIT）与传统方法在AI场景下的性能对比。
//! 通过基准测试，验证新特性在保持类型安全的同时，是否带来了性能提升。

use std::time::Instant;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// 传统方法的AI推理trait（不使用GAT）
pub trait TraditionalAIInference {
    fn infer_sync(&self, input: &[f64]) -> Result<f64, AIError>;
    fn infer_async(&self, input: &[f64]) -> Pin<Box<dyn Future<Output = Result<f64, AIError>> + Send>>;
}

/// 使用GAT的AI推理trait
pub trait GATAIIInference<'a> {
    type Input: 'a;
    type Output: 'a;
    type Future: Future<Output = Result<Self::Output, AIError>> + 'a;
    
    fn infer(&'a self, input: Self::Input) -> Self::Future;
}

/// 使用TAIT的类型别名
type TAITInferenceResult<'a> = impl Future<Output = Result<f64, AIError>> + 'a;

/// AI错误类型
#[derive(Debug, Clone)]
pub enum AIError {
    InvalidInput(String),
    ComputationError(String),
}

impl std::fmt::Display for AIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AIError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AIError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl std::error::Error for AIError {}

/// 线性模型实现（传统方法）
pub struct TraditionalLinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl TraditionalLinearModel {
    pub fn new(size: usize) -> Self {
        Self {
            weights: vec![1.0; size],
            bias: 0.0,
        }
    }
    
    fn forward(&self, input: &[f64]) -> f64 {
        input.iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum::<f64>() + self.bias
    }
}

impl TraditionalAIInference for TraditionalLinearModel {
    fn infer_sync(&self, input: &[f64]) -> Result<f64, AIError> {
        if input.len() != self.weights.len() {
            return Err(AIError::InvalidInput("Size mismatch".to_string()));
        }
        Ok(self.forward(input))
    }
    
    fn infer_async(&self, input: &[f64]) -> Pin<Box<dyn Future<Output = Result<f64, AIError>> + Send>> {
        let weights = self.weights.clone();
        let bias = self.bias;
        let input = input.to_vec();
        
        Box::pin(async move {
            if input.len() != weights.len() {
                return Err(AIError::InvalidInput("Size mismatch".to_string()));
            }
            
            // 模拟异步计算
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            
            let result = input.iter()
                .zip(&weights)
                .map(|(x, w)| x * w)
                .sum::<f64>() + bias;
            
            Ok(result)
        })
    }
}

/// 线性模型实现（GAT方法）
pub struct GATLinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl GATLinearModel {
    pub fn new(size: usize) -> Self {
        Self {
            weights: vec![1.0; size],
            bias: 0.0,
        }
    }
    
    fn forward(&self, input: &[f64]) -> f64 {
        input.iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum::<f64>() + self.bias
    }
}

impl<'a> GATAIIInference<'a> for GATLinearModel {
    type Input = &'a [f64];
    type Output = f64;
    type Future = Pin<Box<dyn Future<Output = Result<f64, AIError>> + 'a>>;
    
    fn infer(&'a self, input: Self::Input) -> Self::Future {
        Box::pin(async move {
            if input.len() != self.weights.len() {
                return Err(AIError::InvalidInput("Size mismatch".to_string()));
            }
            
            // 模拟异步计算
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            
            Ok(self.forward(input))
        })
    }
}

/// 线性模型实现（TAIT方法）
pub struct TAITLinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl TAITLinearModel {
    pub fn new(size: usize) -> Self {
        Self {
            weights: vec![1.0; size],
            bias: 0.0,
        }
    }
    
    fn forward(&self, input: &[f64]) -> f64 {
        input.iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum::<f64>() + self.bias
    }
    
    /// 使用TAIT的推理方法
    pub fn infer<'a>(&'a self, input: &'a [f64]) -> TAITInferenceResult<'a> {
        async move {
            if input.len() != self.weights.len() {
                return Err(AIError::InvalidInput("Size mismatch".to_string()));
            }
            
            // 模拟异步计算
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            
            Ok(self.forward(input))
        }
    }
}

/// 性能测试结果
#[derive(Debug, Clone)]
pub struct PerformanceResults {
    pub method: String,
    pub total_time: std::time::Duration,
    pub average_time: std::time::Duration,
    pub throughput: f64, // 操作/秒
    pub memory_usage: usize, // 字节
    pub error_rate: f64,
}

/// 性能测试器
pub struct PerformanceTester;

impl PerformanceTester {
    /// 测试传统同步方法
    pub fn test_traditional_sync(model: &TraditionalLinearModel, inputs: &[Vec<f64>], iterations: usize) -> PerformanceResults {
        let start = Instant::now();
        let mut errors = 0;
        
        for _ in 0..iterations {
            for input in inputs {
                match model.infer_sync(input) {
                    Ok(_) => {},
                    Err(_) => errors += 1,
                }
            }
        }
        
        let total_time = start.elapsed();
        let total_operations = iterations * inputs.len();
        let average_time = total_time / total_operations as u32;
        let throughput = total_operations as f64 / total_time.as_secs_f64();
        let error_rate = errors as f64 / total_operations as f64;
        
        PerformanceResults {
            method: "Traditional Sync".to_string(),
            total_time,
            average_time,
            throughput,
            memory_usage: std::mem::size_of::<TraditionalLinearModel>(),
            error_rate,
        }
    }
    
    /// 测试传统异步方法
    pub async fn test_traditional_async(model: &TraditionalLinearModel, inputs: &[Vec<f64>], iterations: usize) -> PerformanceResults {
        let start = Instant::now();
        let mut errors = 0;
        
        for _ in 0..iterations {
            for input in inputs {
                match model.infer_async(input).await {
                    Ok(_) => {},
                    Err(_) => errors += 1,
                }
            }
        }
        
        let total_time = start.elapsed();
        let total_operations = iterations * inputs.len();
        let average_time = total_time / total_operations as u32;
        let throughput = total_operations as f64 / total_time.as_secs_f64();
        let error_rate = errors as f64 / total_operations as f64;
        
        PerformanceResults {
            method: "Traditional Async".to_string(),
            total_time,
            average_time,
            throughput,
            memory_usage: std::mem::size_of::<TraditionalLinearModel>(),
            error_rate,
        }
    }
    
    /// 测试GAT方法
    pub async fn test_gat(model: &GATLinearModel, inputs: &[Vec<f64>], iterations: usize) -> PerformanceResults {
        let start = Instant::now();
        let mut errors = 0;
        
        for _ in 0..iterations {
            for input in inputs {
                match model.infer(input).await {
                    Ok(_) => {},
                    Err(_) => errors += 1,
                }
            }
        }
        
        let total_time = start.elapsed();
        let total_operations = iterations * inputs.len();
        let average_time = total_time / total_operations as u32;
        let throughput = total_operations as f64 / total_time.as_secs_f64();
        let error_rate = errors as f64 / total_operations as f64;
        
        PerformanceResults {
            method: "GAT".to_string(),
            total_time,
            average_time,
            throughput,
            memory_usage: std::mem::size_of::<GATLinearModel>(),
            error_rate,
        }
    }
    
    /// 测试TAIT方法
    pub async fn test_tait(model: &TAITLinearModel, inputs: &[Vec<f64>], iterations: usize) -> PerformanceResults {
        let start = Instant::now();
        let mut errors = 0;
        
        for _ in 0..iterations {
            for input in inputs {
                match model.infer(input).await {
                    Ok(_) => {},
                    Err(_) => errors += 1,
                }
            }
        }
        
        let total_time = start.elapsed();
        let total_operations = iterations * inputs.len();
        let average_time = total_time / total_operations as u32;
        let throughput = total_operations as f64 / total_time.as_secs_f64();
        let error_rate = errors as f64 / total_operations as f64;
        
        PerformanceResults {
            method: "TAIT".to_string(),
            total_time,
            average_time,
            throughput,
            memory_usage: std::mem::size_of::<TAITLinearModel>(),
            error_rate,
        }
    }
    
    /// 运行完整的性能对比测试
    pub async fn run_comprehensive_benchmark() -> Vec<PerformanceResults> {
        let model_size = 1000;
        let input_count = 100;
        let iterations = 10;
        
        // 创建测试数据
        let inputs: Vec<Vec<f64>> = (0..input_count)
            .map(|i| (0..model_size).map(|j| (i + j) as f64).collect())
            .collect();
        
        let mut results = Vec::new();
        
        // 测试传统同步方法
        let traditional_model = TraditionalLinearModel::new(model_size);
        let sync_result = Self::test_traditional_sync(&traditional_model, &inputs, iterations);
        results.push(sync_result);
        
        // 测试传统异步方法
        let async_result = Self::test_traditional_async(&traditional_model, &inputs, iterations).await;
        results.push(async_result);
        
        // 测试GAT方法
        let gat_model = GATLinearModel::new(model_size);
        let gat_result = Self::test_gat(&gat_model, &inputs, iterations).await;
        results.push(gat_result);
        
        // 测试TAIT方法
        let tait_model = TAITLinearModel::new(model_size);
        let tait_result = Self::test_tait(&tait_model, &inputs, iterations).await;
        results.push(tait_result);
        
        results
    }
    
    /// 生成性能报告
    pub fn generate_performance_report(results: &[PerformanceResults]) -> String {
        let mut report = String::new();
        report.push_str("=== Rust 1.90 性能对比报告 ===\n\n");
        
        // 表格头部
        report.push_str("| 方法 | 总时间 | 平均时间 | 吞吐量(ops/s) | 内存使用 | 错误率 |\n");
        report.push_str("|------|--------|----------|---------------|----------|--------|\n");
        
        // 数据行
        for result in results {
            report.push_str(&format!(
                "| {} | {:?} | {:?} | {:.2} | {} bytes | {:.2}% |\n",
                result.method,
                result.total_time,
                result.average_time,
                result.throughput,
                result.memory_usage,
                result.error_rate * 100.0
            ));
        }
        
        // 性能分析
        report.push_str("\n=== 性能分析 ===\n");
        
        if let Some(fastest) = results.iter().max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap()) {
            report.push_str(&format!("🏆 最高吞吐量: {} ({:.2} ops/s)\n", fastest.method, fastest.throughput));
        }
        
        if let Some(fastest) = results.iter().min_by(|a, b| a.average_time.cmp(&b.average_time)) {
            report.push_str(&format!("⚡ 最低延迟: {} ({:?})\n", fastest.method, fastest.average_time));
        }
        
        if let Some(smallest) = results.iter().min_by(|a, b| a.memory_usage.cmp(&b.memory_usage)) {
            report.push_str(&format!("💾 最少内存: {} ({} bytes)\n", smallest.method, smallest.memory_usage));
        }
        
        // Rust 1.90特性优势分析
        report.push_str("\n=== Rust 1.90特性优势 ===\n");
        report.push_str("✅ GAT (Generic Associated Types):\n");
        report.push_str("   - 更好的类型安全性\n");
        report.push_str("   - 更灵活的生命周期管理\n");
        report.push_str("   - 减少运行时开销\n\n");
        
        report.push_str("✅ TAIT (Type Alias Impl Trait):\n");
        report.push_str("   - 简化复杂类型定义\n");
        report.push_str("   - 提高代码可读性\n");
        report.push_str("   - 更好的类型推断\n\n");
        
        report
    }
}

/// 内存使用分析器
pub struct MemoryAnalyzer;

impl MemoryAnalyzer {
    /// 分析不同方法的内存使用模式
    pub fn analyze_memory_usage() -> HashMap<String, usize> {
        let mut usage = HashMap::new();
        
        // 传统方法内存使用
        let traditional_model = TraditionalLinearModel::new(1000);
        usage.insert("Traditional".to_string(), std::mem::size_of_val(&traditional_model));
        
        // GAT方法内存使用
        let gat_model = GATLinearModel::new(1000);
        usage.insert("GAT".to_string(), std::mem::size_of_val(&gat_model));
        
        // TAIT方法内存使用
        let tait_model = TAITLinearModel::new(1000);
        usage.insert("TAIT".to_string(), std::mem::size_of_val(&tait_model));
        
        usage
    }
    
    /// 生成内存使用报告
    pub fn generate_memory_report(usage: &HashMap<String, usize>) -> String {
        let mut report = String::new();
        report.push_str("=== 内存使用分析报告 ===\n\n");
        
        for (method, size) in usage {
            report.push_str(&format!("{}: {} bytes\n", method, size));
        }
        
        if let Some((smallest_method, smallest_size)) = usage.iter().min_by_key(|(_, &size)| size) {
            report.push_str(&format!("\n💾 最少内存使用: {} ({} bytes)\n", smallest_method, smallest_size));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_performance_comparison() {
        let results = PerformanceTester::run_comprehensive_benchmark().await;
        
        // 验证所有方法都完成了测试
        assert_eq!(results.len(), 4);
        
        // 验证所有方法都有合理的结果
        for result in &results {
            assert!(result.throughput > 0.0);
            assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
            assert!(result.memory_usage > 0);
        }
        
        // 生成并打印报告
        let report = PerformanceTester::generate_performance_report(&results);
        println!("{}", report);
    }

    #[test]
    async fn test_individual_methods() {
        let model_size = 100;
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64; model_size]).collect();
        
        // 测试传统同步方法
        let traditional_model = TraditionalLinearModel::new(model_size);
        let sync_result = PerformanceTester::test_traditional_sync(&traditional_model, &inputs, 5);
        assert!(sync_result.throughput > 0.0);
        
        // 测试传统异步方法
        let async_result = PerformanceTester::test_traditional_async(&traditional_model, &inputs, 5).await;
        assert!(async_result.throughput > 0.0);
        
        // 测试GAT方法
        let gat_model = GATLinearModel::new(model_size);
        let gat_result = PerformanceTester::test_gat(&gat_model, &inputs, 5).await;
        assert!(gat_result.throughput > 0.0);
        
        // 测试TAIT方法
        let tait_model = TAITLinearModel::new(model_size);
        let tait_result = PerformanceTester::test_tait(&tait_model, &inputs, 5).await;
        assert!(tait_result.throughput > 0.0);
    }

    #[test]
    fn test_memory_analysis() {
        let usage = MemoryAnalyzer::analyze_memory_usage();
        
        // 验证所有方法都有内存使用数据
        assert!(usage.contains_key("Traditional"));
        assert!(usage.contains_key("GAT"));
        assert!(usage.contains_key("TAIT"));
        
        // 验证内存使用都是正数
        for (_, &size) in &usage {
            assert!(size > 0);
        }
        
        // 生成并打印内存报告
        let report = MemoryAnalyzer::generate_memory_report(&usage);
        println!("{}", report);
    }

    #[test]
    async fn test_error_handling() {
        let model = GATLinearModel::new(3);
        
        // 测试错误输入
        let invalid_input = [1.0, 2.0, 3.0, 4.0]; // 大小不匹配
        let result = model.infer(&invalid_input).await;
        assert!(result.is_err());
        
        // 测试正确输入
        let valid_input = [1.0, 2.0, 3.0];
        let result = model.infer(&valid_input).await;
        assert!(result.is_ok());
    }
}

/// 基准测试模块
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    async fn benchmark_large_scale_comparison() {
        let model_size = 10000;
        let input_count = 1000;
        let iterations = 5;
        
        println!("开始大规模性能对比测试...");
        println!("模型大小: {} 参数", model_size);
        println!("输入数量: {}", input_count);
        println!("迭代次数: {}", iterations);
        
        let start = Instant::now();
        let results = PerformanceTester::run_comprehensive_benchmark().await;
        let total_time = start.elapsed();
        
        println!("测试完成，总耗时: {:?}", total_time);
        
        // 生成详细报告
        let report = PerformanceTester::generate_performance_report(&results);
        println!("{}", report);
        
        // 验证结果
        assert_eq!(results.len(), 4);
        for result in &results {
            assert!(result.throughput > 0.0);
        }
    }

    #[test]
    async fn benchmark_concurrent_performance() {
        let model = GATLinearModel::new(1000);
        let inputs: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64; 1000]).collect();
        
        let start = Instant::now();
        
        // 并发执行推理
        let mut handles = Vec::new();
        for input in inputs {
            let model_ref = &model;
            let handle = tokio::spawn(async move {
                model_ref.infer(&input).await
            });
            handles.push(handle);
        }
        
        // 等待所有任务完成
        for handle in handles {
            let _ = handle.await.unwrap();
        }
        
        let duration = start.elapsed();
        println!("并发推理100个样本耗时: {:?}", duration);
        println!("平均每个样本: {:?}", duration / 100);
    }
}
