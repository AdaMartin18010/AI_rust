//! 基准测试套件
//! 
//! 提供性能对比、压力测试和性能分析功能

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// 基准测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration: Duration,
    pub operations: u64,
    pub operations_per_second: f64,
    pub memory_usage: u64,
    pub cpu_usage: f64,
    pub gpu_usage: Option<f64>,
    pub error_rate: f64,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

/// 性能对比结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub test_name: String,
    pub results: Vec<BenchmarkResult>,
    pub winner: Option<String>,
    pub improvement_percentage: Option<f64>,
    pub summary: String,
}

/// 压力测试配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestConfig {
    pub duration: Duration,
    pub concurrent_users: u32,
    pub requests_per_second: u32,
    pub ramp_up_duration: Duration,
    pub ramp_down_duration: Duration,
    pub target_error_rate: f64,
    pub max_response_time: Duration,
}

/// 压力测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    pub config: StressTestConfig,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time: Duration,
    pub min_response_time: Duration,
    pub max_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub cpu_usage_avg: f64,
    pub memory_usage_avg: u64,
    pub gpu_usage_avg: Option<f64>,
    pub test_duration: Duration,
    pub passed: bool,
}

/// 基准测试套件
pub struct BenchmarkSuite {
    results: Vec<BenchmarkResult>,
    stress_test_results: Vec<StressTestResult>,
    performance_comparisons: Vec<PerformanceComparison>,
}

impl BenchmarkSuite {
    /// 创建新的基准测试套件
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            stress_test_results: Vec::new(),
            performance_comparisons: Vec::new(),
        }
    }
    
    /// 运行单个基准测试
    pub async fn run_benchmark<F, Fut>(&mut self, name: &str, operations: u64, test_fn: F) -> BenchmarkResult
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<(), String>>,
    {
        println!("🔄 运行基准测试: {}", name);
        
        let start_time = Instant::now();
        let start_memory = self.get_memory_usage();
        let start_cpu = self.get_cpu_usage();
        let start_gpu = self.get_gpu_usage();
        
        let mut error_count = 0u64;
        
        // 运行指定次数的操作
        for _ in 0..operations {
            match test_fn().await {
                Ok(_) => {},
                Err(_) => {
                    error_count += 1;
                }
            }
        }
        
        let duration = start_time.elapsed();
        let end_memory = self.get_memory_usage();
        let end_cpu = self.get_cpu_usage();
        let end_gpu = self.get_gpu_usage();
        
        let operations_per_second = operations as f64 / duration.as_secs_f64();
        let error_rate = (error_count as f64 / operations as f64) * 100.0;
        
        let result = BenchmarkResult {
            name: name.to_string(),
            duration,
            operations,
            operations_per_second,
            memory_usage: end_memory.saturating_sub(start_memory),
            cpu_usage: (start_cpu + end_cpu) / 2.0,
            gpu_usage: if let (Some(start), Some(end)) = (start_gpu, end_gpu) {
                Some((start + end) / 2.0)
            } else {
                None
            },
            error_rate,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        self.results.push(result.clone());
        
        println!("   ✅ 完成: {:.2} ops/sec, {:.2}ms 平均耗时, {:.2}% 错误率", 
                result.operations_per_second, 
                duration.as_millis() as f64 / operations as f64,
                result.error_rate);
        
        result
    }
    
    /// 运行压力测试
    pub async fn run_stress_test(&mut self, config: StressTestConfig, test_fn: impl Fn() -> Result<(), String>) -> StressTestResult {
        println!("🔥 运行压力测试: {}秒, {}并发用户", 
                config.duration.as_secs(), config.concurrent_users);
        
        let start_time = Instant::now();
        let mut total_requests = 0u64;
        let mut successful_requests = 0u64;
        let mut failed_requests = 0u64;
        let mut response_times = Vec::new();
        
        let mut cpu_samples = Vec::new();
        let mut memory_samples = Vec::new();
        let mut gpu_samples = Vec::new();
        
        // 模拟压力测试
        let end_time = start_time + config.duration;
        let mut last_sample = start_time;
        
        while Instant::now() < end_time {
            let current_time = Instant::now();
            
            // 模拟并发请求
            for _ in 0..config.concurrent_users {
                if total_requests % (config.requests_per_second as u64) == 0 {
                    let request_start = Instant::now();
                    
                    match test_fn() {
                        Ok(_) => {
                            successful_requests += 1;
                            response_times.push(request_start.elapsed());
                        },
                        Err(_) => {
                            failed_requests += 1;
                            response_times.push(request_start.elapsed());
                        }
                    }
                    
                    total_requests += 1;
                }
            }
            
            // 定期采样系统资源
            if current_time.duration_since(last_sample) >= Duration::from_secs(1) {
                cpu_samples.push(self.get_cpu_usage());
                memory_samples.push(self.get_memory_usage());
                if let Some(gpu) = self.get_gpu_usage() {
                    gpu_samples.push(gpu);
                }
                last_sample = current_time;
            }
            
            // 控制请求速率
            tokio::time::sleep(Duration::from_millis(1000 / config.requests_per_second as u64)).await;
        }
        
        let test_duration = start_time.elapsed();
        let requests_per_second = total_requests as f64 / test_duration.as_secs_f64();
        let error_rate = (failed_requests as f64 / total_requests as f64) * 100.0;
        
        // 计算响应时间统计
        response_times.sort();
        let average_response_time = if !response_times.is_empty() {
            response_times.iter().sum::<Duration>() / response_times.len() as u32
        } else {
            Duration::from_millis(0)
        };
        
        let min_response_time = response_times.first().copied().unwrap_or_default();
        let max_response_time = response_times.last().copied().unwrap_or_default();
        
        let p95_index = (response_times.len() as f64 * 0.95) as usize;
        let p95_response_time = response_times.get(p95_index).copied().unwrap_or_default();
        
        let p99_index = (response_times.len() as f64 * 0.99) as usize;
        let p99_response_time = response_times.get(p99_index).copied().unwrap_or_default();
        
        // 计算平均资源使用率
        let cpu_usage_avg = if !cpu_samples.is_empty() {
            cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64
        } else {
            0.0
        };
        
        let memory_usage_avg = if !memory_samples.is_empty() {
            memory_samples.iter().sum::<u64>() / memory_samples.len() as u64
        } else {
            0
        };
        
        let gpu_usage_avg = if !gpu_samples.is_empty() {
            Some(gpu_samples.iter().sum::<f64>() / gpu_samples.len() as f64)
        } else {
            None
        };
        
        // 判断测试是否通过
        let passed = error_rate <= config.target_error_rate && 
                    average_response_time <= config.max_response_time;
        
        let result = StressTestResult {
            config: config.clone(),
            total_requests,
            successful_requests,
            failed_requests,
            average_response_time,
            min_response_time,
            max_response_time,
            p95_response_time,
            p99_response_time,
            requests_per_second,
            error_rate,
            cpu_usage_avg,
            memory_usage_avg,
            gpu_usage_avg,
            test_duration,
            passed,
        };
        
        self.stress_test_results.push(result.clone());
        
        println!("   📊 压力测试结果:");
        println!("      • 总请求数: {}", total_requests);
        println!("      • 成功率: {:.2}%", (successful_requests as f64 / total_requests as f64) * 100.0);
        println!("      • 平均响应时间: {:.2}ms", average_response_time.as_millis());
        println!("      • 请求速率: {:.2} req/s", requests_per_second);
        println!("      • 测试状态: {}", if passed { "✅ 通过" } else { "❌ 失败" });
        
        result
    }
    
    /// 比较性能结果
    pub fn compare_performance(&mut self, test_name: &str, result_names: Vec<&str>) -> PerformanceComparison {
        let mut results = Vec::new();
        
        for name in &result_names {
            if let Some(result) = self.results.iter().find(|r| r.name == *name) {
                results.push(result.clone());
            }
        }
        
        if results.len() < 2 {
            return PerformanceComparison {
                test_name: test_name.to_string(),
                results,
                winner: None,
                improvement_percentage: None,
                summary: "需要至少2个结果进行比较".to_string(),
            };
        }
        
        // 按性能排序（操作数/秒）
        results.sort_by(|a, b| b.operations_per_second.partial_cmp(&a.operations_per_second).unwrap());
        
        let winner = results.first().map(|r| r.name.clone());
        let baseline = results.last().unwrap();
        let best = results.first().unwrap();
        
        let improvement_percentage = if baseline.operations_per_second > 0.0 {
            Some(((best.operations_per_second - baseline.operations_per_second) / baseline.operations_per_second) * 100.0)
        } else {
            None
        };
        
        let summary = format!(
            "最佳性能: {} ({:.2} ops/sec), 基准: {} ({:.2} ops/sec), 提升: {:.2}%",
            best.name, best.operations_per_second,
            baseline.name, baseline.operations_per_second,
            improvement_percentage.unwrap_or(0.0)
        );
        
        let comparison = PerformanceComparison {
            test_name: test_name.to_string(),
            results,
            winner,
            improvement_percentage,
            summary,
        };
        
        self.performance_comparisons.push(comparison.clone());
        comparison
    }
    
    /// 生成性能报告
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("# 基准测试报告\n\n");
        
        // 基准测试结果
        report.push_str("## 基准测试结果\n\n");
        for result in &self.results {
            report.push_str(&format!("### {}\n", result.name));
            report.push_str(&format!("- 操作数: {}\n", result.operations));
            report.push_str(&format!("- 操作数/秒: {:.2}\n", result.operations_per_second));
            report.push_str(&format!("- 耗时: {:.2}ms\n", result.duration.as_millis()));
            report.push_str(&format!("- 内存使用: {} bytes\n", result.memory_usage));
            report.push_str(&format!("- CPU使用率: {:.2}%\n", result.cpu_usage));
            if let Some(gpu) = result.gpu_usage {
                report.push_str(&format!("- GPU使用率: {:.2}%\n", gpu));
            }
            report.push_str(&format!("- 错误率: {:.2}%\n\n", result.error_rate));
        }
        
        // 压力测试结果
        if !self.stress_test_results.is_empty() {
            report.push_str("## 压力测试结果\n\n");
            for result in &self.stress_test_results {
                report.push_str(&format!("### 压力测试 - {}并发用户\n", result.config.concurrent_users));
                report.push_str(&format!("- 总请求数: {}\n", result.total_requests));
                report.push_str(&format!("- 成功请求数: {}\n", result.successful_requests));
                report.push_str(&format!("- 失败请求数: {}\n", result.failed_requests));
                report.push_str(&format!("- 平均响应时间: {:.2}ms\n", result.average_response_time.as_millis()));
                report.push_str(&format!("- P95响应时间: {:.2}ms\n", result.p95_response_time.as_millis()));
                report.push_str(&format!("- P99响应时间: {:.2}ms\n", result.p99_response_time.as_millis()));
                report.push_str(&format!("- 请求速率: {:.2} req/s\n", result.requests_per_second));
                report.push_str(&format!("- 错误率: {:.2}%\n", result.error_rate));
                report.push_str(&format!("- 测试状态: {}\n\n", if result.passed { "通过" } else { "失败" }));
            }
        }
        
        // 性能对比
        if !self.performance_comparisons.is_empty() {
            report.push_str("## 性能对比\n\n");
            for comparison in &self.performance_comparisons {
                report.push_str(&format!("### {}\n", comparison.test_name));
                report.push_str(&format!("{}\n\n", comparison.summary));
            }
        }
        
        report
    }
    
    /// 获取所有基准测试结果
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }
    
    /// 获取所有压力测试结果
    pub fn get_stress_test_results(&self) -> &[StressTestResult] {
        &self.stress_test_results
    }
    
    /// 获取所有性能对比结果
    pub fn get_performance_comparisons(&self) -> &[PerformanceComparison] {
        &self.performance_comparisons
    }
    
    // 私有辅助方法
    
    fn get_memory_usage(&self) -> u64 {
        // 模拟内存使用量获取
        // 在实际实现中，这里会调用系统API
        1024 * 1024 * 100 // 100MB
    }
    
    fn get_cpu_usage(&self) -> f64 {
        // 模拟CPU使用率获取
        // 在实际实现中，这里会调用系统API
        45.0
    }
    
    fn get_gpu_usage(&self) -> Option<f64> {
        // 模拟GPU使用率获取
        // 在实际实现中，这里会调用GPU API
        Some(65.0)
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert_eq!(suite.results.len(), 0);
        assert_eq!(suite.stress_test_results.len(), 0);
        assert_eq!(suite.performance_comparisons.len(), 0);
    }
    
    #[tokio::test]
    async fn test_run_benchmark() {
        let mut suite = BenchmarkSuite::new();
        
        let result = suite.run_benchmark("test_benchmark", 100, || async {
            // 模拟一些工作
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(())
        }).await;
        
        assert_eq!(result.name, "test_benchmark");
        assert_eq!(result.operations, 100);
        assert!(result.operations_per_second > 0.0);
    }
    
    #[tokio::test]
    async fn test_stress_test_config() {
        let config = StressTestConfig {
            duration: Duration::from_secs(10),
            concurrent_users: 10,
            requests_per_second: 100,
            ramp_up_duration: Duration::from_secs(2),
            ramp_down_duration: Duration::from_secs(2),
            target_error_rate: 1.0,
            max_response_time: Duration::from_millis(100),
        };
        
        assert_eq!(config.duration.as_secs(), 10);
        assert_eq!(config.concurrent_users, 10);
        assert_eq!(config.requests_per_second, 100);
    }
    
    #[tokio::test]
    async fn test_performance_comparison() {
        let mut suite = BenchmarkSuite::new();
        
        // 添加一些测试结果
        suite.results.push(BenchmarkResult {
            name: "test1".to_string(),
            duration: Duration::from_millis(100),
            operations: 100,
            operations_per_second: 1000.0,
            memory_usage: 1024,
            cpu_usage: 50.0,
            gpu_usage: None,
            error_rate: 0.0,
            timestamp: 0,
            metadata: HashMap::new(),
        });
        
        suite.results.push(BenchmarkResult {
            name: "test2".to_string(),
            duration: Duration::from_millis(200),
            operations: 100,
            operations_per_second: 500.0,
            memory_usage: 2048,
            cpu_usage: 60.0,
            gpu_usage: None,
            error_rate: 0.0,
            timestamp: 0,
            metadata: HashMap::new(),
        });
        
        let comparison = suite.compare_performance("性能对比测试", vec!["test1", "test2"]);
        
        assert_eq!(comparison.test_name, "性能对比测试");
        assert_eq!(comparison.results.len(), 2);
        assert!(comparison.winner.is_some());
        assert!(comparison.improvement_percentage.is_some());
    }
}
