//! 性能优化和监控系统
//! 
//! 本示例展示了一个完整的性能优化和监控系统，包括：
//! - 性能指标收集
//! - 实时监控
//! - 性能分析
//! - 自动优化
//! - 报告生成

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use tracing::{info, warn, error, debug};

/// 性能指标类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Timer,
}

/// 性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub name: String,
    pub metric_type: MetricType,
    pub value: f64,
    pub timestamp: u64,
    pub tags: HashMap<String, String>,
    pub unit: String,
}

/// 性能统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub name: String,
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub timestamp: u64,
}

/// 性能报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub report_id: String,
    pub start_time: u64,
    pub end_time: u64,
    pub duration: u64,
    pub metrics: Vec<PerformanceMetric>,
    pub stats: Vec<PerformanceStats>,
    pub summary: PerformanceSummary,
}

/// 性能摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time: f64,
    pub max_response_time: f64,
    pub min_response_time: f64,
    pub throughput: f64,
    pub error_rate: f64,
}

/// 性能监控器
pub struct PerformanceMonitor {
    pub metrics: Arc<RwLock<HashMap<String, Vec<PerformanceMetric>>>>,
    pub stats: Arc<RwLock<HashMap<String, PerformanceStats>>>,
    pub report_sender: mpsc::UnboundedSender<PerformanceReport>,
    pub report_receiver: Arc<RwLock<mpsc::UnboundedReceiver<PerformanceReport>>>,
    pub start_time: Instant,
}

impl PerformanceMonitor {
    /// 创建新的性能监控器
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
            report_sender: sender,
            report_receiver: Arc::new(RwLock::new(receiver)),
            start_time: Instant::now(),
        }
    }
    
    /// 记录性能指标
    pub async fn record_metric(&self, metric: PerformanceMetric) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        let metric_list = metrics.entry(metric.name.clone()).or_insert_with(Vec::new);
        metric_list.push(metric.clone());
        
        // 保持最近1000个指标
        if metric_list.len() > 1000 {
            metric_list.drain(0..100);
        }
        
        debug!("Recorded metric: {} = {}", metric.name, metric.value);
        Ok(())
    }
    
    /// 记录计数器指标
    pub async fn increment_counter(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<()> {
        let metric = PerformanceMetric {
            name: name.to_string(),
            metric_type: MetricType::Counter,
            value,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            tags,
            unit: "count".to_string(),
        };
        
        self.record_metric(metric).await
    }
    
    /// 记录仪表盘指标
    pub async fn set_gauge(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<()> {
        let metric = PerformanceMetric {
            name: name.to_string(),
            metric_type: MetricType::Gauge,
            value,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            tags,
            unit: "gauge".to_string(),
        };
        
        self.record_metric(metric).await
    }
    
    /// 记录直方图指标
    pub async fn record_histogram(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<()> {
        let metric = PerformanceMetric {
            name: name.to_string(),
            metric_type: MetricType::Histogram,
            value,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            tags,
            unit: "histogram".to_string(),
        };
        
        self.record_metric(metric).await
    }
    
    /// 记录时间指标
    pub async fn record_timer(&self, name: &str, duration: Duration, tags: HashMap<String, String>) -> Result<()> {
        let metric = PerformanceMetric {
            name: name.to_string(),
            metric_type: MetricType::Timer,
            value: duration.as_millis() as f64,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            tags,
            unit: "ms".to_string(),
        };
        
        self.record_metric(metric).await
    }
    
    /// 计算性能统计
    pub async fn calculate_stats(&self, metric_name: &str) -> Result<PerformanceStats> {
        let metrics = self.metrics.read().await;
        let metric_list = metrics.get(metric_name)
            .ok_or_else(|| anyhow!("Metric not found: {}", metric_name))?;
        
        if metric_list.is_empty() {
            return Err(anyhow!("No metrics found for: {}", metric_name));
        }
        
        let values: Vec<f64> = metric_list.iter().map(|m| m.value).collect();
        let count = values.len() as u64;
        let sum: f64 = values.iter().sum();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg = sum / count as f64;
        
        // 计算百分位数
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50 = percentile(&sorted_values, 0.5);
        let p95 = percentile(&sorted_values, 0.95);
        let p99 = percentile(&sorted_values, 0.99);
        
        let stats = PerformanceStats {
            name: metric_name.to_string(),
            count,
            sum,
            min,
            max,
            avg,
            p50,
            p95,
            p99,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };
        
        // 存储统计信息
        {
            let mut stats_map = self.stats.write().await;
            stats_map.insert(metric_name.to_string(), stats.clone());
        }
        
        Ok(stats)
    }
    
    /// 生成性能报告
    pub async fn generate_report(&self, report_id: &str) -> Result<PerformanceReport> {
        let start_time = self.start_time.elapsed().as_secs();
        let end_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        
        let metrics = self.metrics.read().await;
        let stats = self.stats.read().await;
        
        // 收集所有指标
        let all_metrics: Vec<PerformanceMetric> = metrics
            .values()
            .flatten()
            .cloned()
            .collect();
        
        // 收集所有统计信息
        let all_stats: Vec<PerformanceStats> = stats.values().cloned().collect();
        
        // 计算摘要
        let summary = self.calculate_summary(&all_metrics, &all_stats).await?;
        
        let report = PerformanceReport {
            report_id: report_id.to_string(),
            start_time,
            end_time,
            duration: end_time - start_time,
            metrics: all_metrics,
            stats: all_stats,
            summary,
        };
        
        // 发送报告
        self.report_sender.send(report.clone())?;
        
        info!("Generated performance report: {}", report_id);
        Ok(report)
    }
    
    /// 计算性能摘要
    async fn calculate_summary(
        &self,
        metrics: &[PerformanceMetric],
        stats: &[PerformanceStats],
    ) -> Result<PerformanceSummary> {
        let mut total_requests = 0u64;
        let mut successful_requests = 0u64;
        let mut failed_requests = 0u64;
        let mut response_times = Vec::new();
        
        for metric in metrics {
            match metric.name.as_str() {
                "http_requests_total" => {
                    total_requests += metric.value as u64;
                }
                "http_requests_success" => {
                    successful_requests += metric.value as u64;
                }
                "http_requests_error" => {
                    failed_requests += metric.value as u64;
                }
                "http_request_duration" => {
                    response_times.push(metric.value);
                }
                _ => {}
            }
        }
        
        let avg_response_time = if response_times.is_empty() {
            0.0
        } else {
            response_times.iter().sum::<f64>() / response_times.len() as f64
        };
        
        let max_response_time = response_times.iter().fold(0.0, |a, &b| a.max(b));
        let min_response_time = response_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let duration_secs = self.start_time.elapsed().as_secs() as f64;
        let throughput = if duration_secs > 0.0 {
            total_requests as f64 / duration_secs
        } else {
            0.0
        };
        
        let error_rate = if total_requests > 0 {
            failed_requests as f64 / total_requests as f64
        } else {
            0.0
        };
        
        Ok(PerformanceSummary {
            total_requests,
            successful_requests,
            failed_requests,
            avg_response_time,
            max_response_time,
            min_response_time,
            throughput,
            error_rate,
        })
    }
    
    /// 获取所有指标
    pub async fn get_all_metrics(&self) -> HashMap<String, Vec<PerformanceMetric>> {
        self.metrics.read().await.clone()
    }
    
    /// 获取所有统计信息
    pub async fn get_all_stats(&self) -> HashMap<String, PerformanceStats> {
        self.stats.read().await.clone()
    }
    
    /// 清理旧指标
    pub async fn cleanup_old_metrics(&self, max_age_seconds: u64) -> Result<()> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        
        let mut metrics = self.metrics.write().await;
        
        for (name, metric_list) in metrics.iter_mut() {
            metric_list.retain(|metric| {
                current_time - metric.timestamp < max_age_seconds
            });
            
            debug!("Cleaned up old metrics for: {}, remaining: {}", name, metric_list.len());
        }
        
        Ok(())
    }
    
    /// 获取系统信息
    pub async fn get_system_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        
        // 运行时间
        let uptime = self.start_time.elapsed().as_secs();
        info.insert("uptime_seconds".to_string(), uptime.to_string());
        
        // 内存使用
        if let Ok(memory_info) = get_memory_info() {
            info.insert("memory_used_mb".to_string(), memory_info.used_mb.to_string());
            info.insert("memory_total_mb".to_string(), memory_info.total_mb.to_string());
        }
        
        // CPU使用率
        if let Ok(cpu_usage) = get_cpu_usage() {
            info.insert("cpu_usage_percent".to_string(), cpu_usage.to_string());
        }
        
        info
    }
}

/// 内存信息
#[derive(Debug, Clone)]
struct MemoryInfo {
    pub used_mb: f64,
    pub total_mb: f64,
}

/// 获取内存信息
fn get_memory_info() -> Result<MemoryInfo> {
    // 简化实现，实际应该读取系统信息
    Ok(MemoryInfo {
        used_mb: 512.0,
        total_mb: 8192.0,
    })
}

/// 获取CPU使用率
fn get_cpu_usage() -> Result<f64> {
    // 简化实现，实际应该读取系统信息
    Ok(25.5)
}

/// 计算百分位数
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    
    let index = (sorted_values.len() as f64 * p) as usize;
    let index = index.min(sorted_values.len() - 1);
    
    sorted_values[index]
}

/// 性能优化器
pub struct PerformanceOptimizer {
    pub monitor: Arc<PerformanceMonitor>,
    pub optimization_rules: Vec<OptimizationRule>,
}

/// 优化规则
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    pub name: String,
    pub condition: OptimizationCondition,
    pub action: OptimizationAction,
    pub enabled: bool,
}

/// 优化条件
#[derive(Debug, Clone)]
pub enum OptimizationCondition {
    ResponseTimeGreaterThan(f64),
    ErrorRateGreaterThan(f64),
    ThroughputLessThan(f64),
    MemoryUsageGreaterThan(f64),
    CpuUsageGreaterThan(f64),
}

/// 优化动作
#[derive(Debug, Clone)]
pub enum OptimizationAction {
    ScaleUp,
    ScaleDown,
    RestartService,
    ClearCache,
    AdjustTimeout,
    LogWarning,
}

impl PerformanceOptimizer {
    /// 创建新的性能优化器
    pub fn new(monitor: Arc<PerformanceMonitor>) -> Self {
        Self {
            monitor,
            optimization_rules: Vec::new(),
        }
    }
    
    /// 添加优化规则
    pub fn add_rule(&mut self, rule: OptimizationRule) {
        self.optimization_rules.push(rule);
    }
    
    /// 检查并执行优化
    pub async fn check_and_optimize(&self) -> Result<Vec<String>> {
        let mut actions_taken = Vec::new();
        
        for rule in &self.optimization_rules {
            if !rule.enabled {
                continue;
            }
            
            if self.check_condition(&rule.condition).await? {
                let action_result = self.execute_action(&rule.action).await?;
                actions_taken.push(format!("Rule '{}': {}", rule.name, action_result));
                
                info!("Applied optimization rule: {} - {}", rule.name, action_result);
            }
        }
        
        Ok(actions_taken)
    }
    
    /// 检查优化条件
    async fn check_condition(&self, condition: &OptimizationCondition) -> Result<bool> {
        match condition {
            OptimizationCondition::ResponseTimeGreaterThan(threshold) => {
                let stats = self.monitor.calculate_stats("http_request_duration").await?;
                Ok(stats.avg > *threshold)
            }
            OptimizationCondition::ErrorRateGreaterThan(threshold) => {
                let summary = self.monitor.generate_report("temp").await?.summary;
                Ok(summary.error_rate > *threshold)
            }
            OptimizationCondition::ThroughputLessThan(threshold) => {
                let summary = self.monitor.generate_report("temp").await?.summary;
                Ok(summary.throughput < *threshold)
            }
            OptimizationCondition::MemoryUsageGreaterThan(threshold) => {
                let system_info = self.monitor.get_system_info().await;
                if let Some(memory_used) = system_info.get("memory_used_mb") {
                    if let Some(memory_total) = system_info.get("memory_total_mb") {
                        let usage_percent = memory_used.parse::<f64>()? / memory_total.parse::<f64>()? * 100.0;
                        return Ok(usage_percent > *threshold);
                    }
                }
                Ok(false)
            }
            OptimizationCondition::CpuUsageGreaterThan(threshold) => {
                let system_info = self.monitor.get_system_info().await;
                if let Some(cpu_usage) = system_info.get("cpu_usage_percent") {
                    return Ok(cpu_usage.parse::<f64>()? > *threshold);
                }
                Ok(false)
            }
        }
    }
    
    /// 执行优化动作
    async fn execute_action(&self, action: &OptimizationAction) -> Result<String> {
        match action {
            OptimizationAction::ScaleUp => {
                // 模拟扩容
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok("Scaled up service".to_string())
            }
            OptimizationAction::ScaleDown => {
                // 模拟缩容
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok("Scaled down service".to_string())
            }
            OptimizationAction::RestartService => {
                // 模拟重启服务
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok("Restarted service".to_string())
            }
            OptimizationAction::ClearCache => {
                // 模拟清理缓存
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok("Cleared cache".to_string())
            }
            OptimizationAction::AdjustTimeout => {
                // 模拟调整超时
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok("Adjusted timeout".to_string())
            }
            OptimizationAction::LogWarning => {
                warn!("Performance optimization warning triggered");
                Ok("Logged warning".to_string())
            }
        }
    }
}

/// 性能测试器
pub struct PerformanceTester {
    pub monitor: Arc<PerformanceMonitor>,
    pub test_scenarios: Vec<TestScenario>,
}

/// 测试场景
#[derive(Debug, Clone)]
pub struct TestScenario {
    pub name: String,
    pub requests_per_second: u32,
    pub duration: Duration,
    pub endpoint: String,
    pub payload: Option<String>,
}

impl PerformanceTester {
    /// 创建新的性能测试器
    pub fn new(monitor: Arc<PerformanceMonitor>) -> Self {
        Self {
            monitor,
            test_scenarios: Vec::new(),
        }
    }
    
    /// 添加测试场景
    pub fn add_scenario(&mut self, scenario: TestScenario) {
        self.test_scenarios.push(scenario);
    }
    
    /// 运行性能测试
    pub async fn run_tests(&self) -> Result<Vec<TestResult>> {
        let mut results = Vec::new();
        
        for scenario in &self.test_scenarios {
            let result = self.run_scenario(scenario).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// 运行单个测试场景
    async fn run_scenario(&self, scenario: &TestScenario) -> Result<TestResult> {
        info!("Running test scenario: {}", scenario.name);
        
        let start_time = Instant::now();
        let mut total_requests = 0u64;
        let mut successful_requests = 0u64;
        let mut failed_requests = 0u64;
        let mut response_times = Vec::new();
        
        let interval = Duration::from_secs(1) / scenario.requests_per_second;
        let mut interval_timer = tokio::time::interval(interval);
        
        while start_time.elapsed() < scenario.duration {
            interval_timer.tick().await;
            
            let request_start = Instant::now();
            
            // 模拟HTTP请求
            let success = self.simulate_request(&scenario.endpoint, scenario.payload.as_deref()).await;
            let response_time = request_start.elapsed();
            
            total_requests += 1;
            
            if success {
                successful_requests += 1;
            } else {
                failed_requests += 1;
            }
            
            response_times.push(response_time.as_millis() as f64);
            
            // 记录指标
            let mut tags = HashMap::new();
            tags.insert("scenario".to_string(), scenario.name.clone());
            tags.insert("endpoint".to_string(), scenario.endpoint.clone());
            
            self.monitor.record_timer("http_request_duration", response_time, tags.clone()).await?;
            self.monitor.increment_counter("http_requests_total", 1.0, tags.clone()).await?;
            
            if success {
                self.monitor.increment_counter("http_requests_success", 1.0, tags).await?;
            } else {
                self.monitor.increment_counter("http_requests_error", 1.0, tags).await?;
            }
        }
        
        let avg_response_time = if response_times.is_empty() {
            0.0
        } else {
            response_times.iter().sum::<f64>() / response_times.len() as f64
        };
        
        let max_response_time = response_times.iter().fold(0.0, |a, &b| a.max(b));
        let min_response_time = response_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let actual_duration = start_time.elapsed();
        let throughput = total_requests as f64 / actual_duration.as_secs_f64();
        
        let error_rate = if total_requests > 0 {
            failed_requests as f64 / total_requests as f64
        } else {
            0.0
        };
        
        Ok(TestResult {
            scenario_name: scenario.name.clone(),
            total_requests,
            successful_requests,
            failed_requests,
            avg_response_time,
            max_response_time,
            min_response_time,
            throughput,
            error_rate,
            duration: actual_duration,
        })
    }
    
    /// 模拟HTTP请求
    async fn simulate_request(&self, endpoint: &str, payload: Option<&str>) -> bool {
        // 模拟网络延迟
        let delay = Duration::from_millis(rand::random::<u64>() % 100 + 10);
        tokio::time::sleep(delay).await;
        
        // 模拟成功率（90%成功）
        rand::random::<f64>() < 0.9
    }
}

/// 测试结果
#[derive(Debug, Clone)]
pub struct TestResult {
    pub scenario_name: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time: f64,
    pub max_response_time: f64,
    pub min_response_time: f64,
    pub throughput: f64,
    pub error_rate: f64,
    pub duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        // 测试记录指标
        let mut tags = HashMap::new();
        tags.insert("endpoint".to_string(), "/api/test".to_string());
        
        monitor.record_timer("test_timer", Duration::from_millis(100), tags.clone()).await.unwrap();
        monitor.increment_counter("test_counter", 1.0, tags).await.unwrap();
        
        // 测试计算统计
        let stats = monitor.calculate_stats("test_timer").await.unwrap();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.avg, 100.0);
        
        // 测试生成报告
        let report = monitor.generate_report("test_report").await.unwrap();
        assert_eq!(report.report_id, "test_report");
        assert!(!report.metrics.is_empty());
    }

    #[tokio::test]
    async fn test_performance_optimizer() {
        let monitor = Arc::new(PerformanceMonitor::new());
        let mut optimizer = PerformanceOptimizer::new(monitor.clone());
        
        // 添加优化规则
        let rule = OptimizationRule {
            name: "test_rule".to_string(),
            condition: OptimizationCondition::ResponseTimeGreaterThan(1000.0),
            action: OptimizationAction::LogWarning,
            enabled: true,
        };
        
        optimizer.add_rule(rule);
        
        // 记录高响应时间
        let mut tags = HashMap::new();
        tags.insert("endpoint".to_string(), "/api/slow".to_string());
        monitor.record_timer("http_request_duration", Duration::from_millis(1500), tags).await.unwrap();
        
        // 检查优化
        let actions = optimizer.check_and_optimize().await.unwrap();
        assert!(!actions.is_empty());
    }

    #[tokio::test]
    async fn test_performance_tester() {
        let monitor = Arc::new(PerformanceMonitor::new());
        let mut tester = PerformanceTester::new(monitor);
        
        // 添加测试场景
        let scenario = TestScenario {
            name: "test_scenario".to_string(),
            requests_per_second: 10,
            duration: Duration::from_secs(1),
            endpoint: "/api/test".to_string(),
            payload: None,
        };
        
        tester.add_scenario(scenario);
        
        // 运行测试
        let results = tester.run_tests().await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].total_requests > 0);
    }
}