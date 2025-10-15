//! 日志和监控系统实现
//! 
//! 本示例展示了一个完整的日志和监控系统，包括：
//! - 结构化日志记录
//! - 性能指标收集
//! - 健康检查
//! - 错误追踪
//! - 实时监控

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use prometheus::{Counter, Histogram, Gauge, Encoder, TextEncoder, register_counter, register_histogram, register_gauge};

/// 日志级别
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// 日志条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: LogLevel,
    pub message: String,
    pub module: String,
    pub file: String,
    pub line: u32,
    pub fields: HashMap<String, serde_json::Value>,
}

/// 性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub request_count: u64,
    pub error_count: u64,
    pub average_response_time: f64,
    pub p95_response_time: f64,
    pub p99_response_time: f64,
    pub memory_usage: u64,
    pub cpu_usage: f64,
    pub active_connections: u64,
}

/// 健康检查状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub uptime: Duration,
    pub version: String,
    pub checks: HashMap<String, CheckResult>,
}

/// 检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub status: String,
    pub message: String,
    pub duration: Duration,
}

/// 日志收集器
pub struct LogCollector {
    logs: Arc<RwLock<Vec<LogEntry>>>,
    max_logs: usize,
}

impl LogCollector {
    pub fn new(max_logs: usize) -> Self {
        Self {
            logs: Arc::new(RwLock::new(Vec::new())),
            max_logs,
        }
    }
    
    pub async fn add_log(&self, entry: LogEntry) {
        let mut logs = self.logs.write().await;
        logs.push(entry);
        
        // 保持日志数量在限制内
        if logs.len() > self.max_logs {
            logs.drain(0..logs.len() - self.max_logs);
        }
    }
    
    pub async fn get_logs(&self, limit: usize) -> Vec<LogEntry> {
        let logs = self.logs.read().await;
        logs.iter().rev().take(limit).cloned().collect()
    }
    
    pub async fn get_logs_by_level(&self, level: LogLevel, limit: usize) -> Vec<LogEntry> {
        let logs = self.logs.read().await;
        logs.iter()
            .rev()
            .filter(|log| matches!(log.level, level))
            .take(limit)
            .cloned()
            .collect()
    }
    
    pub async fn clear_logs(&self) {
        let mut logs = self.logs.write().await;
        logs.clear();
    }
}

/// 性能监控器
pub struct PerformanceMonitor {
    metrics: Arc<RwLock<PerformanceMetrics>>,
    request_times: Arc<RwLock<Vec<f64>>>,
    start_time: Instant,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics {
                request_count: 0,
                error_count: 0,
                average_response_time: 0.0,
                p95_response_time: 0.0,
                p99_response_time: 0.0,
                memory_usage: 0,
                cpu_usage: 0.0,
                active_connections: 0,
            })),
            request_times: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
        }
    }
    
    pub async fn record_request(&self, duration: Duration, success: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.request_count += 1;
        
        if !success {
            metrics.error_count += 1;
        }
        
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let mut request_times = self.request_times.write().await;
        request_times.push(duration_ms);
        
        // 保持最近1000个请求的时间
        if request_times.len() > 1000 {
            request_times.drain(0..request_times.len() - 1000);
        }
        
        // 计算统计信息
        if !request_times.is_empty() {
            let mut sorted_times = request_times.clone();
            sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            metrics.average_response_time = request_times.iter().sum::<f64>() / request_times.len() as f64;
            
            if sorted_times.len() > 0 {
                let p95_index = (sorted_times.len() as f64 * 0.95) as usize;
                let p99_index = (sorted_times.len() as f64 * 0.99) as usize;
                
                metrics.p95_response_time = sorted_times[p95_index.min(sorted_times.len() - 1)];
                metrics.p99_response_time = sorted_times[p99_index.min(sorted_times.len() - 1)];
            }
        }
    }
    
    pub async fn update_system_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        
        // 获取内存使用情况
        if let Ok(memory_info) = sysinfo::System::new_all().memory() {
            metrics.memory_usage = memory_info.used();
        }
        
        // 获取CPU使用情况
        if let Ok(cpu_info) = sysinfo::System::new_all().cpu_usage() {
            metrics.cpu_usage = cpu_info as f64;
        }
    }
    
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }
    
    pub fn get_uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// 健康检查器
pub struct HealthChecker {
    checks: HashMap<String, Box<dyn HealthCheck + Send + Sync>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: HashMap::new(),
        }
    }
    
    pub fn add_check(&mut self, name: String, check: Box<dyn HealthCheck + Send + Sync>) {
        self.checks.insert(name, check);
    }
    
    pub async fn run_checks(&self) -> HealthStatus {
        let mut results = HashMap::new();
        let mut overall_status = "healthy";
        
        for (name, check) in &self.checks {
            let start = Instant::now();
            let result = check.check().await;
            let duration = start.elapsed();
            
            let check_result = CheckResult {
                status: if result.is_ok() { "healthy".to_string() } else { "unhealthy".to_string() },
                message: result.unwrap_or_else(|e| e.to_string()),
                duration,
            };
            
            if check_result.status != "healthy" {
                overall_status = "unhealthy";
            }
            
            results.insert(name.clone(), check_result);
        }
        
        HealthStatus {
            status: overall_status.to_string(),
            timestamp: chrono::Utc::now(),
            uptime: Duration::from_secs(0), // 将在外部设置
            version: env!("CARGO_PKG_VERSION").to_string(),
            checks: results,
        }
    }
}

/// 健康检查trait
pub trait HealthCheck {
    async fn check(&self) -> Result<String, String>;
}

/// 数据库健康检查
pub struct DatabaseHealthCheck {
    connection_string: String,
}

impl DatabaseHealthCheck {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }
}

impl HealthCheck for DatabaseHealthCheck {
    async fn check(&self) -> Result<String, String> {
        // 模拟数据库连接检查
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        if self.connection_string.contains("postgresql://") {
            Ok("Database connection healthy".to_string())
        } else {
            Err("Invalid database connection string".to_string())
        }
    }
}

/// Redis健康检查
pub struct RedisHealthCheck {
    connection_string: String,
}

impl RedisHealthCheck {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }
}

impl HealthCheck for RedisHealthCheck {
    async fn check(&self) -> Result<String, String> {
        // 模拟Redis连接检查
        tokio::time::sleep(Duration::from_millis(5)).await;
        
        if self.connection_string.contains("redis://") {
            Ok("Redis connection healthy".to_string())
        } else {
            Err("Invalid Redis connection string".to_string())
        }
    }
}

/// 内存健康检查
pub struct MemoryHealthCheck {
    max_memory_mb: u64,
}

impl MemoryHealthCheck {
    pub fn new(max_memory_mb: u64) -> Self {
        Self { max_memory_mb }
    }
}

impl HealthCheck for MemoryHealthCheck {
    async fn check(&self) -> Result<String, String> {
        let system = sysinfo::System::new_all();
        let memory_usage = system.memory().used();
        let memory_usage_mb = memory_usage / 1024 / 1024;
        
        if memory_usage_mb > self.max_memory_mb {
            Err(format!("Memory usage too high: {}MB > {}MB", memory_usage_mb, self.max_memory_mb))
        } else {
            Ok(format!("Memory usage healthy: {}MB", memory_usage_mb))
        }
    }
}

/// 监控系统
pub struct MonitoringSystem {
    log_collector: LogCollector,
    performance_monitor: PerformanceMonitor,
    health_checker: HealthChecker,
    prometheus_metrics: PrometheusMetrics,
}

impl MonitoringSystem {
    pub fn new() -> Self {
        let mut health_checker = HealthChecker::new();
        
        // 添加健康检查
        health_checker.add_check(
            "database".to_string(),
            Box::new(DatabaseHealthCheck::new("postgresql://localhost:5432/ai_rust_db".to_string())),
        );
        
        health_checker.add_check(
            "redis".to_string(),
            Box::new(RedisHealthCheck::new("redis://localhost:6379".to_string())),
        );
        
        health_checker.add_check(
            "memory".to_string(),
            Box::new(MemoryHealthCheck::new(1024)), // 1GB限制
        );
        
        Self {
            log_collector: LogCollector::new(10000),
            performance_monitor: PerformanceMonitor::new(),
            health_checker,
            prometheus_metrics: PrometheusMetrics::new(),
        }
    }
    
    pub async fn record_request(&self, duration: Duration, success: bool) {
        self.performance_monitor.record_request(duration, success).await;
        self.prometheus_metrics.record_request(duration, success).await;
    }
    
    pub async fn add_log(&self, level: LogLevel, message: String, module: String, file: String, line: u32, fields: HashMap<String, serde_json::Value>) {
        let entry = LogEntry {
            timestamp: chrono::Utc::now(),
            level,
            message,
            module,
            file,
            line,
            fields,
        };
        
        self.log_collector.add_log(entry).await;
    }
    
    pub async fn get_health_status(&self) -> HealthStatus {
        let mut status = self.health_checker.run_checks().await;
        status.uptime = self.performance_monitor.get_uptime();
        status
    }
    
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_monitor.get_metrics().await
    }
    
    pub async fn get_logs(&self, limit: usize) -> Vec<LogEntry> {
        self.log_collector.get_logs(limit).await
    }
    
    pub async fn get_logs_by_level(&self, level: LogLevel, limit: usize) -> Vec<LogEntry> {
        self.log_collector.get_logs_by_level(level, limit).await
    }
    
    pub async fn get_prometheus_metrics(&self) -> String {
        self.prometheus_metrics.get_metrics().await
    }
    
    pub async fn update_system_metrics(&self) {
        self.performance_monitor.update_system_metrics().await;
    }
}

/// Prometheus指标
pub struct PrometheusMetrics {
    request_counter: Counter,
    error_counter: Counter,
    response_time_histogram: Histogram,
    active_connections_gauge: Gauge,
    memory_usage_gauge: Gauge,
    cpu_usage_gauge: Gauge,
}

impl PrometheusMetrics {
    pub fn new() -> Self {
        let request_counter = register_counter!("http_requests_total", "Total number of HTTP requests").unwrap();
        let error_counter = register_counter!("http_errors_total", "Total number of HTTP errors").unwrap();
        let response_time_histogram = register_histogram!("http_request_duration_seconds", "HTTP request duration").unwrap();
        let active_connections_gauge = register_gauge!("active_connections", "Number of active connections").unwrap();
        let memory_usage_gauge = register_gauge!("memory_usage_bytes", "Memory usage in bytes").unwrap();
        let cpu_usage_gauge = register_gauge!("cpu_usage_percent", "CPU usage percentage").unwrap();
        
        Self {
            request_counter,
            error_counter,
            response_time_histogram,
            active_connections_gauge,
            memory_usage_gauge,
            cpu_usage_gauge,
        }
    }
    
    pub async fn record_request(&self, duration: Duration, success: bool) {
        self.request_counter.inc();
        
        if !success {
            self.error_counter.inc();
        }
        
        self.response_time_histogram.observe(duration.as_secs_f64());
    }
    
    pub async fn update_connections(&self, count: u64) {
        self.active_connections_gauge.set(count as f64);
    }
    
    pub async fn update_memory_usage(&self, bytes: u64) {
        self.memory_usage_gauge.set(bytes as f64);
    }
    
    pub async fn update_cpu_usage(&self, percent: f64) {
        self.cpu_usage_gauge.set(percent);
    }
    
    pub async fn get_metrics(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

/// 初始化日志系统
pub fn init_logging() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    Ok(())
}

/// 日志宏
#[macro_export]
macro_rules! log_info {
    ($monitoring:expr, $($arg:tt)*) => {
        {
            let message = format!($($arg)*);
            $monitoring.add_log(
                crate::monitoring::LogLevel::Info,
                message,
                module_path!().to_string(),
                file!().to_string(),
                line!(),
                std::collections::HashMap::new(),
            ).await;
            info!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! log_warn {
    ($monitoring:expr, $($arg:tt)*) => {
        {
            let message = format!($($arg)*);
            $monitoring.add_log(
                crate::monitoring::LogLevel::Warn,
                message,
                module_path!().to_string(),
                file!().to_string(),
                line!(),
                std::collections::HashMap::new(),
            ).await;
            warn!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! log_error {
    ($monitoring:expr, $($arg:tt)*) => {
        {
            let message = format!($($arg)*);
            $monitoring.add_log(
                crate::monitoring::LogLevel::Error,
                message,
                module_path!().to_string(),
                file!().to_string(),
                line!(),
                std::collections::HashMap::new(),
            ).await;
            error!($($arg)*);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_log_collector() {
        let collector = LogCollector::new(100);
        
        let entry = LogEntry {
            timestamp: chrono::Utc::now(),
            level: LogLevel::Info,
            message: "Test message".to_string(),
            module: "test".to_string(),
            file: "test.rs".to_string(),
            line: 1,
            fields: HashMap::new(),
        };
        
        collector.add_log(entry).await;
        
        let logs = collector.get_logs(10).await;
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].message, "Test message");
    }

    #[test]
    async fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        monitor.record_request(Duration::from_millis(100), true).await;
        monitor.record_request(Duration::from_millis(200), false).await;
        
        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.request_count, 2);
        assert_eq!(metrics.error_count, 1);
        assert!(metrics.average_response_time > 0.0);
    }

    #[test]
    async fn test_health_checker() {
        let mut checker = HealthChecker::new();
        checker.add_check(
            "test".to_string(),
            Box::new(TestHealthCheck),
        );
        
        let status = checker.run_checks().await;
        assert_eq!(status.checks.len(), 1);
        assert!(status.checks.contains_key("test"));
    }

    struct TestHealthCheck;

    impl HealthCheck for TestHealthCheck {
        async fn check(&self) -> Result<String, String> {
            Ok("Test check passed".to_string())
        }
    }

    #[test]
    async fn test_monitoring_system() {
        let system = MonitoringSystem::new();
        
        system.record_request(Duration::from_millis(50), true).await;
        
        let metrics = system.get_performance_metrics().await;
        assert_eq!(metrics.request_count, 1);
        
        let health = system.get_health_status().await;
        assert!(!health.checks.is_empty());
    }
}

/// 性能基准测试
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn benchmark_log_collector() {
        let collector = LogCollector::new(10000);
        
        let start = Instant::now();
        for i in 0..1000 {
            let entry = LogEntry {
                timestamp: chrono::Utc::now(),
                level: LogLevel::Info,
                message: format!("Test message {}", i),
                module: "benchmark".to_string(),
                file: "benchmark.rs".to_string(),
                line: i as u32,
                fields: HashMap::new(),
            };
            collector.add_log(entry).await;
        }
        let duration = start.elapsed();
        
        println!("Added 1000 logs in {:?}", duration);
        println!("Average time per log: {:?}", duration / 1000);
    }

    #[tokio::test]
    async fn benchmark_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        let start = Instant::now();
        for i in 0..1000 {
            let duration = Duration::from_millis(i % 100);
            let success = i % 10 != 0; // 10% error rate
            monitor.record_request(duration, success).await;
        }
        let duration = start.elapsed();
        
        println!("Recorded 1000 requests in {:?}", duration);
        println!("Average time per request: {:?}", duration / 1000);
    }

    #[tokio::test]
    async fn benchmark_health_checks() {
        let mut checker = HealthChecker::new();
        checker.add_check(
            "test".to_string(),
            Box::new(TestHealthCheck),
        );
        
        let start = Instant::now();
        for _ in 0..100 {
            let _ = checker.run_checks().await;
        }
        let duration = start.elapsed();
        
        println!("Ran 100 health checks in {:?}", duration);
        println!("Average time per check: {:?}", duration / 100);
    }

    struct TestHealthCheck;

    impl HealthCheck for TestHealthCheck {
        async fn check(&self) -> Result<String, String> {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok("Test check passed".to_string())
        }
    }
}
