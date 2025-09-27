//! 监控仪表板模块

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// 监控指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub timestamp: u64,
    pub labels: HashMap<String, String>,
}

/// 系统资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: f64,
    pub gpu_memory_usage: f64,
    pub timestamp: u64,
}

/// 应用性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time: f64,
    pub requests_per_second: f64,
    pub timestamp: u64,
}

/// 监控仪表板管理器
#[allow(dead_code)]
#[allow(unused_variables)]
pub struct MonitoringDashboard {
    metrics: Arc<RwLock<HashMap<String, Vec<Metric>>>>,
    system_metrics: Arc<Mutex<Vec<SystemMetrics>>>,
    app_metrics: Arc<Mutex<Vec<ApplicationMetrics>>>,
    
    // 统计计数器
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    
    // 响应时间统计
    response_times: Arc<Mutex<Vec<f64>>>,
    
    // 监控状态
    monitoring_active: Arc<Mutex<bool>>,
    start_time: Instant,
}

impl MonitoringDashboard {
    /// 创建新的监控仪表板
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            system_metrics: Arc::new(Mutex::new(Vec::new())),
            app_metrics: Arc::new(Mutex::new(Vec::new())),
            
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            
            response_times: Arc::new(Mutex::new(Vec::new())),
            monitoring_active: Arc::new(Mutex::new(false)),
            start_time: Instant::now(),
        }
    }
    
    /// 开始监控
    pub async fn start_monitoring(&self) -> Result<(), String> {
        let mut active = self.monitoring_active.lock().unwrap();
        *active = true;
        Ok(())
    }
    
    /// 停止监控
    pub fn stop_monitoring(&self) {
        let mut active = self.monitoring_active.lock().unwrap();
        *active = false;
    }
    
    /// 记录指标
    pub fn record_metric(&self, name: &str, value: f64, labels: Option<HashMap<String, String>>) {
        let metric = Metric {
            name: name.to_string(),
            value,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            labels: labels.unwrap_or_default(),
        };
        
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.entry(name.to_string()).or_insert_with(Vec::new).push(metric);
            
            // 保留最近1000个指标
            if let Some(metric_list) = metrics.get_mut(name) {
                if metric_list.len() > 1000 {
                    metric_list.remove(0);
                }
            }
        }
    }
    
    /// 记录请求
    pub fn record_request(&self, success: bool, response_time: f64) {
        self.total_requests.fetch_add(1, Ordering::SeqCst);
        
        if success {
            self.successful_requests.fetch_add(1, Ordering::SeqCst);
        } else {
            self.failed_requests.fetch_add(1, Ordering::SeqCst);
        }
        
        // 记录响应时间
        if let Ok(mut times) = self.response_times.lock() {
            times.push(response_time);
            if times.len() > 1000 {
                times.remove(0);
            }
        }
    }
    
    /// 获取系统指标
    pub async fn get_system_metrics(&self) -> Vec<SystemMetrics> {
        self.system_metrics.lock().unwrap().clone()
    }
    
    /// 获取应用指标
    pub async fn get_application_metrics(&self) -> Vec<ApplicationMetrics> {
        let total = self.total_requests.load(Ordering::SeqCst);
        let successful = self.successful_requests.load(Ordering::SeqCst);
        let failed = self.failed_requests.load(Ordering::SeqCst);
        
        let avg_response_time = {
            let times = self.response_times.lock().unwrap();
            if times.is_empty() {
                0.0
            } else {
                times.iter().sum::<f64>() / times.len() as f64
            }
        };
        
        let uptime = self.start_time.elapsed().as_secs_f64();
        let rps = if uptime > 0.0 { total as f64 / uptime } else { 0.0 };
        
        vec![ApplicationMetrics {
            total_requests: total,
            successful_requests: successful,
            failed_requests: failed,
            average_response_time: avg_response_time,
            requests_per_second: rps,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }]
    }
    
    /// 获取指标
    pub fn get_metrics(&self, name: Option<&str>) -> HashMap<String, Vec<Metric>> {
        let metrics = self.metrics.read().unwrap();
        if let Some(name) = name {
            metrics.get(name).map(|v| {
                let mut result = HashMap::new();
                result.insert(name.to_string(), v.clone());
                result
            }).unwrap_or_default()
        } else {
            metrics.clone()
        }
    }
    
    /// 获取仪表板数据
    pub async fn get_dashboard_data(&self) -> DashboardData {
        let system_metrics = self.get_system_metrics().await;
        let app_metrics = self.get_application_metrics().await;
        
        DashboardData {
            system_metrics,
            application_metrics: app_metrics,
            uptime: self.start_time.elapsed().as_secs(),
        }
    }
}

/// 仪表板数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub system_metrics: Vec<SystemMetrics>,
    pub application_metrics: Vec<ApplicationMetrics>,
    pub uptime: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_monitoring_dashboard_creation() {
        let dashboard = MonitoringDashboard::new();
        assert_eq!(dashboard.total_requests.load(Ordering::SeqCst), 0);
    }
    
    #[tokio::test]
    async fn test_metric_recording() {
        let dashboard = MonitoringDashboard::new();
        
        let mut labels = HashMap::new();
        labels.insert("service".to_string(), "api".to_string());
        
        dashboard.record_metric("request_count", 100.0, Some(labels));
        
        let metrics = dashboard.get_metrics(Some("request_count"));
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics["request_count"].len(), 1);
        assert_eq!(metrics["request_count"][0].value, 100.0);
    }
    
    #[tokio::test]
    async fn test_request_recording() {
        let dashboard = MonitoringDashboard::new();
        
        dashboard.record_request(true, 50.0);
        dashboard.record_request(false, 100.0);
        
        assert_eq!(dashboard.total_requests.load(Ordering::SeqCst), 2);
        assert_eq!(dashboard.successful_requests.load(Ordering::SeqCst), 1);
        assert_eq!(dashboard.failed_requests.load(Ordering::SeqCst), 1);
    }
}