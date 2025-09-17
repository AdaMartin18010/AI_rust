use std::time::{Duration, Instant};
//use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub endpoint: String,
    pub duration_ms: u64,
    pub status_code: u16,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_requests: u64,
    pub total_errors: u64,
    pub avg_duration_ms: f64,
    pub p50_duration_ms: u64,
    pub p95_duration_ms: u64,
    pub p99_duration_ms: u64,
    pub requests_per_second: f64,
}

#[derive(Debug)]
pub struct MetricsCollector {
    requests: Arc<Mutex<Vec<RequestMetrics>>>,
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    pub fn record_request(&self, endpoint: String, duration: Duration, status_code: u16) {
        let metric = RequestMetrics {
            endpoint,
            duration_ms: duration.as_millis() as u64,
            status_code,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        if let Ok(mut requests) = self.requests.lock() {
            requests.push(metric);
            // 保持最近1000个请求的记录
            if requests.len() > 1000 {
                requests.remove(0);
            }
        }
    }

    pub fn get_stats(&self) -> PerformanceStats {
        let requests = self.requests.lock().unwrap();
        let total_requests = requests.len() as u64;
        
        if total_requests == 0 {
            return PerformanceStats {
                total_requests: 0,
                total_errors: 0,
                avg_duration_ms: 0.0,
                p50_duration_ms: 0,
                p95_duration_ms: 0,
                p99_duration_ms: 0,
                requests_per_second: 0.0,
            };
        }

        let total_errors = requests.iter()
            .filter(|r| r.status_code >= 400)
            .count() as u64;

        let mut durations: Vec<u64> = requests.iter()
            .map(|r| r.duration_ms)
            .collect();
        durations.sort();

        let avg_duration_ms = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
        
        let p50_idx = (durations.len() as f64 * 0.5) as usize;
        let p95_idx = (durations.len() as f64 * 0.95) as usize;
        let p99_idx = (durations.len() as f64 * 0.99) as usize;
        
        let p50_duration_ms = durations.get(p50_idx).copied().unwrap_or(0);
        let p95_duration_ms = durations.get(p95_idx).copied().unwrap_or(0);
        let p99_duration_ms = durations.get(p99_idx).copied().unwrap_or(0);

        let elapsed_seconds = self.start_time.elapsed().as_secs_f64().max(1.0);
        let requests_per_second = total_requests as f64 / elapsed_seconds;

        PerformanceStats {
            total_requests,
            total_errors,
            avg_duration_ms,
            p50_duration_ms,
            p95_duration_ms,
            p99_duration_ms,
            requests_per_second,
        }
    }

    pub fn get_recent_requests(&self, limit: usize) -> Vec<RequestMetrics> {
        let requests = self.requests.lock().unwrap();
        requests.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
