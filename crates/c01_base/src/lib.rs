//! C01_base: 基础工具与示例（数理/工程基石）

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunningAverage {
    count: u64,
    sum: f64,
}

impl RunningAverage {
    pub fn new() -> Self {
        Self { count: 0, sum: 0.0 }
    }

    pub fn add_sample(&mut self, value: f64) {
        self.count = self.count.saturating_add(1);
        self.sum += value;
    }

    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 { None } else { Some(self.sum / self.count as f64) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_average_works() {
        let mut ra = RunningAverage::new();
        assert_eq!(ra.mean(), None);
        ra.add_sample(1.0);
        ra.add_sample(2.0);
        ra.add_sample(3.0);
        assert_eq!(ra.mean(), Some(2.0));
    }
}


