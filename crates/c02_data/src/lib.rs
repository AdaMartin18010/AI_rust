//! c02_data: 数据获取/清洗/标注/可视化 基础接口

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub id: u64,
    pub value: f64,
}

pub fn normalize_min_max(values: &[f64]) -> Option<Vec<f64>> {
    let (min, max) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    if !min.is_finite() || !max.is_finite() || (max - min).abs() < f64::EPSILON {
        return None;
    }
    Some(values.iter().map(|&x| (x - min) / (max - min)).collect())
}

/// Z-score标准化 (均值为0，标准差为1)
pub fn normalize_z_score(values: &[f64]) -> Option<Vec<f64>> {
    if values.is_empty() {
        return None;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    
    if std_dev < f64::EPSILON {
        return None;
    }
    
    Some(values.iter().map(|&x| (x - mean) / std_dev).collect())
}

/// 计算数据统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64,
    pub q75: f64,
}

pub fn calculate_stats(values: &[f64]) -> Option<DataStats> {
    if values.is_empty() {
        return None;
    }
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let median = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    } else {
        sorted[count / 2]
    };
    
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();
    
    let min = sorted[0];
    let max = sorted[count - 1];
    let q25 = sorted[count / 4];
    let q75 = sorted[3 * count / 4];
    
    Some(DataStats {
        count,
        mean,
        median,
        std_dev,
        min,
        max,
        q25,
        q75,
    })
}

/// 数据清洗：移除异常值 (使用IQR方法)
pub fn remove_outliers(values: &[f64], iqr_multiplier: f64) -> Vec<f64> {
    if let Some(stats) = calculate_stats(values) {
        let iqr = stats.q75 - stats.q25;
        let lower_bound = stats.q25 - iqr_multiplier * iqr;
        let upper_bound = stats.q75 + iqr_multiplier * iqr;
        
        values.iter()
            .filter(|&&x| x >= lower_bound && x <= upper_bound)
            .copied()
            .collect()
    } else {
        values.to_vec()
    }
}

/// 数据分箱
pub fn create_bins(values: &[f64], num_bins: usize) -> Option<Vec<(f64, f64, usize)>> {
    if values.is_empty() || num_bins == 0 {
        return None;
    }
    
    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if (max - min).abs() < f64::EPSILON {
        return None;
    }
    
    let bin_width = (max - min) / num_bins as f64;
    let mut bins = vec![(0.0, 0.0, 0); num_bins];
    
    for i in 0..num_bins {
        let start = min + i as f64 * bin_width;
        let end = if i == num_bins - 1 { max } else { start + bin_width };
        bins[i] = (start, end, 0);
    }
    
    for &value in values {
        let bin_index = if value == max {
            num_bins - 1
        } else {
            ((value - min) / bin_width) as usize
        };
        
        if bin_index < num_bins {
            bins[bin_index].2 += 1;
        }
    }
    
    Some(bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic() {
        let v = vec![0.0, 5.0, 10.0];
        let out = normalize_min_max(&v).unwrap();
        assert_eq!(out, vec![0.0, 0.5, 1.0]);
    }
    
    #[test]
    fn normalize_z_score_basic() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = normalize_z_score(&v).unwrap();
        let mean = out.iter().sum::<f64>() / out.len() as f64;
        assert!((mean).abs() < 1e-10); // 均值应该接近0
    }
    
    #[test]
    fn calculate_stats_basic() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_stats(&v).unwrap();
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }
    
    #[test]
    fn remove_outliers_basic() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0是异常值
        let cleaned = remove_outliers(&v, 1.5);
        assert!(!cleaned.contains(&100.0));
        assert!(cleaned.len() < v.len());
    }
    
    #[test]
    fn create_bins_basic() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bins = create_bins(&v, 3).unwrap();
        assert_eq!(bins.len(), 3);
        let total_count: usize = bins.iter().map(|(_, _, count)| count).sum();
        assert_eq!(total_count, v.len());
    }
}


