//! c03_ml_basics: 线性回归的最小实现（闭式解）

pub fn linear_regression_fit(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    if x.len() != y.len() || x.is_empty() { return None; }
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|v| v * v).sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < f64::EPSILON { return None; }
    let w1 = (n * sum_xy - sum_x * sum_y) / denom;
    let w0 = (sum_y - w1 * sum_x) / n;
    Some((w0, w1))
}

/// 预测函数
pub fn linear_regression_predict(x: f64, w0: f64, w1: f64) -> f64 {
    w0 + w1 * x
}

/// 计算R²（决定系数）
pub fn r_squared(x: &[f64], y: &[f64], w0: f64, w1: f64) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() { return None; }
    
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x.iter().zip(y).map(|(&xi, &yi)| {
        let y_pred = linear_regression_predict(xi, w0, w1);
        (yi - y_pred).powi(2)
    }).sum();
    
    if ss_tot.abs() < f64::EPSILON { return None; }
    Some(1.0 - ss_res / ss_tot)
}

/// K-均值聚类（简化版，一维数据）
pub fn kmeans_1d(data: &[f64], k: usize, max_iter: usize) -> Option<Vec<f64>> {
    if data.is_empty() || k == 0 || k > data.len() { return None; }
    
    let mut centers = vec![0.0; k];
    let data_min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let data_max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // 初始化聚类中心
    for i in 0..k {
        centers[i] = data_min + (data_max - data_min) * i as f64 / (k - 1) as f64;
    }
    
    for _ in 0..max_iter {
        let mut clusters: Vec<Vec<f64>> = vec![Vec::new(); k];
        
        // 分配数据点到最近的聚类中心
        for &point in data {
            let mut min_dist = f64::INFINITY;
            let mut closest_center = 0;
            
            for (i, &center) in centers.iter().enumerate() {
                let dist = (point - center).abs();
                if dist < min_dist {
                    min_dist = dist;
                    closest_center = i;
                }
            }
            
            clusters[closest_center].push(point);
        }
        
        // 更新聚类中心
        let mut changed = false;
        for (i, cluster) in clusters.iter().enumerate() {
            if !cluster.is_empty() {
                let new_center = cluster.iter().sum::<f64>() / cluster.len() as f64;
                if (centers[i] - new_center).abs() > f64::EPSILON {
                    centers[i] = new_center;
                    changed = true;
                }
            }
        }
        
        if !changed { break; }
    }
    
    Some(centers)
}

/// 朴素贝叶斯分类器（高斯朴素贝叶斯，用于连续特征）
#[derive(Debug, Clone)]
pub struct GaussianNaiveBayes {
    pub class_priors: Vec<f64>,
    pub class_means: Vec<Vec<f64>>,
    pub class_vars: Vec<Vec<f64>>,
    pub classes: Vec<i32>,
}

impl GaussianNaiveBayes {
    pub fn new() -> Self {
        Self {
            class_priors: Vec::new(),
            class_means: Vec::new(),
            class_vars: Vec::new(),
            classes: Vec::new(),
        }
    }
    
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[i32]) -> Result<(), &'static str> {
        if features.len() != labels.len() || features.is_empty() {
            return Err("Features and labels must have the same length and be non-empty");
        }
        
        let num_features = features[0].len();
        let mut unique_classes = labels.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        
        self.classes = unique_classes.clone();
        let num_classes = self.classes.len();
        
        self.class_priors = vec![0.0; num_classes];
        self.class_means = vec![vec![0.0; num_features]; num_classes];
        self.class_vars = vec![vec![0.0; num_features]; num_classes];
        
        // 计算类别先验概率和特征均值
        for (class_idx, &class) in self.classes.iter().enumerate() {
            let mut class_samples: Vec<&Vec<f64>> = Vec::new();
            for (feature, label) in features.iter().zip(labels.iter()) {
                if *label == class {
                    class_samples.push(feature);
                }
            }
                
            self.class_priors[class_idx] = class_samples.len() as f64 / features.len() as f64;
            
            // 计算每个特征的均值
            for feature_idx in 0..num_features {
                let feature_sum: f64 = class_samples.iter()
                    .map(|sample| sample[feature_idx])
                    .sum();
                self.class_means[class_idx][feature_idx] = feature_sum / class_samples.len() as f64;
            }
            
            // 计算每个特征的方差
            for feature_idx in 0..num_features {
                let mean = self.class_means[class_idx][feature_idx];
                let variance: f64 = class_samples.iter()
                    .map(|sample| (sample[feature_idx] - mean).powi(2))
                    .sum::<f64>() / class_samples.len() as f64;
                self.class_vars[class_idx][feature_idx] = variance.max(1e-9); // 避免除零
            }
        }
        
        Ok(())
    }
    
    pub fn predict(&self, features: &[f64]) -> Option<i32> {
        if features.len() != self.class_means[0].len() {
            return None;
        }
        
        let mut max_posterior = f64::NEG_INFINITY;
        let mut predicted_class = 0;
        
        for (class_idx, &class) in self.classes.iter().enumerate() {
            let mut log_posterior = self.class_priors[class_idx].ln();
            
            for (feature_idx, &feature_value) in features.iter().enumerate() {
                let mean = self.class_means[class_idx][feature_idx];
                let var = self.class_vars[class_idx][feature_idx];
                
                // 高斯概率密度函数的对数
                let log_likelihood = -0.5 * ((feature_value - mean).powi(2) / var + var.ln() + (2.0 * std::f64::consts::PI).ln());
                log_posterior += log_likelihood;
            }
            
            if log_posterior > max_posterior {
                max_posterior = log_posterior;
                predicted_class = class;
            }
        }
        
        Some(predicted_class)
    }
}

impl Default for GaussianNaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_line() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![3.0, 5.0, 7.0, 9.0]; // y = 1 + 2x
        let (w0, w1) = linear_regression_fit(&x, &y).unwrap();
        assert!((w0 - 1.0).abs() < 1e-9);
        assert!((w1 - 2.0).abs() < 1e-9);
    }
    
    #[test]
    fn test_r_squared() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![3.0, 5.0, 7.0, 9.0]; // 完美线性关系
        let (w0, w1) = linear_regression_fit(&x, &y).unwrap();
        let r2 = r_squared(&x, &y, w0, w1).unwrap();
        assert!((r2 - 1.0).abs() < 1e-9); // R²应该接近1
    }
    
    #[test]
    fn test_kmeans_1d() {
        let data = vec![1.0, 2.0, 8.0, 9.0, 10.0]; // 两个明显的聚类
        let centers = kmeans_1d(&data, 2, 100).unwrap();
        assert_eq!(centers.len(), 2);
        // 聚类中心应该大致在1.5和9附近
        assert!(centers.iter().any(|&c| (c - 1.5).abs() < 1.0));
        assert!(centers.iter().any(|&c| (c - 9.0).abs() < 1.0));
    }
    
    #[test]
    fn test_gaussian_naive_bayes() {
        let mut nb = GaussianNaiveBayes::new();
        
        // 简单的二分类数据
        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![10.0, 11.0],
            vec![11.0, 12.0],
            vec![12.0, 13.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        
        nb.fit(&features, &labels).unwrap();
        
        // 测试预测
        let prediction1 = nb.predict(&[1.5, 2.5]).unwrap();
        let prediction2 = nb.predict(&[10.5, 11.5]).unwrap();
        
        assert_eq!(prediction1, 0);
        assert_eq!(prediction2, 1);
    }
}