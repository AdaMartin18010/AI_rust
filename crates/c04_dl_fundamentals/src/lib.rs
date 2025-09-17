//! c04_dl_fundamentals: 深度学习基础功能

/// ReLU激活函数
pub fn relu(x: f64) -> f64 { 
    if x > 0.0 { x } else { 0.0 } 
}

/// ReLU的导数
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Sigmoid激活函数
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid的导数
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Tanh激活函数
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Tanh的导数
pub fn tanh_derivative(x: f64) -> f64 {
    let t = x.tanh();
    1.0 - t * t
}

/// 均方误差损失函数
pub fn mse_loss(predictions: &[f64], targets: &[f64]) -> Option<f64> {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return None;
    }
    
    let sum_squared_errors: f64 = predictions.iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| (pred - target).powi(2))
        .sum();
    
    Some(sum_squared_errors / predictions.len() as f64)
}

/// 交叉熵损失函数（用于分类）
pub fn cross_entropy_loss(predictions: &[f64], targets: &[i32]) -> Option<f64> {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return None;
    }
    
    let epsilon = 1e-15; // 避免log(0)
    let loss: f64 = predictions.iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| {
            let prob = sigmoid(pred).max(epsilon).min(1.0 - epsilon);
            if target == 1 {
                -prob.ln()
            } else {
                -(1.0 - prob).ln()
            }
        })
        .sum();
    
    Some(loss / predictions.len() as f64)
}

/// 简单的全连接层
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub input_size: usize,
    pub output_size: usize,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // 使用Xavier初始化
        let xavier_std = (2.0 / (input_size + output_size) as f64).sqrt();
        
        let mut weights = vec![vec![0.0; input_size]; output_size];
        let mut biases = vec![0.0; output_size];
        
        // 简单的随机初始化（实际应用中应使用更好的随机数生成器）
        for i in 0..output_size {
            for j in 0..input_size {
                weights[i][j] = (i as f64 * 0.1 + j as f64 * 0.01) * xavier_std;
            }
            biases[i] = (i as f64 * 0.01) * xavier_std;
        }
        
        Self {
            weights,
            biases,
            input_size,
            output_size,
        }
    }
    
    pub fn forward(&self, inputs: &[f64]) -> Option<Vec<f64>> {
        if inputs.len() != self.input_size {
            return None;
        }
        
        let mut outputs = vec![0.0; self.output_size];
        
        for i in 0..self.output_size {
            let mut sum = self.biases[i];
            for j in 0..self.input_size {
                sum += self.weights[i][j] * inputs[j];
            }
            outputs[i] = sum;
        }
        
        Some(outputs)
    }
    
    pub fn forward_with_activation(&self, inputs: &[f64], activation: fn(f64) -> f64) -> Option<Vec<f64>> {
        self.forward(inputs).map(|outputs| {
            outputs.into_iter().map(activation).collect()
        })
    }
}

/// 简单的神经网络
#[derive(Debug, Clone)]
pub struct SimpleNeuralNetwork {
    pub layers: Vec<DenseLayer>,
    pub activation: fn(f64) -> f64,
}

impl SimpleNeuralNetwork {
    pub fn new(layer_sizes: &[usize], activation: fn(f64) -> f64) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        
        Self { layers, activation }
    }
    
    pub fn forward(&self, inputs: &[f64]) -> Option<Vec<f64>> {
        let mut current_inputs = inputs.to_vec();
        
        for (i, layer) in self.layers.iter().enumerate() {
            let is_last_layer = i == self.layers.len() - 1;
            
            if is_last_layer {
                // 最后一层不使用激活函数（用于回归）或使用sigmoid（用于二分类）
                current_inputs = layer.forward(&current_inputs)?;
            } else {
                current_inputs = layer.forward_with_activation(&current_inputs, self.activation)?;
            }
        }
        
        Some(current_inputs)
    }
    
    pub fn predict(&self, inputs: &[f64]) -> Option<f64> {
        let outputs = self.forward(inputs)?;
        outputs.first().copied()
    }
}

/// 梯度下降优化器
pub struct GradientDescent {
    pub learning_rate: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
    
    pub fn update_weights(&self, weights: &mut Vec<Vec<f64>>, gradients: &[Vec<f64>]) {
        for (weight_row, grad_row) in weights.iter_mut().zip(gradients.iter()) {
            for (weight, &grad) in weight_row.iter_mut().zip(grad_row.iter()) {
                *weight -= self.learning_rate * grad;
            }
        }
    }
    
    pub fn update_biases(&self, biases: &mut Vec<f64>, gradients: &[f64]) {
        for (bias, &grad) in biases.iter_mut().zip(gradients.iter()) {
            *bias -= self.learning_rate * grad;
        }
    }
}

/// 简单的反向传播（数值梯度）
pub fn numerical_gradient<F>(f: F, x: &[f64], h: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut gradients = vec![0.0; x.len()];
    
    for i in 0..x.len() {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        
        x_plus[i] += h;
        x_minus[i] -= h;
        
        gradients[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
    }
    
    gradients
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relu_works() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(3.5), 3.5);
    }
    
    #[test]
    fn test_activation_functions() {
        // ReLU测试
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(2.0), 2.0);
        assert_eq!(relu_derivative(-1.0), 0.0);
        assert_eq!(relu_derivative(2.0), 1.0);
        
        // Sigmoid测试
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.9);
        assert!(sigmoid(-10.0) < 0.1);
        
        // Tanh测试
        assert!((tanh(0.0) - 0.0).abs() < 1e-10);
        assert!(tanh(10.0) > 0.9);
        assert!(tanh(-10.0) < -0.9);
    }
    
    #[test]
    fn test_loss_functions() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 1.9, 3.1];
        
        let mse = mse_loss(&predictions, &targets).unwrap();
        assert!(mse > 0.0);
        assert!(mse < 0.1); // 应该很小
        
        let predictions_cls = vec![0.8, 0.3, 0.9];
        let targets_cls = vec![1, 0, 1];
        
        let ce_loss = cross_entropy_loss(&predictions_cls, &targets_cls).unwrap();
        assert!(ce_loss > 0.0);
    }
    
    #[test]
    fn test_dense_layer() {
        let layer = DenseLayer::new(2, 3);
        assert_eq!(layer.input_size, 2);
        assert_eq!(layer.output_size, 3);
        
        let inputs = vec![1.0, 2.0];
        let outputs = layer.forward(&inputs).unwrap();
        assert_eq!(outputs.len(), 3);
        
        let outputs_with_activation = layer.forward_with_activation(&inputs, relu).unwrap();
        assert_eq!(outputs_with_activation.len(), 3);
        assert!(outputs_with_activation.iter().all(|&x| x >= 0.0)); // ReLU输出应该非负
    }
    
    #[test]
    fn test_neural_network() {
        let network = SimpleNeuralNetwork::new(&[2, 4, 1], relu);
        assert_eq!(network.layers.len(), 2);
        
        let inputs = vec![1.0, 2.0];
        let outputs = network.forward(&inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        
        let prediction = network.predict(&inputs).unwrap();
        assert!(prediction.is_finite());
    }
    
    #[test]
    fn test_gradient_descent() {
        let optimizer = GradientDescent::new(0.1);
        
        let mut weights = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let gradients = vec![vec![0.5, 0.3], vec![0.2, 0.1]];
        
        optimizer.update_weights(&mut weights, &gradients);
        
        // 检查权重是否被正确更新
        assert!((weights[0][0] - 0.95).abs() < 1e-10);
        assert!((weights[0][1] - 1.97).abs() < 1e-10);
    }
    
    #[test]
    fn test_numerical_gradient() {
        // 测试函数: f(x) = x^2
        let f = |x: &[f64]| x[0] * x[0];
        let x = vec![2.0];
        let gradients = numerical_gradient(f, &x, 1e-6);
        
        // 对于f(x) = x^2，在x=2处的导数应该是4
        assert!((gradients[0] - 4.0).abs() < 1e-5);
    }
}


