//! WebAssembly AI推理示例
//! 
//! 本示例展示了如何在WebAssembly环境中运行AI推理：
//! - 客户端AI计算能力
//! - 隐私保护的本地AI处理
//! - 离线AI功能支持
//! - 边缘计算优化

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// WebAssembly AI推理引擎
#[allow(unused)]
pub struct WasmAIEngine {
    models: HashMap<String, WasmModel>,
    device: WasmDevice,
}

#[allow(unused)]
pub struct WasmModel {
    pub id: String,
    pub name: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub enum WasmDevice {
    Cpu,
    // 未来可能支持WebGPU
    WebGpu,
}

// 推理请求和响应
#[derive(Debug, Serialize, Deserialize)]
pub struct WasmInferenceRequest {
    pub model_id: String,
    pub input_data: Vec<f32>,
    pub parameters: Option<HashMap<String, f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WasmInferenceResponse {
    pub predictions: Vec<f32>,
    pub confidence: f32,
    pub processing_time_ms: u64,
    pub memory_usage_mb: f32,
}

// 错误类型
#[derive(Debug, thiserror::Error)]
pub enum WasmAIError {
    #[error("模型未找到: {0}")]
    ModelNotFound(String),
    #[error("输入数据无效: {0}")]
    InvalidInput(String),
    #[error("内存不足")]
    OutOfMemory,
    #[error("推理执行失败: {0}")]
    InferenceFailed(String),
}

impl WasmAIEngine {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            device: WasmDevice::Cpu,
        }
    }
    
    pub fn load_model(&mut self, model: WasmModel) -> Result<(), WasmAIError> {
        self.models.insert(model.id.clone(), model);
        Ok(())
    }
    
    pub fn infer(&self, request: &WasmInferenceRequest) -> Result<WasmInferenceResponse, WasmAIError> {
        let start_time = std::time::Instant::now();
        
        // 获取模型
        let model = self.models.get(&request.model_id)
            .ok_or_else(|| WasmAIError::ModelNotFound(request.model_id.clone()))?;
        
        // 验证输入数据
        let expected_size: usize = model.input_shape.iter().product();
        if request.input_data.len() != expected_size {
            return Err(WasmAIError::InvalidInput(format!(
                "期望输入大小: {}, 实际: {}", expected_size, request.input_data.len()
            )));
        }
        
        // 执行简单的线性变换 (模拟神经网络推理)
        let output = self.simple_linear_transform(
            &request.input_data,
            &model.weights,
            &model.bias,
            &model.input_shape,
            &model.output_shape,
        )?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        let memory_usage = self.estimate_memory_usage(&request.input_data, &output);
        
        Ok(WasmInferenceResponse {
            predictions: output,
            confidence: 0.95, // 示例置信度
            processing_time_ms: processing_time,
            memory_usage_mb: memory_usage,
        })
    }
    
    // 简单的线性变换实现 (模拟神经网络前向传播)
    fn simple_linear_transform(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<Vec<f32>, WasmAIError> {
        let input_size = input_shape.iter().product::<usize>();
        let output_size = output_shape.iter().product::<usize>();
        
        if weights.len() != input_size * output_size {
            return Err(WasmAIError::InferenceFailed(
                "权重矩阵大小不匹配".to_string()
            ));
        }
        
        if bias.len() != output_size {
            return Err(WasmAIError::InferenceFailed(
                "偏置向量大小不匹配".to_string()
            ));
        }
        
        let mut output = vec![0.0; output_size];
        
        // 矩阵乘法: output = input * weights + bias
        for i in 0..output_size {
            let mut sum = bias[i];
            for j in 0..input_size {
                sum += input[j] * weights[i * input_size + j];
            }
            // 应用激活函数 (ReLU)
            output[i] = sum.max(0.0);
        }
        
        Ok(output)
    }
    
    fn estimate_memory_usage(&self, input: &[f32], output: &[f32]) -> f32 {
        let input_memory = input.len() * 4; // f32 = 4 bytes
        let output_memory = output.len() * 4;
        let total_memory = input_memory + output_memory;
        total_memory as f32 / (1024.0 * 1024.0) // 转换为MB
    }
    
    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
    
    pub fn get_model_info(&self, model_id: &str) -> Option<&WasmModel> {
        self.models.get(model_id)
    }
}

// 预训练模型工厂
pub struct ModelFactory;

impl ModelFactory {
    // 创建一个简单的分类模型
    pub fn create_simple_classifier() -> WasmModel {
        WasmModel {
            id: "simple_classifier".to_string(),
            name: "简单分类器".to_string(),
            input_shape: vec![10], // 10个特征
            output_shape: vec![3], // 3个类别
            weights: vec![
                // 权重矩阵 (3x10)
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
            ],
            bias: vec![0.1, 0.2, 0.3], // 偏置向量
        }
    }
    
    // 创建一个图像处理模型
    pub fn create_image_processor() -> WasmModel {
        WasmModel {
            id: "image_processor".to_string(),
            name: "图像处理器".to_string(),
            input_shape: vec![28, 28, 1], // 28x28灰度图像
            output_shape: vec![10], // 10个数字类别
            weights: vec![0.1; 28 * 28 * 10], // 简化的权重
            bias: vec![0.0; 10],
        }
    }
    
    // 创建一个文本嵌入模型
    pub fn create_text_embedder() -> WasmModel {
        WasmModel {
            id: "text_embedder".to_string(),
            name: "文本嵌入器".to_string(),
            input_shape: vec![512], // 512维输入
            output_shape: vec![256], // 256维嵌入
            weights: vec![0.1; 512 * 256],
            bias: vec![0.0; 256],
        }
    }
}

// 性能监控
pub struct PerformanceMonitor {
    inference_count: u64,
    total_time_ms: u64,
    total_memory_mb: f32,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            inference_count: 0,
            total_time_ms: 0,
            total_memory_mb: 0.0,
        }
    }
    
    pub fn record_inference(&mut self, time_ms: u64, memory_mb: f32) {
        self.inference_count += 1;
        self.total_time_ms += time_ms;
        self.total_memory_mb += memory_mb;
    }
    
    pub fn get_stats(&self) -> PerformanceStats {
        let avg_time_ms = if self.inference_count > 0 {
            self.total_time_ms as f64 / self.inference_count as f64
        } else {
            0.0
        };
        
        let avg_memory_mb = if self.inference_count > 0 {
            self.total_memory_mb / self.inference_count as f32
        } else {
            0.0
        };
        
        PerformanceStats {
            inference_count: self.inference_count,
            average_time_ms: avg_time_ms,
            average_memory_mb: avg_memory_mb,
            total_time_ms: self.total_time_ms,
            total_memory_mb: self.total_memory_mb,
        }
    }
}

#[derive(Debug)]
pub struct PerformanceStats {
    pub inference_count: u64,
    pub average_time_ms: f64,
    pub average_memory_mb: f32,
    pub total_time_ms: u64,
    pub total_memory_mb: f32,
}

// 主函数演示
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 WebAssembly AI推理演示");
    println!("================================");
    
    // 创建AI引擎
    let mut engine = WasmAIEngine::new();
    
    // 加载预训练模型
    let classifier = ModelFactory::create_simple_classifier();
    let image_processor = ModelFactory::create_image_processor();
    let text_embedder = ModelFactory::create_text_embedder();
    
    engine.load_model(classifier)?;
    engine.load_model(image_processor)?;
    engine.load_model(text_embedder)?;
    
    println!("📦 已加载模型:");
    for model_id in engine.list_models() {
        if let Some(model) = engine.get_model_info(&model_id) {
            println!("  - {}: {} (输入: {:?}, 输出: {:?})", 
                model.id, model.name, model.input_shape, model.output_shape);
        }
    }
    
    // 创建性能监控器
    let mut monitor = PerformanceMonitor::new();
    
    // 执行推理测试
    println!("\n🚀 执行推理测试:");
    
    // 测试分类器
    let classification_request = WasmInferenceRequest {
        model_id: "simple_classifier".to_string(),
        input_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        parameters: None,
    };
    
    let response = engine.infer(&classification_request)?;
    monitor.record_inference(response.processing_time_ms, response.memory_usage_mb);
    
    println!("  分类结果: {:?}", response.predictions);
    println!("  置信度: {:.2}%", response.confidence * 100.0);
    println!("  处理时间: {}ms", response.processing_time_ms);
    println!("  内存使用: {:.2}MB", response.memory_usage_mb);
    
    // 测试图像处理器
    let image_request = WasmInferenceRequest {
        model_id: "image_processor".to_string(),
        input_data: vec![0.5; 28 * 28], // 28x28图像数据
        parameters: None,
    };
    
    let response = engine.infer(&image_request)?;
    monitor.record_inference(response.processing_time_ms, response.memory_usage_mb);
    
    println!("\n  图像处理结果: {:?}", response.predictions);
    println!("  处理时间: {}ms", response.processing_time_ms);
    
    // 测试文本嵌入器
    let text_request = WasmInferenceRequest {
        model_id: "text_embedder".to_string(),
        input_data: vec![0.1; 512], // 512维文本特征
        parameters: None,
    };
    
    let response = engine.infer(&text_request)?;
    monitor.record_inference(response.processing_time_ms, response.memory_usage_mb);
    
    println!("\n  文本嵌入结果: {:?}", &response.predictions[..10]); // 只显示前10维
    println!("  处理时间: {}ms", response.processing_time_ms);
    
    // 显示性能统计
    let stats = monitor.get_stats();
    println!("\n📊 性能统计:");
    println!("  推理次数: {}", stats.inference_count);
    println!("  平均处理时间: {:.2}ms", stats.average_time_ms);
    println!("  平均内存使用: {:.2}MB", stats.average_memory_mb);
    println!("  总处理时间: {}ms", stats.total_time_ms);
    println!("  总内存使用: {:.2}MB", stats.total_memory_mb);
    
    println!("\n✅ WebAssembly AI推理演示完成！");
    println!("\n🌟 WebAssembly AI的优势：");
    println!("   - 客户端AI计算，保护隐私");
    println!("   - 离线AI功能支持");
    println!("   - 边缘计算优化");
    println!("   - 跨平台兼容性");
    println!("   - 接近原生性能");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wasm_ai_engine_creation() {
        let engine = WasmAIEngine::new();
        assert_eq!(engine.list_models().len(), 0);
    }
    
    #[test]
    fn test_model_loading() {
        let mut engine = WasmAIEngine::new();
        let model = ModelFactory::create_simple_classifier();
        
        assert!(engine.load_model(model).is_ok());
        assert_eq!(engine.list_models().len(), 1);
    }
    
    #[test]
    fn test_inference() {
        let mut engine = WasmAIEngine::new();
        let model = ModelFactory::create_simple_classifier();
        engine.load_model(model).unwrap();
        
        let request = WasmInferenceRequest {
            model_id: "simple_classifier".to_string(),
            input_data: vec![1.0; 10],
            parameters: None,
        };
        
        let response = engine.infer(&request);
        assert!(response.is_ok());
        
        let response = response.unwrap();
        assert_eq!(response.predictions.len(), 3);
        assert!(response.confidence > 0.0);
    }
    
    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        
        monitor.record_inference(100, 1.5);
        monitor.record_inference(200, 2.0);
        
        let stats = monitor.get_stats();
        assert_eq!(stats.inference_count, 2);
        assert_eq!(stats.average_time_ms, 150.0);
        assert_eq!(stats.average_memory_mb, 1.75);
    }
}
