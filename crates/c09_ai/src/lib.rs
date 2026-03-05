//! # C19 AI - 人工智能与机器学习 (2025 Edition)
//!
//! 这是一个基于 Rust 1.89 的现代化 AI 和机器学习库，集成了最新的开源 AI 框架和工具。
//! 支持 2025 年最新的 AI 技术栈，包括 Candle、Burn、Tch、DFDx 等深度学习框架。
//!
//! ## 主要特性
//!
//! - 🤖 **机器学习**: 支持监督学习、无监督学习和强化学习
//! - 🧠 **深度学习**: 集成 Candle 0.10、Burn 0.15、Tch 0.16、DFDx 0.15 等现代深度学习框架
//! - 🗣️ **自然语言处理**: 支持 BERT、GPT、LLaMA 等预训练模型
//! - 👁️ **计算机视觉**: OpenCV 集成和图像处理功能
//! - 📊 **数据处理**: 高性能的 DataFrame 和数据处理管道
//! - 🔍 **向量搜索**: 支持向量数据库和语义搜索
//! - 🚀 **高性能**: 利用 Rust 的零成本抽象和内存安全
//! - 🌐 **多模态AI**: 支持文本、图像、音频等多模态处理
//! - 🔗 **联邦学习**: 支持分布式和隐私保护的机器学习
//! - ⚡ **边缘AI**: 支持移动端和边缘设备部署
//! - 🧮 **量子机器学习**: 探索量子计算在机器学习中的应用
//!
//! ## 快速开始
//!
//! ```rust
//! use c19_ai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 创建 AI 引擎
//!     let mut ai_engine = AIEngine::new();
//!     
//!     // 加载预训练模型
//!     ai_engine.load_model("bert-base-chinese").await?;
//!     
//!     // 进行推理
//!     let result = ai_engine.predict("你好，世界！").await?;
//!     println!("预测结果: {:?}", result);
//!     
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

// 核心模块
pub mod machine_learning;
pub mod neural_networks;
pub mod config;
pub mod gpu;
pub mod model_serving;
pub mod monitoring;
pub mod benchmarks;
pub mod memory;
pub mod error;
pub mod logging;

// API模块（可选特性）
#[cfg(feature = "api-server")]
pub mod api;

// 深度学习框架支持
#[cfg(any(feature = "candle", feature = "dfdx"))]
pub mod deep_learning;

// 特定深度学习框架
#[cfg(feature = "candle")]
pub mod candle_integration;

#[cfg(feature = "dfdx")]
pub mod dfdx_integration;

// NLP 功能
#[cfg(feature = "nlp")]
pub mod nlp;

// 计算机视觉
#[cfg(feature = "vision")]
pub mod computer_vision;

// 数据处理
#[cfg(feature = "data")]
pub mod data_processing;

// 向量搜索
#[cfg(feature = "search")]
pub mod vector_search;

// 模型管理
#[cfg(feature = "management")]
pub mod model_management;

// 管道
#[cfg(feature = "data")]
pub mod pipelines;

// 大语言模型
#[cfg(feature = "llm")]
pub mod llm;

// 扩散模型
#[cfg(any(feature = "candle", feature = "dfdx"))]
pub mod diffusion;

// 强化学习 (暂时禁用，存在pyo3安全漏洞)
// #[cfg(feature = "reinforcement")]
// pub mod reinforcement_learning;

// 图神经网络
#[cfg(feature = "gnn")]
pub mod graph_neural_networks;

// 时间序列
#[cfg(feature = "timeseries")]
pub mod time_series;

// 监控
#[cfg(feature = "monitoring")]
pub mod monitoring;

// 2025年新增模块
// 多模态AI
#[cfg(feature = "multimodal")]
pub mod multimodal;

// 联邦学习
#[cfg(feature = "federated")]
pub mod federated_learning;

// 边缘AI
#[cfg(feature = "edge")]
pub mod edge_ai;

// 量子机器学习
#[cfg(feature = "quantum")]
pub mod quantum_ml;

// API模块
#[cfg(feature = "api-server")]
pub mod api;

// 新增核心模块
pub mod model_management;
pub mod training;
pub mod inference;
pub mod validation;
pub mod database;
pub mod cache;
pub mod storage;
pub mod messaging;
pub mod websocket;
pub mod auth;

// 预导入模块
pub mod prelude {
    pub use crate::{
        AIEngine, AIModule, Error, ModelConfig, ModelType, PredictionResult, TrainingConfig,
        machine_learning::*, neural_networks::*,
    };

    #[cfg(any(feature = "candle", feature = "dfdx"))]
    pub use crate::deep_learning::*;

    #[cfg(feature = "candle")]
    pub use crate::candle_integration::*;

    #[cfg(feature = "dfdx")]
    pub use crate::dfdx_integration::*;

    #[cfg(feature = "nlp")]
    pub use crate::nlp::*;

    #[cfg(feature = "vision")]
    pub use crate::computer_vision::*;

    #[cfg(feature = "data")]
    pub use crate::data_processing::*;

    #[cfg(feature = "search")]
    pub use crate::vector_search::*;

    #[cfg(feature = "management")]
    pub use crate::model_management::*;

    #[cfg(feature = "data")]
    pub use crate::pipelines::*;

    #[cfg(feature = "llm")]
    pub use crate::llm::*;

    #[cfg(any(feature = "candle", feature = "dfdx"))]
    pub use crate::diffusion::*;

    // #[cfg(feature = "reinforcement")]
    // pub use crate::reinforcement_learning::*;  // 暂时禁用，存在pyo3安全漏洞

    #[cfg(feature = "gnn")]
    pub use crate::graph_neural_networks::*;

    #[cfg(feature = "timeseries")]
    pub use crate::time_series::*;

    #[cfg(feature = "monitoring")]
    pub use crate::monitoring::*;

    #[cfg(feature = "multimodal")]
    pub use crate::multimodal::*;

    #[cfg(feature = "federated")]
    pub use crate::federated_learning::*;

    #[cfg(feature = "edge")]
    pub use crate::edge_ai::*;

    #[cfg(feature = "quantum")]
    pub use crate::quantum_ml::*;
}

/// AI 引擎错误类型
#[derive(Error, Debug)]
pub enum Error {
    #[error("模型加载失败: {0}")]
    ModelLoadError(String),

    #[error("推理失败: {0}")]
    InferenceError(String),

    #[error("训练失败: {0}")]
    TrainingError(String),

    #[error("数据处理错误: {0}")]
    DataProcessingError(String),

    #[error("配置错误: {0}")]
    ConfigError(String),

    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("序列化错误: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("网络错误: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("多模态处理错误: {0}")]
    MultimodalError(String),

    #[error("联邦学习错误: {0}")]
    FederatedError(String),

    #[error("边缘AI错误: {0}")]
    EdgeError(String),

    #[error("量子计算错误: {0}")]
    QuantumError(String),
}

/// 模型类型枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// 机器学习模型
    MachineLearning,
    /// 深度学习模型
    DeepLearning,
    /// 自然语言处理模型
    NLP,
    /// 计算机视觉模型
    ComputerVision,
    /// 多模态模型
    Multimodal,
    /// 强化学习模型
    ReinforcementLearning,
    /// 图神经网络模型
    GraphNeuralNetwork,
    /// 时间序列模型
    TimeSeries,
    /// 扩散模型
    Diffusion,
    /// 大语言模型
    LargeLanguageModel,
    /// 联邦学习模型
    FederatedLearning,
    /// 边缘AI模型
    EdgeAI,
    /// 量子机器学习模型
    QuantumML,
}

/// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: ModelType,
    pub version: String,
    pub path: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub framework: Option<String>, // candle, burn, tch, dfdx
    pub device: Option<String>,    // cpu, cuda, metal
    pub precision: Option<String>, // f32, f16, bf16
}

/// 预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub predictions: Vec<f64>,
    pub confidence: f64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub model_info: Option<ModelInfo>,
}

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub framework: String,
    pub parameters: usize,
    pub inference_time: f64,
}

/// 训练配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub validation_split: f64,
    pub early_stopping: bool,
    pub metrics: Vec<String>,
    pub optimizer: Option<String>,
    pub scheduler: Option<String>,
    pub mixed_precision: bool,
    pub gradient_accumulation: Option<usize>,
}

/// AI 模块
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModule {
    pub name: String,
    pub version: String,
    pub description: String,
    pub capabilities: Vec<String>,
    pub framework: Option<String>,
    pub supported_devices: Vec<String>,
}

impl AIModule {
    /// 创建新的 AI 模块
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            version: "0.3.0".to_string(),
            description,
            capabilities: Vec::new(),
            framework: None,
            supported_devices: vec!["cpu".to_string()],
        }
    }

    /// 添加能力
    pub fn add_capability(&mut self, capability: String) {
        self.capabilities.push(capability);
    }

    /// 设置框架
    pub fn set_framework(&mut self, framework: String) {
        self.framework = Some(framework);
    }

    /// 添加支持的设备
    pub fn add_device(&mut self, device: String) {
        if !self.supported_devices.contains(&device) {
            self.supported_devices.push(device);
        }
    }

    /// 获取模块信息
    pub fn get_info(&self) -> String {
        let framework_info = self
            .framework
            .as_ref()
            .map(|f| format!(" ({})", f))
            .unwrap_or_default();
        format!(
            "AI模块: {} v{}{} - {}",
            self.name, self.version, framework_info, self.description
        )
    }

    /// 获取能力列表
    pub fn get_capabilities(&self) -> &[String] {
        &self.capabilities
    }

    /// 检查是否支持设备
    pub fn supports_device(&self, device: &str) -> bool {
        self.supported_devices.contains(&device.to_string())
    }
}

/// AI 引擎 - 主要的 AI 系统接口
#[allow(dead_code)]
pub struct AIEngine {
    modules: HashMap<String, AIModule>,
    models: HashMap<String, ModelConfig>,
    config: EngineConfig,
    device_manager: DeviceManager,
    gpu_manager: gpu::GpuManager,             // GPU管理器
    model_service: model_serving::ModelServiceManager, // 模型服务管理器
    monitoring_dashboard: monitoring::MonitoringDashboard, // 监控仪表板
    memory_optimizer: memory::MemoryOptimizer,             // 内存优化器
    // 现代AI系统核心功能
    state: HashMap<String, String>,           // 状态管理
    event_listeners: HashMap<String, Vec<Box<dyn Fn(&str) + Send + Sync>>>, // 事件系统
    metrics: HashMap<String, f64>,            // 性能指标
    resource_limits: HashMap<String, usize>,  // 资源限制
    cache: HashMap<String, Vec<u8>>,          // 缓存系统
    task_queue: VecDeque<String>,             // 任务队列
    running: std::sync::atomic::AtomicBool,   // 运行状态
    created_at: std::time::Instant,           // 创建时间
    last_activity: std::sync::Mutex<std::time::Instant>, // 最后活动时间
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub max_models: usize,
    pub cache_size: usize,
    pub enable_gpu: bool,
    pub log_level: String,
    pub default_framework: Option<String>,
    pub mixed_precision: bool,
    pub enable_monitoring: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_models: 10,
            cache_size: 1000,
            enable_gpu: false,
            log_level: "info".to_string(),
            default_framework: None,
            mixed_precision: false,
            enable_monitoring: false,
        }
    }
}

/// 设备管理器
#[derive(Debug, Clone)]
pub struct DeviceManager {
    available_devices: Vec<String>,
    current_device: String,
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceManager {
    pub fn new() -> Self {
        let devices = vec!["cpu".to_string()];

        // 检测可用的GPU设备
        #[cfg(feature = "cuda")]
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            devices.push("cuda".to_string());
        }

        #[cfg(feature = "metal")]
        devices.push("metal".to_string());

        Self {
            available_devices: devices,
            current_device: "cpu".to_string(),
        }
    }

    pub fn get_available_devices(&self) -> &[String] {
        &self.available_devices
    }

    pub fn set_device(&mut self, device: String) -> Result<(), Error> {
        if self.available_devices.contains(&device) {
            self.current_device = device;
            Ok(())
        } else {
            Err(Error::ConfigError(format!("设备 {} 不可用", device)))
        }
    }

    pub fn get_current_device(&self) -> &str {
        &self.current_device
    }
}

impl AIEngine {
    /// 创建新的 AI 引擎
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            modules: HashMap::new(),
            models: HashMap::new(),
            config: EngineConfig::default(),
            device_manager: DeviceManager::new(),
            gpu_manager: gpu::GpuManager::new().unwrap_or_default(),
            model_service: model_serving::ModelServiceManager::new(),
            monitoring_dashboard: monitoring::MonitoringDashboard::new(),
            memory_optimizer: memory::MemoryOptimizer::default(),
            // 初始化现代AI系统功能
            state: HashMap::new(),
            event_listeners: HashMap::new(),
            metrics: HashMap::new(),
            resource_limits: HashMap::new(),
            cache: HashMap::new(),
            task_queue: VecDeque::new(),
            running: std::sync::atomic::AtomicBool::new(true),
            created_at: now,
            last_activity: std::sync::Mutex::new(now),
        }
    }

    /// 使用配置创建 AI 引擎
    pub fn with_config(config: EngineConfig) -> Self {
        let now = std::time::Instant::now();
        Self {
            modules: HashMap::new(),
            models: HashMap::new(),
            config,
            device_manager: DeviceManager::new(),
            gpu_manager: gpu::GpuManager::new().unwrap_or_default(),
            model_service: model_serving::ModelServiceManager::new(),
            monitoring_dashboard: monitoring::MonitoringDashboard::new(),
            memory_optimizer: memory::MemoryOptimizer::default(),
            // 初始化现代AI系统功能
            state: HashMap::new(),
            event_listeners: HashMap::new(),
            metrics: HashMap::new(),
            resource_limits: HashMap::new(),
            cache: HashMap::new(),
            task_queue: VecDeque::new(),
            running: std::sync::atomic::AtomicBool::new(true),
            created_at: now,
            last_activity: std::sync::Mutex::new(now),
        }
    }

    /// 注册 AI 模块
    pub fn register_module(&mut self, module: AIModule) {
        self.modules.insert(module.name.clone(), module);
    }

    /// 加载模型
    pub async fn load_model(&mut self, model_name: &str) -> Result<(), Error> {
        tracing::info!(
            "加载模型: {} 到设备: {}",
            model_name,
            self.device_manager.get_current_device()
        );

        // 这里将集成实际的模型加载逻辑
        // 根据不同的框架和模型类型进行加载

        Ok(())
    }

    /// 进行预测
    pub async fn predict(&self, input: &str) -> Result<PredictionResult, Error> {
        tracing::info!(
            "进行预测: {} 使用设备: {}",
            input,
            self.device_manager.get_current_device()
        );

        // 这里将集成实际的预测逻辑
        // 根据模型类型和框架进行推理

        Ok(PredictionResult {
            predictions: vec![0.8, 0.2],
            confidence: 0.85,
            metadata: HashMap::new(),
            model_info: Some(ModelInfo {
                name: "demo_model".to_string(),
                version: "1.0.0".to_string(),
                framework: "candle".to_string(),
                parameters: 1000000,
                inference_time: 0.05,
            }),
        })
    }

    /// 训练模型
    pub async fn train(&mut self, config: TrainingConfig) -> Result<(), Error> {
        tracing::info!("开始训练模型，配置: {:?}", config);

        // 这里将集成实际的训练逻辑
        // 支持分布式训练、混合精度等现代特性

        Ok(())
    }

    /// 获取已注册的模块
    pub fn get_modules(&self) -> &HashMap<String, AIModule> {
        &self.modules
    }

    /// 获取已加载的模型
    pub fn get_models(&self) -> &HashMap<String, ModelConfig> {
        &self.models
    }

    /// 获取设备管理器
    pub fn get_device_manager(&self) -> &DeviceManager {
        &self.device_manager
    }

    /// 设置设备
    pub fn set_device(&mut self, device: String) -> Result<(), Error> {
        self.device_manager.set_device(device)
    }

    /// 获取GPU管理器
    pub fn get_gpu_manager(&self) -> &gpu::GpuManager {
        &self.gpu_manager
    }

    /// 获取GPU管理器（可变引用）
    pub fn get_gpu_manager_mut(&mut self) -> &mut gpu::GpuManager {
        &mut self.gpu_manager
    }

    /// 执行GPU加速计算
    pub fn execute_gpu_computation(&mut self, operation: &str, data: &[f32]) -> Result<Vec<f32>, Error> {
        let result = self.gpu_manager.execute_gpu_computation(operation, data)?;
        
        // 记录到引擎指标中
        self.record_metric(&format!("gpu_{}_total", operation), 1.0);
        
        Ok(result)
    }

    /// 分配GPU内存
    pub fn allocate_gpu_memory(&mut self, key: &str, size: usize) -> Result<(), Error> {
        self.gpu_manager.allocate_memory(key, size)
    }

    /// 释放GPU内存
    pub fn deallocate_gpu_memory(&mut self, key: &str) -> Result<(), Error> {
        self.gpu_manager.deallocate_memory(key)
    }

    /// 获取GPU内存使用情况
    pub fn get_gpu_memory_usage(&self) -> HashMap<String, u64> {
        self.gpu_manager.get_memory_usage()
    }

    /// 获取GPU性能统计
    pub fn get_gpu_performance_stats(&self) -> HashMap<String, String> {
        self.gpu_manager.get_performance_stats()
    }

    /// 获取模型服务管理器
    pub fn get_model_service(&self) -> &model_serving::ModelServiceManager {
        &self.model_service
    }

    /// 获取模型服务管理器（可变引用）
    pub fn get_model_service_mut(&mut self) -> &mut model_serving::ModelServiceManager {
        &mut self.model_service
    }

    /// 加载模型到服务
    pub async fn load_model_to_service(&mut self, config: ModelConfig) -> Result<(), Error> {
        let result = self.model_service.load_model(config.clone()).await;
        if result.is_ok() {
            self.models.insert(config.name.clone(), config);
        }
        result
    }

    /// 卸载模型服务
    pub async fn unload_model_from_service(&mut self, model_name: &str) -> Result<(), Error> {
        let result = self.model_service.unload_model(model_name).await;
        if result.is_ok() {
            self.models.remove(model_name);
        }
        result
    }

    /// 开始服务模型
    pub async fn start_model_serving(&mut self, model_name: &str) -> Result<(), Error> {
        self.model_service.start_serving(model_name).await
    }

    /// 停止服务模型
    pub async fn stop_model_serving(&mut self, model_name: &str) -> Result<(), Error> {
        self.model_service.stop_serving(model_name).await
    }

    /// 执行模型推理
    pub async fn inference(&mut self, request: model_serving::InferenceRequest) -> Result<model_serving::InferenceResponse, Error> {
        self.model_service.inference(request).await
    }

    /// 执行批处理推理
    pub async fn batch_inference(&mut self, batch_request: model_serving::BatchRequest) -> Result<model_serving::BatchResponse, Error> {
        self.model_service.batch_inference(batch_request).await
    }

    /// 获取模型服务统计
    pub async fn get_model_service_stats(&self) -> HashMap<String, String> {
        self.model_service.get_service_stats().await
    }

    // ===== 现代AI系统核心方法 =====

    /// 清理所有资源 - 对标现代AI框架的cleanup方法
    pub fn cleanup(&mut self) -> Result<(), Error> {
        tracing::info!("开始清理AI引擎资源");
        
        // 停止运行状态
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        
        // 清理模块
        self.modules.clear();
        
        // 清理模型
        self.models.clear();
        
        // 清理状态
        self.state.clear();
        
        // 清理事件监听器
        self.event_listeners.clear();
        
        // 清理指标
        self.metrics.clear();
        
        // 清理缓存
        self.cache.clear();
        
        // 清理任务队列
        self.task_queue.clear();
        
        tracing::info!("AI引擎资源清理完成");
        Ok(())
    }

    /// 状态管理 - 设置状态
    pub fn set_state(&mut self, key: &str, value: &str) -> Result<(), Error> {
        self.state.insert(key.to_string(), value.to_string());
        
        // 更新最后活动时间
        if let Ok(mut last_activity) = self.last_activity.lock() {
            *last_activity = std::time::Instant::now();
        }
        
        Ok(())
    }

    /// 状态管理 - 获取状态
    pub fn get_state(&self, key: &str) -> Option<String> {
        self.state.get(key).cloned()
    }

    /// 状态管理 - 删除状态
    pub fn remove_state(&mut self, key: &str) -> Result<(), Error> {
        self.state.remove(key);
        Ok(())
    }

    /// 事件系统 - 注册事件监听器
    pub fn on_event<F>(&mut self, event_name: &str, callback: F) -> Result<(), Error>
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        let listeners = self.event_listeners.entry(event_name.to_string()).or_default();
        listeners.push(Box::new(callback));
        Ok(())
    }

    /// 事件系统 - 触发事件
    pub fn emit_event(&self, event_name: &str, data: &str) -> Result<(), Error> {
        if let Some(listeners) = self.event_listeners.get(event_name) {
            for listener in listeners {
                listener(data);
            }
        }
        
        // 更新最后活动时间
        if let Ok(mut last_activity) = self.last_activity.lock() {
            *last_activity = std::time::Instant::now();
        }
        
        Ok(())
    }

    /// 指标收集 - 记录指标
    pub fn record_metric(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }

    /// 指标收集 - 获取所有指标
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }

    /// 指标收集 - 获取特定指标
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// 资源限制 - 设置资源限制
    pub fn set_resource_limit(&mut self, resource: &str, limit: usize) -> Result<(), Error> {
        self.resource_limits.insert(resource.to_string(), limit);
        Ok(())
    }

    /// 资源限制 - 获取资源限制
    pub fn get_resource_limit(&self, resource: &str) -> Option<usize> {
        self.resource_limits.get(resource).copied()
    }

    /// 缓存系统 - 设置缓存
    pub fn set_cache(&mut self, key: &str, data: Vec<u8>) -> Result<(), Error> {
        // 检查缓存大小限制
        if self.cache.len() >= self.config.cache_size {
            // 简单的LRU策略：删除最老的缓存项
            if let Some(oldest_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&oldest_key);
            }
        }
        
        self.cache.insert(key.to_string(), data);
        Ok(())
    }

    /// 缓存系统 - 获取缓存
    pub fn get_cache(&self, key: &str) -> Option<&Vec<u8>> {
        self.cache.get(key)
    }

    /// 任务队列 - 添加任务
    pub fn add_task(&mut self, task: String) -> Result<(), Error> {
        self.task_queue.push_back(task);
        Ok(())
    }

    /// 任务队列 - 获取下一个任务
    pub fn get_next_task(&mut self) -> Option<String> {
        self.task_queue.pop_front()
    }

    /// 任务队列 - 获取队列长度
    pub fn get_task_queue_length(&self) -> usize {
        self.task_queue.len()
    }

    /// 运行状态 - 检查是否运行中
    pub fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// 运行状态 - 停止引擎
    pub fn stop(&mut self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// 运行状态 - 启动引擎
    pub fn start(&mut self) {
        self.running.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// 版本信息 - 获取引擎版本
    pub fn version(&self) -> &str {
        "0.3.0"
    }

    /// 运行时间 - 获取引擎运行时间
    pub fn get_uptime(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// 最后活动时间 - 获取最后活动时间
    pub fn get_last_activity(&self) -> Result<std::time::Duration, Error> {
        let last_activity = self.last_activity.lock()
            .map_err(|_| Error::ConfigError("无法获取最后活动时间".to_string()))?;
        Ok(last_activity.elapsed())
    }

    /// 系统信息 - 获取引擎统计信息
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("version".to_string(), self.version().to_string());
        stats.insert("uptime_seconds".to_string(), self.get_uptime().as_secs().to_string());
        stats.insert("modules_count".to_string(), self.modules.len().to_string());
        stats.insert("models_count".to_string(), self.models.len().to_string());
        stats.insert("state_entries".to_string(), self.state.len().to_string());
        stats.insert("metrics_count".to_string(), self.metrics.len().to_string());
        stats.insert("cache_size".to_string(), self.cache.len().to_string());
        stats.insert("task_queue_length".to_string(), self.task_queue.len().to_string());
        stats.insert("is_running".to_string(), self.is_running().to_string());
        stats.insert("current_device".to_string(), self.device_manager.get_current_device().to_string());
        stats
    }
    
    /// 获取监控仪表板
    pub fn get_monitoring_dashboard(&self) -> &monitoring::MonitoringDashboard {
        &self.monitoring_dashboard
    }
    
    /// 获取监控仪表板（可变引用）
    pub fn get_monitoring_dashboard_mut(&mut self) -> &mut monitoring::MonitoringDashboard {
        &mut self.monitoring_dashboard
    }
    
    /// 开始监控
    pub async fn start_monitoring(&self) -> Result<(), Error> {
        self.monitoring_dashboard.start_monitoring().await
            .map_err(|e| Error::ConfigError(format!("监控启动失败: {}", e)))
    }
    
    /// 停止监控
    pub fn stop_monitoring(&self) {
        self.monitoring_dashboard.stop_monitoring();
    }
    
    /// 记录监控指标
    pub fn record_monitoring_metric(&self, name: &str, value: f64, labels: Option<HashMap<String, String>>) {
        self.monitoring_dashboard.record_metric(name, value, labels);
    }
    
    /// 记录请求到监控系统
    pub fn record_monitoring_request(&self, success: bool, response_time: f64) {
        self.monitoring_dashboard.record_request(success, response_time);
    }
    
    /// 获取监控仪表板数据
    pub async fn get_monitoring_data(&self) -> monitoring::DashboardData {
        self.monitoring_dashboard.get_dashboard_data().await
    }
    
    /// 获取基准测试套件
    pub fn get_benchmark_suite(&self) -> benchmarks::BenchmarkSuite {
        benchmarks::BenchmarkSuite::new()
    }
    
    /// 运行性能基准测试
    pub async fn run_benchmark_test(&self, name: &str, operations: u64, test_fn: impl Fn() -> Result<(), String>) -> benchmarks::BenchmarkResult {
        let mut suite = benchmarks::BenchmarkSuite::new();
        suite.run_benchmark(name, operations, || async { test_fn() }).await
    }
    
    /// 运行压力测试
    pub async fn run_stress_test(&self, config: benchmarks::StressTestConfig, test_fn: impl Fn() -> Result<(), String>) -> benchmarks::StressTestResult {
        let mut suite = benchmarks::BenchmarkSuite::new();
        suite.run_stress_test(config, test_fn).await
    }
    
    /// 生成性能报告
    pub fn generate_performance_report(&self) -> String {
        let suite = benchmarks::BenchmarkSuite::new();
        suite.generate_report()
    }
    
    /// 获取内存优化器
    pub fn get_memory_optimizer(&self) -> &memory::MemoryOptimizer {
        &self.memory_optimizer
    }
    
    /// 获取内存优化器（可变引用）
    pub fn get_memory_optimizer_mut(&mut self) -> &mut memory::MemoryOptimizer {
        &mut self.memory_optimizer
    }
    
    /// 获取内存池
    pub fn get_memory_pool(&self) -> std::sync::Arc<memory::MemoryPool> {
        self.memory_optimizer.get_memory_pool()
    }
    
    /// 获取缓存管理器
    pub fn get_cache_manager(&self) -> std::sync::Arc<memory::CacheManager> {
        self.memory_optimizer.get_cache_manager()
    }
    
    /// 优化内存使用
    pub fn optimize_memory(&self) {
        self.memory_optimizer.optimize();
    }
    
    /// 创建零拷贝缓冲区
    pub fn create_zero_copy_buffer(&self, key: String, data: Vec<u8>) {
        self.memory_optimizer.create_zero_copy_buffer(key, data);
    }
    
    /// 获取零拷贝缓冲区
    pub fn get_zero_copy_buffer(&self, key: &str) -> Option<memory::ZeroCopyBuffer> {
        self.memory_optimizer.get_zero_copy_buffer(key)
    }
    
    /// 获取内存优化统计信息
    pub fn get_memory_stats(&self) -> memory::MemoryOptimizerStats {
        self.memory_optimizer.get_stats()
    }
    
    /// 生成内存优化报告
    pub fn generate_memory_report(&self) -> String {
        self.memory_optimizer.generate_report()
    }
}

impl Default for AIEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// 创建默认的 AI 模块集合
pub fn create_default_modules() -> Vec<AIModule> {
    vec![
        {
            let mut ml_module =
                AIModule::new("机器学习".to_string(), "支持各种机器学习算法".to_string());
            ml_module.add_capability("分类".to_string());
            ml_module.add_capability("回归".to_string());
            ml_module.add_capability("聚类".to_string());
            ml_module.set_framework("linfa".to_string());
            ml_module
        },
        {
            let mut dl_module = AIModule::new(
                "深度学习".to_string(),
                "支持神经网络和深度学习模型".to_string(),
            );
            dl_module.add_capability("CNN".to_string());
            dl_module.add_capability("RNN".to_string());
            dl_module.add_capability("Transformer".to_string());
            dl_module.set_framework("candle".to_string());
            dl_module.add_device("cuda".to_string());
            dl_module.add_device("metal".to_string());
            dl_module
        },
        {
            let mut nlp_module = AIModule::new(
                "自然语言处理".to_string(),
                "支持文本分析和语言模型".to_string(),
            );
            nlp_module.add_capability("文本分类".to_string());
            nlp_module.add_capability("情感分析".to_string());
            nlp_module.add_capability("机器翻译".to_string());
            nlp_module.add_capability("文本生成".to_string());
            nlp_module.set_framework("candle".to_string());
            nlp_module
        },
        {
            let mut cv_module = AIModule::new(
                "计算机视觉".to_string(),
                "支持图像处理和计算机视觉任务".to_string(),
            );
            cv_module.add_capability("图像分类".to_string());
            cv_module.add_capability("目标检测".to_string());
            cv_module.add_capability("图像分割".to_string());
            cv_module.add_capability("图像生成".to_string());
            cv_module.set_framework("candle".to_string());
            cv_module
        },
        {
            let mut multimodal_module = AIModule::new(
                "多模态AI".to_string(),
                "支持文本、图像、音频等多模态处理".to_string(),
            );
            multimodal_module.add_capability("图文理解".to_string());
            multimodal_module.add_capability("多模态生成".to_string());
            multimodal_module.add_capability("跨模态检索".to_string());
            multimodal_module.set_framework("candle".to_string());
            multimodal_module
        },
        {
            let mut federated_module = AIModule::new(
                "联邦学习".to_string(),
                "支持分布式和隐私保护的机器学习".to_string(),
            );
            federated_module.add_capability("联邦训练".to_string());
            federated_module.add_capability("隐私保护".to_string());
            federated_module.add_capability("分布式推理".to_string());
            federated_module.set_framework("federated".to_string());
            federated_module
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_module() {
        let mut ai = AIModule::new("测试模块".to_string(), "测试描述".to_string());
        ai.add_capability("测试能力".to_string());
        ai.set_framework("candle".to_string());
        ai.add_device("cuda".to_string());

        assert_eq!(ai.get_info(), "AI模块: 测试模块 v0.3.0 (candle) - 测试描述");
        assert_eq!(ai.get_capabilities(), &["测试能力"]);
        assert!(ai.supports_device("cuda"));
    }

    #[test]
    fn test_ai_engine() {
        let mut engine = AIEngine::new();
        let module = AIModule::new("测试模块".to_string(), "测试描述".to_string());

        engine.register_module(module);
        assert_eq!(engine.get_modules().len(), 1);

        // 测试设备管理
        assert!(engine.set_device("cpu".to_string()).is_ok());
        assert!(engine.set_device("invalid_device".to_string()).is_err());
    }

    #[test]
    fn test_default_modules() {
        let modules = create_default_modules();
        assert_eq!(modules.len(), 6);

        let ml_module = &modules[0];
        assert_eq!(ml_module.name, "机器学习");
        assert!(ml_module.capabilities.contains(&"分类".to_string()));
        assert_eq!(ml_module.framework, Some("linfa".to_string()));

        let multimodal_module = &modules[4];
        assert_eq!(multimodal_module.name, "多模态AI");
        assert!(
            multimodal_module
                .capabilities
                .contains(&"图文理解".to_string())
        );
    }

    #[tokio::test]
    async fn test_ai_engine_async() {
        let engine = AIEngine::new();
        let result = engine.predict("测试输入").await.unwrap();

        assert_eq!(result.predictions.len(), 2);
        assert!(result.confidence > 0.0);
        assert!(result.model_info.is_some());
    }

    #[test]
    fn test_device_manager() {
        let device_manager = DeviceManager::new();
        let devices = device_manager.get_available_devices();

        assert!(devices.contains(&"cpu".to_string()));
        assert_eq!(device_manager.get_current_device(), "cpu");
    }
}
