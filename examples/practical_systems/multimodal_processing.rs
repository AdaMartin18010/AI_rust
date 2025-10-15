//! 多模态AI处理系统实现
//! 
//! 本示例展示了一个完整的多模态AI处理系统，包括：
//! - 文本、图像、音频的统一处理
//! - 跨模态理解和生成
//! - 多模态融合和推理
//! - 实时处理和流式输出
//! - 性能优化和缓存

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::time::{Instant, SystemTime};

/// 多模态数据类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultimodalData {
    Text(String),
    Image(Vec<u8>),
    Audio(Vec<u8>),
    Video(Vec<u8>),
    Mixed {
        text: Option<String>,
        image: Option<Vec<u8>>,
        audio: Option<Vec<u8>>,
        video: Option<Vec<u8>>,
    },
}

/// 多模态特征向量
#[derive(Debug, Clone)]
pub struct MultimodalFeatures {
    pub text_features: Option<Vec<f64>>,
    pub image_features: Option<Vec<f64>>,
    pub audio_features: Option<Vec<f64>>,
    pub video_features: Option<Vec<f64>>,
    pub fused_features: Vec<f64>,
    pub confidence: f64,
}

/// 多模态处理结果
#[derive(Debug, Clone)]
pub struct MultimodalResult {
    pub input: MultimodalData,
    pub features: MultimodalFeatures,
    pub output: MultimodalData,
    pub processing_time: std::time::Duration,
    pub confidence: f64,
    pub modalities_used: Vec<String>,
}

/// 文本编码器trait
pub trait TextEncoder: Send + Sync {
    async fn encode(&self, text: &str) -> Result<Vec<f64>>;
    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}

/// 图像编码器trait
pub trait ImageEncoder: Send + Sync {
    async fn encode(&self, image: &[u8]) -> Result<Vec<f64>>;
    async fn encode_batch(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f64>>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
    fn supported_formats(&self) -> Vec<String>;
}

/// 音频编码器trait
pub trait AudioEncoder: Send + Sync {
    async fn encode(&self, audio: &[u8]) -> Result<Vec<f64>>;
    async fn encode_batch(&self, audios: &[Vec<u8>]) -> Result<Vec<Vec<f64>>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
    fn supported_formats(&self) -> Vec<String>;
}

/// 多模态融合器trait
pub trait MultimodalFusion: Send + Sync {
    async fn fuse(&self, features: &MultimodalFeatures) -> Result<Vec<f64>>;
    async fn fuse_batch(&self, features_list: &[MultimodalFeatures]) -> Result<Vec<Vec<f64>>>;
    fn output_dimension(&self) -> usize;
    fn fusion_strategy(&self) -> FusionStrategy;
}

/// 融合策略
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    Concatenation,
    WeightedAverage,
    Attention,
    CrossModal,
}

/// 简单文本编码器实现
pub struct SimpleTextEncoder {
    pub dimension: usize,
    pub model_name: String,
    pub cache: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl SimpleTextEncoder {
    pub fn new(dimension: usize, model_name: String) -> Self {
        Self {
            dimension,
            model_name,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 文本预处理
    fn preprocess_text(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }
    
    /// 生成文本嵌入
    fn generate_text_embedding(&self, text: &str) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let processed_text = self.preprocess_text(text);
        let mut hasher = DefaultHasher::new();
        processed_text.hash(&mut hasher);
        let base_hash = hasher.finish();
        
        let mut embedding = vec![0.0; self.dimension];
        
        for i in 0..self.dimension {
            let mut hasher = DefaultHasher::new();
            (base_hash, i as u64, processed_text.len()).hash(&mut hasher);
            let hash = hasher.finish();
            embedding[i] = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
        
        // L2归一化
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        
        embedding
    }
}

impl TextEncoder for SimpleTextEncoder {
    async fn encode(&self, text: &str) -> Result<Vec<f64>> {
        // 检查缓存
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }
        
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
        
        let embedding = self.generate_text_embedding(text);
        
        // 存储到缓存
        {
            let mut cache = self.cache.write().await;
            cache.insert(text.to_string(), embedding.clone());
        }
        
        Ok(embedding)
    }
    
    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        let mut embeddings = Vec::new();
        
        // 并发处理
        let mut handles = Vec::new();
        for text in texts {
            let encoder = self;
            let text = text.clone();
            let handle = tokio::spawn(async move {
                encoder.encode(&text).await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            embeddings.push(handle.await??);
        }
        
        Ok(embeddings)
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// 简单图像编码器实现
pub struct SimpleImageEncoder {
    pub dimension: usize,
    pub model_name: String,
    pub supported_formats: Vec<String>,
    pub cache: Arc<RwLock<HashMap<Vec<u8>, Vec<f64>>>>,
}

impl SimpleImageEncoder {
    pub fn new(dimension: usize, model_name: String) -> Self {
        Self {
            dimension,
            model_name,
            supported_formats: vec!["jpg".to_string(), "png".to_string(), "bmp".to_string()],
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 生成图像嵌入
    fn generate_image_embedding(&self, image: &[u8]) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        image.hash(&mut hasher);
        let base_hash = hasher.finish();
        
        let mut embedding = vec![0.0; self.dimension];
        
        for i in 0..self.dimension {
            let mut hasher = DefaultHasher::new();
            (base_hash, i as u64, image.len()).hash(&mut hasher);
            let hash = hasher.finish();
            embedding[i] = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
        
        // L2归一化
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        
        embedding
    }
}

impl ImageEncoder for SimpleImageEncoder {
    async fn encode(&self, image: &[u8]) -> Result<Vec<f64>> {
        // 检查缓存
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(image) {
                return Ok(cached.clone());
            }
        }
        
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        let embedding = self.generate_image_embedding(image);
        
        // 存储到缓存
        {
            let mut cache = self.cache.write().await;
            cache.insert(image.to_vec(), embedding.clone());
        }
        
        Ok(embedding)
    }
    
    async fn encode_batch(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f64>>> {
        let mut embeddings = Vec::new();
        
        // 并发处理
        let mut handles = Vec::new();
        for image in images {
            let encoder = self;
            let image = image.clone();
            let handle = tokio::spawn(async move {
                encoder.encode(&image).await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            embeddings.push(handle.await??);
        }
        
        Ok(embeddings)
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
    
    fn supported_formats(&self) -> Vec<String> {
        self.supported_formats.clone()
    }
}

/// 简单音频编码器实现
pub struct SimpleAudioEncoder {
    pub dimension: usize,
    pub model_name: String,
    pub supported_formats: Vec<String>,
    pub cache: Arc<RwLock<HashMap<Vec<u8>, Vec<f64>>>>,
}

impl SimpleAudioEncoder {
    pub fn new(dimension: usize, model_name: String) -> Self {
        Self {
            dimension,
            model_name,
            supported_formats: vec!["wav".to_string(), "mp3".to_string(), "flac".to_string()],
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 生成音频嵌入
    fn generate_audio_embedding(&self, audio: &[u8]) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        audio.hash(&mut hasher);
        let base_hash = hasher.finish();
        
        let mut embedding = vec![0.0; self.dimension];
        
        for i in 0..self.dimension {
            let mut hasher = DefaultHasher::new();
            (base_hash, i as u64, audio.len()).hash(&mut hasher);
            let hash = hasher.finish();
            embedding[i] = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
        
        // L2归一化
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        
        embedding
    }
}

impl AudioEncoder for SimpleAudioEncoder {
    async fn encode(&self, audio: &[u8]) -> Result<Vec<f64>> {
        // 检查缓存
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(audio) {
                return Ok(cached.clone());
            }
        }
        
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(40)).await;
        
        let embedding = self.generate_audio_embedding(audio);
        
        // 存储到缓存
        {
            let mut cache = self.cache.write().await;
            cache.insert(audio.to_vec(), embedding.clone());
        }
        
        Ok(embedding)
    }
    
    async fn encode_batch(&self, audios: &[Vec<u8>]) -> Result<Vec<Vec<f64>>> {
        let mut embeddings = Vec::new();
        
        // 并发处理
        let mut handles = Vec::new();
        for audio in audios {
            let encoder = self;
            let audio = audio.clone();
            let handle = tokio::spawn(async move {
                encoder.encode(&audio).await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            embeddings.push(handle.await??);
        }
        
        Ok(embeddings)
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
    
    fn supported_formats(&self) -> Vec<String> {
        self.supported_formats.clone()
    }
}

/// 多模态融合器实现
pub struct AttentionFusion {
    pub output_dimension: usize,
    pub fusion_strategy: FusionStrategy,
    pub attention_weights: Vec<f64>,
}

impl AttentionFusion {
    pub fn new(output_dimension: usize, fusion_strategy: FusionStrategy) -> Self {
        Self {
            output_dimension,
            fusion_strategy,
            attention_weights: vec![0.25, 0.25, 0.25, 0.25], // 文本、图像、音频、视频的权重
        }
    }
    
    /// 计算注意力权重
    fn compute_attention_weights(&self, features: &MultimodalFeatures) -> Vec<f64> {
        let mut weights = vec![0.0; 4];
        let mut total_confidence = 0.0;
        
        if let Some(_) = features.text_features {
            weights[0] = 0.3;
            total_confidence += 0.3;
        }
        if let Some(_) = features.image_features {
            weights[1] = 0.3;
            total_confidence += 0.3;
        }
        if let Some(_) = features.audio_features {
            weights[2] = 0.2;
            total_confidence += 0.2;
        }
        if let Some(_) = features.video_features {
            weights[3] = 0.2;
            total_confidence += 0.2;
        }
        
        // 归一化权重
        if total_confidence > 0.0 {
            for weight in &mut weights {
                *weight /= total_confidence;
            }
        }
        
        weights
    }
    
    /// 注意力融合
    fn attention_fusion(&self, features: &MultimodalFeatures) -> Vec<f64> {
        let attention_weights = self.compute_attention_weights(features);
        let mut fused = vec![0.0; self.output_dimension];
        
        let mut weight_index = 0;
        
        if let Some(text_features) = &features.text_features {
            let weight = attention_weights[weight_index];
            for (i, &feature) in text_features.iter().enumerate() {
                if i < fused.len() {
                    fused[i] += feature * weight;
                }
            }
            weight_index += 1;
        }
        
        if let Some(image_features) = &features.image_features {
            let weight = attention_weights[weight_index];
            for (i, &feature) in image_features.iter().enumerate() {
                if i < fused.len() {
                    fused[i] += feature * weight;
                }
            }
            weight_index += 1;
        }
        
        if let Some(audio_features) = &features.audio_features {
            let weight = attention_weights[weight_index];
            for (i, &feature) in audio_features.iter().enumerate() {
                if i < fused.len() {
                    fused[i] += feature * weight;
                }
            }
            weight_index += 1;
        }
        
        if let Some(video_features) = &features.video_features {
            let weight = attention_weights[weight_index];
            for (i, &feature) in video_features.iter().enumerate() {
                if i < fused.len() {
                    fused[i] += feature * weight;
                }
            }
        }
        
        fused
    }
    
    /// 连接融合
    fn concatenation_fusion(&self, features: &MultimodalFeatures) -> Vec<f64> {
        let mut fused = Vec::new();
        
        if let Some(text_features) = &features.text_features {
            fused.extend(text_features);
        }
        if let Some(image_features) = &features.image_features {
            fused.extend(image_features);
        }
        if let Some(audio_features) = &features.audio_features {
            fused.extend(audio_features);
        }
        if let Some(video_features) = &features.video_features {
            fused.extend(video_features);
        }
        
        // 截断或填充到目标维度
        if fused.len() > self.output_dimension {
            fused.truncate(self.output_dimension);
        } else {
            fused.resize(self.output_dimension, 0.0);
        }
        
        fused
    }
}

impl MultimodalFusion for AttentionFusion {
    async fn fuse(&self, features: &MultimodalFeatures) -> Result<Vec<f64>> {
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
        
        let fused = match self.fusion_strategy {
            FusionStrategy::Attention => self.attention_fusion(features),
            FusionStrategy::Concatenation => self.concatenation_fusion(features),
            FusionStrategy::WeightedAverage => {
                // 加权平均融合
                let mut fused = vec![0.0; self.output_dimension];
                let mut total_weight = 0.0;
                
                if let Some(text_features) = &features.text_features {
                    let weight = 0.4;
                    for (i, &feature) in text_features.iter().enumerate() {
                        if i < fused.len() {
                            fused[i] += feature * weight;
                        }
                    }
                    total_weight += weight;
                }
                
                if let Some(image_features) = &features.image_features {
                    let weight = 0.3;
                    for (i, &feature) in image_features.iter().enumerate() {
                        if i < fused.len() {
                            fused[i] += feature * weight;
                        }
                    }
                    total_weight += weight;
                }
                
                if let Some(audio_features) = &features.audio_features {
                    let weight = 0.2;
                    for (i, &feature) in audio_features.iter().enumerate() {
                        if i < fused.len() {
                            fused[i] += feature * weight;
                        }
                    }
                    total_weight += weight;
                }
                
                if let Some(video_features) = &features.video_features {
                    let weight = 0.1;
                    for (i, &feature) in video_features.iter().enumerate() {
                        if i < fused.len() {
                            fused[i] += feature * weight;
                        }
                    }
                    total_weight += weight;
                }
                
                // 归一化
                if total_weight > 0.0 {
                    for x in &mut fused {
                        *x /= total_weight;
                    }
                }
                
                fused
            },
            FusionStrategy::CrossModal => {
                // 跨模态融合（简化实现）
                self.attention_fusion(features)
            },
        };
        
        Ok(fused)
    }
    
    async fn fuse_batch(&self, features_list: &[MultimodalFeatures]) -> Result<Vec<Vec<f64>>> {
        let mut fused_list = Vec::new();
        
        // 并发处理
        let mut handles = Vec::new();
        for features in features_list {
            let fusion = self;
            let features = features.clone();
            let handle = tokio::spawn(async move {
                fusion.fuse(&features).await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            fused_list.push(handle.await??);
        }
        
        Ok(fused_list)
    }
    
    fn output_dimension(&self) -> usize {
        self.output_dimension
    }
    
    fn fusion_strategy(&self) -> FusionStrategy {
        self.fusion_strategy.clone()
    }
}

/// 多模态AI处理系统
pub struct MultimodalAIProcessor {
    pub text_encoder: Arc<dyn TextEncoder>,
    pub image_encoder: Arc<dyn ImageEncoder>,
    pub audio_encoder: Arc<dyn AudioEncoder>,
    pub fusion: Arc<dyn MultimodalFusion>,
    pub cache: Arc<RwLock<HashMap<String, MultimodalResult>>>,
}

impl MultimodalAIProcessor {
    pub fn new(
        text_encoder: Arc<dyn TextEncoder>,
        image_encoder: Arc<dyn ImageEncoder>,
        audio_encoder: Arc<dyn AudioEncoder>,
        fusion: Arc<dyn MultimodalFusion>,
    ) -> Self {
        Self {
            text_encoder,
            image_encoder,
            audio_encoder,
            fusion,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 处理多模态输入
    pub async fn process(&self, input: MultimodalData) -> Result<MultimodalResult> {
        let start_time = Instant::now();
        
        // 生成缓存键
        let cache_key = self.generate_cache_key(&input);
        
        // 检查缓存
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        
        // 提取特征
        let features = self.extract_features(&input).await?;
        
        // 融合特征
        let fused_features = self.fusion.fuse(&features).await?;
        
        // 生成输出
        let output = self.generate_output(&input, &fused_features).await?;
        
        // 计算置信度
        let confidence = self.calculate_confidence(&features);
        
        // 确定使用的模态
        let modalities_used = self.get_modalities_used(&input);
        
        let processing_time = start_time.elapsed();
        
        let result = MultimodalResult {
            input: input.clone(),
            features: MultimodalFeatures {
                text_features: features.text_features,
                image_features: features.image_features,
                audio_features: features.audio_features,
                video_features: features.video_features,
                fused_features,
                confidence: features.confidence,
            },
            output,
            processing_time,
            confidence,
            modalities_used,
        };
        
        // 存储到缓存
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, result.clone());
        }
        
        Ok(result)
    }
    
    /// 提取多模态特征
    async fn extract_features(&self, input: &MultimodalData) -> Result<MultimodalFeatures> {
        let mut text_features = None;
        let mut image_features = None;
        let mut audio_features = None;
        let mut video_features = None;
        
        match input {
            MultimodalData::Text(text) => {
                text_features = Some(self.text_encoder.encode(text).await?);
            },
            MultimodalData::Image(image) => {
                image_features = Some(self.image_encoder.encode(image).await?);
            },
            MultimodalData::Audio(audio) => {
                audio_features = Some(self.audio_encoder.encode(audio).await?);
            },
            MultimodalData::Video(video) => {
                // 简化处理，将视频当作图像处理
                image_features = Some(self.image_encoder.encode(video).await?);
            },
            MultimodalData::Mixed { text, image, audio, video } => {
                // 并行处理所有模态
                let mut handles = Vec::new();
                
                if let Some(text) = text {
                    let encoder = self.text_encoder.clone();
                    let text = text.clone();
                    let handle = tokio::spawn(async move {
                        encoder.encode(&text).await
                    });
                    handles.push(("text", handle));
                }
                
                if let Some(image) = image {
                    let encoder = self.image_encoder.clone();
                    let image = image.clone();
                    let handle = tokio::spawn(async move {
                        encoder.encode(&image).await
                    });
                    handles.push(("image", handle));
                }
                
                if let Some(audio) = audio {
                    let encoder = self.audio_encoder.clone();
                    let audio = audio.clone();
                    let handle = tokio::spawn(async move {
                        encoder.encode(&audio).await
                    });
                    handles.push(("audio", handle));
                }
                
                if let Some(video) = video {
                    let encoder = self.image_encoder.clone();
                    let video = video.clone();
                    let handle = tokio::spawn(async move {
                        encoder.encode(&video).await
                    });
                    handles.push(("video", handle));
                }
                
                // 收集结果
                for (modality, handle) in handles {
                    match handle.await? {
                        Ok(features) => {
                            match modality {
                                "text" => text_features = Some(features),
                                "image" => image_features = Some(features),
                                "audio" => audio_features = Some(features),
                                "video" => video_features = Some(features),
                                _ => {},
                            }
                        },
                        Err(e) => {
                            eprintln!("处理{}模态时出错: {}", modality, e);
                        }
                    }
                }
            }
        }
        
        Ok(MultimodalFeatures {
            text_features,
            image_features,
            audio_features,
            video_features,
            fused_features: Vec::new(), // 将在融合时填充
            confidence: 0.0, // 将在计算时填充
        })
    }
    
    /// 生成输出
    async fn generate_output(&self, input: &MultimodalData, fused_features: &[f64]) -> Result<MultimodalData> {
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // 根据输入类型生成相应的输出
        match input {
            MultimodalData::Text(_) => {
                Ok(MultimodalData::Text(format!(
                    "基于文本输入生成的分析结果。特征维度: {}",
                    fused_features.len()
                )))
            },
            MultimodalData::Image(_) => {
                Ok(MultimodalData::Text(format!(
                    "基于图像输入生成的描述。特征维度: {}",
                    fused_features.len()
                )))
            },
            MultimodalData::Audio(_) => {
                Ok(MultimodalData::Text(format!(
                    "基于音频输入生成的转录。特征维度: {}",
                    fused_features.len()
                )))
            },
            MultimodalData::Video(_) => {
                Ok(MultimodalData::Text(format!(
                    "基于视频输入生成的分析。特征维度: {}",
                    fused_features.len()
                )))
            },
            MultimodalData::Mixed { .. } => {
                Ok(MultimodalData::Text(format!(
                    "基于多模态输入生成的综合分析。特征维度: {}",
                    fused_features.len()
                )))
            }
        }
    }
    
    /// 计算置信度
    fn calculate_confidence(&self, features: &MultimodalFeatures) -> f64 {
        let mut confidence = 0.0;
        let mut modality_count = 0;
        
        if features.text_features.is_some() {
            confidence += 0.3;
            modality_count += 1;
        }
        if features.image_features.is_some() {
            confidence += 0.3;
            modality_count += 1;
        }
        if features.audio_features.is_some() {
            confidence += 0.2;
            modality_count += 1;
        }
        if features.video_features.is_some() {
            confidence += 0.2;
            modality_count += 1;
        }
        
        // 多模态融合提升置信度
        if modality_count > 1 {
            confidence *= 1.2;
        }
        
        confidence.min(1.0)
    }
    
    /// 获取使用的模态
    fn get_modalities_used(&self, input: &MultimodalData) -> Vec<String> {
        match input {
            MultimodalData::Text(_) => vec!["text".to_string()],
            MultimodalData::Image(_) => vec!["image".to_string()],
            MultimodalData::Audio(_) => vec!["audio".to_string()],
            MultimodalData::Video(_) => vec!["video".to_string()],
            MultimodalData::Mixed { text, image, audio, video } => {
                let mut modalities = Vec::new();
                if text.is_some() { modalities.push("text".to_string()); }
                if image.is_some() { modalities.push("image".to_string()); }
                if audio.is_some() { modalities.push("audio".to_string()); }
                if video.is_some() { modalities.push("video".to_string()); }
                modalities
            }
        }
    }
    
    /// 生成缓存键
    fn generate_cache_key(&self, input: &MultimodalData) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        match input {
            MultimodalData::Text(text) => text.hash(&mut hasher),
            MultimodalData::Image(image) => image.hash(&mut hasher),
            MultimodalData::Audio(audio) => audio.hash(&mut hasher),
            MultimodalData::Video(video) => video.hash(&mut hasher),
            MultimodalData::Mixed { text, image, audio, video } => {
                text.hash(&mut hasher);
                image.hash(&mut hasher);
                audio.hash(&mut hasher);
                video.hash(&mut hasher);
            }
        }
        
        format!("multimodal_{}", hasher.finish())
    }
    
    /// 批量处理
    pub async fn process_batch(&self, inputs: &[MultimodalData]) -> Result<Vec<MultimodalResult>> {
        let mut results = Vec::new();
        
        // 并发处理
        let mut handles = Vec::new();
        for input in inputs {
            let processor = self;
            let input = input.clone();
            let handle = tokio::spawn(async move {
                processor.process(input).await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            results.push(handle.await??);
        }
        
        Ok(results)
    }
    
    /// 获取系统统计信息
    pub async fn get_stats(&self) -> HashMap<String, usize> {
        let cache = self.cache.read().await;
        let mut stats = HashMap::new();
        
        stats.insert("cached_results".to_string(), cache.len());
        stats.insert("text_encoder_dimension".to_string(), self.text_encoder.dimension());
        stats.insert("image_encoder_dimension".to_string(), self.image_encoder.dimension());
        stats.insert("audio_encoder_dimension".to_string(), self.audio_encoder.dimension());
        stats.insert("fusion_output_dimension".to_string(), self.fusion.output_dimension());
        
        stats
    }
    
    /// 清理缓存
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_multimodal_processor() {
        // 创建编码器
        let text_encoder = Arc::new(SimpleTextEncoder::new(128, "test-text".to_string()));
        let image_encoder = Arc::new(SimpleImageEncoder::new(256, "test-image".to_string()));
        let audio_encoder = Arc::new(SimpleAudioEncoder::new(192, "test-audio".to_string()));
        let fusion = Arc::new(AttentionFusion::new(512, FusionStrategy::Attention));
        
        // 创建多模态处理器
        let processor = MultimodalAIProcessor::new(
            text_encoder,
            image_encoder,
            audio_encoder,
            fusion,
        );
        
        // 测试文本处理
        let text_input = MultimodalData::Text("这是一个测试文本".to_string());
        let text_result = processor.process(text_input).await.unwrap();
        assert!(!text_result.output.to_string().is_empty());
        assert!(text_result.confidence > 0.0);
        assert_eq!(text_result.modalities_used, vec!["text"]);
        
        // 测试图像处理
        let image_input = MultimodalData::Image(vec![1, 2, 3, 4, 5]);
        let image_result = processor.process(image_input).await.unwrap();
        assert!(!image_result.output.to_string().is_empty());
        assert!(image_result.confidence > 0.0);
        assert_eq!(image_result.modalities_used, vec!["image"]);
        
        // 测试多模态处理
        let mixed_input = MultimodalData::Mixed {
            text: Some("测试文本".to_string()),
            image: Some(vec![1, 2, 3, 4, 5]),
            audio: Some(vec![6, 7, 8, 9, 10]),
            video: None,
        };
        let mixed_result = processor.process(mixed_input).await.unwrap();
        assert!(!mixed_result.output.to_string().is_empty());
        assert!(mixed_result.confidence > 0.0);
        assert!(mixed_result.modalities_used.len() > 1);
        
        println!("文本处理结果: {}", text_result.output.to_string());
        println!("图像处理结果: {}", image_result.output.to_string());
        println!("多模态处理结果: {}", mixed_result.output.to_string());
    }

    #[test]
    async fn test_batch_processing() {
        let text_encoder = Arc::new(SimpleTextEncoder::new(128, "test-text".to_string()));
        let image_encoder = Arc::new(SimpleImageEncoder::new(256, "test-image".to_string()));
        let audio_encoder = Arc::new(SimpleAudioEncoder::new(192, "test-audio".to_string()));
        let fusion = Arc::new(AttentionFusion::new(512, FusionStrategy::Attention));
        
        let processor = MultimodalAIProcessor::new(
            text_encoder,
            image_encoder,
            audio_encoder,
            fusion,
        );
        
        let inputs = vec![
            MultimodalData::Text("第一个文本".to_string()),
            MultimodalData::Text("第二个文本".to_string()),
            MultimodalData::Image(vec![1, 2, 3]),
            MultimodalData::Audio(vec![4, 5, 6]),
        ];
        
        let results = processor.process_batch(&inputs).await.unwrap();
        assert_eq!(results.len(), 4);
        
        for result in &results {
            assert!(result.confidence > 0.0);
            assert!(!result.modalities_used.is_empty());
        }
    }

    #[test]
    async fn test_fusion_strategies() {
        let text_encoder = Arc::new(SimpleTextEncoder::new(128, "test-text".to_string()));
        let image_encoder = Arc::new(SimpleImageEncoder::new(256, "test-image".to_string()));
        let audio_encoder = Arc::new(SimpleAudioEncoder::new(192, "test-audio".to_string()));
        
        // 测试不同的融合策略
        let strategies = vec![
            FusionStrategy::Attention,
            FusionStrategy::Concatenation,
            FusionStrategy::WeightedAverage,
            FusionStrategy::CrossModal,
        ];
        
        for strategy in strategies {
            let fusion = Arc::new(AttentionFusion::new(512, strategy.clone()));
            let processor = MultimodalAIProcessor::new(
                text_encoder.clone(),
                image_encoder.clone(),
                audio_encoder.clone(),
                fusion,
            );
            
            let input = MultimodalData::Mixed {
                text: Some("测试文本".to_string()),
                image: Some(vec![1, 2, 3, 4, 5]),
                audio: Some(vec![6, 7, 8, 9, 10]),
                video: None,
            };
            
            let result = processor.process(input).await.unwrap();
            assert!(result.confidence > 0.0);
            assert_eq!(result.features.fused_features.len(), 512);
            
            println!("融合策略 {:?} 处理结果: 置信度 {:.3}", strategy, result.confidence);
        }
    }
}

/// 性能基准测试
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    async fn benchmark_multimodal_processing() {
        let text_encoder = Arc::new(SimpleTextEncoder::new(256, "benchmark-text".to_string()));
        let image_encoder = Arc::new(SimpleImageEncoder::new(512, "benchmark-image".to_string()));
        let audio_encoder = Arc::new(SimpleAudioEncoder::new(384, "benchmark-audio".to_string()));
        let fusion = Arc::new(AttentionFusion::new(1024, FusionStrategy::Attention));
        
        let processor = MultimodalAIProcessor::new(
            text_encoder,
            image_encoder,
            audio_encoder,
            fusion,
        );
        
        let input = MultimodalData::Mixed {
            text: Some("这是一个用于性能测试的多模态输入，包含文本、图像和音频数据。".to_string()),
            image: Some(vec![1; 1000]),
            audio: Some(vec![2; 2000]),
            video: None,
        };
        
        let start = Instant::now();
        let _result = processor.process(input).await.unwrap();
        let duration = start.elapsed();
        
        println!("多模态处理耗时: {:?}", duration);
    }

    #[test]
    async fn benchmark_batch_multimodal_processing() {
        let text_encoder = Arc::new(SimpleTextEncoder::new(128, "benchmark-text".to_string()));
        let image_encoder = Arc::new(SimpleImageEncoder::new(256, "benchmark-image".to_string()));
        let audio_encoder = Arc::new(SimpleAudioEncoder::new(192, "benchmark-audio".to_string()));
        let fusion = Arc::new(AttentionFusion::new(512, FusionStrategy::Attention));
        
        let processor = MultimodalAIProcessor::new(
            text_encoder,
            image_encoder,
            audio_encoder,
            fusion,
        );
        
        let inputs: Vec<MultimodalData> = (0..100).map(|i| {
            MultimodalData::Mixed {
                text: Some(format!("测试文本 {}", i)),
                image: Some(vec![i as u8; 100]),
                audio: Some(vec![i as u8; 200]),
                video: None,
            }
        }).collect();
        
        let start = Instant::now();
        let _results = processor.process_batch(&inputs).await.unwrap();
        let duration = start.elapsed();
        
        println!("批量多模态处理100个样本耗时: {:?}", duration);
        println!("平均每个样本: {:?}", duration / 100);
    }

    #[test]
    async fn benchmark_caching_performance() {
        let text_encoder = Arc::new(SimpleTextEncoder::new(128, "cache-test".to_string()));
        let image_encoder = Arc::new(SimpleImageEncoder::new(256, "cache-test".to_string()));
        let audio_encoder = Arc::new(SimpleAudioEncoder::new(192, "cache-test".to_string()));
        let fusion = Arc::new(AttentionFusion::new(512, FusionStrategy::Attention));
        
        let processor = MultimodalAIProcessor::new(
            text_encoder,
            image_encoder,
            audio_encoder,
            fusion,
        );
        
        let input = MultimodalData::Text("缓存测试文本".to_string());
        
        // 第一次处理（无缓存）
        let start1 = Instant::now();
        let _result1 = processor.process(input.clone()).await.unwrap();
        let duration1 = start1.elapsed();
        
        // 第二次处理（有缓存）
        let start2 = Instant::now();
        let _result2 = processor.process(input).await.unwrap();
        let duration2 = start2.elapsed();
        
        println!("第一次处理（无缓存）: {:?}", duration1);
        println!("第二次处理（有缓存）: {:?}", duration2);
        println!("缓存加速比: {:.2}x", duration1.as_secs_f64() / duration2.as_secs_f64());
    }
}

/// 辅助方法实现
impl MultimodalData {
    pub fn to_string(&self) -> String {
        match self {
            MultimodalData::Text(text) => text.clone(),
            MultimodalData::Image(_) => "图像数据".to_string(),
            MultimodalData::Audio(_) => "音频数据".to_string(),
            MultimodalData::Video(_) => "视频数据".to_string(),
            MultimodalData::Mixed { .. } => "多模态数据".to_string(),
        }
    }
}
