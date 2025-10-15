//! 增强的RAG (Retrieval-Augmented Generation) 系统实现
//! 
//! 本示例展示了一个功能完整的RAG系统，包括：
//! - 高级文档处理和分块
//! - 多种嵌入模型支持
//! - 智能检索和重排序
//! - 上下文压缩和优化
//! - 多轮对话支持
//! - 性能监控和缓存

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::time::{Instant, SystemTime};

/// 文档块结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub embedding: Option<Vec<f64>>,
    pub chunk_index: usize,
    pub parent_doc_id: String,
    pub overlap_size: usize,
}

/// 检索结果
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub chunk: DocumentChunk,
    pub similarity: f64,
    pub rank: usize,
    pub retrieval_time: std::time::Duration,
}

/// 查询结果
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub answer: String,
    pub sources: Vec<RetrievalResult>,
    pub confidence: f64,
    pub processing_time: std::time::Duration,
    pub token_count: usize,
    pub context_compression_ratio: f64,
}

/// 对话上下文
#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub history: Vec<ConversationTurn>,
    pub current_topic: Option<String>,
    pub user_preferences: HashMap<String, String>,
}

/// 对话轮次
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub user_query: String,
    pub system_response: String,
    pub timestamp: SystemTime,
    pub sources_used: Vec<String>,
}

/// 嵌入模型trait
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f64>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}

/// 语言模型trait
pub trait LanguageModel: Send + Sync {
    async fn generate(&self, context: &str, question: &str) -> Result<String>;
    async fn generate_with_params(&self, context: &str, question: &str, max_tokens: usize, temperature: f64) -> Result<String>;
    async fn generate_streaming(&self, context: &str, question: &str) -> Result<tokio::sync::mpsc::Receiver<String>>;
    fn model_name(&self) -> &str;
    fn max_context_length(&self) -> usize;
}

/// 重排序模型trait
pub trait RerankingModel: Send + Sync {
    async fn rerank(&self, query: &str, documents: &[DocumentChunk]) -> Result<Vec<(DocumentChunk, f64)>>;
    fn model_name(&self) -> &str;
}

/// 高级嵌入模型实现
pub struct AdvancedEmbeddingModel {
    pub dimension: usize,
    pub model_name: String,
    pub cache: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl AdvancedEmbeddingModel {
    pub fn new(dimension: usize, model_name: String) -> Self {
        Self {
            dimension,
            model_name,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 高级文本预处理
    fn preprocess_text(&self, text: &str) -> String {
        // 文本清理和标准化
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?;:".contains(*c))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }
    
    /// 改进的哈希嵌入（实际应用中应使用真正的嵌入模型）
    fn advanced_hash_embedding(&self, text: &str) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let processed_text = self.preprocess_text(text);
        let mut hasher = DefaultHasher::new();
        processed_text.hash(&mut hasher);
        let base_hash = hasher.finish();
        
        let mut embedding = vec![0.0; self.dimension];
        
        // 使用多个哈希函数生成更丰富的嵌入
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

impl EmbeddingModel for AdvancedEmbeddingModel {
    async fn embed(&self, text: &str) -> Result<Vec<f64>> {
        // 检查缓存
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }
        
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
        
        let embedding = self.advanced_hash_embedding(text);
        
        // 存储到缓存
        {
            let mut cache = self.cache.write().await;
            cache.insert(text.to_string(), embedding.clone());
        }
        
        Ok(embedding)
    }
    
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        let mut embeddings = Vec::new();
        
        // 批量处理，利用并发
        let mut handles = Vec::new();
        for text in texts {
            let model = self;
            let text = text.clone();
            let handle = tokio::spawn(async move {
                model.embed(&text).await
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

/// 高级语言模型实现
pub struct AdvancedLanguageModel {
    pub model_name: String,
    pub max_tokens: usize,
    pub max_context_length: usize,
    pub cache: Arc<RwLock<HashMap<String, String>>>,
}

impl AdvancedLanguageModel {
    pub fn new(model_name: String, max_tokens: usize, max_context_length: usize) -> Self {
        Self {
            model_name,
            max_tokens,
            max_context_length,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 上下文压缩
    fn compress_context(&self, context: &str, max_length: usize) -> String {
        if context.len() <= max_length {
            return context.to_string();
        }
        
        // 简单的上下文压缩策略
        let sentences: Vec<&str> = context.split('.').collect();
        let mut compressed = String::new();
        let mut current_length = 0;
        
        for sentence in sentences {
            if current_length + sentence.len() + 1 > max_length {
                break;
            }
            compressed.push_str(sentence);
            compressed.push('.');
            current_length += sentence.len() + 1;
        }
        
        compressed
    }
    
    /// 改进的模板生成
    fn advanced_generate(&self, context: &str, question: &str) -> String {
        // 压缩上下文
        let compressed_context = self.compress_context(context, self.max_context_length / 2);
        
        // 分析问题类型
        let question_type = self.analyze_question_type(question);
        
        match question_type {
            QuestionType::Factual => {
                format!(
                    "根据提供的文档信息：\n\n{}\n\n问题：{}\n\n答案：{}",
                    compressed_context,
                    question,
                    self.generate_factual_answer(&compressed_context, question)
                )
            },
            QuestionType::Analytical => {
                format!(
                    "基于以下信息进行分析：\n\n{}\n\n问题：{}\n\n分析：{}",
                    compressed_context,
                    question,
                    self.generate_analytical_answer(&compressed_context, question)
                )
            },
            QuestionType::Comparative => {
                format!(
                    "比较分析：\n\n{}\n\n问题：{}\n\n比较结果：{}",
                    compressed_context,
                    question,
                    self.generate_comparative_answer(&compressed_context, question)
                )
            },
        }
    }
    
    /// 分析问题类型
    fn analyze_question_type(&self, question: &str) -> QuestionType {
        let question_lower = question.to_lowercase();
        
        if question_lower.contains("什么是") || question_lower.contains("定义") {
            QuestionType::Factual
        } else if question_lower.contains("分析") || question_lower.contains("为什么") {
            QuestionType::Analytical
        } else if question_lower.contains("比较") || question_lower.contains("区别") {
            QuestionType::Comparative
        } else {
            QuestionType::Factual
        }
    }
    
    /// 生成事实性答案
    fn generate_factual_answer(&self, context: &str, question: &str) -> String {
        if context.contains("人工智能") {
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        } else if context.contains("机器学习") {
            "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。"
        } else if context.contains("深度学习") {
            "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。"
        } else {
            "根据提供的上下文信息，相关内容在文档中有所涉及。"
        }.to_string()
    }
    
    /// 生成分析性答案
    fn generate_analytical_answer(&self, context: &str, question: &str) -> String {
        "基于提供的信息，可以进行深入分析。相关技术在不同领域都有重要应用，具有广阔的发展前景。".to_string()
    }
    
    /// 生成比较性答案
    fn generate_comparative_answer(&self, context: &str, question: &str) -> String {
        "通过比较分析，可以发现不同方法各有优势，选择合适的技术方案需要根据具体应用场景来决定。".to_string()
    }
}

#[derive(Debug, Clone)]
enum QuestionType {
    Factual,
    Analytical,
    Comparative,
}

impl LanguageModel for AdvancedLanguageModel {
    async fn generate(&self, context: &str, question: &str) -> Result<String> {
        self.generate_with_params(context, question, self.max_tokens, 0.7).await
    }
    
    async fn generate_with_params(&self, context: &str, question: &str, _max_tokens: usize, _temperature: f64) -> Result<String> {
        // 检查缓存
        let cache_key = format!("{}|||{}", context, question);
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
        
        let response = self.advanced_generate(context, question);
        
        // 存储到缓存
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, response.clone());
        }
        
        Ok(response)
    }
    
    async fn generate_streaming(&self, context: &str, question: &str) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        let response = self.generate(context, question).await?;
        let words: Vec<&str> = response.split_whitespace().collect();
        
        tokio::spawn(async move {
            for word in words {
                if tx.send(word.to_string()).await.is_err() {
                    break;
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
        });
        
        Ok(rx)
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
    
    fn max_context_length(&self) -> usize {
        self.max_context_length
    }
}

/// 重排序模型实现
pub struct SimpleRerankingModel {
    pub model_name: String,
}

impl SimpleRerankingModel {
    pub fn new(model_name: String) -> Self {
        Self { model_name }
    }
    
    /// 计算查询与文档的相关性分数
    fn calculate_relevance_score(&self, query: &str, document: &DocumentChunk) -> f64 {
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let doc_words: Vec<&str> = document.content.split_whitespace().collect();
        
        let mut score = 0.0;
        for query_word in &query_words {
            for doc_word in &doc_words {
                if query_word.to_lowercase() == doc_word.to_lowercase() {
                    score += 1.0;
                }
            }
        }
        
        // 归一化
        score / (query_words.len() * doc_words.len()) as f64
    }
}

impl RerankingModel for SimpleRerankingModel {
    async fn rerank(&self, query: &str, documents: &[DocumentChunk]) -> Result<Vec<(DocumentChunk, f64)>> {
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        let mut scored_docs = Vec::new();
        for doc in documents {
            let score = self.calculate_relevance_score(query, doc);
            scored_docs.push((doc.clone(), score));
        }
        
        // 按分数排序
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(scored_docs)
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// 增强的RAG系统
pub struct EnhancedRAGSystem {
    pub vector_store: Arc<RwLock<HashMap<String, DocumentChunk>>>,
    pub embedding_model: Arc<dyn EmbeddingModel>,
    pub language_model: Arc<dyn LanguageModel>,
    pub reranking_model: Arc<dyn RerankingModel>,
    pub conversation_context: Arc<RwLock<ConversationContext>>,
    pub top_k: usize,
    pub similarity_threshold: f64,
    pub rerank_top_k: usize,
    pub cache: Arc<RwLock<HashMap<String, QueryResult>>>,
}

impl EnhancedRAGSystem {
    pub fn new(
        embedding_model: Arc<dyn EmbeddingModel>,
        language_model: Arc<dyn LanguageModel>,
        reranking_model: Arc<dyn RerankingModel>,
        top_k: usize,
        similarity_threshold: f64,
        rerank_top_k: usize,
    ) -> Self {
        Self {
            vector_store: Arc::new(RwLock::new(HashMap::new())),
            embedding_model,
            language_model,
            reranking_model,
            conversation_context: Arc::new(RwLock::new(ConversationContext {
                history: Vec::new(),
                current_topic: None,
                user_preferences: HashMap::new(),
            })),
            top_k,
            similarity_threshold,
            rerank_top_k,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 智能文档分块
    pub async fn chunk_document(&self, doc_id: &str, content: &str, chunk_size: usize, overlap: usize) -> Vec<DocumentChunk> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        
        let mut start = 0;
        while start < words.len() {
            let end = (start + chunk_size).min(words.len());
            let chunk_words = &words[start..end];
            let chunk_content = chunk_words.join(" ");
            
            let chunk = DocumentChunk {
                id: format!("{}_{}", doc_id, chunk_index),
                content: chunk_content,
                metadata: HashMap::new(),
                embedding: None,
                chunk_index,
                parent_doc_id: doc_id.to_string(),
                overlap_size: overlap,
            };
            
            chunks.push(chunk);
            chunk_index += 1;
            start = end.saturating_sub(overlap);
        }
        
        chunks
    }
    
    /// 添加文档（自动分块）
    pub async fn add_document(&self, doc_id: &str, content: &str, metadata: HashMap<String, String>) -> Result<()> {
        let chunks = self.chunk_document(doc_id, content, 200, 50).await;
        
        let mut store = self.vector_store.write().await;
        
        // 批量生成嵌入
        let contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.embedding_model.embed_batch(&contents).await?;
        
        for (mut chunk, embedding) in chunks.into_iter().zip(embeddings) {
            chunk.embedding = Some(embedding);
            chunk.metadata = metadata.clone();
            store.insert(chunk.id.clone(), chunk);
        }
        
        Ok(())
    }
    
    /// 高级检索（包含重排序）
    pub async fn retrieve_documents(&self, query: &str) -> Result<Vec<RetrievalResult>> {
        let start_time = Instant::now();
        
        // 生成查询嵌入
        let query_embedding = self.embedding_model.embed(query).await?;
        
        let store = self.vector_store.read().await;
        let mut candidates = Vec::new();
        
        // 初始检索
        for (id, chunk) in store.iter() {
            if let Some(chunk_embedding) = &chunk.embedding {
                let similarity = Self::cosine_similarity(&query_embedding, chunk_embedding);
                
                if similarity >= self.similarity_threshold {
                    candidates.push(RetrievalResult {
                        chunk: chunk.clone(),
                        similarity,
                        rank: 0,
                        retrieval_time: start_time.elapsed(),
                    });
                }
            }
        }
        
        // 按相似度排序
        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        
        // 取前top_k个进行重排序
        let top_candidates: Vec<DocumentChunk> = candidates.iter()
            .take(self.top_k)
            .map(|r| r.chunk.clone())
            .collect();
        
        // 重排序
        let reranked = self.reranking_model.rerank(query, &top_candidates).await?;
        
        // 构建最终结果
        let mut final_results = Vec::new();
        for (i, (chunk, rerank_score)) in reranked.iter().take(self.rerank_top_k).enumerate() {
            final_results.push(RetrievalResult {
                chunk: chunk.clone(),
                similarity: rerank_score * 0.7 + candidates.iter().find(|r| r.chunk.id == chunk.id).map(|r| r.similarity).unwrap_or(0.0) * 0.3,
                rank: i + 1,
                retrieval_time: start_time.elapsed(),
            });
        }
        
        Ok(final_results)
    }
    
    /// 上下文压缩
    fn compress_context(&self, results: &[RetrievalResult], max_length: usize) -> String {
        let mut context_parts = Vec::new();
        let mut current_length = 0;
        
        for result in results {
            let part = format!(
                "文档 {} (相似度: {:.3}):\n{}\n",
                result.rank,
                result.similarity,
                result.chunk.content
            );
            
            if current_length + part.len() > max_length {
                break;
            }
            
            context_parts.push(part);
            current_length += part.len();
        }
        
        context_parts.join("\n---\n\n")
    }
    
    /// 执行增强查询
    pub async fn query(&self, question: &str) -> Result<QueryResult> {
        let start_time = Instant::now();
        
        // 检查缓存
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(question) {
                return Ok(cached.clone());
            }
        }
        
        // 1. 检索相关文档
        let retrieval_results = self.retrieve_documents(question).await?;
        
        if retrieval_results.is_empty() {
            return Err(anyhow!("未找到相关文档"));
        }
        
        // 2. 压缩上下文
        let max_context_length = self.language_model.max_context_length();
        let context = self.compress_context(&retrieval_results, max_context_length);
        let compression_ratio = context.len() as f64 / retrieval_results.iter().map(|r| r.chunk.content.len()).sum::<usize>() as f64;
        
        // 3. 生成答案
        let answer = self.language_model.generate(&context, question).await?;
        
        // 4. 计算置信度
        let confidence = retrieval_results.iter()
            .map(|r| r.similarity)
            .sum::<f64>() / retrieval_results.len() as f64;
        
        // 5. 计算token数量（简单估算）
        let token_count = answer.split_whitespace().count();
        
        let processing_time = start_time.elapsed();
        
        let result = QueryResult {
            answer,
            sources: retrieval_results,
            confidence,
            processing_time,
            token_count,
            context_compression_ratio: compression_ratio,
        };
        
        // 存储到缓存
        {
            let mut cache = self.cache.write().await;
            cache.insert(question.to_string(), result.clone());
        }
        
        // 更新对话上下文
        {
            let mut context = self.conversation_context.write().await;
            context.history.push(ConversationTurn {
                user_query: question.to_string(),
                system_response: result.answer.clone(),
                timestamp: SystemTime::now(),
                sources_used: result.sources.iter().map(|s| s.chunk.id.clone()).collect(),
            });
        }
        
        Ok(result)
    }
    
    /// 多轮对话查询
    pub async fn query_with_context(&self, question: &str) -> Result<QueryResult> {
        let context = {
            let conv_context = self.conversation_context.read().await;
            if conv_context.history.is_empty() {
                return self.query(question).await;
            }
            
            // 构建对话历史上下文
            let history_context = conv_context.history.iter()
                .rev()
                .take(3) // 最近3轮对话
                .map(|turn| format!("Q: {}\nA: {}", turn.user_query, turn.system_response))
                .collect::<Vec<_>>()
                .join("\n\n");
            
            format!("对话历史：\n{}\n\n当前问题：{}", history_context, question)
        };
        
        // 使用增强的查询
        self.query(&context).await
    }
    
    /// 流式生成
    pub async fn query_streaming(&self, question: &str) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let retrieval_results = self.retrieve_documents(question).await?;
        let context = self.compress_context(&retrieval_results, self.language_model.max_context_length());
        
        self.language_model.generate_streaming(&context, question).await
    }
    
    /// 计算余弦相似度
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// 获取系统统计信息
    pub async fn get_stats(&self) -> HashMap<String, usize> {
        let store = self.vector_store.read().await;
        let cache = self.cache.read().await;
        let context = self.conversation_context.read().await;
        
        let mut stats = HashMap::new();
        stats.insert("total_chunks".to_string(), store.len());
        stats.insert("embedded_chunks".to_string(), 
                    store.values().filter(|c| c.embedding.is_some()).count());
        stats.insert("cached_queries".to_string(), cache.len());
        stats.insert("conversation_turns".to_string(), context.history.len());
        
        stats
    }
    
    /// 清理缓存
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
    
    /// 清理对话历史
    pub async fn clear_conversation_history(&self) {
        let mut context = self.conversation_context.write().await;
        context.history.clear();
        context.current_topic = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_enhanced_rag_system() {
        // 创建模型
        let embedding_model = Arc::new(AdvancedEmbeddingModel::new(128, "test-embedding".to_string()));
        let language_model = Arc::new(AdvancedLanguageModel::new("test-llm".to_string(), 1000, 4000));
        let reranking_model = Arc::new(SimpleRerankingModel::new("test-reranker".to_string()));
        
        // 创建增强RAG系统
        let rag = EnhancedRAGSystem::new(
            embedding_model,
            language_model,
            reranking_model,
            5, // top_k
            0.1, // similarity_threshold
            3, // rerank_top_k
        );
        
        // 添加测试文档
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "AI".to_string());
        metadata.insert("source".to_string(), "test".to_string());
        
        rag.add_document(
            "doc1",
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。人工智能包括机器学习、深度学习、自然语言处理等多个子领域。",
            metadata
        ).await.unwrap();
        
        // 执行查询
        let result = rag.query("什么是人工智能？").await.unwrap();
        
        // 验证结果
        assert!(!result.answer.is_empty());
        assert!(!result.sources.is_empty());
        assert!(result.confidence > 0.0);
        assert!(result.processing_time.as_millis() > 0);
        assert!(result.token_count > 0);
        assert!(result.context_compression_ratio > 0.0 && result.context_compression_ratio <= 1.0);
        
        println!("查询结果: {}", result.answer);
        println!("置信度: {:.3}", result.confidence);
        println!("处理时间: {:?}", result.processing_time);
        println!("Token数量: {}", result.token_count);
        println!("上下文压缩比: {:.3}", result.context_compression_ratio);
    }

    #[test]
    async fn test_conversation_context() {
        let embedding_model = Arc::new(AdvancedEmbeddingModel::new(128, "test-embedding".to_string()));
        let language_model = Arc::new(AdvancedLanguageModel::new("test-llm".to_string(), 1000, 4000));
        let reranking_model = Arc::new(SimpleRerankingModel::new("test-reranker".to_string()));
        
        let rag = EnhancedRAGSystem::new(
            embedding_model,
            language_model,
            reranking_model,
            5, 0.1, 3
        );
        
        // 添加文档
        rag.add_document(
            "doc1",
            "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
            HashMap::new()
        ).await.unwrap();
        
        // 第一轮对话
        let result1 = rag.query("什么是机器学习？").await.unwrap();
        assert!(!result1.answer.is_empty());
        
        // 第二轮对话（带上下文）
        let result2 = rag.query_with_context("它有哪些应用？").await.unwrap();
        assert!(!result2.answer.is_empty());
        
        // 检查对话历史
        let stats = rag.get_stats().await;
        assert_eq!(stats["conversation_turns"], 2);
    }

    #[test]
    async fn test_streaming_generation() {
        let embedding_model = Arc::new(AdvancedEmbeddingModel::new(128, "test-embedding".to_string()));
        let language_model = Arc::new(AdvancedLanguageModel::new("test-llm".to_string(), 1000, 4000));
        let reranking_model = Arc::new(SimpleRerankingModel::new("test-reranker".to_string()));
        
        let rag = EnhancedRAGSystem::new(
            embedding_model,
            language_model,
            reranking_model,
            5, 0.1, 3
        );
        
        rag.add_document(
            "doc1",
            "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。",
            HashMap::new()
        ).await.unwrap();
        
        let mut stream = rag.query_streaming("什么是深度学习？").await.unwrap();
        let mut response = String::new();
        
        while let Some(word) = stream.recv().await {
            response.push_str(&word);
            response.push(' ');
        }
        
        assert!(!response.is_empty());
        println!("流式响应: {}", response);
    }

    #[test]
    async fn test_document_chunking() {
        let embedding_model = Arc::new(AdvancedEmbeddingModel::new(128, "test-embedding".to_string()));
        let language_model = Arc::new(AdvancedLanguageModel::new("test-llm".to_string(), 1000, 4000));
        let reranking_model = Arc::new(SimpleRerankingModel::new("test-reranker".to_string()));
        
        let rag = EnhancedRAGSystem::new(
            embedding_model,
            language_model,
            reranking_model,
            5, 0.1, 3
        );
        
        let content = "这是一个很长的文档内容，需要被分成多个块进行处理。每个块都应该包含足够的信息来回答相关问题，同时保持合理的长度。分块策略对于RAG系统的性能至关重要。";
        
        let chunks = rag.chunk_document("test_doc", content, 20, 5).await;
        
        assert!(!chunks.is_empty());
        assert!(chunks.len() > 1); // 应该被分成多个块
        
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
            assert_eq!(chunk.parent_doc_id, "test_doc");
            assert!(!chunk.content.is_empty());
        }
    }
}

/// 性能基准测试
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    async fn benchmark_enhanced_rag_query() {
        let embedding_model = Arc::new(AdvancedEmbeddingModel::new(256, "benchmark-embedding".to_string()));
        let language_model = Arc::new(AdvancedLanguageModel::new("benchmark-llm".to_string(), 2000, 8000));
        let reranking_model = Arc::new(SimpleRerankingModel::new("benchmark-reranker".to_string()));
        
        let rag = EnhancedRAGSystem::new(
            embedding_model,
            language_model,
            reranking_model,
            10, 0.1, 5
        );
        
        // 添加大量文档
        for i in 0..100 {
            let content = format!(
                "这是第{}个文档，包含关于人工智能、机器学习、深度学习、自然语言处理、计算机视觉等技术的详细内容。这些技术在现代AI系统中发挥着重要作用。",
                i
            );
            rag.add_document(&format!("doc_{}", i), &content, HashMap::new()).await.unwrap();
        }
        
        let start = Instant::now();
        let result = rag.query("什么是人工智能？").await.unwrap();
        let duration = start.elapsed();
        
        println!("增强RAG查询耗时: {:?}", duration);
        println!("检索到 {} 个相关文档", result.sources.len());
        println!("置信度: {:.3}", result.confidence);
        println!("上下文压缩比: {:.3}", result.context_compression_ratio);
    }

    #[test]
    async fn benchmark_batch_embedding() {
        let model = AdvancedEmbeddingModel::new(512, "benchmark-embedding".to_string());
        let texts: Vec<String> = (0..1000).map(|i| format!("text {}", i)).collect();
        
        let start = Instant::now();
        let _embeddings = model.embed_batch(&texts).await.unwrap();
        let duration = start.elapsed();
        
        println!("批量嵌入1000个文本耗时: {:?}", duration);
        println!("平均每个文本: {:?}", duration / 1000);
    }

    #[test]
    async fn benchmark_caching_performance() {
        let embedding_model = Arc::new(AdvancedEmbeddingModel::new(128, "cache-test".to_string()));
        let language_model = Arc::new(AdvancedLanguageModel::new("cache-test".to_string(), 1000, 4000));
        let reranking_model = Arc::new(SimpleRerankingModel::new("cache-test".to_string()));
        
        let rag = EnhancedRAGSystem::new(
            embedding_model,
            language_model,
            reranking_model,
            5, 0.1, 3
        );
        
        rag.add_document(
            "doc1",
            "缓存测试文档内容，用于验证缓存机制的性能提升效果。",
            HashMap::new()
        ).await.unwrap();
        
        let query = "什么是缓存？";
        
        // 第一次查询（无缓存）
        let start1 = Instant::now();
        let _result1 = rag.query(query).await.unwrap();
        let duration1 = start1.elapsed();
        
        // 第二次查询（有缓存）
        let start2 = Instant::now();
        let _result2 = rag.query(query).await.unwrap();
        let duration2 = start2.elapsed();
        
        println!("第一次查询（无缓存）: {:?}", duration1);
        println!("第二次查询（有缓存）: {:?}", duration2);
        println!("缓存加速比: {:.2}x", duration1.as_secs_f64() / duration2.as_secs_f64());
    }
}
