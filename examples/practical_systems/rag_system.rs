//! 实用的RAG (Retrieval-Augmented Generation) 系统实现
//! 
//! 本示例展示了一个完整的RAG系统，包括：
//! - 文档嵌入和存储
//! - 语义检索
//! - 上下文构建
//! - 答案生成
//! - 性能监控

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};

/// 文档结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub embedding: Option<Vec<f64>>,
}

/// 检索结果
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub document: Document,
    pub similarity: f64,
    pub rank: usize,
}

/// 查询结果
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub answer: String,
    pub sources: Vec<RetrievalResult>,
    pub confidence: f64,
    pub processing_time: std::time::Duration,
}

/// 嵌入模型trait
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f64>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>>;
}

/// 语言模型trait
pub trait LanguageModel: Send + Sync {
    async fn generate(&self, context: &str, question: &str) -> Result<String>;
    async fn generate_with_params(&self, context: &str, question: &str, max_tokens: usize, temperature: f64) -> Result<String>;
}

/// 简单的嵌入模型实现（用于演示）
pub struct SimpleEmbeddingModel {
    pub dimension: usize,
}

impl SimpleEmbeddingModel {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
    
    /// 简单的文本哈希嵌入（实际应用中应使用真正的嵌入模型）
    fn simple_hash_embedding(&self, text: &str) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        let mut embedding = vec![0.0; self.dimension];
        for i in 0..self.dimension {
            let seed = hash.wrapping_add(i as u64);
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            embedding[i] = (hasher.finish() as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
        
        // 归一化
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        
        embedding
    }
}

impl EmbeddingModel for SimpleEmbeddingModel {
    async fn embed(&self, text: &str) -> Result<Vec<f64>> {
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        Ok(self.simple_hash_embedding(text))
    }
    
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }
}

/// 简单的语言模型实现（用于演示）
pub struct SimpleLanguageModel {
    pub max_tokens: usize,
}

impl SimpleLanguageModel {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }
    
    /// 简单的模板生成（实际应用中应使用真正的LLM）
    fn simple_generate(&self, context: &str, question: &str) -> String {
        format!(
            "基于提供的上下文信息：\n\n{}\n\n问题：{}\n\n答案：根据上下文，{}。",
            context,
            question,
            if context.contains("人工智能") {
                "这是一个关于人工智能的问题，相关内容在上下文中有所涉及"
            } else if context.contains("机器学习") {
                "这是一个关于机器学习的问题，相关内容在上下文中有所涉及"
            } else {
                "相关信息在上下文中可以找到"
            }
        )
    }
}

impl LanguageModel for SimpleLanguageModel {
    async fn generate(&self, context: &str, question: &str) -> Result<String> {
        self.generate_with_params(context, question, self.max_tokens, 0.7).await
    }
    
    async fn generate_with_params(&self, context: &str, question: &str, _max_tokens: usize, _temperature: f64) -> Result<String> {
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        Ok(self.simple_generate(context, question))
    }
}

/// RAG系统主结构
pub struct RAGSystem {
    pub vector_store: Arc<RwLock<HashMap<String, Document>>>,
    pub embedding_model: Arc<dyn EmbeddingModel>,
    pub language_model: Arc<dyn LanguageModel>,
    pub top_k: usize,
    pub similarity_threshold: f64,
}

impl RAGSystem {
    pub fn new(
        embedding_model: Arc<dyn EmbeddingModel>,
        language_model: Arc<dyn LanguageModel>,
        top_k: usize,
        similarity_threshold: f64,
    ) -> Self {
        Self {
            vector_store: Arc::new(RwLock::new(HashMap::new())),
            embedding_model,
            language_model,
            top_k,
            similarity_threshold,
        }
    }
    
    /// 添加文档到向量存储
    pub async fn add_document(&self, document: Document) -> Result<()> {
        let mut store = self.vector_store.write().await;
        
        // 生成嵌入
        let embedding = self.embedding_model.embed(&document.content).await?;
        let mut doc = document;
        doc.embedding = Some(embedding);
        
        store.insert(doc.id.clone(), doc);
        Ok(())
    }
    
    /// 批量添加文档
    pub async fn add_documents(&self, documents: Vec<Document>) -> Result<()> {
        let mut store = self.vector_store.write().await;
        
        // 批量生成嵌入
        let contents: Vec<String> = documents.iter().map(|d| d.content.clone()).collect();
        let embeddings = self.embedding_model.embed_batch(&contents).await?;
        
        for (mut doc, embedding) in documents.into_iter().zip(embeddings) {
            doc.embedding = Some(embedding);
            store.insert(doc.id.clone(), doc);
        }
        
        Ok(())
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
    
    /// 检索相关文档
    pub async fn retrieve_documents(&self, query: &str) -> Result<Vec<RetrievalResult>> {
        let start_time = std::time::Instant::now();
        
        // 生成查询嵌入
        let query_embedding = self.embedding_model.embed(query).await?;
        
        let store = self.vector_store.read().await;
        let mut results = Vec::new();
        
        // 计算相似度
        for (id, doc) in store.iter() {
            if let Some(doc_embedding) = &doc.embedding {
                let similarity = Self::cosine_similarity(&query_embedding, doc_embedding);
                
                if similarity >= self.similarity_threshold {
                    results.push(RetrievalResult {
                        document: doc.clone(),
                        similarity,
                        rank: 0, // 将在排序后设置
                    });
                }
            }
        }
        
        // 按相似度排序
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        
        // 设置排名并限制数量
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i + 1;
        }
        
        let final_results = results.into_iter().take(self.top_k).collect();
        
        let processing_time = start_time.elapsed();
        println!("检索耗时: {:?}", processing_time);
        
        Ok(final_results)
    }
    
    /// 构建上下文
    fn build_context(&self, results: &[RetrievalResult]) -> String {
        let mut context_parts = Vec::new();
        
        for (i, result) in results.iter().enumerate() {
            context_parts.push(format!(
                "文档 {} (相似度: {:.3}):\n{}\n",
                i + 1,
                result.similarity,
                result.document.content
            ));
        }
        
        context_parts.join("\n---\n\n")
    }
    
    /// 执行完整查询
    pub async fn query(&self, question: &str) -> Result<QueryResult> {
        let start_time = std::time::Instant::now();
        
        // 1. 检索相关文档
        let retrieval_results = self.retrieve_documents(question).await?;
        
        if retrieval_results.is_empty() {
            return Err(anyhow!("未找到相关文档"));
        }
        
        // 2. 构建上下文
        let context = self.build_context(&retrieval_results);
        
        // 3. 生成答案
        let answer = self.language_model.generate(&context, question).await?;
        
        // 4. 计算置信度（基于检索结果的平均相似度）
        let confidence = retrieval_results.iter()
            .map(|r| r.similarity)
            .sum::<f64>() / retrieval_results.len() as f64;
        
        let processing_time = start_time.elapsed();
        
        Ok(QueryResult {
            answer,
            sources: retrieval_results,
            confidence,
            processing_time,
        })
    }
    
    /// 获取系统统计信息
    pub async fn get_stats(&self) -> HashMap<String, usize> {
        let store = self.vector_store.read().await;
        let mut stats = HashMap::new();
        
        stats.insert("total_documents".to_string(), store.len());
        stats.insert("embedded_documents".to_string(), 
                    store.values().filter(|d| d.embedding.is_some()).count());
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_rag_system() {
        // 创建模型
        let embedding_model = Arc::new(SimpleEmbeddingModel::new(128));
        let language_model = Arc::new(SimpleLanguageModel::new(1000));
        
        // 创建RAG系统
        let rag = RAGSystem::new(embedding_model, language_model, 3, 0.1);
        
        // 添加测试文档
        let documents = vec![
            Document {
                id: "doc1".to_string(),
                content: "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。".to_string(),
                metadata: HashMap::new(),
                embedding: None,
            },
            Document {
                id: "doc2".to_string(),
                content: "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。".to_string(),
                metadata: HashMap::new(),
                embedding: None,
            },
            Document {
                id: "doc3".to_string(),
                content: "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。".to_string(),
                metadata: HashMap::new(),
                embedding: None,
            },
        ];
        
        // 添加文档
        rag.add_documents(documents).await.unwrap();
        
        // 执行查询
        let result = rag.query("什么是人工智能？").await.unwrap();
        
        // 验证结果
        assert!(!result.answer.is_empty());
        assert!(!result.sources.is_empty());
        assert!(result.confidence > 0.0);
        assert!(result.processing_time.as_millis() > 0);
        
        println!("查询结果: {}", result.answer);
        println!("置信度: {:.3}", result.confidence);
        println!("处理时间: {:?}", result.processing_time);
        println!("检索到 {} 个相关文档", result.sources.len());
    }

    #[test]
    async fn test_embedding_model() {
        let model = SimpleEmbeddingModel::new(64);
        
        let embedding1 = model.embed("hello world").await.unwrap();
        let embedding2 = model.embed("hello world").await.unwrap();
        let embedding3 = model.embed("different text").await.unwrap();
        
        assert_eq!(embedding1.len(), 64);
        assert_eq!(embedding1, embedding2); // 相同文本应该产生相同嵌入
        assert_ne!(embedding1, embedding3); // 不同文本应该产生不同嵌入
        
        // 测试归一化
        let norm: f64 = embedding1.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    async fn test_language_model() {
        let model = SimpleLanguageModel::new(1000);
        
        let context = "人工智能是计算机科学的一个分支。";
        let question = "什么是人工智能？";
        
        let answer = model.generate(context, question).await.unwrap();
        
        assert!(!answer.is_empty());
        assert!(answer.contains("人工智能"));
        assert!(answer.contains(question));
    }

    #[test]
    async fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        
        assert!((RAGSystem::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(RAGSystem::cosine_similarity(&a, &c).abs() < 1e-6);
    }
}

/// 性能基准测试
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    async fn benchmark_embedding_generation() {
        let model = SimpleEmbeddingModel::new(512);
        let texts: Vec<String> = (0..100).map(|i| format!("text {}", i)).collect();
        
        let start = Instant::now();
        let _embeddings = model.embed_batch(&texts).await.unwrap();
        let duration = start.elapsed();
        
        println!("生成100个嵌入耗时: {:?}", duration);
        println!("平均每个嵌入: {:?}", duration / 100);
    }

    #[test]
    async fn benchmark_rag_query() {
        let embedding_model = Arc::new(SimpleEmbeddingModel::new(128));
        let language_model = Arc::new(SimpleLanguageModel::new(1000));
        let rag = RAGSystem::new(embedding_model, language_model, 5, 0.1);
        
        // 添加大量文档
        let documents: Vec<Document> = (0..1000).map(|i| Document {
            id: format!("doc_{}", i),
            content: format!("这是第{}个文档，包含一些关于人工智能和机器学习的内容。", i),
            metadata: HashMap::new(),
            embedding: None,
        }).collect();
        
        rag.add_documents(documents).await.unwrap();
        
        let start = Instant::now();
        let _result = rag.query("什么是机器学习？").await.unwrap();
        let duration = start.elapsed();
        
        println!("RAG查询耗时: {:?}", duration);
    }
}
