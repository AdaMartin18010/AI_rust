//! 实用AI系统综合演示程序
//! 
//! 本程序展示了Week 2完成的所有实用AI系统，包括：
//! - 增强的RAG系统
//! - 多模态处理系统
//! - Agent系统框架
//! - 性能监控和评估

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

mod enhanced_rag_system;
mod multimodal_processing;
mod agent_system_framework;

use enhanced_rag_system::*;
use multimodal_processing::*;
use agent_system_framework::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 实用AI系统综合演示");
    println!("========================");
    
    // 1. 演示增强RAG系统
    println!("\n📚 1. 增强RAG系统演示");
    await demonstrate_enhanced_rag().await?;
    
    // 2. 演示多模态处理系统
    println!("\n🎭 2. 多模态处理系统演示");
    await demonstrate_multimodal_processing().await?;
    
    // 3. 演示Agent系统框架
    println!("\n🤖 3. Agent系统框架演示");
    await demonstrate_agent_system().await?;
    
    // 4. 综合性能测试
    println!("\n⚡ 4. 综合性能测试");
    await demonstrate_performance_testing().await?;
    
    // 5. 系统集成演示
    println!("\n🔗 5. 系统集成演示");
    await demonstrate_system_integration().await?;
    
    println!("\n✅ 所有演示完成！");
    println!("\n📈 Week 2成果总结：");
    println!("   • 增强RAG系统：智能检索、重排序、上下文压缩");
    println!("   • 多模态处理：文本、图像、音频统一处理");
    println!("   • Agent系统：感知-推理-规划-执行循环");
    println!("   • 性能优化：缓存、并发、批量处理");
    println!("   • 系统集成：完整的AI应用框架");
    
    Ok(())
}

/// 演示增强RAG系统
async fn demonstrate_enhanced_rag() -> Result<(), Box<dyn std::error::Error>> {
    println!("创建增强RAG系统...");
    
    // 创建模型
    let embedding_model = Arc::new(AdvancedEmbeddingModel::new(256, "enhanced-embedding".to_string()));
    let language_model = Arc::new(AdvancedLanguageModel::new("enhanced-llm".to_string(), 2000, 8000));
    let reranking_model = Arc::new(SimpleRerankingModel::new("enhanced-reranker".to_string()));
    
    // 创建增强RAG系统
    let rag = EnhancedRAGSystem::new(
        embedding_model,
        language_model,
        reranking_model,
        10, // top_k
        0.1, // similarity_threshold
        5, // rerank_top_k
    );
    
    // 添加文档
    let documents = vec![
        ("ai_intro", "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。人工智能包括机器学习、深度学习、自然语言处理、计算机视觉等多个子领域。"),
        ("ml_basics", "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。机器学习算法通过训练数据来学习模式，然后对新数据进行预测或决策。"),
        ("dl_fundamentals", "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。"),
        ("nlp_overview", "自然语言处理是人工智能的一个重要分支，专注于让计算机理解、解释和生成人类语言。NLP技术包括文本分析、机器翻译、情感分析、问答系统等。"),
        ("cv_applications", "计算机视觉是人工智能的一个分支，致力于让计算机能够理解和解释视觉信息。计算机视觉技术广泛应用于图像识别、物体检测、人脸识别、自动驾驶等领域。"),
    ];
    
    for (doc_id, content) in documents {
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "AI".to_string());
        metadata.insert("source".to_string(), "demo".to_string());
        
        rag.add_document(doc_id, content, metadata).await?;
    }
    
    println!("✅ 文档添加完成");
    
    // 执行查询
    let queries = vec![
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "自然语言处理有哪些应用？",
        "计算机视觉在哪些领域有应用？",
    ];
    
    for query in queries {
        println!("\n🔍 查询: {}", query);
        let result = rag.query(query).await?;
        
        println!("📝 答案: {}", result.answer);
        println!("🎯 置信度: {:.3}", result.confidence);
        println!("⏱️ 处理时间: {:?}", result.processing_time);
        println!("📊 Token数量: {}", result.token_count);
        println!("🗜️ 上下文压缩比: {:.3}", result.context_compression_ratio);
        println!("📚 检索到 {} 个相关文档", result.sources.len());
    }
    
    // 多轮对话演示
    println!("\n💬 多轮对话演示:");
    let _result1 = rag.query("什么是机器学习？").await?;
    let result2 = rag.query_with_context("它有哪些主要算法？").await?;
    println!("📝 上下文回答: {}", result2.answer);
    
    // 获取统计信息
    let stats = rag.get_stats().await;
    println!("\n📊 系统统计:");
    for (key, value) in stats {
        println!("   {}: {}", key, value);
    }
    
    Ok(())
}

/// 演示多模态处理系统
async fn demonstrate_multimodal_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("创建多模态处理系统...");
    
    // 创建编码器
    let text_encoder = Arc::new(SimpleTextEncoder::new(256, "demo-text".to_string()));
    let image_encoder = Arc::new(SimpleImageEncoder::new(512, "demo-image".to_string()));
    let audio_encoder = Arc::new(SimpleAudioEncoder::new(384, "demo-audio".to_string()));
    let fusion = Arc::new(AttentionFusion::new(1024, FusionStrategy::Attention));
    
    // 创建多模态处理器
    let processor = MultimodalAIProcessor::new(
        text_encoder,
        image_encoder,
        audio_encoder,
        fusion,
    );
    
    println!("✅ 多模态处理器创建完成");
    
    // 测试不同模态的处理
    let inputs = vec![
        MultimodalData::Text("这是一个关于人工智能的文本描述，包含了机器学习和深度学习的基本概念。".to_string()),
        MultimodalData::Image(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        MultimodalData::Audio(vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        MultimodalData::Mixed {
            text: Some("多模态输入示例".to_string()),
            image: Some(vec![21, 22, 23, 24, 25]),
            audio: Some(vec![26, 27, 28, 29, 30]),
            video: None,
        },
    ];
    
    for (i, input) in inputs.iter().enumerate() {
        println!("\n🎭 处理模态 {}: {:?}", i + 1, input);
        let result = processor.process(input.clone()).await?;
        
        println!("📝 输出: {}", result.output.to_string());
        println!("🎯 置信度: {:.3}", result.confidence);
        println!("⏱️ 处理时间: {:?}", result.processing_time);
        println!("🔧 使用模态: {:?}", result.modalities_used);
        println!("📊 融合特征维度: {}", result.features.fused_features.len());
    }
    
    // 批量处理演示
    println!("\n📦 批量处理演示:");
    let batch_inputs = vec![
        MultimodalData::Text("批量处理文本1".to_string()),
        MultimodalData::Text("批量处理文本2".to_string()),
        MultimodalData::Image(vec![1, 2, 3]),
        MultimodalData::Audio(vec![4, 5, 6]),
    ];
    
    let batch_results = processor.process_batch(&batch_inputs).await?;
    println!("✅ 批量处理完成，处理了 {} 个样本", batch_results.len());
    
    // 获取统计信息
    let stats = processor.get_stats().await;
    println!("\n📊 多模态系统统计:");
    for (key, value) in stats {
        println!("   {}: {}", key, value);
    }
    
    Ok(())
}

/// 演示Agent系统框架
async fn demonstrate_agent_system() -> Result<(), Box<dyn std::error::Error>> {
    println!("创建Agent系统框架...");
    
    // 创建Agent组件
    let perception = Arc::new(SimplePerception::new(10.0));
    let reasoning = Arc::new(SimpleReasoning::new("demo-reasoning".to_string()));
    let planning = Arc::new(SimplePlanning::new(5));
    let mut execution = SimpleExecution::new(std::time::Duration::from_secs(30));
    let memory = Arc::new(SimpleMemory::new(1000));
    
    // 添加工具
    execution.add_tool(Arc::new(CalculatorTool::new()));
    execution.add_tool(Arc::new(WebSearchTool::new("demo-api-key".to_string())));
    
    // 创建Agent
    let agent = Arc::new(AIAgent::new(
        "demo_agent".to_string(),
        "演示Agent".to_string(),
        perception,
        reasoning,
        planning,
        Arc::new(execution),
        memory,
    ));
    
    println!("✅ Agent创建完成");
    
    // 测试Agent组件
    println!("\n🧠 测试Agent组件:");
    
    // 测试感知
    let environment = HashMap::new();
    let perception_result = agent.perception.perceive(&environment).await?;
    println!("👁️ 感知结果: {:?}", perception_result.observations);
    
    // 测试推理
    let reasoning_result = agent.reasoning.reason(&perception_result, &[]).await?;
    println!("🤔 推理结果: 目标 = {}, 置信度 = {:.3}", reasoning_result.goal, reasoning_result.confidence);
    
    // 测试规划
    let plan = agent.planning.plan(&reasoning_result.goal, &HashMap::new()).await?;
    println!("📋 规划结果: {} 个动作", plan.len());
    
    // 测试执行
    if let Some(action) = plan.first() {
        let execution_result = agent.execution.execute(action).await?;
        println!("⚡ 执行结果: {}", execution_result.result);
    }
    
    // 测试记忆系统
    println!("\n🧠 测试记忆系统:");
    let memory_item = MemoryItem {
        id: "demo_memory".to_string(),
        content: "这是一个演示记忆".to_string(),
        timestamp: std::time::SystemTime::now(),
        importance: 0.8,
        memory_type: MemoryType::Episodic,
        tags: vec!["demo".to_string()],
    };
    
    agent.memory.store(memory_item).await?;
    let retrieved = agent.memory.retrieve("演示", 5).await?;
    println!("📚 检索到 {} 个相关记忆", retrieved.len());
    
    // 测试工具
    println!("\n🔧 测试工具:");
    let calculator = CalculatorTool::new();
    let mut params = HashMap::new();
    params.insert("operation".to_string(), "multiply".to_string());
    params.insert("a".to_string(), "6.0".to_string());
    params.insert("b".to_string(), "7.0".to_string());
    
    let calc_result = calculator.execute(&params).await?;
    println!("🧮 计算器结果: {}", calc_result);
    
    // 创建多Agent系统
    println!("\n👥 创建多Agent系统:");
    let multi_agent_system = MultiAgentSystem::new();
    multi_agent_system.add_agent(agent, "worker".to_string()).await;
    
    let system_stats = multi_agent_system.get_system_stats().await;
    println!("📊 多Agent系统统计:");
    for (key, value) in system_stats {
        println!("   {}: {}", key, value);
    }
    
    Ok(())
}

/// 演示性能测试
async fn demonstrate_performance_testing() -> Result<(), Box<dyn std::error::Error>> {
    println!("开始综合性能测试...");
    
    // RAG系统性能测试
    println!("\n📚 RAG系统性能测试:");
    let embedding_model = Arc::new(AdvancedEmbeddingModel::new(128, "perf-test".to_string()));
    let language_model = Arc::new(AdvancedLanguageModel::new("perf-test".to_string(), 1000, 4000));
    let reranking_model = Arc::new(SimpleRerankingModel::new("perf-test".to_string()));
    
    let rag = EnhancedRAGSystem::new(
        embedding_model,
        language_model,
        reranking_model,
        5, 0.1, 3
    );
    
    // 添加大量文档
    for i in 0..100 {
        let content = format!(
            "这是第{}个测试文档，包含关于人工智能、机器学习、深度学习、自然语言处理、计算机视觉等技术的详细内容。",
            i
        );
        rag.add_document(&format!("perf_doc_{}", i), &content, HashMap::new()).await?;
    }
    
    let start = Instant::now();
    let _result = rag.query("什么是人工智能？").await?;
    let rag_duration = start.elapsed();
    println!("   RAG查询耗时: {:?}", rag_duration);
    
    // 多模态处理性能测试
    println!("\n🎭 多模态处理性能测试:");
    let text_encoder = Arc::new(SimpleTextEncoder::new(128, "perf-text".to_string()));
    let image_encoder = Arc::new(SimpleImageEncoder::new(256, "perf-image".to_string()));
    let audio_encoder = Arc::new(SimpleAudioEncoder::new(192, "perf-audio".to_string()));
    let fusion = Arc::new(AttentionFusion::new(512, FusionStrategy::Attention));
    
    let processor = MultimodalAIProcessor::new(
        text_encoder,
        image_encoder,
        audio_encoder,
        fusion,
    );
    
    let multimodal_input = MultimodalData::Mixed {
        text: Some("性能测试多模态输入".to_string()),
        image: Some(vec![1; 1000]),
        audio: Some(vec![2; 2000]),
        video: None,
    };
    
    let start = Instant::now();
    let _result = processor.process(multimodal_input).await?;
    let multimodal_duration = start.elapsed();
    println!("   多模态处理耗时: {:?}", multimodal_duration);
    
    // Agent系统性能测试
    println!("\n🤖 Agent系统性能测试:");
    let perception = Arc::new(SimplePerception::new(10.0));
    let reasoning = Arc::new(SimpleReasoning::new("perf-reasoning".to_string()));
    let planning = Arc::new(SimplePlanning::new(5));
    let mut execution = SimpleExecution::new(std::time::Duration::from_secs(30));
    let memory = Arc::new(SimpleMemory::new(1000));
    
    execution.add_tool(Arc::new(CalculatorTool::new()));
    
    let agent = AIAgent::new(
        "perf_agent".to_string(),
        "性能测试Agent".to_string(),
        perception,
        reasoning,
        planning,
        Arc::new(execution),
        memory,
    );
    
    let start = Instant::now();
    
    // 执行一个完整的Agent循环
    agent.perceive().await?;
    let reasoning_result = agent.reason().await?;
    let plan = agent.plan(&reasoning_result.goal).await?;
    
    for action in plan {
        let _result = agent.execute(&action).await?;
    }
    
    let agent_duration = start.elapsed();
    println!("   Agent循环耗时: {:?}", agent_duration);
    
    // 性能总结
    println!("\n📈 性能测试总结:");
    println!("   RAG系统: {:?}", rag_duration);
    println!("   多模态处理: {:?}", multimodal_duration);
    println!("   Agent系统: {:?}", agent_duration);
    
    Ok(())
}

/// 演示系统集成
async fn demonstrate_system_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("开始系统集成演示...");
    
    // 创建集成系统
    let integrated_system = IntegratedAISystem::new().await?;
    
    // 演示集成功能
    println!("\n🔗 集成功能演示:");
    
    // 1. RAG + 多模态处理
    let rag_multimodal_result = integrated_system.process_rag_with_multimodal(
        "什么是人工智能？",
        MultimodalData::Text("请结合图像信息回答".to_string())
    ).await?;
    println!("📚 RAG+多模态结果: {}", rag_multimodal_result);
    
    // 2. Agent + RAG
    let agent_rag_result = integrated_system.agent_with_rag_query(
        "agent1",
        "请搜索关于机器学习的信息"
    ).await?;
    println!("🤖 Agent+RAG结果: {}", agent_rag_result);
    
    // 3. 多模态 + Agent
    let multimodal_agent_result = integrated_system.process_multimodal_with_agent(
        MultimodalData::Mixed {
            text: Some("分析这个多模态输入".to_string()),
            image: Some(vec![1, 2, 3, 4, 5]),
            audio: Some(vec![6, 7, 8, 9, 10]),
            video: None,
        },
        "agent1"
    ).await?;
    println!("🎭 多模态+Agent结果: {}", multimodal_agent_result);
    
    // 获取集成系统统计
    let stats = integrated_system.get_integrated_stats().await;
    println!("\n📊 集成系统统计:");
    for (key, value) in stats {
        println!("   {}: {}", key, value);
    }
    
    Ok(())
}

/// 集成AI系统
pub struct IntegratedAISystem {
    pub rag_system: Arc<EnhancedRAGSystem>,
    pub multimodal_processor: Arc<MultimodalAIProcessor>,
    pub agent_system: Arc<MultiAgentSystem>,
}

impl IntegratedAISystem {
    pub async fn new() -> Result<Self> {
        // 创建RAG系统
        let embedding_model = Arc::new(AdvancedEmbeddingModel::new(256, "integrated-embedding".to_string()));
        let language_model = Arc::new(AdvancedLanguageModel::new("integrated-llm".to_string(), 2000, 8000));
        let reranking_model = Arc::new(SimpleRerankingModel::new("integrated-reranker".to_string()));
        
        let rag_system = Arc::new(EnhancedRAGSystem::new(
            embedding_model,
            language_model,
            reranking_model,
            10, 0.1, 5
        ));
        
        // 创建多模态处理器
        let text_encoder = Arc::new(SimpleTextEncoder::new(256, "integrated-text".to_string()));
        let image_encoder = Arc::new(SimpleImageEncoder::new(512, "integrated-image".to_string()));
        let audio_encoder = Arc::new(SimpleAudioEncoder::new(384, "integrated-audio".to_string()));
        let fusion = Arc::new(AttentionFusion::new(1024, FusionStrategy::Attention));
        
        let multimodal_processor = Arc::new(MultimodalAIProcessor::new(
            text_encoder,
            image_encoder,
            audio_encoder,
            fusion,
        ));
        
        // 创建Agent系统
        let agent_system = Arc::new(MultiAgentSystem::new());
        
        // 添加Agent
        let perception = Arc::new(SimplePerception::new(10.0));
        let reasoning = Arc::new(SimpleReasoning::new("integrated-reasoning".to_string()));
        let planning = Arc::new(SimplePlanning::new(5));
        let mut execution = SimpleExecution::new(std::time::Duration::from_secs(30));
        let memory = Arc::new(SimpleMemory::new(1000));
        
        execution.add_tool(Arc::new(CalculatorTool::new()));
        execution.add_tool(Arc::new(WebSearchTool::new("integrated-api-key".to_string())));
        
        let agent = Arc::new(AIAgent::new(
            "agent1".to_string(),
            "集成Agent".to_string(),
            perception,
            reasoning,
            planning,
            Arc::new(execution),
            memory,
        ));
        
        agent_system.add_agent(agent, "worker".to_string()).await;
        
        Ok(Self {
            rag_system,
            multimodal_processor,
            agent_system,
        })
    }
    
    /// RAG + 多模态处理
    pub async fn process_rag_with_multimodal(&self, query: &str, multimodal_input: MultimodalData) -> Result<String> {
        // 1. 处理多模态输入
        let multimodal_result = self.multimodal_processor.process(multimodal_input).await?;
        
        // 2. 使用多模态结果增强RAG查询
        let enhanced_query = format!("{} (多模态上下文: {})", query, multimodal_result.output.to_string());
        
        // 3. 执行RAG查询
        let rag_result = self.rag_system.query(&enhanced_query).await?;
        
        Ok(format!("RAG结果: {}\n多模态上下文: {}", rag_result.answer, multimodal_result.output.to_string()))
    }
    
    /// Agent + RAG查询
    pub async fn agent_with_rag_query(&self, agent_id: &str, query: &str) -> Result<String> {
        // 1. 执行RAG查询
        let rag_result = self.rag_system.query(query).await?;
        
        // 2. 让Agent处理RAG结果
        let agents = self.agent_system.agents.read().await;
        if let Some(agent) = agents.get(agent_id) {
            // 存储RAG结果到Agent记忆
            let memory_item = MemoryItem {
                id: format!("rag_result_{}", SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()),
                content: rag_result.answer.clone(),
                timestamp: SystemTime::now(),
                importance: 0.9,
                memory_type: MemoryType::Semantic,
                tags: vec!["rag".to_string(), "query".to_string()],
            };
            
            agent.memory.store(memory_item).await?;
            
            // Agent处理查询
            let reasoning_result = agent.reasoning.reason(
                &Perception {
                    timestamp: SystemTime::now(),
                    environment_state: HashMap::new(),
                    observations: vec![query.to_string()],
                    confidence: 0.8,
                },
                &[]
            ).await?;
            
            Ok(format!("Agent处理结果: {}\nRAG信息: {}", reasoning_result.goal, rag_result.answer))
        } else {
            Err(anyhow!("Agent不存在: {}", agent_id))
        }
    }
    
    /// 多模态 + Agent处理
    pub async fn process_multimodal_with_agent(&self, input: MultimodalData, agent_id: &str) -> Result<String> {
        // 1. 处理多模态输入
        let multimodal_result = self.multimodal_processor.process(input).await?;
        
        // 2. 让Agent处理多模态结果
        let agents = self.agent_system.agents.read().await;
        if let Some(agent) = agents.get(agent_id) {
            // 存储多模态结果到Agent记忆
            let memory_item = MemoryItem {
                id: format!("multimodal_result_{}", SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()),
                content: multimodal_result.output.to_string(),
                timestamp: SystemTime::now(),
                importance: 0.8,
                memory_type: MemoryType::Episodic,
                tags: vec!["multimodal".to_string(), "processing".to_string()],
            };
            
            agent.memory.store(memory_item).await?;
            
            // Agent处理多模态结果
            let reasoning_result = agent.reasoning.reason(
                &Perception {
                    timestamp: SystemTime::now(),
                    environment_state: HashMap::new(),
                    observations: vec![multimodal_result.output.to_string()],
                    confidence: multimodal_result.confidence,
                },
                &[]
            ).await?;
            
            Ok(format!("Agent分析: {}\n多模态结果: {}", reasoning_result.goal, multimodal_result.output.to_string()))
        } else {
            Err(anyhow!("Agent不存在: {}", agent_id))
        }
    }
    
    /// 获取集成系统统计
    pub async fn get_integrated_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        // RAG系统统计
        let rag_stats = self.rag_system.get_stats().await;
        for (key, value) in rag_stats {
            stats.insert(format!("rag_{}", key), value);
        }
        
        // 多模态系统统计
        let multimodal_stats = self.multimodal_processor.get_stats().await;
        for (key, value) in multimodal_stats {
            stats.insert(format!("multimodal_{}", key), value);
        }
        
        // Agent系统统计
        let agent_stats = self.agent_system.get_system_stats().await;
        for (key, value) in agent_stats {
            stats.insert(format!("agent_{}", key), value);
        }
        
        stats
    }
}
