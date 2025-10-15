//! 基础Agent系统框架实现
//! 
//! 本示例展示了一个完整的AI Agent系统框架，包括：
//! - 感知、推理、规划、执行循环
//! - 工具调用和外部API集成
//! - 记忆系统和知识管理
//! - 多Agent协作和通信
//! - 安全边界和治理机制

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::time::{Instant, SystemTime};

/// Agent状态
#[derive(Debug, Clone, PartialEq)]
pub enum AgentState {
    Idle,
    Perceiving,
    Reasoning,
    Planning,
    Executing,
    Waiting,
    Error(String),
}

/// Agent动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentAction {
    Move { x: f64, y: f64 },
    Communicate { target: String, message: String },
    UseTool { tool_name: String, parameters: HashMap<String, String> },
    Query { query: String },
    Wait { duration: std::time::Duration },
    Terminate,
}

/// 感知结果
#[derive(Debug, Clone)]
pub struct Perception {
    pub timestamp: SystemTime,
    pub environment_state: HashMap<String, String>,
    pub observations: Vec<String>,
    pub confidence: f64,
}

/// 推理结果
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    pub goal: String,
    pub plan: Vec<AgentAction>,
    pub confidence: f64,
    pub reasoning_steps: Vec<String>,
}

/// 执行结果
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub action: AgentAction,
    pub success: bool,
    pub result: String,
    pub execution_time: std::time::Duration,
    pub error: Option<String>,
}

/// 记忆项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub timestamp: SystemTime,
    pub importance: f64,
    pub memory_type: MemoryType,
    pub tags: Vec<String>,
}

/// 记忆类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Episodic,    // 情节记忆
    Semantic,    // 语义记忆
    Procedural,  // 程序记忆
    Working,     // 工作记忆
}

/// 工具trait
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Vec<String>;
    async fn execute(&self, parameters: &HashMap<String, String>) -> Result<String>;
}

/// 感知器trait
pub trait AgentPerception: Send + Sync {
    async fn perceive(&self, environment: &HashMap<String, String>) -> Result<Perception>;
    fn perception_range(&self) -> f64;
}

/// 推理器trait
pub trait AgentReasoning: Send + Sync {
    async fn reason(&self, perception: &Perception, memory: &[MemoryItem]) -> Result<ReasoningResult>;
    fn reasoning_model(&self) -> &str;
}

/// 规划器trait
pub trait AgentPlanning: Send + Sync {
    async fn plan(&self, goal: &str, current_state: &HashMap<String, String>) -> Result<Vec<AgentAction>>;
    fn planning_horizon(&self) -> usize;
}

/// 执行器trait
pub trait AgentExecution: Send + Sync {
    async fn execute(&self, action: &AgentAction) -> Result<ExecutionResult>;
    fn execution_timeout(&self) -> std::time::Duration;
}

/// 记忆系统trait
pub trait AgentMemory: Send + Sync {
    async fn store(&self, item: MemoryItem) -> Result<()>;
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>>;
    async fn update(&self, id: &str, item: MemoryItem) -> Result<()>;
    async fn delete(&self, id: &str) -> Result<()>;
    async fn clear(&self) -> Result<()>;
}

/// 简单工具实现
pub struct CalculatorTool;

impl CalculatorTool {
    pub fn new() -> Self {
        Self
    }
}

impl AgentTool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }
    
    fn description(&self) -> &str {
        "执行基本数学计算"
    }
    
    fn parameters(&self) -> Vec<String> {
        vec!["operation".to_string(), "a".to_string(), "b".to_string()]
    }
    
    async fn execute(&self, parameters: &HashMap<String, String>) -> Result<String> {
        let operation = parameters.get("operation").ok_or_else(|| anyhow!("缺少operation参数"))?;
        let a: f64 = parameters.get("a").ok_or_else(|| anyhow!("缺少a参数"))?.parse()?;
        let b: f64 = parameters.get("b").ok_or_else(|| anyhow!("缺少b参数"))?.parse()?;
        
        // 模拟异步处理
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        let result = match operation.as_str() {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Err(anyhow!("除零错误"));
                }
                a / b
            },
            _ => return Err(anyhow!("不支持的操作: {}", operation)),
        };
        
        Ok(format!("{} {} {} = {}", a, operation, b, result))
    }
}

/// 网络搜索工具
pub struct WebSearchTool {
    pub api_key: String,
}

impl WebSearchTool {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
}

impl AgentTool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }
    
    fn description(&self) -> &str {
        "搜索网络信息"
    }
    
    fn parameters(&self) -> Vec<String> {
        vec!["query".to_string(), "limit".to_string()]
    }
    
    async fn execute(&self, parameters: &HashMap<String, String>) -> Result<String> {
        let query = parameters.get("query").ok_or_else(|| anyhow!("缺少query参数"))?;
        let limit: usize = parameters.get("limit").unwrap_or(&"5".to_string()).parse()?;
        
        // 模拟网络搜索
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        // 模拟搜索结果
        let results = (0..limit).map(|i| {
            format!("搜索结果 {}: 关于'{}'的相关信息", i + 1, query)
        }).collect::<Vec<_>>().join("\n");
        
        Ok(format!("搜索查询: '{}'\n\n{}", query, results))
    }
}

/// 简单感知器实现
pub struct SimplePerception {
    pub perception_range: f64,
}

impl SimplePerception {
    pub fn new(perception_range: f64) -> Self {
        Self { perception_range }
    }
}

impl AgentPerception for SimplePerception {
    async fn perceive(&self, environment: &HashMap<String, String>) -> Result<Perception> {
        // 模拟异步感知
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        let observations = environment.iter()
            .map(|(key, value)| format!("{}: {}", key, value))
            .collect();
        
        Ok(Perception {
            timestamp: SystemTime::now(),
            environment_state: environment.clone(),
            observations,
            confidence: 0.8,
        })
    }
    
    fn perception_range(&self) -> f64 {
        self.perception_range
    }
}

/// 简单推理器实现
pub struct SimpleReasoning {
    pub model_name: String,
}

impl SimpleReasoning {
    pub fn new(model_name: String) -> Self {
        Self { model_name }
    }
}

impl AgentReasoning for SimpleReasoning {
    async fn reason(&self, perception: &Perception, memory: &[MemoryItem]) -> Result<ReasoningResult> {
        // 模拟异步推理
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // 简单的目标推理
        let goal = if perception.observations.iter().any(|obs| obs.contains("问题")) {
            "解决问题".to_string()
        } else if perception.observations.iter().any(|obs| obs.contains("任务")) {
            "完成任务".to_string()
        } else {
            "探索环境".to_string()
        };
        
        // 简单的规划
        let plan = vec![
            AgentAction::Query { query: "分析当前情况".to_string() },
            AgentAction::UseTool { 
                tool_name: "calculator".to_string(), 
                parameters: HashMap::new() 
            },
            AgentAction::Communicate { 
                target: "user".to_string(), 
                message: "任务完成".to_string() 
            },
        ];
        
        let reasoning_steps = vec![
            "分析感知信息".to_string(),
            "检索相关记忆".to_string(),
            "确定目标".to_string(),
            "制定计划".to_string(),
        ];
        
        Ok(ReasoningResult {
            goal,
            plan,
            confidence: 0.7,
            reasoning_steps,
        })
    }
    
    fn reasoning_model(&self) -> &str {
        &self.model_name
    }
}

/// 简单规划器实现
pub struct SimplePlanning {
    pub planning_horizon: usize,
}

impl SimplePlanning {
    pub fn new(planning_horizon: usize) -> Self {
        Self { planning_horizon }
    }
}

impl AgentPlanning for SimplePlanning {
    async fn plan(&self, goal: &str, current_state: &HashMap<String, String>) -> Result<Vec<AgentAction>> {
        // 模拟异步规划
        tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
        
        let mut plan = Vec::new();
        
        match goal {
            "解决问题" => {
                plan.push(AgentAction::Query { query: "分析问题".to_string() });
                plan.push(AgentAction::UseTool { 
                    tool_name: "calculator".to_string(), 
                    parameters: HashMap::new() 
                });
                plan.push(AgentAction::Communicate { 
                    target: "user".to_string(), 
                    message: "问题已解决".to_string() 
                });
            },
            "完成任务" => {
                plan.push(AgentAction::Query { query: "了解任务要求".to_string() });
                plan.push(AgentAction::UseTool { 
                    tool_name: "web_search".to_string(), 
                    parameters: HashMap::new() 
                });
                plan.push(AgentAction::Communicate { 
                    target: "user".to_string(), 
                    message: "任务已完成".to_string() 
                });
            },
            _ => {
                plan.push(AgentAction::Query { query: "探索环境".to_string() });
                plan.push(AgentAction::Wait { duration: std::time::Duration::from_secs(1) });
            }
        }
        
        // 限制规划长度
        plan.truncate(self.planning_horizon);
        
        Ok(plan)
    }
    
    fn planning_horizon(&self) -> usize {
        self.planning_horizon
    }
}

/// 简单执行器实现
pub struct SimpleExecution {
    pub timeout: std::time::Duration,
    pub tools: HashMap<String, Arc<dyn AgentTool>>,
}

impl SimpleExecution {
    pub fn new(timeout: std::time::Duration) -> Self {
        Self {
            timeout,
            tools: HashMap::new(),
        }
    }
    
    pub fn add_tool(&mut self, tool: Arc<dyn AgentTool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }
}

impl AgentExecution for SimpleExecution {
    async fn execute(&self, action: &AgentAction) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        
        let result = match action {
            AgentAction::Move { x, y } => {
                // 模拟移动
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                format!("移动到位置 ({}, {})", x, y)
            },
            AgentAction::Communicate { target, message } => {
                // 模拟通信
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                format!("向 {} 发送消息: {}", target, message)
            },
            AgentAction::UseTool { tool_name, parameters } => {
                // 使用工具
                if let Some(tool) = self.tools.get(tool_name) {
                    tool.execute(parameters).await?
                } else {
                    return Err(anyhow!("工具不存在: {}", tool_name));
                }
            },
            AgentAction::Query { query } => {
                // 模拟查询
                tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
                format!("查询: {}", query)
            },
            AgentAction::Wait { duration } => {
                // 等待
                tokio::time::sleep(*duration).await;
                format!("等待了 {:?}", duration)
            },
            AgentAction::Terminate => {
                "Agent终止".to_string()
            }
        };
        
        let execution_time = start_time.elapsed();
        
        Ok(ExecutionResult {
            action: action.clone(),
            success: true,
            result,
            execution_time,
            error: None,
        })
    }
    
    fn execution_timeout(&self) -> std::time::Duration {
        self.timeout
    }
}

/// 简单记忆系统实现
pub struct SimpleMemory {
    pub memories: Arc<RwLock<HashMap<String, MemoryItem>>>,
    pub max_memories: usize,
}

impl SimpleMemory {
    pub fn new(max_memories: usize) -> Self {
        Self {
            memories: Arc::new(RwLock::new(HashMap::new())),
            max_memories,
        }
    }
    
    /// 计算记忆相似度
    fn calculate_similarity(&self, query: &str, memory: &MemoryItem) -> f64 {
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let memory_words: Vec<&str> = memory.content.split_whitespace().collect();
        
        let mut matches = 0;
        for query_word in &query_words {
            for memory_word in &memory_words {
                if query_word.to_lowercase() == memory_word.to_lowercase() {
                    matches += 1;
                    break;
                }
            }
        }
        
        matches as f64 / query_words.len() as f64
    }
}

impl AgentMemory for SimpleMemory {
    async fn store(&self, item: MemoryItem) -> Result<()> {
        let mut memories = self.memories.write().await;
        
        // 检查内存限制
        if memories.len() >= self.max_memories {
            // 删除最不重要的记忆
            if let Some(least_important) = memories.iter()
                .min_by(|a, b| a.1.importance.partial_cmp(&b.1.importance).unwrap()) {
                memories.remove(least_important.0);
            }
        }
        
        memories.insert(item.id.clone(), item);
        Ok(())
    }
    
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>> {
        let memories = self.memories.read().await;
        let mut scored_memories: Vec<(MemoryItem, f64)> = Vec::new();
        
        for memory in memories.values() {
            let similarity = self.calculate_similarity(query, memory);
            if similarity > 0.1 { // 相似度阈值
                scored_memories.push((memory.clone(), similarity));
            }
        }
        
        // 按相似度排序
        scored_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // 返回前limit个
        let result = scored_memories.into_iter()
            .take(limit)
            .map(|(memory, _)| memory)
            .collect();
        
        Ok(result)
    }
    
    async fn update(&self, id: &str, item: MemoryItem) -> Result<()> {
        let mut memories = self.memories.write().await;
        memories.insert(id.to_string(), item);
        Ok(())
    }
    
    async fn delete(&self, id: &str) -> Result<()> {
        let mut memories = self.memories.write().await;
        memories.remove(id);
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        let mut memories = self.memories.write().await;
        memories.clear();
        Ok(())
    }
}

/// AI Agent主结构
pub struct AIAgent {
    pub id: String,
    pub name: String,
    pub state: Arc<RwLock<AgentState>>,
    pub perception: Arc<dyn AgentPerception>,
    pub reasoning: Arc<dyn AgentReasoning>,
    pub planning: Arc<dyn AgentPlanning>,
    pub execution: Arc<dyn AgentExecution>,
    pub memory: Arc<dyn AgentMemory>,
    pub message_sender: mpsc::UnboundedSender<AgentMessage>,
    pub message_receiver: Arc<RwLock<mpsc::UnboundedReceiver<AgentMessage>>>,
}

/// Agent消息
#[derive(Debug, Clone)]
pub struct AgentMessage {
    pub from: String,
    pub to: String,
    pub content: String,
    pub message_type: MessageType,
    pub timestamp: SystemTime,
}

/// 消息类型
#[derive(Debug, Clone)]
pub enum MessageType {
    Request,
    Response,
    Notification,
    Error,
}

impl AIAgent {
    pub fn new(
        id: String,
        name: String,
        perception: Arc<dyn AgentPerception>,
        reasoning: Arc<dyn AgentReasoning>,
        planning: Arc<dyn AgentPlanning>,
        execution: Arc<dyn AgentExecution>,
        memory: Arc<dyn AgentMemory>,
    ) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        Self {
            id,
            name,
            state: Arc::new(RwLock::new(AgentState::Idle)),
            perception,
            reasoning,
            planning,
            execution,
            memory,
            message_sender: sender,
            message_receiver: Arc::new(RwLock::new(receiver)),
        }
    }
    
    /// 主循环
    pub async fn run(&self) -> Result<()> {
        loop {
            // 1. 感知
            self.perceive().await?;
            
            // 2. 推理
            let reasoning_result = self.reason().await?;
            
            // 3. 规划
            let plan = self.plan(&reasoning_result.goal).await?;
            
            // 4. 执行
            for action in plan {
                let result = self.execute(&action).await?;
                
                // 存储执行结果到记忆
                let memory_item = MemoryItem {
                    id: format!("execution_{}", SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()),
                    content: format!("执行动作: {:?}, 结果: {}", action, result.result),
                    timestamp: SystemTime::now(),
                    importance: if result.success { 0.8 } else { 0.3 },
                    memory_type: MemoryType::Episodic,
                    tags: vec!["execution".to_string()],
                };
                
                self.memory.store(memory_item).await?;
                
                if !result.success {
                    break;
                }
            }
            
            // 5. 等待一段时间
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }
    
    /// 感知阶段
    async fn perceive(&self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = AgentState::Perceiving;
        drop(state);
        
        let environment = HashMap::new(); // 简化的环境
        let perception = self.perception.perceive(&environment).await?;
        
        // 存储感知结果到记忆
        let memory_item = MemoryItem {
            id: format!("perception_{}", SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()),
            content: format!("感知结果: {:?}", perception.observations),
            timestamp: SystemTime::now(),
            importance: 0.6,
            memory_type: MemoryType::Episodic,
            tags: vec!["perception".to_string()],
        };
        
        self.memory.store(memory_item).await?;
        
        Ok(())
    }
    
    /// 推理阶段
    async fn reason(&self) -> Result<ReasoningResult> {
        let mut state = self.state.write().await;
        *state = AgentState::Reasoning;
        drop(state);
        
        // 获取最近的感知和记忆
        let recent_memories = self.memory.retrieve("perception", 5).await?;
        
        // 简化的环境状态
        let environment = HashMap::new();
        let perception = self.perception.perceive(&environment).await?;
        
        self.reasoning.reason(&perception, &recent_memories).await
    }
    
    /// 规划阶段
    async fn plan(&self, goal: &str) -> Result<Vec<AgentAction>> {
        let mut state = self.state.write().await;
        *state = AgentState::Planning;
        drop(state);
        
        let current_state = HashMap::new();
        self.planning.plan(goal, &current_state).await
    }
    
    /// 执行阶段
    async fn execute(&self, action: &AgentAction) -> Result<ExecutionResult> {
        let mut state = self.state.write().await;
        *state = AgentState::Executing;
        drop(state);
        
        let result = self.execution.execute(action).await?;
        
        let mut state = self.state.write().await;
        *state = AgentState::Idle;
        
        Ok(result)
    }
    
    /// 发送消息
    pub fn send_message(&self, to: &str, content: &str, message_type: MessageType) -> Result<()> {
        let message = AgentMessage {
            from: self.id.clone(),
            to: to.to_string(),
            content: content.to_string(),
            message_type,
            timestamp: SystemTime::now(),
        };
        
        self.message_sender.send(message)?;
        Ok(())
    }
    
    /// 接收消息
    pub async fn receive_message(&self) -> Option<AgentMessage> {
        let mut receiver = self.message_receiver.write().await;
        receiver.recv().await
    }
    
    /// 获取Agent状态
    pub async fn get_state(&self) -> AgentState {
        let state = self.state.read().await;
        state.clone()
    }
    
    /// 获取Agent统计信息
    pub async fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        // 获取记忆统计
        let memories = self.memory.retrieve("", 1000).await.unwrap_or_default();
        stats.insert("total_memories".to_string(), memories.len());
        
        // 按类型统计记忆
        let episodic_count = memories.iter().filter(|m| matches!(m.memory_type, MemoryType::Episodic)).count();
        let semantic_count = memories.iter().filter(|m| matches!(m.memory_type, MemoryType::Semantic)).count();
        let procedural_count = memories.iter().filter(|m| matches!(m.memory_type, MemoryType::Procedural)).count();
        
        stats.insert("episodic_memories".to_string(), episodic_count);
        stats.insert("semantic_memories".to_string(), semantic_count);
        stats.insert("procedural_memories".to_string(), procedural_count);
        
        stats
    }
}

/// 多Agent系统
pub struct MultiAgentSystem {
    pub agents: Arc<RwLock<HashMap<String, Arc<AIAgent>>>>,
    pub coordinator: Arc<RwLock<SystemCoordinator>>,
}

/// 系统协调器
pub struct SystemCoordinator {
    pub system_state: HashMap<String, String>,
    pub agent_roles: HashMap<String, String>,
    pub communication_rules: Vec<CommunicationRule>,
}

/// 通信规则
#[derive(Debug, Clone)]
pub struct CommunicationRule {
    pub from_role: String,
    pub to_role: String,
    pub message_type: MessageType,
    pub priority: usize,
}

impl MultiAgentSystem {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            coordinator: Arc::new(RwLock::new(SystemCoordinator {
                system_state: HashMap::new(),
                agent_roles: HashMap::new(),
                communication_rules: Vec::new(),
            })),
        }
    }
    
    /// 添加Agent
    pub async fn add_agent(&self, agent: Arc<AIAgent>, role: String) {
        let mut agents = self.agents.write().await;
        let mut coordinator = self.coordinator.write().await;
        
        agents.insert(agent.id.clone(), agent);
        coordinator.agent_roles.insert(agent.id.clone(), role);
    }
    
    /// 启动系统
    pub async fn start(&self) -> Result<()> {
        let agents = self.agents.read().await;
        let mut handles = Vec::new();
        
        for (id, agent) in agents.iter() {
            let agent = agent.clone();
            let handle = tokio::spawn(async move {
                if let Err(e) = agent.run().await {
                    eprintln!("Agent {} 运行出错: {}", id, e);
                }
            });
            handles.push(handle);
        }
        
        // 等待所有Agent完成
        for handle in handles {
            let _ = handle.await;
        }
        
        Ok(())
    }
    
    /// 获取系统统计信息
    pub async fn get_system_stats(&self) -> HashMap<String, usize> {
        let agents = self.agents.read().await;
        let mut stats = HashMap::new();
        
        stats.insert("total_agents".to_string(), agents.len());
        
        // 统计各Agent的状态
        let mut idle_count = 0;
        let mut active_count = 0;
        
        for agent in agents.values() {
            match agent.get_state().await {
                AgentState::Idle => idle_count += 1,
                _ => active_count += 1,
            }
        }
        
        stats.insert("idle_agents".to_string(), idle_count);
        stats.insert("active_agents".to_string(), active_count);
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_agent_creation() {
        let perception = Arc::new(SimplePerception::new(10.0));
        let reasoning = Arc::new(SimpleReasoning::new("test-reasoning".to_string()));
        let planning = Arc::new(SimplePlanning::new(5));
        let mut execution = SimpleExecution::new(std::time::Duration::from_secs(30));
        let memory = Arc::new(SimpleMemory::new(1000));
        
        // 添加工具
        execution.add_tool(Arc::new(CalculatorTool::new()));
        execution.add_tool(Arc::new(WebSearchTool::new("test-key".to_string())));
        
        let agent = AIAgent::new(
            "agent1".to_string(),
            "测试Agent".to_string(),
            perception,
            reasoning,
            planning,
            Arc::new(execution),
            memory,
        );
        
        assert_eq!(agent.id, "agent1");
        assert_eq!(agent.name, "测试Agent");
        assert_eq!(agent.get_state().await, AgentState::Idle);
    }

    #[test]
    async fn test_agent_memory() {
        let memory = SimpleMemory::new(100);
        
        let memory_item = MemoryItem {
            id: "test1".to_string(),
            content: "这是一个测试记忆".to_string(),
            timestamp: SystemTime::now(),
            importance: 0.8,
            memory_type: MemoryType::Episodic,
            tags: vec!["test".to_string()],
        };
        
        // 存储记忆
        memory.store(memory_item).await.unwrap();
        
        // 检索记忆
        let retrieved = memory.retrieve("测试", 5).await.unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].content, "这是一个测试记忆");
    }

    #[test]
    async fn test_agent_tools() {
        let calculator = CalculatorTool::new();
        
        let mut params = HashMap::new();
        params.insert("operation".to_string(), "add".to_string());
        params.insert("a".to_string(), "5.0".to_string());
        params.insert("b".to_string(), "3.0".to_string());
        
        let result = calculator.execute(&params).await.unwrap();
        assert_eq!(result, "5.0 add 3.0 = 8");
        
        let web_search = WebSearchTool::new("test-key".to_string());
        let mut search_params = HashMap::new();
        search_params.insert("query".to_string(), "Rust AI".to_string());
        search_params.insert("limit".to_string(), "3".to_string());
        
        let search_result = web_search.execute(&search_params).await.unwrap();
        assert!(search_result.contains("Rust AI"));
    }

    #[test]
    async fn test_multi_agent_system() {
        let system = MultiAgentSystem::new();
        
        // 创建Agent
        let perception = Arc::new(SimplePerception::new(10.0));
        let reasoning = Arc::new(SimpleReasoning::new("test-reasoning".to_string()));
        let planning = Arc::new(SimplePlanning::new(5));
        let mut execution = SimpleExecution::new(std::time::Duration::from_secs(30));
        let memory = Arc::new(SimpleMemory::new(1000));
        
        execution.add_tool(Arc::new(CalculatorTool::new()));
        
        let agent = Arc::new(AIAgent::new(
            "agent1".to_string(),
            "测试Agent".to_string(),
            perception,
            reasoning,
            planning,
            Arc::new(execution),
            memory,
        ));
        
        // 添加Agent到系统
        system.add_agent(agent, "worker".to_string()).await;
        
        // 获取系统统计
        let stats = system.get_system_stats().await;
        assert_eq!(stats["total_agents"], 1);
    }

    #[test]
    async fn test_agent_communication() {
        let perception = Arc::new(SimplePerception::new(10.0));
        let reasoning = Arc::new(SimpleReasoning::new("test-reasoning".to_string()));
        let planning = Arc::new(SimplePlanning::new(5));
        let execution = SimpleExecution::new(std::time::Duration::from_secs(30));
        let memory = Arc::new(SimpleMemory::new(1000));
        
        let agent = AIAgent::new(
            "agent1".to_string(),
            "测试Agent".to_string(),
            perception,
            reasoning,
            planning,
            Arc::new(execution),
            memory,
        );
        
        // 发送消息
        agent.send_message("agent2", "你好", MessageType::Request).unwrap();
        
        // 接收消息
        let message = agent.receive_message().await;
        assert!(message.is_some());
        let msg = message.unwrap();
        assert_eq!(msg.from, "agent1");
        assert_eq!(msg.to, "agent2");
        assert_eq!(msg.content, "你好");
    }
}

/// 性能基准测试
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    async fn benchmark_agent_cycle() {
        let perception = Arc::new(SimplePerception::new(10.0));
        let reasoning = Arc::new(SimpleReasoning::new("benchmark-reasoning".to_string()));
        let planning = Arc::new(SimplePlanning::new(5));
        let mut execution = SimpleExecution::new(std::time::Duration::from_secs(30));
        let memory = Arc::new(SimpleMemory::new(1000));
        
        execution.add_tool(Arc::new(CalculatorTool::new()));
        
        let agent = AIAgent::new(
            "benchmark_agent".to_string(),
            "基准测试Agent".to_string(),
            perception,
            reasoning,
            planning,
            Arc::new(execution),
            memory,
        );
        
        let start = Instant::now();
        
        // 执行一个完整的感知-推理-规划-执行循环
        agent.perceive().await.unwrap();
        let reasoning_result = agent.reason().await.unwrap();
        let plan = agent.plan(&reasoning_result.goal).await.unwrap();
        
        for action in plan {
            let _result = agent.execute(&action).await.unwrap();
        }
        
        let duration = start.elapsed();
        println!("完整Agent循环耗时: {:?}", duration);
    }

    #[test]
    async fn benchmark_memory_operations() {
        let memory = SimpleMemory::new(10000);
        
        // 批量存储记忆
        let start = Instant::now();
        for i in 0..1000 {
            let memory_item = MemoryItem {
                id: format!("test_{}", i),
                content: format!("测试记忆内容 {}", i),
                timestamp: SystemTime::now(),
                importance: 0.5,
                memory_type: MemoryType::Episodic,
                tags: vec!["test".to_string()],
            };
            memory.store(memory_item).await.unwrap();
        }
        let store_duration = start.elapsed();
        
        // 检索记忆
        let start = Instant::now();
        let _retrieved = memory.retrieve("测试", 100).await.unwrap();
        let retrieve_duration = start.elapsed();
        
        println!("存储1000个记忆耗时: {:?}", store_duration);
        println!("检索100个记忆耗时: {:?}", retrieve_duration);
    }

    #[test]
    async fn benchmark_tool_execution() {
        let calculator = CalculatorTool::new();
        
        let start = Instant::now();
        for i in 0..100 {
            let mut params = HashMap::new();
            params.insert("operation".to_string(), "add".to_string());
            params.insert("a".to_string(), i.to_string());
            params.insert("b".to_string(), (i + 1).to_string());
            
            let _result = calculator.execute(&params).await.unwrap();
        }
        let duration = start.elapsed();
        
        println!("执行100次计算耗时: {:?}", duration);
        println!("平均每次计算: {:?}", duration / 100);
    }
}
