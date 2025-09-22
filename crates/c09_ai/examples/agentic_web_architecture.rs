//! Agentic Web架构示例
//! 
//! 本示例展示了Agentic Web的核心概念和实现：
//! - AI代理驱动的Web交互
//! - 自主规划、协调和执行复杂任务
//! - 代理间协作与协议标准化
//! - 智能性、交互性和经济性三维度

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

// 代理类型定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    WebNavigator,      // 网页导航代理
    DataProcessor,     // 数据处理代理
    ContentGenerator,  // 内容生成代理
    TaskCoordinator,   // 任务协调代理
    ResourceManager,   // 资源管理代理
    SecurityMonitor,   // 安全监控代理
}

// 任务状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,      // 待处理
    InProgress,   // 进行中
    Completed,    // 已完成
    Failed,       // 失败
    Cancelled,    // 已取消
}

// 任务优先级
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

// 任务定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub name: String,
    pub description: String,
    pub agent_type: AgentType,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub input_data: HashMap<String, serde_json::Value>,
    pub output_data: Option<HashMap<String, serde_json::Value>>,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub dependencies: Vec<String>,
    pub estimated_duration_ms: u64,
    pub actual_duration_ms: Option<u64>,
    pub error_message: Option<String>,
}

// 代理能力
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapability {
    pub name: String,
    pub description: String,
    pub input_types: Vec<String>,
    pub output_types: Vec<String>,
    pub max_concurrent_tasks: usize,
    pub estimated_processing_time_ms: u64,
}

// 代理信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    pub agent_type: AgentType,
    pub capabilities: Vec<AgentCapability>,
    pub status: AgentStatus,
    pub current_tasks: Vec<String>,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub average_processing_time_ms: f64,
    pub last_heartbeat: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Available,    // 可用
    Busy,         // 忙碌
    Offline,      // 离线
    Maintenance,  // 维护中
}

// 代理接口
#[async_trait]
pub trait Agent: Send + Sync {
    async fn get_info(&self) -> AgentInfo;
    async fn can_handle_task(&self, task: &Task) -> bool;
    async fn execute_task(&self, task: &Task) -> Result<TaskResult, AgentError>;
    async fn get_capabilities(&self) -> Vec<AgentCapability>;
    async fn heartbeat(&self) -> Result<(), AgentError>;
}

// 任务结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub success: bool,
    pub output_data: Option<HashMap<String, serde_json::Value>>,
    pub processing_time_ms: u64,
    pub error_message: Option<String>,
    pub metrics: HashMap<String, f64>,
}

// 代理错误
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("任务执行失败: {0}")]
    TaskExecutionError(String),
    #[error("能力不足: {0}")]
    InsufficientCapability(String),
    #[error("资源不足: {0}")]
    InsufficientResource(String),
    #[error("通信错误: {0}")]
    CommunicationError(String),
    #[error("超时: {0}")]
    Timeout(String),
}

// 代理注册表
pub struct AgentRegistry {
    agents: Arc<RwLock<HashMap<String, Arc<dyn Agent>>>>,
    agent_info: Arc<RwLock<HashMap<String, AgentInfo>>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            agent_info: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_agent(&self, agent: Arc<dyn Agent>) -> Result<(), AgentError> {
        let info = agent.get_info().await;
        let agent_id = info.id.clone();
        
        {
            let mut agents = self.agents.write().await;
            agents.insert(agent_id.clone(), agent);
        }
        
        {
            let mut agent_info = self.agent_info.write().await;
            agent_info.insert(agent_id, info);
        }
        
        Ok(())
    }
    
    pub async fn find_agent_for_task(&self, task: &Task) -> Option<Arc<dyn Agent>> {
        let agents = self.agents.read().await;
        
        for (_, agent) in agents.iter() {
            if agent.can_handle_task(task).await {
                return Some(Arc::clone(agent));
            }
        }
        
        None
    }
    
    pub async fn get_available_agents(&self) -> Vec<AgentInfo> {
        let agent_info = self.agent_info.read().await;
        agent_info.values()
            .filter(|info| info.status == AgentStatus::Available)
            .cloned()
            .collect()
    }
    
    pub async fn update_agent_status(&self, agent_id: &str, status: AgentStatus) {
        let mut agent_info = self.agent_info.write().await;
        if let Some(info) = agent_info.get_mut(agent_id) {
            info.status = status;
        }
    }
}

// 任务调度器
pub struct TaskScheduler {
    task_queue: Arc<RwLock<Vec<Task>>>,
    running_tasks: Arc<RwLock<HashMap<String, Task>>>,
    completed_tasks: Arc<RwLock<Vec<Task>>>,
    agent_registry: Arc<AgentRegistry>,
    event_sender: broadcast::Sender<TaskEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskEvent {
    TaskCreated(Task),
    TaskStarted(String),
    TaskCompleted(String, TaskResult),
    TaskFailed(String, String),
    AgentAssigned(String, String),
}

impl TaskScheduler {
    pub fn new(agent_registry: Arc<AgentRegistry>) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        
        Self {
            task_queue: Arc::new(RwLock::new(Vec::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(Vec::new())),
            agent_registry,
            event_sender,
        }
    }
    
    pub async fn submit_task(&self, mut task: Task) -> Result<String, AgentError> {
        task.id = Uuid::new_v4().to_string();
        task.created_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        {
            let mut queue = self.task_queue.write().await;
            queue.push(task.clone());
        }
        
        // 按优先级排序
        self.sort_task_queue().await;
        
        // 发送任务创建事件
        let _ = self.event_sender.send(TaskEvent::TaskCreated(task.clone()));
        
        // 尝试立即分配任务
        self.try_assign_tasks().await;
        
        Ok(task.id)
    }
    
    async fn sort_task_queue(&self) {
        let mut queue = self.task_queue.write().await;
        queue.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
    
    async fn try_assign_tasks(&self) {
        let mut queue = self.task_queue.write().await;
        let running_tasks = self.running_tasks.write().await;
        
        let mut tasks_to_assign = Vec::new();
        let mut remaining_tasks = Vec::new();
        
        for task in queue.drain(..) {
            if task.status == TaskStatus::Pending {
                tasks_to_assign.push(task);
            } else {
                remaining_tasks.push(task);
            }
        }
        
        *queue = remaining_tasks;
        drop(queue);
        drop(running_tasks);
        
        for task in tasks_to_assign {
            if let Some(agent) = self.agent_registry.find_agent_for_task(&task).await {
                self.assign_task_to_agent(task, agent).await;
            } else {
                // 没有可用代理，重新加入队列
                let mut queue = self.task_queue.write().await;
                queue.push(task);
            }
        }
    }
    
    async fn assign_task_to_agent(&self, mut task: Task, agent: Arc<dyn Agent>) {
        task.status = TaskStatus::InProgress;
        task.started_at = Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
        
        {
            let mut running_tasks = self.running_tasks.write().await;
            running_tasks.insert(task.id.clone(), task.clone());
        }
        
        // 发送任务开始事件
        let _ = self.event_sender.send(TaskEvent::TaskStarted(task.id.clone()));
        
        // 同步执行任务（简化版本）
        let result = agent.execute_task(&task).await;
        self.handle_task_completion(task, result).await;
    }
    
    async fn handle_task_completion(&self, mut task: Task, result: Result<TaskResult, AgentError>) {
        let _task_result = match result {
            Ok(result) => {
                task.status = TaskStatus::Completed;
                task.completed_at = Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
                task.output_data = result.output_data.clone();
                task.actual_duration_ms = Some(result.processing_time_ms);
                
                let _ = self.event_sender.send(TaskEvent::TaskCompleted(task.id.clone(), result));
                Ok(())
            }
            Err(error) => {
                task.status = TaskStatus::Failed;
                task.completed_at = Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
                task.error_message = Some(error.to_string());
                
                let _ = self.event_sender.send(TaskEvent::TaskFailed(task.id.clone(), error.to_string()));
                Err(error)
            }
        };
        
        // 从运行任务中移除
        {
            let mut running_tasks = self.running_tasks.write().await;
            running_tasks.remove(&task.id);
        }
        
        // 添加到已完成任务
        {
            let mut completed_tasks = self.completed_tasks.write().await;
            completed_tasks.push(task);
        }
        
        // 注意：这里不递归调用try_assign_tasks以避免无限递归
        // 在实际应用中，可以通过事件系统或其他机制来处理新任务分配
    }
    
    pub async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        // 检查运行中的任务
        {
            let running_tasks = self.running_tasks.read().await;
            if let Some(task) = running_tasks.get(task_id) {
                return Some(task.status.clone());
            }
        }
        
        // 检查已完成的任务
        {
            let completed_tasks = self.completed_tasks.read().await;
            if let Some(task) = completed_tasks.iter().find(|t| t.id == task_id) {
                return Some(task.status.clone());
            }
        }
        
        // 检查队列中的任务
        {
            let queue = self.task_queue.read().await;
            if let Some(task) = queue.iter().find(|t| t.id == task_id) {
                return Some(task.status.clone());
            }
        }
        
        None
    }
    
    pub async fn get_system_stats(&self) -> SystemStats {
        let queue = self.task_queue.read().await;
        let running_tasks = self.running_tasks.read().await;
        let completed_tasks = self.completed_tasks.read().await;
        let available_agents = self.agent_registry.get_available_agents().await;
        
        SystemStats {
            pending_tasks: queue.len(),
            running_tasks: running_tasks.len(),
            completed_tasks: completed_tasks.len(),
            available_agents: available_agents.len(),
            total_agents: self.agent_registry.agent_info.read().await.len(),
        }
    }
    
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<TaskEvent> {
        self.event_sender.subscribe()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub pending_tasks: usize,
    pub running_tasks: usize,
    pub completed_tasks: usize,
    pub available_agents: usize,
    pub total_agents: usize,
}

// 克隆实现
impl Clone for TaskScheduler {
    fn clone(&self) -> Self {
        Self {
            task_queue: Arc::clone(&self.task_queue),
            running_tasks: Arc::clone(&self.running_tasks),
            completed_tasks: Arc::clone(&self.completed_tasks),
            agent_registry: Arc::clone(&self.agent_registry),
            event_sender: self.event_sender.clone(),
        }
    }
}

// 示例代理实现：网页导航代理
pub struct WebNavigatorAgent {
    info: AgentInfo,
    http_client: reqwest::Client,
}

impl WebNavigatorAgent {
    pub fn new() -> Self {
        let info = AgentInfo {
            id: Uuid::new_v4().to_string(),
            name: "Web Navigator Agent".to_string(),
            agent_type: AgentType::WebNavigator,
            capabilities: vec![
                AgentCapability {
                    name: "web_scraping".to_string(),
                    description: "网页内容抓取".to_string(),
                    input_types: vec!["url".to_string()],
                    output_types: vec!["html_content".to_string(), "text_content".to_string()],
                    max_concurrent_tasks: 5,
                    estimated_processing_time_ms: 2000,
                },
                AgentCapability {
                    name: "form_interaction".to_string(),
                    description: "表单交互".to_string(),
                    input_types: vec!["form_data".to_string()],
                    output_types: vec!["form_result".to_string()],
                    max_concurrent_tasks: 3,
                    estimated_processing_time_ms: 3000,
                },
            ],
            status: AgentStatus::Available,
            current_tasks: Vec::new(),
            completed_tasks: 0,
            failed_tasks: 0,
            average_processing_time_ms: 0.0,
            last_heartbeat: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };
        
        Self {
            info,
            http_client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Agent for WebNavigatorAgent {
    async fn get_info(&self) -> AgentInfo {
        self.info.clone()
    }
    
    async fn can_handle_task(&self, task: &Task) -> bool {
        matches!(task.agent_type, AgentType::WebNavigator) && 
        self.info.status == AgentStatus::Available &&
        self.info.current_tasks.len() < self.info.capabilities[0].max_concurrent_tasks
    }
    
    async fn execute_task(&self, task: &Task) -> Result<TaskResult, AgentError> {
        let start_time = std::time::Instant::now();
        
        match task.name.as_str() {
            "web_scraping" => {
                if let Some(url) = task.input_data.get("url") {
                    if let Some(url_str) = url.as_str() {
                        match self.http_client.get(url_str).send().await {
                            Ok(response) => {
                                match response.text().await {
                                    Ok(html_content) => {
                                        let mut output_data = HashMap::new();
                                        output_data.insert("html_content".to_string(), serde_json::Value::String(html_content.clone()));
                                        output_data.insert("text_content".to_string(), serde_json::Value::String(html_content));
                                        
                                        let processing_time = start_time.elapsed().as_millis() as u64;
                                        
                                        Ok(TaskResult {
                                            task_id: task.id.clone(),
                                            success: true,
                                            output_data: Some(output_data),
                                            processing_time_ms: processing_time,
                                            error_message: None,
                                            metrics: HashMap::new(),
                                        })
                                    }
                                    Err(e) => Err(AgentError::TaskExecutionError(e.to_string()))
                                }
                            }
                            Err(e) => Err(AgentError::TaskExecutionError(e.to_string()))
                        }
                    } else {
                        Err(AgentError::TaskExecutionError("Invalid URL format".to_string()))
                    }
                } else {
                    Err(AgentError::TaskExecutionError("Missing URL parameter".to_string()))
                }
            }
            _ => Err(AgentError::InsufficientCapability(format!("Unknown task: {}", task.name)))
        }
    }
    
    async fn get_capabilities(&self) -> Vec<AgentCapability> {
        self.info.capabilities.clone()
    }
    
    async fn heartbeat(&self) -> Result<(), AgentError> {
        // 更新心跳时间
        Ok(())
    }
}

// 示例代理实现：数据处理代理
pub struct DataProcessorAgent {
    info: AgentInfo,
}

impl DataProcessorAgent {
    pub fn new() -> Self {
        let info = AgentInfo {
            id: Uuid::new_v4().to_string(),
            name: "Data Processor Agent".to_string(),
            agent_type: AgentType::DataProcessor,
            capabilities: vec![
                AgentCapability {
                    name: "data_cleaning".to_string(),
                    description: "数据清洗".to_string(),
                    input_types: vec!["raw_data".to_string()],
                    output_types: vec!["cleaned_data".to_string()],
                    max_concurrent_tasks: 10,
                    estimated_processing_time_ms: 1000,
                },
                AgentCapability {
                    name: "data_analysis".to_string(),
                    description: "数据分析".to_string(),
                    input_types: vec!["dataset".to_string()],
                    output_types: vec!["analysis_result".to_string()],
                    max_concurrent_tasks: 5,
                    estimated_processing_time_ms: 5000,
                },
            ],
            status: AgentStatus::Available,
            current_tasks: Vec::new(),
            completed_tasks: 0,
            failed_tasks: 0,
            average_processing_time_ms: 0.0,
            last_heartbeat: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };
        
        Self { info }
    }
}

#[async_trait]
impl Agent for DataProcessorAgent {
    async fn get_info(&self) -> AgentInfo {
        self.info.clone()
    }
    
    async fn can_handle_task(&self, task: &Task) -> bool {
        matches!(task.agent_type, AgentType::DataProcessor) && 
        self.info.status == AgentStatus::Available &&
        self.info.current_tasks.len() < self.info.capabilities[0].max_concurrent_tasks
    }
    
    async fn execute_task(&self, task: &Task) -> Result<TaskResult, AgentError> {
        let start_time = std::time::Instant::now();
        
        match task.name.as_str() {
            "data_cleaning" => {
                // 模拟数据清洗
                tokio::time::sleep(Duration::from_millis(500)).await;
                
                let mut output_data = HashMap::new();
                output_data.insert("cleaned_data".to_string(), serde_json::Value::String("Cleaned data".to_string()));
                
                let processing_time = start_time.elapsed().as_millis() as u64;
                
                Ok(TaskResult {
                    task_id: task.id.clone(),
                    success: true,
                    output_data: Some(output_data),
                    processing_time_ms: processing_time,
                    error_message: None,
                    metrics: HashMap::new(),
                })
            }
            "data_analysis" => {
                // 模拟数据分析
                tokio::time::sleep(Duration::from_millis(2000)).await;
                
                let mut output_data = HashMap::new();
                output_data.insert("analysis_result".to_string(), serde_json::Value::String("Analysis complete".to_string()));
                
                let processing_time = start_time.elapsed().as_millis() as u64;
                
                Ok(TaskResult {
                    task_id: task.id.clone(),
                    success: true,
                    output_data: Some(output_data),
                    processing_time_ms: processing_time,
                    error_message: None,
                    metrics: HashMap::new(),
                })
            }
            _ => Err(AgentError::InsufficientCapability(format!("Unknown task: {}", task.name)))
        }
    }
    
    async fn get_capabilities(&self) -> Vec<AgentCapability> {
        self.info.capabilities.clone()
    }
    
    async fn heartbeat(&self) -> Result<(), AgentError> {
        Ok(())
    }
}

// 主函数演示
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Agentic Web架构演示");
    println!("================================");
    
    // 创建代理注册表
    let agent_registry = Arc::new(AgentRegistry::new());
    
    // 创建任务调度器
    let scheduler = Arc::new(TaskScheduler::new(Arc::clone(&agent_registry)));
    
    // 注册代理
    let web_agent = Arc::new(WebNavigatorAgent::new());
    let data_agent = Arc::new(DataProcessorAgent::new());
    
    agent_registry.register_agent(web_agent).await?;
    agent_registry.register_agent(data_agent).await?;
    
    println!("📋 已注册代理:");
    let available_agents = agent_registry.get_available_agents().await;
    for agent in &available_agents {
        println!("  - {} ({})", agent.name, agent.id);
        for capability in &agent.capabilities {
            println!("    * {}: {}", capability.name, capability.description);
        }
    }
    
    // 创建任务
    let web_task = Task {
        id: String::new(), // 将由调度器生成
        name: "web_scraping".to_string(),
        description: "抓取网页内容".to_string(),
        agent_type: AgentType::WebNavigator,
        priority: TaskPriority::High,
        status: TaskStatus::Pending,
        input_data: {
            let mut data = HashMap::new();
            data.insert("url".to_string(), serde_json::Value::String("https://example.com".to_string()));
            data
        },
        output_data: None,
        created_at: 0,
        started_at: None,
        completed_at: None,
        dependencies: Vec::new(),
        estimated_duration_ms: 2000,
        actual_duration_ms: None,
        error_message: None,
    };
    
    let data_task = Task {
        id: String::new(),
        name: "data_cleaning".to_string(),
        description: "清洗数据".to_string(),
        agent_type: AgentType::DataProcessor,
        priority: TaskPriority::Medium,
        status: TaskStatus::Pending,
        input_data: {
            let mut data = HashMap::new();
            data.insert("raw_data".to_string(), serde_json::Value::String("Raw data to clean".to_string()));
            data
        },
        output_data: None,
        created_at: 0,
        started_at: None,
        completed_at: None,
        dependencies: Vec::new(),
        estimated_duration_ms: 1000,
        actual_duration_ms: None,
        error_message: None,
    };
    
    // 提交任务
    println!("\n🚀 提交任务:");
    let web_task_id = scheduler.submit_task(web_task).await?;
    let data_task_id = scheduler.submit_task(data_task).await?;
    
    println!("  - 网页抓取任务: {}", web_task_id);
    println!("  - 数据清洗任务: {}", data_task_id);
    
    // 监听事件
    let mut event_receiver = scheduler.subscribe_to_events();
    let _scheduler_clone = Arc::clone(&scheduler);
    
    tokio::spawn(async move {
        while let Ok(event) = event_receiver.recv().await {
            match event {
                TaskEvent::TaskCreated(task) => {
                    println!("📝 任务已创建: {} ({})", task.name, task.id);
                }
                TaskEvent::TaskStarted(task_id) => {
                    println!("▶️ 任务已开始: {}", task_id);
                }
                TaskEvent::TaskCompleted(task_id, result) => {
                    println!("✅ 任务已完成: {} (耗时: {}ms)", task_id, result.processing_time_ms);
                }
                TaskEvent::TaskFailed(task_id, error) => {
                    println!("❌ 任务失败: {} - {}", task_id, error);
                }
                TaskEvent::AgentAssigned(task_id, agent_id) => {
                    println!("🤝 代理已分配: 任务 {} -> 代理 {}", task_id, agent_id);
                }
            }
        }
    });
    
    // 等待任务完成
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // 显示系统统计
    let stats = scheduler.get_system_stats().await;
    println!("\n📊 系统统计:");
    println!("  - 待处理任务: {}", stats.pending_tasks);
    println!("  - 运行中任务: {}", stats.running_tasks);
    println!("  - 已完成任务: {}", stats.completed_tasks);
    println!("  - 可用代理: {}", stats.available_agents);
    println!("  - 总代理数: {}", stats.total_agents);
    
    // 检查任务状态
    println!("\n🔍 任务状态检查:");
    if let Some(status) = scheduler.get_task_status(&web_task_id).await {
        println!("  - 网页抓取任务: {:?}", status);
    }
    if let Some(status) = scheduler.get_task_status(&data_task_id).await {
        println!("  - 数据清洗任务: {:?}", status);
    }
    
    println!("\n✅ Agentic Web架构演示完成！");
    println!("\n🌟 Agentic Web的优势：");
    println!("   - AI代理驱动的智能交互");
    println!("   - 自主任务规划和执行");
    println!("   - 代理间协作和协调");
    println!("   - 智能性、交互性和经济性");
    println!("   - 可扩展的代理生态系统");
    println!("   - 实时任务调度和监控");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_agent_registry() {
        let registry = AgentRegistry::new();
        let agent = Arc::new(WebNavigatorAgent::new());
        
        assert!(registry.register_agent(agent).await.is_ok());
        assert_eq!(registry.get_available_agents().await.len(), 1);
    }
    
    #[tokio::test]
    async fn test_task_scheduler() {
        let registry = Arc::new(AgentRegistry::new());
        let agent = Arc::new(DataProcessorAgent::new());
        registry.register_agent(agent).await.unwrap();
        
        let scheduler = Arc::new(TaskScheduler::new(registry));
        
        let task = Task {
            id: String::new(),
            name: "data_cleaning".to_string(),
            description: "Test task".to_string(),
            agent_type: AgentType::DataProcessor,
            priority: TaskPriority::Medium,
            status: TaskStatus::Pending,
            input_data: HashMap::new(),
            output_data: None,
            created_at: 0,
            started_at: None,
            completed_at: None,
            dependencies: Vec::new(),
            estimated_duration_ms: 1000,
            actual_duration_ms: None,
            error_message: None,
        };
        
        let task_id = scheduler.submit_task(task).await.unwrap();
        assert!(!task_id.is_empty());
    }
    
    #[tokio::test]
    async fn test_web_navigator_agent() {
        let agent = WebNavigatorAgent::new();
        let info = agent.get_info().await;
        
        assert_eq!(info.agent_type, AgentType::WebNavigator);
        assert!(!info.capabilities.is_empty());
    }
}
