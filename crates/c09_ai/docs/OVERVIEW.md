# 概览：人工智能（c19_ai）

> 对齐声明：术语统一见 `docs/02_knowledge_structures/2025_ai_知识术语表_GLOSSARY.md`；指标与报告口径见 `docs/03_tech_trends/2025_ai_rust_technology_trends_comprehensive_report.md` §Z.7；监控指标需由 `reports/` CSV 通过 `scripts/repro/` 再生。

本模块聚焦传统机器学习、深度学习、图网络、强化学习、时序与多模态等主题，并整合推理/训练/监控链路。

---

## 目录导航

- 顶层与说明
  - 顶层说明：`README.md`
  - 项目总结：`PROJECT_COMPLETION_REPORT_2025.md`

- 示例与场景
  - `examples/{basic_ml,deep_learning,nlp_pipeline,graph_neural_network,reinforcement_learning,vector_search,...}.rs`

- 源码结构
  - `src/machine_learning/*`、`src/deep_learning/*`、`src/llm/*`
  - `src/model_management/*`、`src/monitoring/*`、`src/pipelines/*`
  - `src/time_series/*`、`src/graph_neural_networks/*`、`src/vector_search/*`

---

## 快速开始

1) 运行 `examples/basic_ml.rs` 与 `examples/deep_learning.rs`
2) 查看 `src/llm/*` 下的聊天/补全/嵌入接口
3) 使用 `src/monitoring/*` 观察性能/指标

---

## 🔗 快速导航

- 模型理论：`../../formal_rust/language/18_model/01_model_theory.md`
- 分布式系统：`../c20_distributed/docs/FAQ.md`
- WebAssembly：`../../formal_rust/language/16_webassembly/FAQ.md`
- IoT系统：`../../formal_rust/language/17_iot/FAQ.md`
- 区块链：`../../formal_rust/language/15_blockchain/FAQ.md`

## 模型注册与部署（端到端示例）

### 注册流程

**模型产物管理**：

- **架构定义**：模型结构、层配置、激活函数
- **权重存储**：模型参数、量化权重（INT8/INT4）、检查点
- **版本控制**：语义化版本（major.minor.patch）、版本标签
- **数字签名**：模型完整性校验、防篡改验证

**元数据管理**：

- **任务类型**：分类、回归、生成、检索等
- **性能指标**：准确率、召回率、F1、延迟、吞吐量
- **依赖关系**：框架版本、库版本、硬件要求
- **训练信息**：数据集、训练参数、训练时长

**存储架构**：

- **对象存储**：S3/GCS兼容存储，支持版本化
- **索引数据库**：PostgreSQL/MySQL，支持标签检索
- **元数据存储**：JSON/YAML格式，便于查询和更新

### 部署与路由

**推理服务部署**：

- **多副本部署**：Kubernetes Deployment，支持水平扩展
- **健康检查**：liveness/readiness探针，自动故障恢复
- **灰度发布**：金丝雀部署，小流量验证新模型
- **A/B测试**：多版本并行运行，按比例分流

**路由策略**：

- **标签路由**：按模型标签（生产/实验/测试）路由
- **版本路由**：按模型版本路由，支持版本回滚
- **权重路由**：按流量权重分配，支持渐进式切换
- **智能路由**：基于请求特征（延迟、成本、质量）动态选择

**实现示例**：

```rust
// 模型注册
pub struct ModelRegistry {
    storage: Arc<ObjectStorage>,
    metadata_db: Arc<Database>,
}

impl ModelRegistry {
    pub async fn register_model(
        &self,
        model: ModelArtifact,
        metadata: ModelMetadata
    ) -> Result<ModelVersion> {
        // 1. 上传模型文件到对象存储
        let storage_path = self.storage.upload(&model).await?;
        
        // 2. 计算模型签名
        let signature = self.compute_signature(&model).await?;
        
        // 3. 创建版本记录
        let version = ModelVersion::new(
            model.name.clone(),
            model.version.clone(),
            storage_path,
            signature,
        );
        
        // 4. 存储元数据
        self.metadata_db.insert_metadata(&version, &metadata).await?;
        
        Ok(version)
    }
}

// 模型路由
pub struct ModelRouter {
    models: HashMap<String, Vec<ModelEndpoint>>,
    routing_strategy: RoutingStrategy,
}

impl ModelRouter {
    pub async fn route_request(
        &self,
        request: InferenceRequest
    ) -> Result<ModelEndpoint> {
        match &self.routing_strategy {
            RoutingStrategy::TagBased(tag) => {
                self.find_by_tag(tag).await
            }
            RoutingStrategy::VersionBased(version) => {
                self.find_by_version(version).await
            }
            RoutingStrategy::WeightBased(weights) => {
                self.select_by_weight(weights).await
            }
            RoutingStrategy::SmartRouting => {
                self.smart_route(&request).await
            }
        }
    }
}
```

**观测与监控**：

- **性能指标**：QPS、P95/P99延迟、错误率、吞吐量
- **SLO监控**：服务级别目标监控，自动告警
- **资源监控**：CPU、内存、GPU利用率
- **成本监控**：推理成本、存储成本、计算成本

**与 `c18_model` 互链**：

- **形式化验证**：将形式化性质（幂等、单调、上界时延）映射为属性测试与 CI 门禁
- **部署前验证**：部署前跑通 `c18_model` 的验证用例，生成合规报告
- **持续验证**：在CI/CD流程中集成模型验证，确保模型符合形式化规范

**参考**：详见 `10_production_deployment/README.md` 和 `docs/05_practical_guides/2025_rust_ai_practical_guide.md` §8
