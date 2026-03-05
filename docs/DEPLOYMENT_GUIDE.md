# 🚀 AI-Rust 部署运维指南

## 概述

本指南提供AI-Rust项目的完整部署和运维方案，涵盖本地开发、Docker容器化、Kubernetes编排、监控告警等内容。

**更新日期**: 2025年12月3日
**支持环境**: Linux, macOS, Windows
**Rust版本**: 1.90+

---

## 📋 目录

- [🚀 AI-Rust 部署运维指南](#-ai-rust-部署运维指南)
  - [概述](#概述)
  - [📋 目录](#-目录)
  - [🔧 环境准备](#-环境准备)
    - [系统要求](#系统要求)
    - [软件依赖](#软件依赖)
  - [💻 本地部署](#-本地部署)
    - [1. 克隆项目](#1-克隆项目)
    - [2. 配置环境变量](#2-配置环境变量)
    - [3. 构建项目](#3-构建项目)
    - [4. 运行服务](#4-运行服务)
    - [5. 验证部署](#5-验证部署)
  - [🐳 Docker部署](#-docker部署)
    - [1. 构建Docker镜像](#1-构建docker镜像)
    - [2. 构建和运行](#2-构建和运行)
    - [3. Docker Compose部署](#3-docker-compose部署)
    - [4. Nginx配置](#4-nginx配置)
  - [☸️ Kubernetes部署](#️-kubernetes部署)
    - [1. Kubernetes配置](#1-kubernetes配置)
    - [2. 水平自动扩缩容](#2-水平自动扩缩容)
    - [3. 部署到Kubernetes](#3-部署到kubernetes)
  - [📊 监控和日志](#-监控和日志)
    - [1. Prometheus监控](#1-prometheus监控)
    - [2. Grafana仪表板](#2-grafana仪表板)
    - [3. 结构化日志](#3-结构化日志)
    - [4. 日志聚合](#4-日志聚合)
  - [⚡ 性能调优](#-性能调优)
    - [1. 系统级优化](#1-系统级优化)
    - [2. 应用级优化](#2-应用级优化)
    - [3. 资源限制](#3-资源限制)
  - [🔍 故障排查](#-故障排查)
    - [1. 常见问题](#1-常见问题)
      - [高内存使用](#高内存使用)
      - [高CPU使用](#高cpu使用)
      - [请求超时](#请求超时)
    - [2. 调试技巧](#2-调试技巧)
  - [🔒 安全加固](#-安全加固)
    - [1. HTTPS配置](#1-https配置)
    - [2. 认证和授权](#2-认证和授权)
    - [3. 速率限制](#3-速率限制)
  - [📝 部署检查清单](#-部署检查清单)
    - [部署前](#部署前)
    - [部署中](#部署中)
    - [部署后](#部署后)
  - [🔗 参考资源](#-参考资源)

---

## 🔧 环境准备

### 系统要求

**最低配置**:

- CPU: 4核
- 内存: 8GB
- 存储: 50GB
- 操作系统: Ubuntu 20.04+ / macOS 12+ / Windows 10+

**推荐配置**:

- CPU: 16核 (AVX2/AVX512支持)
- 内存: 32GB
- 存储: 200GB SSD
- GPU: NVIDIA GPU (可选，用于加速)

---

### 软件依赖

```bash
# Rust (1.90+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup update

# 系统依赖 (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    git

# 系统依赖 (macOS)
brew install pkg-config openssl

# 系统依赖 (Windows)
# 安装 Visual Studio Build Tools
# 下载地址: https://visualstudio.microsoft.com/downloads/
```

---

## 💻 本地部署

### 1. 克隆项目

```bash
git clone https://github.com/your-org/ai-rust.git
cd ai-rust
```

---

### 2. 配置环境变量

```bash
# 创建配置文件
cp .env.example .env

# 编辑配置
cat > .env <<EOF
# 服务器配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
WORKERS=4

# 模型配置
MODEL_CACHE_DIR=./models
DEFAULT_MODEL=embedding

# 日志配置
RUST_LOG=info
LOG_FORMAT=json

# 数据库配置 (可选)
DATABASE_URL=postgresql://localhost/ai_rust

# Redis配置 (可选)
REDIS_URL=redis://localhost:6379
EOF
```

---

### 3. 构建项目

```bash
# 开发构建
cargo build

# 发布构建
cargo build --release

# 验证构建
./target/release/ai_service --version
```

---

### 4. 运行服务

```bash
# 直接运行
./target/release/ai_service

# 使用配置文件
./target/release/ai_service --config config.toml

# 后台运行
nohup ./target/release/ai_service > logs/app.log 2>&1 &
```

---

### 5. 验证部署

```bash
# 健康检查
curl http://localhost:8080/health

# 推理测试
curl -X POST http://localhost:8080/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embedding",
    "input": "Hello, world!"
  }'
```

---

## 🐳 Docker部署

### 1. 构建Docker镜像

**Dockerfile**:

```dockerfile
# 构建阶段
FROM rust:1.90-slim as builder

WORKDIR /app

# 安装依赖
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY crates ./crates

# 构建发布版本
RUN cargo build --release --bin ai_service

# 运行阶段
FROM debian:bookworm-slim

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd -m -u 1000 airust

# 复制二进制文件
COPY --from=builder /app/target/release/ai_service /usr/local/bin/

# 创建数据目录
RUN mkdir -p /data/models && chown -R airust:airust /data

# 切换用户
USER airust

# 工作目录
WORKDIR /app

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["ai_service"]
```

---

### 2. 构建和运行

```bash
# 构建镜像
docker build -t ai-rust:latest .

# 运行容器
docker run -d \
  --name ai-rust \
  -p 8080:8080 \
  -v $(pwd)/models:/data/models \
  -e RUST_LOG=info \
  ai-rust:latest

# 查看日志
docker logs -f ai-rust

# 进入容器
docker exec -it ai-rust bash
```

---

### 3. Docker Compose部署

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  ai-service:
    build: .
    image: ai-rust:latest
    container_name: ai-rust
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./models:/data/models
      - ./logs:/app/logs
    environment:
      - RUST_LOG=info
      - SERVER_PORT=8080
      - WORKERS=4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
    networks:
      - ai-network

  # 数据库 (可选)
  postgres:
    image: postgres:15-alpine
    container_name: postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=ai_rust
      - POSTGRES_USER=airust
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - ai-network

  # Redis缓存 (可选)
  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - ai-network

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - ai-service
    networks:
      - ai-network

volumes:
  postgres-data:
  redis-data:

networks:
  ai-network:
    driver: bridge
```

**启动服务**:

```bash
# 启动所有服务
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 停止并删除数据
docker-compose down -v
```

---

### 4. Nginx配置

**nginx/nginx.conf**:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream ai_backend {
        least_conn;
        server ai-service:8080 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # 请求大小限制
        client_max_body_size 10M;

        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # 代理到后端
        location / {
            proxy_pass http://ai_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # 健康检查
        location /health {
            proxy_pass http://ai_backend/health;
            access_log off;
        }

        # 静态文件
        location /static {
            alias /var/www/static;
            expires 7d;
        }
    }
}
```

---

## ☸️ Kubernetes部署

### 1. Kubernetes配置

**k8s/deployment.yaml**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-rust
  labels:
    app: ai-rust
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-rust
  template:
    metadata:
      labels:
        app: ai-rust
    spec:
      containers:
      - name: ai-rust
        image: ai-rust:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: RUST_LOG
          value: "info"
        - name: SERVER_PORT
          value: "8080"
        - name: WORKERS
          value: "4"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /data/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ai-rust-service
spec:
  selector:
    app: ai-rust
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  accessModes:
  - ReadOnlyMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

---

### 2. 水平自动扩缩容

**k8s/hpa.yaml**:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-rust-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-rust
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

### 3. 部署到Kubernetes

```bash
# 创建命名空间
kubectl create namespace ai-rust

# 应用配置
kubectl apply -f k8s/deployment.yaml -n ai-rust
kubectl apply -f k8s/hpa.yaml -n ai-rust

# 查看部署状态
kubectl get deployments -n ai-rust
kubectl get pods -n ai-rust
kubectl get services -n ai-rust

# 查看日志
kubectl logs -f deployment/ai-rust -n ai-rust

# 扩缩容
kubectl scale deployment ai-rust --replicas=5 -n ai-rust

# 滚动更新
kubectl set image deployment/ai-rust ai-rust=ai-rust:v2 -n ai-rust
kubectl rollout status deployment/ai-rust -n ai-rust

# 回滚
kubectl rollout undo deployment/ai-rust -n ai-rust
```

---

## 📊 监控和日志

### 1. Prometheus监控

**monitoring/prometheus.yml**:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-rust'
    static_configs:
      - targets: ['ai-service:8080']
    metrics_path: '/metrics'
```

**在应用中暴露指标**:

```rust
use prometheus::{Encoder, TextEncoder, Registry};
use axum::{routing::get, Router};

pub fn metrics_handler(registry: Registry) -> impl Handler {
    move || async move {
        let encoder = TextEncoder::new();
        let metric_families = registry.gather();
        let mut buffer = vec![];
        encoder.encode(&metric_families, &mut buffer).unwrap();

        (
            [(header::CONTENT_TYPE, encoder.format_type())],
            buffer
        )
    }
}

// 添加到路由
let app = Router::new()
    .route("/metrics", get(metrics_handler(registry)));
```

---

### 2. Grafana仪表板

**monitoring/grafana/dashboard.json**:

```json
{
  "dashboard": {
    "title": "AI-Rust Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(http_requests_total[5m])"
        }]
      },
      {
        "title": "Latency P95",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(http_requests_failed_total[5m])"
        }]
      }
    ]
  }
}
```

---

### 3. 结构化日志

```rust
use tracing::{info, error, instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn setup_logging() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer().json())
        .init();
}

#[instrument(skip(model))]
pub async fn infer_with_logging(
    model: &Model,
    input: Input,
) -> Result<Output> {
    info!("Starting inference");

    let result = model.infer(input).await;

    match &result {
        Ok(_) => info!("Inference successful"),
        Err(e) => error!("Inference failed: {}", e),
    }

    result
}
```

---

### 4. 日志聚合

**使用ELK Stack**:

```yaml
# docker-compose.yml
services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.11.0
    volumes:
      - ./logstash/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
```

---

## ⚡ 性能调优

### 1. 系统级优化

```bash
# 增加文件描述符限制
ulimit -n 65535

# 优化TCP设置
sudo sysctl -w net.core.somaxconn=1024
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=2048

# 启用透明大页
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

---

### 2. 应用级优化

**配置文件优化**:

```toml
[server]
workers = 16  # CPU核心数
max_connections = 10000
keepalive_timeout = 75

[inference]
batch_size = 32
timeout_seconds = 30
cache_size = 1000

[performance]
thread_pool_size = 8
io_threads = 4
```

---

### 3. 资源限制

**使用cgroups限制资源**:

```bash
# 创建cgroup
sudo cgcreate -g memory,cpu:ai-rust

# 设置内存限制 (8GB)
sudo cgset -r memory.limit_in_bytes=8589934592 ai-rust

# 设置CPU限制 (4核)
sudo cgset -r cpu.shares=4096 ai-rust

# 在cgroup中运行
sudo cgexec -g memory,cpu:ai-rust ./target/release/ai_service
```

---

## 🔍 故障排查

### 1. 常见问题

#### 高内存使用

```bash
# 检查内存使用
ps aux | grep ai_service
top -p $(pgrep ai_service)

# 使用heaptrack分析
heaptrack ./target/release/ai_service
heaptrack_gui heaptrack.ai_service.*
```

**解决方案**:

- 启用对象池
- 减少批处理大小
- 增加缓存驱逐策略

---

#### 高CPU使用

```bash
# 使用perf分析
perf record -g -p $(pgrep ai_service)
perf report
```

**解决方案**:

- 优化热点代码
- 启用SIMD
- 减少锁竞争

---

#### 请求超时

```bash
# 检查日志
tail -f logs/app.log | grep "timeout"

# 检查网络延迟
ping ai-service
traceroute ai-service
```

**解决方案**:

- 增加超时设置
- 优化推理速度
- 启用批处理

---

### 2. 调试技巧

```bash
# 启用详细日志
RUST_LOG=debug ./target/release/ai_service

# 使用gdb调试
gdb ./target/release/ai_service
(gdb) run
(gdb) bt  # 查看堆栈

# 使用rust-gdb
rust-gdb ./target/release/ai_service
```

---

## 🔒 安全加固

### 1. HTTPS配置

```bash
# 生成SSL证书
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes
```

**Nginx HTTPS配置**:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://ai_backend;
    }
}
```

---

### 2. 认证和授权

```rust
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
}

pub async fn auth_middleware(
    req: Request,
    next: Next,
) -> Result<Response> {
    let token = req.headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .ok_or(AuthError::MissingToken)?;

    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(SECRET.as_ref()),
        &Validation::default(),
    )?;

    Ok(next.run(req).await)
}
```

---

### 3. 速率限制

```rust
use governor::{Quota, RateLimiter};

pub struct RateLimitMiddleware {
    limiter: Arc<RateLimiter<String, DefaultDirectRateLimiter>>,
}

impl RateLimitMiddleware {
    pub fn new(requests_per_second: u32) -> Self {
        let quota = Quota::per_second(nonzero!(requests_per_second));
        let limiter = Arc::new(RateLimiter::direct(quota));

        Self { limiter }
    }

    pub async fn check(&self, key: String) -> Result<()> {
        self.limiter.check_key(&key)
            .map_err(|_| Error::RateLimitExceeded)?;
        Ok(())
    }
}
```

---

## 📝 部署检查清单

### 部署前

- [ ] 代码审查完成
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 性能测试通过
- [ ] 安全扫描通过

### 部署中

- [ ] 备份数据
- [ ] 蓝绿部署/金丝雀发布
- [ ] 健康检查
- [ ] 监控告警

### 部署后

- [ ] 验证功能
- [ ] 检查日志
- [ ] 监控指标
- [ ] 回滚准备

---

## 🔗 参考资源

- [Rust部署指南](https://doc.rust-lang.org/book/ch14-04-installing-binaries.html)
- [Docker最佳实践](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes文档](https://kubernetes.io/docs/)
- [Prometheus文档](https://prometheus.io/docs/)

---

*最后更新: 2025年12月3日*
*维护者: AI-Rust项目团队*
