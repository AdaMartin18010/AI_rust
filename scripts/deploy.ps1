# AI-Rust项目部署脚本
# 用于Windows PowerShell

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "test", "deploy", "start", "stop", "restart", "status", "logs", "cleanup", "help")]
    [string]$Action = "deploy"
)

# 颜色函数
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# 检查依赖
function Test-Dependencies {
    Write-Info "检查系统依赖..."
    
    # 检查Docker
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker未安装，请先安装Docker Desktop"
        exit 1
    }
    
    # 检查Docker Compose
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    }
    
    # 检查Rust
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Error "Rust未安装，请先安装Rust"
        exit 1
    }
    
    Write-Success "所有依赖检查通过"
}

# 构建项目
function Build-Project {
    Write-Info "构建项目..."
    
    # 清理之前的构建
    cargo clean
    
    # 构建项目
    cargo build --release
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "项目构建成功"
    } else {
        Write-Error "项目构建失败"
        exit 1
    }
}

# 运行测试
function Test-Project {
    Write-Info "运行测试..."
    
    cargo test --release
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "所有测试通过"
    } else {
        Write-Error "测试失败"
        exit 1
    }
}

# 构建Docker镜像
function Build-DockerImages {
    Write-Info "构建Docker镜像..."
    
    # 构建主服务镜像
    docker build -t ai-rust-web:latest .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker镜像构建成功"
    } else {
        Write-Error "Docker镜像构建失败"
        exit 1
    }
}

# 启动服务
function Start-Services {
    Write-Info "启动服务..."
    
    # 停止现有服务
    docker-compose down
    
    # 启动服务
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "服务启动成功"
    } else {
        Write-Error "服务启动失败"
        exit 1
    }
}

# 等待服务就绪
function Wait-ForServices {
    Write-Info "等待服务就绪..."
    
    # 等待主服务
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "主服务已就绪"
                break
            }
        } catch {
            # 忽略错误，继续等待
        }
        
        if ($attempt -eq $maxAttempts) {
            Write-Error "主服务启动超时"
            exit 1
        }
        
        Write-Info "等待主服务启动... ($attempt/$maxAttempts)"
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    # 等待数据库
    $attempt = 1
    while ($attempt -le $maxAttempts) {
        try {
            $result = docker-compose exec -T postgres pg_isready -U ai_rust 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "数据库已就绪"
                break
            }
        } catch {
            # 忽略错误，继续等待
        }
        
        if ($attempt -eq $maxAttempts) {
            Write-Error "数据库启动超时"
            exit 1
        }
        
        Write-Info "等待数据库启动... ($attempt/$maxAttempts)"
        Start-Sleep -Seconds 2
        $attempt++
    }
}

# 初始化数据库
function Initialize-Database {
    Write-Info "初始化数据库..."
    
    # 等待数据库完全启动
    Start-Sleep -Seconds 5
    
    # 执行数据库初始化脚本
    docker-compose exec -T postgres psql -U ai_rust -d ai_rust -f /docker-entrypoint-initdb.d/init-db.sql
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "数据库初始化成功"
    } else {
        Write-Warning "数据库初始化可能失败，请检查日志"
    }
}

# 健康检查
function Test-Health {
    Write-Info "执行健康检查..."
    
    # 检查主服务
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "主服务健康检查通过"
        }
    } catch {
        Write-Error "主服务健康检查失败"
        exit 1
    }
    
    # 检查Nginx
    try {
        $response = Invoke-WebRequest -Uri "http://localhost/nginx-health" -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "Nginx健康检查通过"
        }
    } catch {
        Write-Warning "Nginx健康检查失败"
    }
    
    # 检查Prometheus
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "Prometheus健康检查通过"
        }
    } catch {
        Write-Warning "Prometheus健康检查失败"
    }
    
    # 检查Grafana
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "Grafana健康检查通过"
        }
    } catch {
        Write-Warning "Grafana健康检查失败"
    }
}

# 显示服务状态
function Show-Status {
    Write-Info "服务状态："
    
    Write-Host ""
    Write-Host "=== Docker容器状态 ===" -ForegroundColor Cyan
    docker-compose ps
    
    Write-Host ""
    Write-Host "=== 服务访问地址 ===" -ForegroundColor Cyan
    Write-Host "主服务: http://localhost:8080"
    Write-Host "Nginx: http://localhost"
    Write-Host "Prometheus: http://localhost:9090"
    Write-Host "Grafana: http://localhost:3000 (admin/admin)"
    Write-Host "Jaeger: http://localhost:16686"
    
    Write-Host ""
    Write-Host "=== 数据库连接信息 ===" -ForegroundColor Cyan
    Write-Host "Host: localhost"
    Write-Host "Port: 5432"
    Write-Host "Database: ai_rust"
    Write-Host "Username: ai_rust"
    Write-Host "Password: ai_rust_password"
}

# 停止服务
function Stop-Services {
    Write-Info "停止服务..."
    
    docker-compose down
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "服务已停止"
    } else {
        Write-Error "服务停止失败"
        exit 1
    }
}

# 清理资源
function Remove-Resources {
    Write-Info "清理资源..."
    
    # 停止服务
    docker-compose down -v
    
    # 删除镜像
    docker rmi ai-rust-web:latest 2>$null
    
    # 清理未使用的资源
    docker system prune -f
    
    Write-Success "资源清理完成"
}

# 显示帮助
function Show-Help {
    Write-Host "AI-Rust项目部署脚本" -ForegroundColor Green
    Write-Host ""
    Write-Host "用法: .\deploy.ps1 [选项]"
    Write-Host ""
    Write-Host "选项:"
    Write-Host "  build      构建项目"
    Write-Host "  test       运行测试"
    Write-Host "  deploy     完整部署"
    Write-Host "  start      启动服务"
    Write-Host "  stop       停止服务"
    Write-Host "  restart    重启服务"
    Write-Host "  status     显示状态"
    Write-Host "  logs       查看日志"
    Write-Host "  cleanup    清理资源"
    Write-Host "  help       显示帮助"
    Write-Host ""
    Write-Host "示例:"
    Write-Host "  .\deploy.ps1 deploy    # 完整部署"
    Write-Host "  .\deploy.ps1 start     # 启动服务"
    Write-Host "  .\deploy.ps1 status    # 查看状态"
}

# 查看日志
function Show-Logs {
    Write-Info "查看服务日志..."
    
    docker-compose logs -f
}

# 主函数
function Main {
    switch ($Action) {
        "build" {
            Test-Dependencies
            Build-Project
        }
        "test" {
            Test-Dependencies
            Test-Project
        }
        "deploy" {
            Test-Dependencies
            Build-Project
            Test-Project
            Build-DockerImages
            Start-Services
            Wait-ForServices
            Initialize-Database
            Test-Health
            Show-Status
        }
        "start" {
            Start-Services
            Wait-ForServices
            Test-Health
            Show-Status
        }
        "stop" {
            Stop-Services
        }
        "restart" {
            Stop-Services
            Start-Services
            Wait-ForServices
            Test-Health
            Show-Status
        }
        "status" {
            Show-Status
        }
        "logs" {
            Show-Logs
        }
        "cleanup" {
            Remove-Resources
        }
        "help" {
            Show-Help
        }
        default {
            Write-Error "未知选项: $Action"
            Show-Help
            exit 1
        }
    }
}

# 执行主函数
Main