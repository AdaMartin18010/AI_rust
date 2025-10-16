#!/bin/bash

# AI-Rust项目部署脚本
# 用于Linux/macOS系统

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查Rust
    if ! command -v cargo &> /dev/null; then
        log_error "Rust未安装，请先安装Rust"
        exit 1
    fi
    
    log_success "所有依赖检查通过"
}

# 构建项目
build_project() {
    log_info "构建项目..."
    
    # 清理之前的构建
    cargo clean
    
    # 构建项目
    cargo build --release
    
    if [ $? -eq 0 ]; then
        log_success "项目构建成功"
    else
        log_error "项目构建失败"
        exit 1
    fi
}

# 运行测试
run_tests() {
    log_info "运行测试..."
    
    cargo test --release
    
    if [ $? -eq 0 ]; then
        log_success "所有测试通过"
    else
        log_error "测试失败"
        exit 1
    fi
}

# 构建Docker镜像
build_docker_images() {
    log_info "构建Docker镜像..."
    
    # 构建主服务镜像
    docker build -t ai-rust-web:latest .
    
    if [ $? -eq 0 ]; then
        log_success "Docker镜像构建成功"
    else
        log_error "Docker镜像构建失败"
        exit 1
    fi
}

# 启动服务
start_services() {
    log_info "启动服务..."
    
    # 停止现有服务
    docker-compose down
    
    # 启动服务
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        log_success "服务启动成功"
    else
        log_error "服务启动失败"
        exit 1
    fi
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    # 等待主服务
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "主服务已就绪"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "主服务启动超时"
            exit 1
        fi
        
        log_info "等待主服务启动... ($attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    # 等待数据库
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if docker-compose exec -T postgres pg_isready -U ai_rust &> /dev/null; then
            log_success "数据库已就绪"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "数据库启动超时"
            exit 1
        fi
        
        log_info "等待数据库启动... ($attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
}

# 初始化数据库
init_database() {
    log_info "初始化数据库..."
    
    # 等待数据库完全启动
    sleep 5
    
    # 执行数据库初始化脚本
    docker-compose exec -T postgres psql -U ai_rust -d ai_rust -f /docker-entrypoint-initdb.d/init-db.sql
    
    if [ $? -eq 0 ]; then
        log_success "数据库初始化成功"
    else
        log_warning "数据库初始化可能失败，请检查日志"
    fi
}

# 健康检查
health_check() {
    log_info "执行健康检查..."
    
    # 检查主服务
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_success "主服务健康检查通过"
    else
        log_error "主服务健康检查失败"
        exit 1
    fi
    
    # 检查Nginx
    if curl -f http://localhost/nginx-health &> /dev/null; then
        log_success "Nginx健康检查通过"
    else
        log_warning "Nginx健康检查失败"
    fi
    
    # 检查Prometheus
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        log_success "Prometheus健康检查通过"
    else
        log_warning "Prometheus健康检查失败"
    fi
    
    # 检查Grafana
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log_success "Grafana健康检查通过"
    else
        log_warning "Grafana健康检查失败"
    fi
}

# 显示服务状态
show_status() {
    log_info "服务状态："
    
    echo ""
    echo "=== Docker容器状态 ==="
    docker-compose ps
    
    echo ""
    echo "=== 服务访问地址 ==="
    echo "主服务: http://localhost:8080"
    echo "Nginx: http://localhost"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000 (admin/admin)"
    echo "Jaeger: http://localhost:16686"
    
    echo ""
    echo "=== 数据库连接信息 ==="
    echo "Host: localhost"
    echo "Port: 5432"
    echo "Database: ai_rust"
    echo "Username: ai_rust"
    echo "Password: ai_rust_password"
}

# 停止服务
stop_services() {
    log_info "停止服务..."
    
    docker-compose down
    
    if [ $? -eq 0 ]; then
        log_success "服务已停止"
    else
        log_error "服务停止失败"
        exit 1
    fi
}

# 清理资源
cleanup() {
    log_info "清理资源..."
    
    # 停止服务
    docker-compose down -v
    
    # 删除镜像
    docker rmi ai-rust-web:latest 2>/dev/null || true
    
    # 清理未使用的资源
    docker system prune -f
    
    log_success "资源清理完成"
}

# 显示帮助
show_help() {
    echo "AI-Rust项目部署脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  build      构建项目"
    echo "  test       运行测试"
    echo "  deploy     完整部署"
    echo "  start      启动服务"
    echo "  stop       停止服务"
    echo "  restart    重启服务"
    echo "  status     显示状态"
    echo "  logs       查看日志"
    echo "  cleanup    清理资源"
    echo "  help       显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 deploy    # 完整部署"
    echo "  $0 start     # 启动服务"
    echo "  $0 status    # 查看状态"
}

# 查看日志
show_logs() {
    log_info "查看服务日志..."
    
    docker-compose logs -f
}

# 主函数
main() {
    case "${1:-deploy}" in
        "build")
            check_dependencies
            build_project
            ;;
        "test")
            check_dependencies
            run_tests
            ;;
        "deploy")
            check_dependencies
            build_project
            run_tests
            build_docker_images
            start_services
            wait_for_services
            init_database
            health_check
            show_status
            ;;
        "start")
            start_services
            wait_for_services
            health_check
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            start_services
            wait_for_services
            health_check
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"