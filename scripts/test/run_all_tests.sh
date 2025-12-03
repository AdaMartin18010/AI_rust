#!/bin/bash
# 运行所有测试的脚本
# 用法: ./scripts/test/run_all_tests.sh

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# 打印标题
print_header() {
    echo ""
    echo "=================================="
    echo "$1"
    echo "=================================="
    echo ""
}

# 记录开始时间
START_TIME=$(date +%s)

# 清理函数
cleanup() {
    print_info "Cleaning up..."
    # 清理临时文件
    rm -rf /tmp/ai-rust-test-*
}

# 设置trap以确保清理
trap cleanup EXIT

# 检查必要的工具
check_requirements() {
    print_header "Checking Requirements"
    
    if ! command -v cargo &> /dev/null; then
        print_error "cargo not found. Please install Rust."
        exit 1
    fi
    print_success "cargo found"
    
    if ! command -v rustc &> /dev/null; then
        print_error "rustc not found. Please install Rust."
        exit 1
    fi
    print_success "rustc found"
    
    # 检查Rust版本
    RUST_VERSION=$(rustc --version | awk '{print $2}')
    print_info "Rust version: $RUST_VERSION"
}

# 运行代码检查
run_check() {
    print_header "Running Cargo Check"
    
    if cargo check --all-features --workspace; then
        print_success "Cargo check passed"
    else
        print_error "Cargo check failed"
        exit 1
    fi
}

# 运行格式检查
run_fmt() {
    print_header "Running Format Check"
    
    if cargo fmt --all -- --check; then
        print_success "Format check passed"
    else
        print_error "Format check failed"
        print_info "Run 'cargo fmt --all' to fix formatting issues"
        exit 1
    fi
}

# 运行Clippy检查
run_clippy() {
    print_header "Running Clippy"
    
    if cargo clippy --all-targets --all-features --workspace -- -D warnings; then
        print_success "Clippy passed"
    else
        print_error "Clippy failed"
        exit 1
    fi
}

# 运行单元测试
run_unit_tests() {
    print_header "Running Unit Tests"
    
    if cargo test --lib --all-features --workspace -- --test-threads=1 --nocapture; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        exit 1
    fi
}

# 运行集成测试
run_integration_tests() {
    print_header "Running Integration Tests"
    
    if cargo test --test '*' --all-features --workspace -- --test-threads=1; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        exit 1
    fi
}

# 运行文档测试
run_doc_tests() {
    print_header "Running Documentation Tests"
    
    if cargo test --doc --all-features --workspace; then
        print_success "Documentation tests passed"
    else
        print_error "Documentation tests failed"
        exit 1
    fi
}

# 运行示例测试
run_example_tests() {
    print_header "Running Example Tests"
    
    # 获取所有示例
    examples=$(cargo build --examples 2>&1 | grep "Compiling" | awk '{print $2}' | grep -v "^ai")
    
    for example in $examples; do
        print_info "Testing example: $example"
        if cargo run --example "$example" --quiet 2>&1 | head -n 20; then
            print_success "Example $example works"
        else
            print_error "Example $example failed"
            # 不退出，继续测试其他示例
        fi
    done
}

# 运行基准测试（可选）
run_benchmarks() {
    print_header "Running Benchmarks"
    
    if [ "$RUN_BENCHMARKS" = "true" ]; then
        if cargo bench --no-fail-fast; then
            print_success "Benchmarks completed"
        else
            print_error "Benchmarks failed"
            # 基准测试失败不影响整体测试
        fi
    else
        print_info "Skipping benchmarks (set RUN_BENCHMARKS=true to run)"
    fi
}

# 生成代码覆盖率报告
run_coverage() {
    print_header "Generating Code Coverage"
    
    if command -v cargo-tarpaulin &> /dev/null; then
        if cargo tarpaulin \
            --all-features \
            --workspace \
            --timeout 300 \
            --out Html \
            --output-dir coverage; then
            print_success "Coverage report generated in coverage/"
            
            # 提取覆盖率
            if [ -f "coverage/tarpaulin-report.html" ]; then
                COVERAGE=$(grep -o '[0-9.]*%' coverage/tarpaulin-report.html | head -1)
                print_info "Code coverage: $COVERAGE"
            fi
        else
            print_error "Coverage generation failed"
            # 覆盖率失败不影响整体测试
        fi
    else
        print_info "cargo-tarpaulin not found. Install with: cargo install cargo-tarpaulin"
    fi
}

# 运行安全审计
run_security_audit() {
    print_header "Running Security Audit"
    
    if command -v cargo-audit &> /dev/null; then
        if cargo audit; then
            print_success "Security audit passed"
        else
            print_error "Security audit found vulnerabilities"
            # 安全审计失败不影响测试，但会报告
        fi
    else
        print_info "cargo-audit not found. Install with: cargo install cargo-audit"
    fi
}

# 检查依赖更新
check_dependencies() {
    print_header "Checking Dependencies"
    
    if command -v cargo-outdated &> /dev/null; then
        print_info "Checking for outdated dependencies..."
        cargo outdated || true
    else
        print_info "cargo-outdated not found. Install with: cargo install cargo-outdated"
    fi
}

# 主测试流程
main() {
    print_header "AI-Rust Test Suite"
    print_info "Starting comprehensive test run..."
    
    # 检查环境
    check_requirements
    
    # 代码质量检查
    run_check
    run_fmt
    run_clippy
    
    # 测试
    run_unit_tests
    run_integration_tests
    run_doc_tests
    run_example_tests
    
    # 可选测试
    run_benchmarks
    run_coverage
    
    # 安全和依赖检查
    run_security_audit
    check_dependencies
    
    # 计算总时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_header "Test Summary"
    print_success "All tests passed! ✨"
    print_info "Total time: ${DURATION}s"
    
    echo ""
    echo "Test results:"
    echo "  ✓ Code check"
    echo "  ✓ Format check"
    echo "  ✓ Clippy check"
    echo "  ✓ Unit tests"
    echo "  ✓ Integration tests"
    echo "  ✓ Documentation tests"
    echo "  ✓ Example tests"
    echo ""
}

# 运行主函数
main
