# Windows版本的测试脚本
# 用法: .\scripts\test\run_all_tests.ps1

# 设置错误时停止
$ErrorActionPreference = "Stop"

# 颜色函数
function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "→ $Message" -ForegroundColor Yellow
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "==================================" -ForegroundColor Cyan
    Write-Host ""
}

# 记录开始时间
$StartTime = Get-Date

# 清理函数
function Cleanup {
    Write-Info "Cleaning up..."
    # 清理临时文件
    Remove-Item -Path "$env:TEMP\ai-rust-test-*" -Force -ErrorAction SilentlyContinue
}

# 检查必要的工具
function Check-Requirements {
    Write-Header "Checking Requirements"
    
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Error-Custom "cargo not found. Please install Rust."
        exit 1
    }
    Write-Success "cargo found"
    
    if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
        Write-Error-Custom "rustc not found. Please install Rust."
        exit 1
    }
    Write-Success "rustc found"
    
    # 检查Rust版本
    $RustVersion = (rustc --version).Split()[1]
    Write-Info "Rust version: $RustVersion"
}

# 运行代码检查
function Run-Check {
    Write-Header "Running Cargo Check"
    
    cargo check --all-features --workspace
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Cargo check passed"
    } else {
        Write-Error-Custom "Cargo check failed"
        exit 1
    }
}

# 运行格式检查
function Run-Fmt {
    Write-Header "Running Format Check"
    
    cargo fmt --all -- --check
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Format check passed"
    } else {
        Write-Error-Custom "Format check failed"
        Write-Info "Run 'cargo fmt --all' to fix formatting issues"
        exit 1
    }
}

# 运行Clippy检查
function Run-Clippy {
    Write-Header "Running Clippy"
    
    cargo clippy --all-targets --all-features --workspace -- -D warnings
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Clippy passed"
    } else {
        Write-Error-Custom "Clippy failed"
        exit 1
    }
}

# 运行单元测试
function Run-UnitTests {
    Write-Header "Running Unit Tests"
    
    cargo test --lib --all-features --workspace -- --test-threads=1 --nocapture
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Unit tests passed"
    } else {
        Write-Error-Custom "Unit tests failed"
        exit 1
    }
}

# 运行集成测试
function Run-IntegrationTests {
    Write-Header "Running Integration Tests"
    
    cargo test --test '*' --all-features --workspace -- --test-threads=1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Integration tests passed"
    } else {
        Write-Error-Custom "Integration tests failed"
        exit 1
    }
}

# 运行文档测试
function Run-DocTests {
    Write-Header "Running Documentation Tests"
    
    cargo test --doc --all-features --workspace
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Documentation tests passed"
    } else {
        Write-Error-Custom "Documentation tests failed"
        exit 1
    }
}

# 运行安全审计
function Run-SecurityAudit {
    Write-Header "Running Security Audit"
    
    if (Get-Command cargo-audit -ErrorAction SilentlyContinue) {
        cargo audit
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Security audit passed"
        } else {
            Write-Error-Custom "Security audit found vulnerabilities"
        }
    } else {
        Write-Info "cargo-audit not found. Install with: cargo install cargo-audit"
    }
}

# 主测试流程
function Main {
    Write-Header "AI-Rust Test Suite"
    Write-Info "Starting comprehensive test run..."
    
    try {
        # 检查环境
        Check-Requirements
        
        # 代码质量检查
        Run-Check
        Run-Fmt
        Run-Clippy
        
        # 测试
        Run-UnitTests
        Run-IntegrationTests
        Run-DocTests
        
        # 安全检查
        Run-SecurityAudit
        
        # 计算总时间
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalSeconds
        
        Write-Header "Test Summary"
        Write-Success "All tests passed! ✨"
        Write-Info "Total time: ${Duration}s"
        
        Write-Host ""
        Write-Host "Test results:"
        Write-Host "  ✓ Code check"
        Write-Host "  ✓ Format check"
        Write-Host "  ✓ Clippy check"
        Write-Host "  ✓ Unit tests"
        Write-Host "  ✓ Integration tests"
        Write-Host "  ✓ Documentation tests"
        Write-Host ""
    }
    finally {
        Cleanup
    }
}

# 运行主函数
Main
