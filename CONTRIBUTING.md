# 🤝 贡献指南

感谢您对AI-Rust项目的关注！我们欢迎所有形式的贡献，包括但不限于代码、文档、问题反馈和功能建议。

---

## 📋 目录

- [🤝 贡献指南](#-贡献指南)
  - [📋 目录](#-目录)
  - [🌟 行为准则](#-行为准则)
    - [我们的承诺](#我们的承诺)
    - [我们的标准](#我们的标准)
  - [🚀 如何贡献](#-如何贡献)
    - [贡献类型](#贡献类型)
  - [💻 开发环境设置](#-开发环境设置)
    - [1. 前置要求](#1-前置要求)
      - [必需](#必需)
      - [可选](#可选)
    - [2. 克隆仓库](#2-克隆仓库)
    - [3. 安装依赖](#3-安装依赖)
    - [4. 构建项目](#4-构建项目)
  - [📝 代码规范](#-代码规范)
    - [Rust代码风格](#rust代码风格)
    - [代码质量标准](#代码质量标准)
      - [1. 文档注释](#1-文档注释)
      - [2. 错误处理](#2-错误处理)
      - [3. 测试覆盖](#3-测试覆盖)
  - [📤 提交流程](#-提交流程)
    - [1. 创建功能分支](#1-创建功能分支)
    - [2. 进行修改](#2-进行修改)
    - [3. 提交代码](#3-提交代码)
    - [4. 推送并创建Pull Request](#4-推送并创建pull-request)
    - [Pull Request检查清单](#pull-request检查清单)
  - [🧪 测试指南](#-测试指南)
    - [运行测试](#运行测试)
    - [自动化测试脚本](#自动化测试脚本)
    - [基准测试](#基准测试)
  - [📚 文档贡献](#-文档贡献)
    - [文档类型](#文档类型)
    - [文档规范](#文档规范)
      - [1. Markdown格式](#1-markdown格式)
      - [2. 代码示例](#2-代码示例)
    - [生成文档](#生成文档)
  - [🔍 代码审查流程](#-代码审查流程)
    - [审查标准](#审查标准)
    - [审查反馈](#审查反馈)
  - [📞 获取帮助](#-获取帮助)
  - [🎖️ 贡献者](#️-贡献者)
  - [📜 许可证](#-许可证)
  - [🙏 致谢](#-致谢)


---

## 🌟 行为准则

### 我们的承诺

为了营造一个开放和友好的环境，我们承诺：

- ✅ 尊重所有贡献者
- ✅ 接受建设性的反馈
- ✅ 关注对社区最有利的事情
- ✅ 对其他社区成员表示同情

### 我们的标准

**鼓励的行为**:

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同情

**不可接受的行为**:

- 使用性化的语言或图像
- 发表侮辱性/贬损性评论
- 公开或私下骚扰
- 未经许可发布他人的私人信息
- 其他不道德或不专业的行为

---

## 🚀 如何贡献

### 贡献类型

1. **报告Bug**
   - 使用[Bug报告模板](https://github.com/your-org/ai-rust/issues/new?template=bug_report.md)
   - 提供详细的复现步骤
   - 包含环境信息

2. **功能请求**
   - 使用[功能请求模板](https://github.com/your-org/ai-rust/issues/new?template=feature_request.md)
   - 清楚说明功能的价值
   - 考虑实现的可行性

3. **代码贡献**
   - Fork项目
   - 创建功能分支
   - 提交Pull Request

4. **文档改进**
   - 修正错误
   - 添加示例
   - 改善说明

5. **性能优化**
   - 提供基准测试
   - 说明优化思路
   - 测试性能提升

---

## 💻 开发环境设置

### 1. 前置要求

#### 必需

- **Rust**: 1.90.0或更高版本
- **Git**: 最新版本
- **操作系统**: Linux, macOS, or Windows

#### 可选

- **Docker**: 用于容器化测试
- **cargo-watch**: 自动重新编译
- **cargo-edit**: 管理依赖
- **cargo-tarpaulin**: 代码覆盖率

---

### 2. 克隆仓库

```bash
# 克隆您fork的仓库
git clone https://github.com/YOUR_USERNAME/ai-rust.git
cd ai-rust

# 添加上游仓库
git remote add upstream https://github.com/your-org/ai-rust.git
```

---

### 3. 安装依赖

```bash
# 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装开发工具
cargo install cargo-watch
cargo install cargo-edit
cargo install cargo-tarpaulin
cargo install cargo-audit

# 安装pre-commit hooks (可选)
./scripts/setup-hooks.sh
```

---

### 4. 构建项目

```bash
# 开发构建
cargo build

# 发布构建
cargo build --release

# 运行测试
cargo test

# 运行示例
cargo run --example gat_ai_inference
```

---

## 📝 代码规范

### Rust代码风格

我们遵循标准的Rust代码风格指南：

1. **格式化**: 使用`rustfmt`

   ```bash
   cargo fmt --all
   ```

2. **Linting**: 使用`clippy`

   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```

3. **命名规范**:
   - 类型: `PascalCase` (e.g., `InferenceEngine`)
   - 函数: `snake_case` (e.g., `run_inference`)
   - 常量: `SCREAMING_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)

---

### 代码质量标准

#### 1. 文档注释

```rust
/// 执行模型推理
///
/// # 参数
///
/// * `input` - 输入张量
///
/// # 返回
///
/// 返回推理结果或错误
///
/// # 示例
///
/// ```
/// let result = model.infer(input)?;
/// ```
pub fn infer(&self, input: Tensor) -> Result<Tensor> {
    // 实现...
}
```

#### 2. 错误处理

```rust
// ✅ 推荐: 使用Result和自定义错误类型
pub fn process_data(data: &[f32]) -> Result<Vec<f32>, ProcessError> {
    if data.is_empty() {
        return Err(ProcessError::EmptyInput);
    }

    // 处理...
    Ok(result)
}

// ❌ 避免: 使用panic!
pub fn process_data(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        panic!("Empty input!");  // 不推荐
    }
    // ...
}
```

#### 3. 测试覆盖

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference() {
        let model = create_test_model();
        let input = create_test_input();

        let result = model.infer(input);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), expected_shape());
    }

    #[test]
    fn test_error_handling() {
        let model = create_test_model();
        let invalid_input = create_invalid_input();

        let result = model.infer(invalid_input);

        assert!(result.is_err());
    }
}
```

---

## 📤 提交流程

### 1. 创建功能分支

```bash
# 从main分支创建新分支
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

分支命名规范:

- `feature/xxx` - 新功能
- `fix/xxx` - Bug修复
- `docs/xxx` - 文档更新
- `perf/xxx` - 性能优化
- `refactor/xxx` - 代码重构

---

### 2. 进行修改

```bash
# 修改文件
# ... 编辑代码 ...

# 运行测试
cargo test

# 运行格式化
cargo fmt --all

# 运行Clippy
cargo clippy --all-targets --all-features
```

---

### 3. 提交代码

提交信息格式:

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**类型 (type)**:

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式(不影响功能)
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 添加测试
- `chore`: 构建过程或辅助工具的变动

**示例**:

```bash
git add .
git commit -m "feat(inference): add batch inference support

- Implement batch processing for inference
- Add benchmarks showing 5x throughput improvement
- Update documentation with usage examples

Closes #123"
```

---

### 4. 推送并创建Pull Request

```bash
# 推送到您的fork
git push origin feature/your-feature-name
```

然后在GitHub上创建Pull Request:

1. 访问您的fork
2. 点击"Pull Request"
3. 选择base分支(通常是`main`)
4. 填写PR模板
5. 提交PR

---

### Pull Request检查清单

提交PR前请确认:

- [ ] 代码已格式化 (`cargo fmt --all`)
- [ ] Clippy检查通过 (`cargo clippy`)
- [ ] 所有测试通过 (`cargo test`)
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 提交信息格式正确
- [ ] PR描述清晰完整

---

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_name

# 运行带输出的测试
cargo test -- --nocapture

# 运行单个文件的测试
cargo test --lib

# 运行集成测试
cargo test --test integration_tests
```

---

### 自动化测试脚本

```bash
# Linux/macOS
./scripts/test/run_all_tests.sh

# Windows
.\scripts\test\run_all_tests.ps1
```

---

### 基准测试

```bash
# 运行所有基准测试
cargo bench

# 运行特定基准测试
cargo bench benchmark_name
```

---

## 📚 文档贡献

### 文档类型

1. **API文档**: Rust文档注释
2. **用户指南**: Markdown文档
3. **示例代码**: `examples/`目录
4. **教程**: `docs/`目录

---

### 文档规范

#### 1. Markdown格式

```markdown
# 一级标题

## 二级标题

### 三级标题

**粗体** 和 *斜体*

- 列表项1
- 列表项2

```rust
// 代码块
fn example() {}
\```

[链接文本](URL)
```

#### 2. 代码示例

- 确保示例可运行
- 添加必要的注释
- 包含错误处理

---

### 生成文档

```bash
# 生成并打开文档
cargo doc --open

# 生成所有文档
cargo doc --no-deps --all-features
```

---

## 🔍 代码审查流程

### 审查标准

PR将根据以下标准审查:

1. **功能性**: 是否实现了预期功能
2. **代码质量**: 是否遵循最佳实践
3. **测试覆盖**: 是否有足够的测试
4. **文档完整性**: 是否更新了文档
5. **性能影响**: 是否影响性能
6. **向后兼容**: 是否破坏现有API

---

### 审查反馈

- 保持开放和尊重的态度
- 及时响应审查意见
- 讨论技术决策
- 必要时更新PR

---

## 📞 获取帮助

如果您有任何问题:

1. **查看文档**: [docs/](docs/)
2. **搜索Issues**: [GitHub Issues](https://github.com/your-org/ai-rust/issues)
3. **讨论区**: [GitHub Discussions](https://github.com/your-org/ai-rust/discussions)
4. **联系维护者**: 通过Issue或讨论区

---

## 🎖️ 贡献者

感谢所有贡献者！

查看[贡献者列表](https://github.com/your-org/ai-rust/graphs/contributors)

---

## 📜 许可证

通过贡献，您同意您的贡献将在MIT许可证下授权。

详见[LICENSE](LICENSE)文件。

---

## 🙏 致谢

感谢您考虑为AI-Rust项目做出贡献！

我们期待您的贡献！🎉

---

*最后更新: 2025年12月3日*
*维护者: AI-Rust项目团队*
