# 贡献指南

感谢您对 SceneWeave 的关注！我们欢迎各种形式的贡献。

---

## 开发环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/LT1st/SceneWave.git
cd SceneWave
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装开发依赖

```bash
pip install pytest pytest-cov ruff black mypy
```

---

## 代码规范

### Python 代码风格

我们使用 **Black** 进行代码格式化：

```bash
black src/ tests/
```

### 代码检查

使用 **Ruff** 进行 linting：

```bash
ruff check src/ tests/
```

### 类型检查

使用 **MyPy** 进行类型检查：

```bash
mypy src/ --ignore-missing-imports
```

---

## 测试

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/core/test_detector.py

# 带覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 测试结构

```
tests/
├── conftest.py              # 共享 fixtures
└── core/
    ├── test_detector.py     # 检测器测试
    ├── test_scorer.py       # 评分器测试
    └── test_reframer.py     # 重构图器测试
```

---

## 提交 PR

### 1. 创建分支

```bash
git checkout -b feature/your-feature-name
```

### 2. 编写代码

- 遵循代码规范
- 添加测试
- 更新文档

### 3. 运行测试

```bash
pytest tests/
```

### 4. 提交代码

```bash
git add .
git commit -m "feat: 添加你的功能描述"
```

### 5. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request。

---

## Commit 规范

我们使用语义化提交信息：

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 代码格式调整
- `refactor:` 重构
- `test:` 添加测试
- `chore:` 构建/工具更新

**示例**
```
feat: 添加智能背景生成功能
fix: 修复构图评分计算错误
docs: 更新 API 文档
```

---

## 项目结构

添加新功能时，请遵循现有结构：

```
src/
├── core/           # 核心算法
├── utils/          # 工具函数
└── models/         # 数据模型
```

---

## 问题反馈

### 报告 Bug

请在 Issues 中提供：
- 问题描述
- 复现步骤
- 期望行为
- 实际行为
- 环境信息 (OS, Python 版本)

### 功能建议

请在 Issues 中：
- 清晰描述功能需求
- 说明使用场景
- 提供可能的实现思路

---

## 行为准则

- 尊重他人
- 接受建设性批评
- 关注社区而非个人

---

## 获取帮助

- 查看 [文档](docs/)
- 提交 [Issue](https://github.com/LT1st/SceneWave/issues)
- 查看 [API 文档](docs/API.md)
