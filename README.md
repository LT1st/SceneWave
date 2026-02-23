# SceneWeave - AI 智能图片重构图工具

> 让每一张照片都成为完美构图

## 产品愿景

帮助不会使用 PS 的普通用户，一键将"随手拍"变成"社交媒体 ready"的完美照片。

---

## 用户痛点分析

| 痛点 | 场景 | 影响 |
|------|------|------|
| 主体太靠边 | 风景照、人像 | 视觉不平衡 |
| 构图太满 | 特写、美食 | 缺乏呼吸感 |
| 比例不合适 | 小红书/IG/Banner | 需要裁剪丢失内容 |
| 不会 PS | 大多数普通用户 | 无法手动调整 |

---

## 核心功能规划

### Phase 1: MVP (当前目标)

```
上传图片 → YOLO识别主体 → 构图评分 → 自动重构图 → 输出优化版本
```

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 主体识别 | YOLOv8 检测主要对象 | P0 |
| 构图评分 | 三分法则 + 视觉平衡评分 | P0 |
| 智能裁剪 | 基于主体位置自动裁剪 | P0 |
| 比例转换 | 支持 1:1 / 4:5 / 16:9 | P0 |
| 简单扩图 | 基于主体位置添加边框/模糊背景 | P1 |

### Phase 2: 进阶功能

| 功能 | 描述 | 技术方案 |
|------|------|----------|
| AI 扩图 | Outpainting 无损扩展 | Stable Diffusion Inpaint |
| 智能背景生成 | 根据主体生成匹配背景 | ControlNet + Depth |
| 批量处理 | 一次处理多张图片 | 异步队列 |
| 构图建议 | 给出多种重构图方案 | 多种裁剪策略 |

### Phase 3: 高级功能

| 功能 | 描述 | 技术方案 |
|------|------|----------|
| 场景理解 | 理解图片语义内容 | CLIP + LLM |
| 风格化扩图 | 按风格扩展背景 | Style Transfer |
| API 服务 | 提供 API 给第三方 | FastAPI + Redis |
| 模板系统 | 预设社交媒体模板 | 配置化 |

---

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    SceneWeave 架构                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   前端 UI    │    │   CLI 工具   │    │   API 接口   │ │
│  │  (Gradio)   │    │  (Python)   │    │  (FastAPI)  │ │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            ▼                            │
│  ┌─────────────────────────────────────────────────────┐│
│  │              核心处理流水线                           ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   ││
│  │  │ 主体检测 │→│构图分析 │→│重构图   │→│ 输出生成 │   ││
│  │  │ (YOLO)  │ │(Scoring)│ │(Reframe)│ │(Export) │   ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   ││
│  └─────────────────────────────────────────────────────┘│
│                            │                            │
│         ┌──────────────────┼──────────────────┐         │
│         ▼                  ▼                  ▼         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  YOLOv8     │    │ OpenCV      │    │ Diffusers   │ │
│  │  (主体检测)  │    │ (图像处理)   │    │ (AI扩图)    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
SceneWeave/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖
├── src/                      # 核心算法 (共享)
│   ├── core/
│   │   ├── detector.py       # YOLO 主体检测
│   │   ├── scorer.py         # 构图评分引擎
│   │   └── reframer.py       # 重构图引擎
│   └── utils/
│       └── image_utils.py    # 图像处理工具
│
├── desktop/                  # 桌面应用
│   ├── windows/
│   │   └── main.py           # Windows (CustomTkinter)
│   └── macos/
│       └── main.py           # macOS (PySide6)
│
├── mobile/                   # 移动应用 (Flutter)
│   ├── lib/                  # Dart 源码
│   │   ├── screens/          # 页面
│   │   ├── widgets/          # 组件
│   │   └── services/         # 服务
│   ├── android/              # Android 配置
│   ├── ios/                  # iOS 配置
│   └── macos/                # macOS 配置
│
├── api/                      # API 服务
│   ├── main.py               # FastAPI 服务
│   └── start.sh              # 启动脚本
│
├── web/
│   └── app.py                # Gradio Web UI
│
├── cli/
│   └── main.py               # 命令行工具
│
├── start.bat                 # Windows 启动脚本
└── start-macos.sh            # macOS 启动脚本
```

---

## 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/LT1st/SceneWave.git
cd SceneWeave

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### CLI 使用

```bash
# 基础使用
python cli/main.py --input image.jpg --output output/

# 指定输出比例
python cli/main.py --input image.jpg --ratio 4:5 --output output/

# 批量处理
python cli/main.py --input images/ --output output/ --batch
```

### Web UI

```bash
python web/app.py
# 打开 http://localhost:7860
```

### 桌面应用 (Windows)

**方式一：双击启动**
```
双击 start.bat
```

**方式二：命令行启动**
```bash
python desktop/windows/main.py
```

### 桌面应用 (macOS)

```bash
# 命令行启动
python desktop/macos/main.py

# 或使用启动脚本
./start-macos.sh
```

### 移动应用 (Flutter)

```bash
cd mobile

# 安装依赖
flutter pub get

# 运行 (iOS/Android/macOS)
flutter run

# 构建 iOS
flutter build ios

# 构建 Android
flutter build apk
```

### API 服务

```bash
# 启动 API 服务
python api/main.py

# 或使用启动脚本
./api/start.sh

# 访问文档
# http://localhost:8000/docs
```

### Docker 部署

```bash
# 构建镜像
docker build -t sceneweave .

# 运行服务
docker-compose up -d

# 访问 API
# http://localhost:8000
```

---

## 开发

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 带覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 代码检查

```bash
# 格式化代码
black src/ tests/

# 代码检查
ruff check src/ tests/

# 类型检查
mypy src/
```

### 配置

配置文件位于 `config/` 目录：

- `default.yaml` - 默认配置
- `development.yaml` - 开发环境

通过环境变量切换：

```bash
export SCENEWEAVE_ENV=development
python api/main.py
```

---

## 文档

- [API 文档](docs/API.md) - REST API 详细说明
- [架构文档](docs/ARCHITECTURE.md) - 系统架构和设计
- [贡献指南](CONTRIBUTING.md) - 如何贡献代码

---

## CI/CD

项目使用 GitHub Actions 进行自动化：

- **测试** - 每次提交自动运行测试
- **Lint** - 代码风格检查
- **覆盖率** - 自动生成覆盖率报告

状态：[![Tests](https://github.com/LT1st/SceneWave/workflows/Tests/badge.svg)](https://github.com/LT1st/SceneWave/actions)

---

## API 设计 (规划)

```python
from sceneweave import Reframer

# 初始化
reframer = Reframer()

# 分析图片
result = reframer.analyze("photo.jpg")
print(result.score)        # 构图评分 0-100
print(result.subjects)     # 检测到的主体
print(result.suggestions)  # 改进建议

# 重构图
output = reframer.reframe(
    "photo.jpg",
    ratio="4:5",           # 目标比例
    strategy="center",     # 策略: center, rule_of_thirds, golden
    padding="blur"         # 填充方式: blur, extend, color
)
output.save("reframed.jpg")
```

---

## 支持的比例预设

| 名称 | 比例 | 适用场景 |
|------|------|----------|
| Square | 1:1 | Instagram 方形 |
| Portrait | 4:5 | Instagram / 小红书竖图 |
| Story | 9:16 | IG Story / 抖音 / 小红书 |
| Landscape | 16:9 | YouTube / Banner |
| Banner | 3:1 | 网站 Banner |
| Cover | 2:3 | 小红书封面 |

---

## 构图评分规则

```python
class CompositionScore:
    """构图评分维度"""

    rule_of_thirds: float   # 三分法则 (0-30分)
    visual_balance: float   # 视觉平衡 (0-25分)
    subject_prominence: float  # 主体突出度 (0-25分)
    breathing_room: float   # 呼吸空间 (0-20分)

    @property
    def total(self) -> float:
        return sum([self.rule_of_thirds, self.visual_balance,
                    self.subject_prominence, self.breathing_room])
```

---

## 开发路线图

```
2024 Q4 ──────────────────────────────────────────────────
  │
  ├─ Week 1-2: MVP 基础
  │   ├── YOLOv8 主体检测
  │   ├── 基础构图评分
  │   └── CLI 工具
  │
  ├─ Week 3-4: 重构图功能
  │   ├── 智能裁剪
  │   ├── 比例转换
  │   └── Web UI (Gradio)
  │
  └─ Week 5-6: 优化与发布
      ├── 性能优化
      ├── 文档完善
      └── v1.0 发布

2025 Q1 ──────────────────────────────────────────────────
  │
  ├─ AI 扩图功能
  │   ├── Stable Diffusion 集成
  │   └── Outpainting 实现
  │
  └─ 产品化
      ├── API 服务
      └── 批量处理
```

---

## 技术选型

| 层级 | 技术 | 原因 |
|------|------|------|
| 主体检测 | YOLOv8 | 快速、准确、易部署 |
| 图像处理 | OpenCV | 成熟稳定 |
| AI 扩图 | Stable Diffusion | 效果最好 |
| Windows UI | CustomTkinter | 简单、现代 |
| macOS UI | PySide6 | Qt、原生体验 |
| 移动端 | Flutter | 跨平台、高性能 |
| Web UI | Gradio | 快速原型 |
| API | FastAPI | 高性能、异步 |
| 部署 | Docker | 标准化 |

---

## 性能目标

| 指标 | 目标值 |
|------|--------|
| 单图处理时间 | < 3s (不含AI扩图) |
| AI 扩图时间 | < 15s |
| 内存占用 | < 2GB (不含SD模型) |
| 并发支持 | 10+ 请求/秒 |

---

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## License

MIT License

---

## 联系方式

- Issues: [GitHub Issues](https://github.com/LT1st/SceneWave/issues)
- Email: your@email.com

---

> Made with by SceneWeave Team
