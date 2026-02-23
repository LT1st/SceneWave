# SceneWeave 架构文档

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         SceneWeave                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Windows    │  │    macOS     │  │  Mobile      │         │
│  │ CustomTkinter│  │   PySide6    │  │   Flutter    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                  │
│                           │                                     │
│                    ┌──────▼──────┐                             │
│                    │ API Service │  (可选)                     │
│                    │  FastAPI    │                             │
│                    └──────┬──────┘                             │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │                    Core Layer                            │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐              │   │
│  │  │ Detector  │ │  Scorer   │ │ Reframer  │              │   │
│  │  │ (YOLOv8)  │ │ (Scoring) │ │ (Crop)    │              │   │
│  │  └───────────┘ └───────────┘ └───────────┘              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │                   Utilities                              │   │
│  │  Logger | Config | Image Utils                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
SceneWeave/
├── src/                      # 核心算法
│   ├── core/                 # 核心模块
│   │   ├── detector.py       # 主体检测 (YOLO)
│   │   ├── scorer.py         # 构图评分
│   │   ├── reframer.py       # 重构图
│   │   └── config.py         # 配置管理
│   └── utils/                # 工具模块
│       ├── logger.py         # 日志系统
│       └── image_utils.py    # 图像工具
│
├── desktop/                  # 桌面应用
│   ├── windows/main.py       # Windows (CustomTkinter)
│   └── macos/main.py         # macOS (PySide6)
│
├── mobile/                   # 移动应用 (Flutter)
│   └── lib/                  # Dart 源码
│       ├── screens/          # 页面
│       ├── widgets/          # 组件
│       └── services/         # 服务 (API 调用)
│
├── api/                      # API 服务
│   └── main.py               # FastAPI
│
├── config/                   # 配置文件
│   ├── default.yaml          # 默认配置
│   └── development.yaml      # 开发配置
│
├── tests/                    # 测试
│   └── core/                 # 核心模块测试
│
└── docs/                     # 文档
    ├── API.md                # API 文档
    └── ARCHITECTURE.md       # 架构文档
```

---

## 核心模块

### 1. Detector (主体检测器)

```python
from src.core import SubjectDetector

detector = SubjectDetector(model_size="n")
subjects = detector.detect("image.jpg")
```

**功能**
- YOLOv8 目标检测
- 80 类物体识别
- 置信度过滤
- 边界框计算

---

### 2. Scorer (构图评分器)

```python
from src.core import CompositionScorer

scorer = CompositionScorer()
score = scorer.score(
    "image.jpg",
    subject_bbox=(100, 100, 200, 200),
    subject_center=(150, 150)
)
```

**评分维度**
- 三分法则 (30分)
- 视觉平衡 (25分)
- 主体突出度 (25分)
- 呼吸空间 (20分)

**评级标准**
- S: 90-100
- A: 80-89
- B: 70-79
- C: 60-69
- D: 0-59

---

### 3. Reframer (重构图器)

```python
from src.core import Reframer
from src.core.reframer import PaddingStrategy

reframer = Reframer()
result = reframer.reframe(
    "image.jpg",
    target_ratio=(4, 5),
    subject_center=(150, 150),
    padding=PaddingStrategy.BLUR
)
```

**功能**
- 智能裁剪
- 比例转换
- 多种填充策略
- 批量处理

---

## 数据流

### 构图分析流程

```
1. 上传图片
   ↓
2. YOLO 检测主体
   ↓
3. 计算主体位置
   ↓
4. 构图评分
   ├── 三分法则评分
   ├── 视觉平衡评分
   ├── 主体突出度评分
   └── 呼吸空间评分
   ↓
5. 生成分析报告
   ↓
6. 绘制检测结果
   ↓
7. 返回结果
```

### 重构图流程

```
1. 上传图片 + 选择比例
   ↓
2. 检测主体 (可选，如果已分析)
   ↓
3. 计算裁剪区域
   ├── 以主体为中心
   └── 考虑目标比例
   ↓
4. 应用填充策略 (如需要)
   ├── 模糊背景
   ├── 纯色填充
   ├── 镜像填充
   └── 边缘延伸
   ↓
5. 执行裁剪/填充
   ↓
6. 返回结果图片
```

---

## 配置系统

配置文件使用 YAML 格式，支持多环境：

```yaml
# config/default.yaml
model:
  yolo_model_size: n
  device: cpu

score:
  rule_of_thirds_weight: 30.0
  visual_balance_weight: 25.0
  ...

api:
  host: 0.0.0.0
  port: 8000
  ...
```

**使用配置**

```python
from src.core.config import load_config

config = load_config()
print(config.model.yolo_model_size)  # "n"
```

---

## 日志系统

```python
from src.utils.logger import get_logger

logger = get_logger("my_module")
logger.info("处理完成")
logger.error("发生错误", exc_info=True)
```

**日志级别**
- DEBUG
- INFO
- WARNING
- ERROR
- CRITICAL

---

## 扩展指南

### 添加新的评分维度

1. 在 `CompositionScore` 中添加新字段
2. 在 `CompositionScorer` 中实现评分逻辑
3. 更新配置文件中的权重

### 添加新的填充策略

1. 在 `PaddingStrategy` 枚举中添加新值
2. 在 `Reframer._apply_padding()` 中实现逻辑
3. 更新 UI 中的选项

### 添加新的比例预设

1. 在 `AspectRatio` 枚举中添加新比例
2. 在 `Reframer.SOCIAL_PRESETS` 中添加映射
3. 更新文档

---

## 性能优化

1. **YOLO 模型选择**
   - `n`: 最快，精度较低
   - `s`: 平衡
   - `m/l/x`: 更高精度，更慢

2. **图像处理**
   - 使用 GPU 加速 (CUDA/MPS)
   - 批量处理减少重复加载

3. **API 缓存**
   - 使用 Redis 缓存检测结果
   - 延迟加载 YOLO 模型

---

## 安全考虑

1. **文件上传**
   - 限制文件大小 (10MB)
   - 验证文件类型
   - 沙化处理

2. **API 安全**
   - 添加认证机制
   - 限流保护
   - HTTPS

3. **隐私**
   - 不存储用户图片
   - 自动清理临时文件
