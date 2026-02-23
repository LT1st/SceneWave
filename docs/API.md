# SceneWeave API 文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API 版本**: v1
- **内容类型**: `application/json`

## 认证

当前版本无需认证。

---

## 端点

### 1. 健康检查

检查 API 服务状态。

**请求**
```http
GET /health
```

**响应**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### 2. 构图分析

分析图片构图，检测主体并计算评分。

**请求**
```http
POST /api/v1/analyze
Content-Type: multipart/form-data
```

**参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |

**响应**
```json
{
  "success": true,
  "subjects": [
    {
      "label": "person",
      "confidence": 0.95,
      "bbox": [100, 100, 200, 200],
      "center": [150.0, 150.0]
    }
  ],
  "score": {
    "rule_of_thirds": 25.5,
    "visual_balance": 20.0,
    "subject_prominence": 22.0,
    "breathing_room": 15.0
  },
  "image_base64": "data:image/jpeg;base64,..."
}
```

**示例**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@photo.jpg"
```

---

### 3. 智能重构图

将图片重构为指定比例。

**请求**
```http
POST /api/v1/reframe
Content-Type: multipart/form-data
```

**参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| ratio_width | int | 是 | 目标宽度比例 (1-21) |
| ratio_height | int | 是 | 目标高度比例 (1-21) |
| padding | string | 否 | 填充策略: none, blur, color, mirror, extend |
| subject_bbox | string | 否 | 主体边界框 JSON: [x1, y1, x2, y2] |
| subject_center | string | 否 | 主体中心点 JSON: [cx, cy] |

**响应**
```json
{
  "success": true,
  "original_size": [1920, 1080],
  "new_size": [1080, 1350],
  "ratio": [4, 5],
  "padding": "blur",
  "image_base64": "data:image/jpeg;base64,..."
}
```

**示例**
```bash
curl -X POST "http://localhost:8000/api/v1/reframe" \
  -F "file=@photo.jpg" \
  -F "ratio_width=4" \
  -F "ratio_height=5" \
  -F "padding=blur"
```

---

### 4. 批量重构图

一次性生成多个比例的重构图版本。

**请求**
```http
POST /api/v1/batch-reframe
Content-Type: multipart/form-data
```

**参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| ratios | string | 是 | 比例数组 JSON: [[1,1],[4,5],[16,9]] |
| padding | string | 否 | 填充策略 |

**响应**
```json
{
  "success": true,
  "results": [
    {
      "ratio": [1, 1],
      "size": [1080, 1080],
      "image_base64": "..."
    },
    {
      "ratio": [4, 5],
      "size": [1080, 1350],
      "image_base64": "..."
    }
  ]
}
```

**示例**
```bash
curl -X POST "http://localhost:8000/api/v1/batch-reframe" \
  -F "file=@photo.jpg" \
  -F 'ratios=[[1,1],[4,5],[16,9]]' \
  -F "padding=blur"
```

---

## 错误响应

所有错误响应遵循以下格式：

```json
{
  "detail": "错误消息"
}
```

**常见错误码**
| 状态码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

---

## 支持的比例

| 比例 | 名称 | 适用场景 |
|------|------|----------|
| 1:1 | 正方形 | Instagram |
| 4:5 | 竖图 | Instagram/小红书 |
| 9:16 | 竖屏 | Story/抖音 |
| 16:9 | 横屏 | YouTube |
| 2:3 | 封面 | 小红书 |
| 3:1 | Banner | 网站 |

---

## 填充策略

| 策略 | 说明 |
|------|------|
| none | 不填充，仅裁剪 |
| blur | 模糊背景填充 |
| color | 纯色填充 (白色) |
| mirror | 镜像填充 |
| extend | 边缘延伸 |

---

## 限制

- 最大上传文件大小: 10MB
- 支持的图片格式: JPG, PNG, WEBP, BMP
- 比例范围: 1-21
