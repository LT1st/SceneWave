"""
SceneWeave API 服务
为移动端和 Web 端提供 REST API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import io
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import SubjectDetector, CompositionScorer, Reframer
from src.core.reframer import PaddingStrategy


# ============================================================================
# Pydantic 模型
# ============================================================================

class SubjectInfo(BaseModel):
    """主体信息"""
    label: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    center: List[float]  # [cx, cy]


class CompositionScore(BaseModel):
    """构图评分"""
    rule_of_thirds: float = Field(description="三分法则得分", ge=0, le=30)
    visual_balance: float = Field(description="视觉平衡得分", ge=0, le=25)
    subject_prominence: float = Field(description="主体突出度得分", ge=0, le=25)
    breathing_room: float = Field(description="呼吸空间得分", ge=0, le=20)

    @property
    def total(self) -> float:
        return self.rule_of_thirds + self.visual_balance + self.subject_prominence + self.breathing_room

    @property
    def grade(self) -> str:
        if self.total >= 90:
            return "S"
        elif self.total >= 80:
            return "A"
        elif self.total >= 70:
            return "B"
        elif self.total >= 60:
            return "C"
        else:
            return "D"


class AnalysisResponse(BaseModel):
    """分析响应"""
    success: bool
    subjects: List[SubjectInfo]
    score: CompositionScore
    image_base64: Optional[str] = None


class ReframeRequest(BaseModel):
    """重构图请求"""
    ratio_width: int = Field(description="目标宽度比例", ge=1, le=21)
    ratio_height: int = Field(description="目标高度比例", ge=1, le=21)
    padding: Literal["none", "blur", "color", "mirror", "extend"] = "blur"


class ReframeResponse(BaseModel):
    """重构图响应"""
    success: bool
    original_size: List[int]  # [width, height]
    new_size: List[int]
    ratio: List[int]
    padding: str
    image_base64: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str


# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="SceneWeave API",
    description="AI 智能图片重构图 API",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
class GlobalState:
    def __init__(self):
        self.detector: Optional[SubjectDetector] = None
        self.reframer = Reframer()
        self.scorer = CompositionScorer()

    def get_detector(self) -> SubjectDetector:
        if self.detector is None:
            self.detector = SubjectDetector(model_size="n")
        return self.detector

state = GlobalState()


# ============================================================================
# 辅助函数
# ============================================================================

def decode_image(image_bytes: bytes) -> np.ndarray:
    """解码图片字节流"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法解码图片")
    return image


def encode_image(image: np.ndarray, format: str = ".jpg") -> bytes:
    """编码图片为字节流"""
    _, encoded = cv2.imencode(format, image)
    return encoded.tobytes()


def image_to_base64(image: np.ndarray, format: str = ".jpg") -> str:
    """图片转 base64"""
    import base64
    _, encoded = cv2.imencode(format, image)
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def save_temp_image(image: np.ndarray) -> str:
    """保存临时图片文件"""
    temp_dir = Path(tempfile.gettempdir()) / "sceneweave"
    temp_dir.mkdir(exist_ok=True)

    temp_path = temp_dir / f"{os.urandom(8).hex()}.jpg"
    cv2.imwrite(str(temp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return str(temp_path)


# ============================================================================
# 路由
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """根路径"""
    return {
        "status": "ok",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    分析图片构图

    - 检测图片中的主体
    - 计算构图评分
    - 返回带标注的图片
    """
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    try:
        # 读取图片
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存临时文件
        temp_path = save_temp_image(image_rgb)

        # 检测主体
        detector = state.get_detector()
        subjects = detector.detect(temp_path)

        # 计算评分
        if subjects:
            main_subject = subjects[0]
            score = state.scorer.score(
                temp_path,
                subject_bbox=main_subject.bbox,
                subject_center=main_subject.center
            )
        else:
            score = state.scorer.score(temp_path)

        # 绘制检测结果
        result_image = detector.draw_detections(temp_path, subjects)

        # 转换为 base64
        image_base64 = image_to_base64(result_image)

        # 构建响应
        subject_list = [
            SubjectInfo(
                label=s.label,
                confidence=float(s.confidence),
                bbox=[int(x) for x in s.bbox],
                center=[float(x) for x in s.center]
            )
            for s in subjects
        ]

        score_data = CompositionScore(
            rule_of_thirds=float(score.rule_of_thirds),
            visual_balance=float(score.visual_balance),
            subject_prominence=float(score.subject_prominence),
            breathing_room=float(score.breathing_room)
        )

        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass

        return AnalysisResponse(
            success=True,
            subjects=subject_list,
            score=score_data,
            image_base64=image_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/api/v1/reframe", response_model=ReframeResponse)
async def reframe_image(
    file: UploadFile = File(...),
    ratio_width: int = 4,
    ratio_height: int = 5,
    padding: str = "blur",
    subject_bbox: Optional[str] = None,  # JSON string: [x1, y1, x2, y2]
    subject_center: Optional[str] = None  # JSON string: [cx, cy]
):
    """
    重构图图片

    - ratio_width: 目标宽度比例
    - ratio_height: 目标高度比例
    - padding: 填充策略 (none, blur, color, mirror, extend)
    - subject_bbox: 主体边界框 (可选, JSON 数组)
    - subject_center: 主体中心点 (可选, JSON 数组)
    """
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    # 验证比例
    if ratio_width < 1 or ratio_height < 1 or ratio_width > 21 or ratio_height > 21:
        raise HTTPException(status_code=400, detail="比例必须在 1-21 之间")

    # 验证填充策略
    valid_paddings = ["none", "blur", "color", "mirror", "extend"]
    if padding not in valid_paddings:
        raise HTTPException(status_code=400, detail=f"填充策略必须是: {', '.join(valid_paddings)}")

    try:
        import json

        # 读取图片
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存临时文件
        temp_path = save_temp_image(image_rgb)

        # 解析主体信息
        subj_center = None
        subj_bbox = None
        if subject_center:
            subj_center = json.loads(subject_center)
        if subject_bbox:
            subj_bbox = json.loads(subject_bbox)
            subj_bbox = tuple(int(x) for x in subj_bbox)

        # 解析填充策略
        padding_strategy = PaddingStrategy(padding)

        # 执行重构图
        result = state.reframer.reframe(
            temp_path,
            target_ratio=(ratio_width, ratio_height),
            subject_center=subj_center,
            subject_bbox=subj_bbox,
            padding=padding_strategy
        )

        # 转换为 base64
        image_base64 = image_to_base64(result.image)

        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass

        return ReframeResponse(
            success=True,
            original_size=list(result.original_size),
            new_size=list(result.new_size),
            ratio=[ratio_width, ratio_height],
            padding=padding,
            image_base64=image_base64
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="subject_bbox 或 subject_center 格式错误")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重构图失败: {str(e)}")


@app.post("/api/v1/batch-reframe")
async def batch_reframe(
    file: UploadFile = File(...),
    ratios: str = "[[1,1],[4,5],[16,9]]",  # JSON array of ratios
    padding: str = "blur"
):
    """
    批量重构图

    一次性生成多个比例的重构图版本
    """
    import json

    try:
        # 解析比例列表
        ratio_list = json.loads(ratios)
        if not isinstance(ratio_list, list):
            raise ValueError("ratios 必须是数组")

        # 读取图片
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存临时文件
        temp_path = save_temp_image(image_rgb)

        # 解析填充策略
        padding_strategy = PaddingStrategy(padding)

        # 生成多个版本
        results = []
        for ratio in ratio_list:
            result = state.reframer.reframe(
                temp_path,
                target_ratio=tuple(ratio),
                padding=padding_strategy
            )
            results.append({
                "ratio": ratio,
                "size": list(result.new_size),
                "image_base64": image_to_base64(result.image)
            })

        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass

        return {
            "success": True,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量重构图失败: {str(e)}")


# ============================================================================
# 启动配置
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
