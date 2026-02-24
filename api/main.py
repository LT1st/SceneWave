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

from src.core import (
    SubjectDetector,
    CompositionScorer,
    Reframer,
    AIOutpainter,
    TraditionalOutpainter,
    OutpaintDirection,
    InpaintMode
)
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


class OutpaintRequest(BaseModel):
    """AI 扩图请求"""
    direction: Literal["left", "right", "top", "bottom", "all"] = "all"
    expand_pixels: int = Field(description="扩展像素数", ge=64, le=1024)
    prompt: str = Field(default="", description="提示词")
    negative_prompt: str = Field(default="blurry, low quality", description="负面提示词")
    use_ai: bool = Field(default=True, description="是否使用 AI 扩图")
    num_inference_steps: int = Field(default=50, description="AI 推理步数")
    guidance_scale: float = Field(default=7.5, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")


class OutpaintResponse(BaseModel):
    """AI 扩图响应"""
    success: bool
    original_size: List[int]
    new_size: List[int]
    direction: str
    expand_pixels: int
    use_ai: bool
    generation_time: float
    image_base64: str


class InpaintRequest(BaseModel):
    """AI 修复请求"""
    mask_bbox: Optional[List[int]] = Field(default=None, description="修复区域 [x1, y1, x2, y2]")
    prompt: str = Field(default="", description="提示词")
    negative_prompt: str = Field(default="blurry, low quality", description="负面提示词")
    mode: Literal["object_removal", "object_replace", "background_fill", "extend"] = "background_fill"
    num_inference_steps: int = Field(default=50, description="AI 推理步数")
    guidance_scale: float = Field(default=7.5, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")


class InpaintResponse(BaseModel):
    """AI 修复响应"""
    success: bool
    original_size: List[int]
    mode: str
    generation_time: float
    image_base64: str


# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="SceneWeave API",
    description="AI 智能图片重构图 API - 支持构图分析、智能重构图、AI 扩图",
    version="2.0.0"
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
        self.ai_outpainter: Optional[AIOutpainter] = None
        self.traditional_outpainter = TraditionalOutpainter()

    def get_detector(self) -> SubjectDetector:
        if self.detector is None:
            self.detector = SubjectDetector(model_size="n")
        return self.detector

    def get_ai_outpainter(self) -> AIOutpainter:
        if self.ai_outpainter is None:
            self.ai_outpainter = AIOutpainter(device="cpu")
        return self.ai_outpainter

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


@app.post("/api/v1/outpaint", response_model=OutpaintResponse)
async def outpaint_image(
    file: UploadFile = File(...),
    direction: str = "all",
    expand_pixels: int = 256,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality",
    use_ai: bool = True,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
):
    """
    AI 扩图 - 智能扩展图片

    - direction: 扩展方向 (left, right, top, bottom, all)
    - expand_pixels: 扩展像素数 (64-1024)
    - prompt: 提示词
    - use_ai: 是否使用 AI (True=Stable Diffusion, False=传统方法)
    - num_inference_steps: AI 推理步数
    - guidance_scale: 引导系数
    - seed: 随机种子
    """
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    # 验证参数
    valid_directions = ["left", "right", "top", "bottom", "all"]
    if direction not in valid_directions:
        raise HTTPException(status_code=400, detail=f"方向必须是: {', '.join(valid_directions)}")

    if expand_pixels < 64 or expand_pixels > 1024:
        raise HTTPException(status_code=400, detail="扩展像素数必须在 64-1024 之间")

    try:
        # 读取图片
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存临时文件
        temp_path = save_temp_image(image_rgb)

        # 执行扩图
        direction_enum = OutpaintDirection(direction)

        if use_ai:
            # 使用 AI 扩图
            outpainter = state.get_ai_outpainter()
            result = outpainter.outpaint(
                temp_path,
                direction=direction_enum,
                expand_pixels=expand_pixels,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
        else:
            # 使用传统方法
            outpainter = state.traditional_outpainter
            result_image = outpainter.extend(image_rgb, direction_enum, expand_pixels)

            from src.core.outpainter import OutpaintResult
            result = OutpaintResult(
                image=result_image,
                original_size=(image_rgb.shape[1], image_rgb.shape[0]),
                new_size=(result_image.shape[1], result_image.shape[0]),
                direction=direction_enum,
                expand_pixels=expand_pixels,
                mask=None,
                generation_time=0
            )

        # 转换为 base64
        image_base64 = image_to_base64(result.image)

        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass

        return OutpaintResponse(
            success=True,
            original_size=list(result.original_size),
            new_size=list(result.new_size),
            direction=direction,
            expand_pixels=result.expand_pixels,
            use_ai=use_ai,
            generation_time=result.generation_time,
            image_base64=image_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 扩图失败: {str(e)}")


@app.post("/api/v1/inpaint", response_model=InpaintResponse)
async def inpaint_image(
    file: UploadFile = File(...),
    mask_file: Optional[UploadFile] = None,
    mask_bbox: Optional[str] = None,  # JSON: [x1, y1, x2, y2]
    prompt: str = "",
    negative_prompt: str = "blurry, low quality",
    mode: str = "background_fill",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
):
    """
    AI 修复 - 修复/替换图片区域

    - mask_file: mask 图片文件 (白色区域为需要修复的区域)
    - mask_bbox: 修复区域边界框 JSON: [x1, y1, x2, y2]
    - prompt: 提示词
    - mode: 修复模式 (object_removal, object_replace, background_fill, extend)
    - num_inference_steps: AI 推理步数
    - guidance_scale: 引导系数
    - seed: 随机种子
    """
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    # 必须提供 mask_file 或 mask_bbox
    if mask_file is None and mask_bbox is None:
        raise HTTPException(status_code=400, detail="必须提供 mask_file 或 mask_bbox")

    try:
        import json

        # 读取图片
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存临时文件
        temp_path = save_temp_image(image_rgb)

        # 处理 mask
        mask_path = None
        mask_bbox_tuple = None

        if mask_file:
            mask_bytes = await mask_file.read()
            mask = decode_image(mask_bytes)
            mask_path = save_temp_image(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

        if mask_bbox:
            mask_bbox_tuple = tuple(int(x) for x in json.loads(mask_bbox))

        # 执行修复
        outpainter = state.get_ai_outpainter()
        mode_enum = InpaintMode(mode)

        result = outpainter.inpaint(
            temp_path,
            mask_path=mask_path,
            mask_bbox=mask_bbox_tuple,
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode_enum,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        # 转换为 base64
        image_base64 = image_to_base64(result.image)

        # 清理临时文件
        try:
            os.remove(temp_path)
            if mask_path:
                os.remove(mask_path)
        except:
            pass

        return InpaintResponse(
            success=True,
            original_size=list(result.original_size),
            mode=mode,
            generation_time=result.generation_time,
            image_base64=image_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 修复失败: {str(e)}")


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
