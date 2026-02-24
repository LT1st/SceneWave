"""
AI 扩图模块 - 使用 Stable Diffusion 进行智能扩展

支持：
- Outpainting: 向外扩展图片
- Inpainting: 修复/替换图片区域
- ControlNet 引导: 保持结构一致性
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Literal
import numpy as np
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OutpaintDirection(Enum):
    """扩图方向"""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ALL = "all"


class InpaintMode(Enum):
    """修复模式"""
    OBJECT_REMOVAL = "object_removal"      # 移除物体
    OBJECT_REPLACE = "object_replace"      # 替换物体
    BACKGROUND_FILL = "background_fill"    # 填充背景
    EXTEND = "extend"                      # 延伸内容


@dataclass
class OutpaintResult:
    """扩图结果"""
    image: np.ndarray                  # 扩图后的图片
    original_size: Tuple[int, int]     # 原始尺寸 (w, h)
    new_size: Tuple[int, int]          # 新尺寸 (w, h)
    direction: OutpaintDirection       # 扩展方向
    expand_pixels: int                 # 扩展像素数
    mask: Optional[np.ndarray]         # 使用的 mask (如有)
    generation_time: float             # 生成时间 (秒)


@dataclass
class InpaintResult:
    """修复结果"""
    image: np.ndarray                  # 修复后的图片
    original_size: Tuple[int, int]     # 原始尺寸
    mask: np.ndarray                   # 修复 mask
    mode: InpaintMode                  # 修复模式
    generation_time: float             # 生成时间


class AIOutpainter:
    """AI 扩图引擎"""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "cpu",
        enable_controlnet: bool = False,
        controlnet_model: Optional[str] = None
    ):
        """
        初始化 AI 扩图引擎

        Args:
            model_id: Stable Diffusion 模型 ID
            device: 运行设备 (cpu, cuda, mps)
            enable_controlnet: 是否启用 ControlNet
            controlnet_model: ControlNet 模型 ID
        """
        self.model_id = model_id
        self.device = device
        self.enable_controlnet = enable_controlnet
        self.controlnet_model = controlnet_model

        # 延迟加载模型
        self.pipe = None
        self.controlnet = None

        logger.info(f"AIOutpainter 初始化 (model={model_id}, device={device})")

    def _load_model(self):
        """延迟加载模型"""
        if self.pipe is not None:
            return

        try:
            from diffusers import StableDiffusionInpaintPipeline, ControlNetModel
            import torch

            logger.info("加载 Stable Diffusion 模型...")

            # 加载 ControlNet (如果启用)
            if self.enable_controlnet and self.controlnet_model:
                self.controlnet = ControlNetModel.from_pretrained(
                    self.controlnet_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )

            # 加载主模型
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )

            # 设置设备
            self.pipe = self.pipe.to(self.device)

            # 优化设置
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                # self.pipe.enable_xformers_memory_efficient_attention()

            logger.info("模型加载完成")

        except ImportError:
            logger.error("未安装 diffusers 库，请运行: pip install diffusers torch")
            raise ImportError("需要安装 diffusers 和 torch")

    def outpaint(
        self,
        image_path: str,
        direction: OutpaintDirection = OutpaintDirection.ALL,
        expand_pixels: int = 256,
        prompt: str = "",
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> OutpaintResult:
        """
        AI 扩图 - 向外扩展图片

        Args:
            image_path: 图片路径
            direction: 扩展方向
            expand_pixels: 扩展像素数
            prompt: 提示词
            negative_prompt: 负面提示词
            num_inference_steps: 推理步数
            guidance_scale: 引导系数
            seed: 随机种子

        Returns:
            OutpaintResult: 扩图结果
        """
        import time
        start_time = time.time()

        # 加载模型
        self._load_model()

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 计算扩展后的尺寸
        if direction == OutpaintDirection.LEFT:
            new_w, new_h = w + expand_pixels, h
            offset_x, offset_y = expand_pixels, 0
        elif direction == OutpaintDirection.RIGHT:
            new_w, new_h = w + expand_pixels, h
            offset_x, offset_y = 0, 0
        elif direction == OutpaintDirection.TOP:
            new_w, new_h = w, h + expand_pixels
            offset_x, offset_y = 0, expand_pixels
        elif direction == OutpaintDirection.BOTTOM:
            new_w, new_h = w, h + expand_pixels
            offset_x, offset_y = 0, 0
        else:  # ALL
            new_w, new_h = w + 2 * expand_pixels, h + 2 * expand_pixels
            offset_x, offset_y = expand_pixels, expand_pixels

        # 创建 mask (白色区域为需要生成的区域)
        mask = np.ones((new_h, new_w), dtype=np.uint8) * 255
        mask[offset_y:offset_y+h, offset_x:offset_x+w] = 0

        # 创建初始化图片 (原图放在中间，周围用边缘填充)
        init_image = cv2.copyMakeBorder(
            image,
            offset_y, new_h - h - offset_y,
            offset_x, new_w - w - offset_x,
            cv2.BORDER_REPLICATE
        )

        # 转换为 PIL Image
        from PIL import Image
        init_image_pil = Image.fromarray(init_image)
        mask_pil = Image.fromarray(mask)

        # 设置随机种子
        import torch
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 执行扩图
        logger.info(f"开始 AI 扩图: {direction.value}, {expand_pixels}px")

        result = self.pipe(
            prompt=prompt or "high quality, detailed, natural extension",
            negative_prompt=negative_prompt,
            image=init_image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=new_h,
            width=new_w
        )

        result_image = np.array(result.images[0])

        generation_time = time.time() - start_time
        logger.info(f"AI 扩图完成，耗时 {generation_time:.2f} 秒")

        return OutpaintResult(
            image=result_image,
            original_size=(w, h),
            new_size=(new_w, new_h),
            direction=direction,
            expand_pixels=expand_pixels,
            mask=mask,
            generation_time=generation_time
        )

    def inpaint(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        mask_bbox: Optional[Tuple[int, int, int, int]] = None,
        prompt: str = "",
        negative_prompt: str = "blurry, low quality",
        mode: InpaintMode = InpaintMode.BACKGROUND_FILL,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> InpaintResult:
        """
        AI 修复 - 修复/替换图片区域

        Args:
            image_path: 图片路径
            mask_path: mask 图片路径 (白色区域为需要修复的区域)
            mask_bbox: 修复区域边界框 (x1, y1, x2, y2)
            prompt: 提示词
            negative_prompt: 负面提示词
            mode: 修复模式
            num_inference_steps: 推理步数
            guidance_scale: 引导系数
            seed: 随机种子

        Returns:
            InpaintResult: 修复结果
        """
        import time
        start_time = time.time()

        # 加载模型
        self._load_model()

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 创建或读取 mask
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法读取 mask: {mask_path}")
        elif mask_bbox:
            x1, y1, x2, y2 = mask_bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
        else:
            raise ValueError("必须提供 mask_path 或 mask_bbox")

        # 根据模式调整提示词
        if mode == InpaintMode.OBJECT_REMOVAL:
            prompt = prompt or "seamless background, natural texture"
        elif mode == InpaintMode.OBJECT_REPLACE:
            pass  # 使用用户提供的 prompt
        elif mode == InpaintMode.BACKGROUND_FILL:
            prompt = prompt or "natural background, consistent with surroundings"
        elif mode == InpaintMode.EXTEND:
            prompt = prompt or "natural extension of surrounding content"

        # 转换为 PIL Image
        from PIL import Image
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)

        # 设置随机种子
        import torch
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 执行修复
        logger.info(f"开始 AI 修复: mode={mode.value}")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        result_image = np.array(result.images[0])

        generation_time = time.time() - start_time
        logger.info(f"AI 修复完成，耗时 {generation_time:.2f} 秒")

        return InpaintResult(
            image=result_image,
            original_size=(w, h),
            mask=mask,
            mode=mode,
            generation_time=generation_time
        )

    def smart_outpaint(
        self,
        image_path: str,
        target_ratio: Tuple[int, int],
        prompt: str = "",
        expand_pixels: int = 256,
        **kwargs
    ) -> OutpaintResult:
        """
        智能扩图 - 自动扩展到目标比例

        Args:
            image_path: 图片路径
            target_ratio: 目标比例 (宽, 高)
            prompt: 提示词
            expand_pixels: 每次扩展的像素数
            **kwargs: 其他参数

        Returns:
            OutpaintResult: 扩图结果
        """
        # 读取图片获取尺寸
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        h, w = image.shape[:2]
        current_ratio = w / h
        target_ratio_val = target_ratio[0] / target_ratio[1]

        # 确定需要扩展的方向
        if abs(current_ratio - target_ratio_val) < 0.01:
            # 已经是目标比例，不需要扩展
            return OutpaintResult(
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                original_size=(w, h),
                new_size=(w, h),
                direction=OutpaintDirection.ALL,
                expand_pixels=0,
                mask=None,
                generation_time=0
            )

        # 计算需要扩展的像素
        if current_ratio < target_ratio_val:
            # 需要加宽
            target_w = int(h * target_ratio_val)
            expand_total = target_w - w
            direction = OutpaintDirection.RIGHT
        else:
            # 需要加高
            target_h = int(w / target_ratio_val)
            expand_total = target_h - h
            direction = OutpaintDirection.BOTTOM

        # 可能需要多次扩展
        current_image_path = image_path
        total_expanded = 0

        while total_expanded < expand_total:
            current_expand = min(expand_pixels, expand_total - total_expanded)

            result = self.outpaint(
                current_image_path,
                direction=direction,
                expand_pixels=current_expand,
                prompt=prompt,
                **kwargs
            )

            # 保存中间结果
            temp_path = f"/tmp/outpaint_temp_{total_expanded}.jpg"
            cv2.imwrite(
                temp_path,
                cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            )
            current_image_path = temp_path
            total_expanded += current_expand

        # 清理临时文件
        import os
        for f in Path("/tmp").glob("outpaint_temp_*.jpg"):
            try:
                os.remove(f)
            except:
                pass

        return result


# ============================================================================
# 传统扩图方法 (无需 AI)
# ============================================================================

class TraditionalOutpainter:
    """传统扩图方法 - 不需要 AI 模型"""

    @staticmethod
    def extend(image: np.ndarray, direction: OutpaintDirection, pixels: int) -> np.ndarray:
        """
        使用边缘延伸扩展图片

        Args:
            image: 输入图片
            direction: 扩展方向
            pixels: 扩展像素数

        Returns:
            np.ndarray: 扩展后的图片
        """
        h, w = image.shape[:2]

        if direction == OutpaintDirection.LEFT:
            return cv2.copyMakeBorder(image, 0, 0, pixels, 0, cv2.BORDER_REPLICATE)
        elif direction == OutpaintDirection.RIGHT:
            return cv2.copyMakeBorder(image, 0, 0, 0, pixels, cv2.BORDER_REPLICATE)
        elif direction == OutpaintDirection.TOP:
            return cv2.copyMakeBorder(image, pixels, 0, 0, 0, cv2.BORDER_REPLICATE)
        elif direction == OutpaintDirection.BOTTOM:
            return cv2.copyMakeBorder(image, 0, pixels, 0, 0, cv2.BORDER_REPLICATE)
        else:  # ALL
            return cv2.copyMakeBorder(
                image, pixels, pixels, pixels, pixels, cv2.BORDER_REPLICATE
            )

    @staticmethod
    def blur_extend(image: np.ndarray, direction: OutpaintDirection, pixels: int) -> np.ndarray:
        """
        使用模糊背景扩展图片

        Args:
            image: 输入图片
            direction: 扩展方向
            pixels: 扩展像素数

        Returns:
            np.ndarray: 扩展后的图片
        """
        # 创建模糊背景
        blurred = cv2.GaussianBlur(image, (51, 51), 0)

        h, w = image.shape[:2]

        if direction == OutpaintDirection.LEFT:
            result = cv2.copyMakeBorder(blurred, 0, 0, pixels, 0, cv2.BORDER_REPLICATE)
            result[:, pixels:] = image
        elif direction == OutpaintDirection.RIGHT:
            result = cv2.copyMakeBorder(blurred, 0, 0, 0, pixels, cv2.BORDER_REPLICATE)
            result[:, :w] = image
        elif direction == OutpaintDirection.TOP:
            result = cv2.copyMakeBorder(blurred, pixels, 0, 0, 0, cv2.BORDER_REPLICATE)
            result[pixels:, :] = image
        elif direction == OutpaintDirection.BOTTOM:
            result = cv2.copyMakeBorder(blurred, 0, pixels, 0, 0, cv2.BORDER_REPLICATE)
            result[:h, :] = image
        else:  # ALL
            result = cv2.copyMakeBorder(
                blurred, pixels, pixels, pixels, pixels, cv2.BORDER_REPLICATE
            )
            result[pixels:pixels+h, pixels:pixels+w] = image

        return result


# 测试代码
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python outpainter.py <image_path> [direction] [pixels]")
        print("示例: python outpainter.py photo.jpg right 256")
        sys.exit(1)

    image_path = sys.argv[1]
    direction = sys.argv[2] if len(sys.argv) > 2 else "all"
    pixels = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    # 使用传统方法快速测试
    outpainter = TraditionalOutpainter()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    direction_enum = OutpaintDirection(direction.lower())
    result = outpainter.extend(image_rgb, direction_enum, pixels)

    # 保存结果
    output = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    output_path = f"outpaint_{direction}_{pixels}px.jpg"
    cv2.imwrite(output_path, output)

    print(f"传统扩图完成: {output_path}")
    print(f"尺寸: {image.shape[:2][::-1]} -> {result.shape[:2][::-1]}")
