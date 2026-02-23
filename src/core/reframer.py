"""
重构图模块 - 智能裁剪和比例转换
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Callable
import cv2
import numpy as np


class AspectRatio(Enum):
    """常用比例预设"""
    SQUARE = (1, 1)           # 1:1 正方形
    PORTRAIT_45 = (4, 5)      # 4:5 Instagram/小红书
    STORY_916 = (9, 16)       # 9:16 Story
    LANDSCAPE_169 = (16, 9)   # 16:9 横屏
    BANNER_31 = (3, 1)        # 3:1 Banner
    COVER_23 = (2, 3)         # 2:3 小红书封面
    CLASSIC_43 = (4, 3)       # 4:3 经典
    CINEMA_219 = (21, 9)      # 21:9 电影


@dataclass
class ReframeResult:
    """重构图结果"""
    image: np.ndarray              # 重构图后的图片
    original_size: Tuple[int, int] # 原始尺寸 (w, h)
    new_size: Tuple[int, int]      # 新尺寸 (w, h)
    crop_box: Tuple[int, int, int, int]  # 裁剪区域 (x1, y1, x2, y2)
    subject_center: Tuple[float, float]  # 主体在新图中的位置
    padding_applied: bool          # 是否应用了填充


class PaddingStrategy(Enum):
    """填充策略"""
    NONE = "none"           # 不填充, 只裁剪
    BLUR = "blur"           # 模糊背景填充
    COLOR = "color"         # 纯色填充
    MIRROR = "mirror"       # 镜像填充
    EXTEND = "extend"       # 边缘延伸


class Reframer:
    """重构图引擎"""

    # 社交媒体预设
    SOCIAL_PRESETS = {
        "instagram_square": AspectRatio.SQUARE,
        "instagram_portrait": AspectRatio.PORTRAIT_45,
        "instagram_story": AspectRatio.STORY_916,
        "youtube_thumbnail": AspectRatio.LANDSCAPE_169,
        "xiaohongshu_cover": AspectRatio.COVER_23,
        "banner": AspectRatio.BANNER_31,
    }

    def __init__(self):
        pass

    def reframe(self,
                image_path: str,
                target_ratio: Tuple[int, int] = (4, 5),
                subject_center: Optional[Tuple[float, float]] = None,
                subject_bbox: Optional[Tuple[int, int, int, int]] = None,
                padding: PaddingStrategy = PaddingStrategy.NONE,
                padding_color: Tuple[int, int, int] = (255, 255, 255)) -> ReframeResult:
        """
        重构图图片

        Args:
            image_path: 图片路径
            target_ratio: 目标比例 (宽, 高)
            subject_center: 主体中心点 (用于智能裁剪)
            subject_bbox: 主体边界框
            padding: 填充策略
            padding_color: 填充颜色 (RGB)

        Returns:
            ReframeResult 重构图结果
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # 计算目标尺寸
        target_w, target_h = target_ratio
        target_aspect = target_w / target_h
        current_aspect = width / height

        # 确定主体位置
        if subject_center is None:
            if subject_bbox:
                x1, y1, x2, y2 = subject_bbox
                subject_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                subject_center = (width / 2, height / 2)

        # 判断是否需要填充
        needs_padding = False
        if current_aspect > target_aspect:
            # 当前图片更宽, 需要在上下填充
            needs_padding = True
        elif current_aspect < target_aspect:
            # 当前图片更高, 需要在左右填充
            needs_padding = True

        # 应用填充策略
        if needs_padding and padding != PaddingStrategy.NONE:
            result_image, padding_applied = self._apply_padding(
                image, target_ratio, subject_center, padding, padding_color
            )
            crop_box = (0, 0, result_image.shape[1], result_image.shape[0])
        else:
            # 智能裁剪
            result_image, crop_box = self._smart_crop(
                image, target_ratio, subject_center, subject_bbox
            )
            padding_applied = False

        # 计算主体在新图中的位置
        new_height, new_width = result_image.shape[:2]
        new_subject_center = (
            subject_center[0] - crop_box[0],
            subject_center[1] - crop_box[1]
        )

        return ReframeResult(
            image=result_image,
            original_size=(width, height),
            new_size=(new_width, new_height),
            crop_box=crop_box,
            subject_center=new_subject_center,
            padding_applied=padding_applied
        )

    def _smart_crop(self,
                    image: np.ndarray,
                    target_ratio: Tuple[int, int],
                    subject_center: Tuple[float, float],
                    subject_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple]:
        """智能裁剪, 保持主体在合适位置"""
        height, width = image.shape[:2]
        target_w, target_h = target_ratio
        target_aspect = target_w / target_h
        current_aspect = width / height

        # 计算裁剪尺寸
        if current_aspect > target_aspect:
            # 图片更宽, 需要裁剪宽度
            new_height = height
            new_width = int(height * target_aspect)
        else:
            # 图片更高, 需要裁剪高度
            new_width = width
            new_height = int(width / target_aspect)

        # 计算裁剪位置 (以主体为中心)
        cx, cy = subject_center

        # 确保 subject_bbox 在裁剪区域内
        if subject_bbox:
            x1, y1, x2, y2 = subject_bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # 裁剪区域必须包含整个 bbox
            new_width = max(new_width, bbox_width + 20)
            new_height = max(new_height, bbox_height + 20)

            # 重新计算裁剪位置
            crop_x1 = max(0, int(cx - new_width / 2))
            crop_y1 = max(0, int(cy - new_height / 2))
        else:
            crop_x1 = max(0, min(int(cx - new_width / 2), width - new_width))
            crop_y1 = max(0, min(int(cy - new_height / 2), height - new_height))

        # 确保不超出边界
        if crop_x1 + new_width > width:
            crop_x1 = width - new_width
        if crop_y1 + new_height > height:
            crop_y1 = height - new_height

        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = crop_x1 + new_width
        crop_y2 = crop_y1 + new_height

        # 执行裁剪
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        return cropped, (crop_x1, crop_y1, crop_x2, crop_y2)

    def _apply_padding(self,
                       image: np.ndarray,
                       target_ratio: Tuple[int, int],
                       subject_center: Tuple[float, float],
                       padding: PaddingStrategy,
                       padding_color: Tuple[int, int, int]) -> Tuple[np.ndarray, bool]:
        """应用填充策略"""
        height, width = image.shape[:2]
        target_w, target_h = target_ratio
        target_aspect = target_w / target_h
        current_aspect = width / height

        # 计算新尺寸
        if current_aspect > target_aspect:
            # 需要增加高度
            new_width = width
            new_height = int(width / target_aspect)
            pad_height = new_height - height
            pad_width = 0
        else:
            # 需要增加宽度
            new_height = height
            new_width = int(height * target_aspect)
            pad_width = new_width - width
            pad_height = 0

        # 生成填充
        if padding == PaddingStrategy.BLUR:
            # 模糊背景填充
            blurred = cv2.GaussianBlur(image, (51, 51), 0)
            result = cv2.resize(blurred, (new_width, new_height))

            # 将原图放到中心
            y_offset = (new_height - height) // 2
            x_offset = (new_width - width) // 2
            result[y_offset:y_offset+height, x_offset:x_offset+width] = image

        elif padding == PaddingStrategy.COLOR:
            # 纯色填充
            result = np.full((new_height, new_width, 3), padding_color, dtype=np.uint8)

            # 将原图放到中心
            y_offset = (new_height - height) // 2
            x_offset = (new_width - width) // 2
            result[y_offset:y_offset+height, x_offset:x_offset+width] = image

        elif padding == PaddingStrategy.MIRROR:
            # 镜像填充
            if pad_height > 0:
                # 上下填充
                top_pad = cv2.flip(image[:pad_height//2 + 10, :], 0)
                bottom_pad = cv2.flip(image[-(pad_height//2 + 10):, :], 0)
                result = cv2.vconcat([top_pad, image, bottom_pad])
                result = cv2.resize(result, (new_width, new_height))
            else:
                # 左右填充
                left_pad = cv2.flip(image[:, :pad_width//2 + 10], 1)
                right_pad = cv2.flip(image[:, -(pad_width//2 + 10):], 1)
                result = cv2.hconcat([left_pad, image, right_pad])
                result = cv2.resize(result, (new_width, new_height))

        else:
            # EXTEND: 边缘延伸
            result = cv2.copyMakeBorder(
                image,
                pad_height // 2 if pad_height > 0 else 0,
                (pad_height + 1) // 2 if pad_height > 0 else 0,
                pad_width // 2 if pad_width > 0 else 0,
                (pad_width + 1) // 2 if pad_width > 0 else 0,
                cv2.BORDER_REPLICATE
            )
            result = cv2.resize(result, (new_width, new_height))

        return result, True

    def reframe_multiple(self,
                        image_path: str,
                        ratios: List[Tuple[int, int]],
                        subject_center: Optional[Tuple[float, float]] = None,
                        subject_bbox: Optional[Tuple[int, int, int, int]] = None,
                        padding: PaddingStrategy = PaddingStrategy.NONE) -> List[ReframeResult]:
        """一次生成多个比例的重构图"""
        results = []
        for ratio in ratios:
            result = self.reframe(
                image_path, ratio, subject_center, subject_bbox, padding
            )
            results.append(result)
        return results


# 测试代码
if __name__ == "__main__":
    reframer = Reframer()

    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        # 测试不同比例
        results = reframer.reframe_multiple(
            image_path,
            ratios=[(1, 1), (4, 5), (16, 9)],
            padding=PaddingStrategy.BLUR
        )

        print(f"\n生成了 {len(results)} 个重构图版本:")
        for i, result in enumerate(results):
            ratio = f"{result.new_size[0]}x{result.new_size[1]}"
            print(f"  {i+1}. 尺寸: {result.original_size} -> {result.new_size}")

            # 保存
            output = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"reframe_{i+1}_{ratio}.jpg", output)

        print("\n结果已保存!")
