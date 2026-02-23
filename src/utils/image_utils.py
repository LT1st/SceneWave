"""图像处理工具函数"""

from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """加载图片为 RGB 格式"""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: str, quality: int = 95) -> None:
    """保存图片"""
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])


def resize_image(image: np.ndarray,
                 max_size: int = 1920,
                 min_size: int = 640) -> np.ndarray:
    """按比例调整图片大小"""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    min_dim = min(h, w)

    if max_dim > max_size:
        scale = max_size / max_dim
    elif min_dim < min_size:
        scale = min_size / min_dim
    else:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_grid(image: np.ndarray,
              divisions: int = 3,
              color: Tuple[int, int, int] = (255, 255, 255),
              thickness: int = 1,
              alpha: float = 0.5) -> np.ndarray:
    """在图片上绘制网格 (用于三分法则可视化)"""
    result = image.copy()
    h, w = result.shape[:2]

    # 绘制垂直线
    for i in range(1, divisions):
        x = w * i // divisions
        cv2.line(result, (x, 0), (x, h), color, thickness)

    # 绘制水平线
    for i in range(1, divisions):
        y = h * i // divisions
        cv2.line(result, (0, y), (w, y), color, thickness)

    # 混合
    return cv2.addWeighted(image, 1 - alpha, result, alpha, 0)


def create_comparison_view(original: np.ndarray,
                           result: np.ndarray,
                           labels: Tuple[str, str] = ("原图", "重构图")) -> np.ndarray:
    """创建对比视图"""
    # 调整尺寸一致
    h = max(original.shape[0], result.shape[0])

    orig_resized = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
    result_resized = cv2.resize(result, (int(result.shape[1] * h / result.shape[0]), h))

    # 添加标签
    def add_label(img, label):
        labeled = np.zeros((img.shape[0] + 40, img.shape[1], 3), dtype=np.uint8)
        labeled[40:, :] = img
        cv2.putText(labeled, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return labeled

    orig_labeled = add_label(orig_resized, labels[0])
    result_labeled = add_label(result_resized, labels[1])

    # 水平拼接
    return np.hstack([orig_labeled, result_labeled])
