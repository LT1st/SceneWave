"""
主体检测模块 - 使用 YOLOv8 检测图片中的主体
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectedSubject:
    """检测到的主体"""
    label: str           # 类别名称
    confidence: float    # 置信度
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]      # 中心点 (cx, cy)
    area: int            # 面积

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class SubjectDetector:
    """主体检测器"""

    # 常见主体类别 (COCO 数据集)
    MAIN_SUBJECTS = {
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake'
    }

    # 人物/动物优先级更高
    PRIORITY_SUBJECTS = {
        'person': 10,
        'dog': 9, 'cat': 9, 'bird': 8,
        'horse': 8, 'cow': 7, 'elephant': 7,
        'car': 6, 'motorcycle': 6, 'bicycle': 5,
    }

    def __init__(self, model_size: str = "n", device: str = None):
        """
        初始化检测器

        Args:
            model_size: YOLO 模型大小 (n/s/m/l/x)
            device: 设备 (cuda/cpu/mps)
        """
        model_name = f"yolov8{model_size}.pt"
        self.model = YOLO(model_name)
        self.device = device

    def detect(self,
               image_path: str,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> List[DetectedSubject]:
        """
        检测图片中的主体

        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值

        Returns:
            检测到的主体列表
        """
        # 运行检测
        results = self.model(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        subjects = []

        # 解析检测结果
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                # 获取类别
                class_id = int(boxes.cls[i].item())
                label = result.names[class_id]

                # 只保留常见主体
                if label not in self.MAIN_SUBJECTS:
                    continue

                # 获取边界框
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # 计算中心和面积
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                subject = DetectedSubject(
                    label=label,
                    confidence=float(boxes.conf[i].item()),
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    area=area
                )
                subjects.append(subject)

        # 按优先级和面积排序
        subjects.sort(key=self._subject_priority, reverse=True)

        return subjects

    def _subject_priority(self, subject: DetectedSubject) -> float:
        """计算主体优先级分数"""
        # 基础分: 优先级类别
        priority_score = self.PRIORITY_SUBJECTS.get(subject.label, 1)

        # 置信度加成
        confidence_score = subject.confidence * 5

        # 面积加成 (归一化)
        area_score = min(subject.area / 100000, 5)  # 最多 5 分

        return priority_score + confidence_score + area_score

    def get_main_subject(self, subjects: List[DetectedSubject]) -> Optional[DetectedSubject]:
        """获取主要主体 (面积最大 + 优先级最高)"""
        if not subjects:
            return None
        return subjects[0]  # 已经按优先级排序

    def draw_detections(self,
                       image_path: str,
                       subjects: List[DetectedSubject],
                       output_path: str = None) -> np.ndarray:
        """
        在图片上绘制检测结果

        Args:
            image_path: 图片路径
            subjects: 检测到的主体
            output_path: 输出路径 (可选)

        Returns:
            绘制后的图片
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        for i, subject in enumerate(subjects):
            x1, y1, x2, y2 = subject.bbox

            # 主要主体用红色, 其他用蓝色
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            thickness = 3 if i == 0 else 2

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # 绘制中心点
            cx, cy = map(int, subject.center)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            # 绘制标签
            label_text = f"{subject.label} {subject.confidence:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if output_path:
            cv2.imwrite(output_path, image)

        return image


# 测试代码
if __name__ == "__main__":
    detector = SubjectDetector(model_size="n")

    # 测试检测
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        subjects = detector.detect(image_path)

        print(f"\n检测到 {len(subjects)} 个主体:")
        for i, s in enumerate(subjects):
            print(f"  {i+1}. {s.label} (置信度: {s.confidence:.2f}, "
                  f"位置: {s.bbox}, 中心: ({s.center[0]:.0f}, {s.center[1]:.0f}))")

        # 绘制结果
        detector.draw_detections(image_path, subjects, "detection_result.jpg")
        print("\n结果已保存到 detection_result.jpg")
