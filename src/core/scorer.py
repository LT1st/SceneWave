"""
构图评分模块 - 基于三分法则和视觉平衡评分
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class CompositionScore:
    """构图评分结果"""
    rule_of_thirds: float      # 三分法则 (0-30分)
    visual_balance: float      # 视觉平衡 (0-25分)
    subject_prominence: float  # 主体突出度 (0-25分)
    breathing_room: float      # 呼吸空间 (0-20分)

    @property
    def total(self) -> float:
        """总分 (0-100)"""
        return (self.rule_of_thirds + self.visual_balance +
                self.subject_prominence + self.breathing_room)

    @property
    def grade(self) -> str:
        """评级"""
        if self.total >= 85:
            return "优秀 ⭐⭐⭐⭐⭐"
        elif self.total >= 70:
            return "良好 ⭐⭐⭐⭐"
        elif self.total >= 55:
            return "一般 ⭐⭐⭐"
        elif self.total >= 40:
            return "待改进 ⭐⭐"
        else:
            return "需要重构 ⭐"

    def to_dict(self) -> dict:
        return {
            "rule_of_thirds": round(self.rule_of_thirds, 1),
            "visual_balance": round(self.visual_balance, 1),
            "subject_prominence": round(self.subject_prominence, 1),
            "breathing_room": round(self.breathing_room, 1),
            "total": round(self.total, 1),
            "grade": self.grade
        }


@dataclass
class CompositionIssue:
    """构图问题"""
    type: str          # 问题类型
    severity: str      # 严重程度: low/medium/high
    description: str   # 描述
    suggestion: str    # 建议


class CompositionScorer:
    """构图评分器"""

    def __init__(self):
        self.issues: List[CompositionIssue] = []

    def score(self,
              image_path: str,
              subject_bbox: Optional[Tuple[int, int, int, int]] = None,
              subject_center: Optional[Tuple[float, float]] = None,
              image_size: Optional[Tuple[int, int]] = None) -> CompositionScore:
        """
        计算构图评分

        Args:
            image_path: 图片路径
            subject_bbox: 主体边界框 (x1, y1, x2, y2)
            subject_center: 主体中心点 (cx, cy)
            image_size: 图片尺寸 (width, height) - 可选, 如果不提供会从图片读取

        Returns:
            CompositionScore 评分结果
        """
        self.issues = []

        # 获取图片尺寸
        if image_size is None:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            height, width = image.shape[:2]
        else:
            width, height = image_size

        # 如果没有主体信息, 计算图片中心作为默认
        if subject_center is None:
            if subject_bbox:
                x1, y1, x2, y2 = subject_bbox
                subject_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                # 使用图片中心
                subject_center = (width / 2, height / 2)

        # 计算各维度分数
        rule_of_thirds = self._score_rule_of_thirds(
            subject_center, width, height
        )
        visual_balance = self._score_visual_balance(
            subject_center, width, height
        )
        subject_prominence = self._score_subject_prominence(
            subject_bbox, width, height
        )
        breathing_room = self._score_breathing_room(
            subject_bbox, subject_center, width, height
        )

        return CompositionScore(
            rule_of_thirds=rule_of_thirds,
            visual_balance=visual_balance,
            subject_prominence=subject_prominence,
            breathing_room=breathing_room
        )

    def _score_rule_of_thirds(self,
                              center: Tuple[float, float],
                              width: int,
                              height: int) -> float:
        """
        三分法则评分 (0-30分)

        主体应该位于三分线的交点附近
        """
        cx, cy = center

        # 四个三分点
        third_points = [
            (width / 3, height / 3),
            (2 * width / 3, height / 3),
            (width / 3, 2 * height / 3),
            (2 * width / 3, 2 * height / 3)
        ]

        # 计算到最近三分点的距离
        min_distance = float('inf')
        for px, py in third_points:
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            min_distance = min(min_distance, dist)

        # 归一化距离
        max_dist = np.sqrt(width ** 2 + height ** 2) / 2
        normalized_dist = min_distance / max_dist

        # 转换为分数 (距离越近分数越高)
        score = max(0, 30 * (1 - normalized_dist * 2))

        # 检测问题
        if score < 15:
            self.issues.append(CompositionIssue(
                type="rule_of_thirds",
                severity="medium" if score > 8 else "high",
                description=f"主体偏离三分点 (得分: {score:.1f}/30)",
                suggestion="考虑将主体移动到三分线交点附近"
            ))

        return score

    def _score_visual_balance(self,
                              center: Tuple[float, float],
                              width: int,
                              height: int) -> float:
        """
        视觉平衡评分 (0-25分)

        主体不应太偏离图片中心
        """
        cx, cy = center
        img_center = (width / 2, height / 2)

        # 计算到图片中心的偏移
        offset_x = abs(cx - img_center[0]) / (width / 2)
        offset_y = abs(cy - img_center[1]) / (height / 2)
        total_offset = np.sqrt(offset_x ** 2 + offset_y ** 2)

        # 偏移越大分数越低
        score = max(0, 25 * (1 - total_offset))

        # 检测问题
        if total_offset > 0.4:
            self.issues.append(CompositionIssue(
                type="visual_balance",
                severity="high" if total_offset > 0.6 else "medium",
                description=f"主体偏离中心 (偏移: {total_offset:.2f})",
                suggestion="主体过于偏离中心, 考虑重构图"
            ))

        return score

    def _score_subject_prominence(self,
                                   bbox: Optional[Tuple[int, int, int, int]],
                                   width: int,
                                   height: int) -> float:
        """
        主体突出度评分 (0-25分)

        主体应该占据适当的面积
        """
        if bbox is None:
            return 15  # 无主体信息, 给中等分数

        x1, y1, x2, y2 = bbox
        subject_area = (x2 - x1) * (y2 - y1)
        image_area = width * height
        area_ratio = subject_area / image_area

        # 主体面积比例的理想范围: 15%-60%
        if area_ratio < 0.05:
            # 太小
            score = area_ratio / 0.05 * 10
            self.issues.append(CompositionIssue(
                type="subject_prominence",
                severity="high",
                description=f"主体过小 (占比: {area_ratio*100:.1f}%)",
                suggestion="主体太小, 考虑裁剪放大"
            ))
        elif area_ratio > 0.75:
            # 太大
            score = max(0, 25 - (area_ratio - 0.6) * 50)
            self.issues.append(CompositionIssue(
                type="subject_prominence",
                severity="medium",
                description=f"主体过大 (占比: {area_ratio*100:.1f}%)",
                suggestion="主体占比过大, 缺乏呼吸空间"
            ))
        elif 0.15 <= area_ratio <= 0.45:
            # 理想范围
            score = 25
        else:
            # 接近理想
            score = 20

        return score

    def _score_breathing_room(self,
                               bbox: Optional[Tuple[int, int, int, int]],
                               center: Tuple[float, float],
                               width: int,
                               height: int) -> float:
        """
        呼吸空间评分 (0-20分)

        主体周围应该有适当的空间
        """
        if bbox is None:
            return 12

        x1, y1, x2, y2 = bbox
        cx, cy = center

        # 计算四个方向的空间
        left_space = x1
        right_space = width - x2
        top_space = y1
        bottom_space = height - y2

        # 检查是否贴边
        edge_threshold = min(width, height) * 0.05
        edges_touching = sum([
            left_space < edge_threshold,
            right_space < edge_threshold,
            top_space < edge_threshold,
            bottom_space < edge_threshold
        ])

        if edges_touching >= 3:
            score = 5
            self.issues.append(CompositionIssue(
                type="breathing_room",
                severity="high",
                description="主体几乎贴边",
                suggestion="主体缺乏呼吸空间, 需要扩图或调整构图"
            ))
        elif edges_touching >= 2:
            score = 10
            self.issues.append(CompositionIssue(
                type="breathing_room",
                severity="medium",
                description="主体部分贴边",
                suggestion="主体贴边, 考虑增加留白"
            ))
        elif edges_touching == 1:
            score = 15
        else:
            score = 20

        return score

    def get_issues(self) -> List[CompositionIssue]:
        """获取检测到的所有构图问题"""
        return self.issues

    def get_suggestions(self) -> List[str]:
        """获取改进建议"""
        return [issue.suggestion for issue in self.issues]


# 测试代码
if __name__ == "__main__":
    scorer = CompositionScorer()

    # 测试评分
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        score = scorer.score(image_path)

        print("\n构图评分结果:")
        print(f"  三分法则: {score.rule_of_thirds:.1f}/30")
        print(f"  视觉平衡: {score.visual_balance:.1f}/25")
        print(f"  主体突出: {score.subject_prominence:.1f}/25")
        print(f"  呼吸空间: {score.breathing_room:.1f}/20")
        print(f"  总分: {score.total:.1f}/100")
        print(f"  评级: {score.grade}")

        if scorer.get_issues():
            print("\n检测到的问题:")
            for issue in scorer.get_issues():
                print(f"  - [{issue.severity}] {issue.description}")
                print(f"    建议: {issue.suggestion}")
