"""
构图评分器测试
"""

import pytest
import numpy as np

from src.core import CompositionScorer, CompositionScore


class TestCompositionScorer:
    """构图评分器测试类"""

    def test_init(self, scorer):
        """测试初始化"""
        assert scorer is not None

    def test_score_with_subject(self, scorer, sample_image_file, sample_subject):
        """测试有主体的评分"""
        score = scorer.score(
            sample_image_file,
            subject_bbox=sample_subject.bbox,
            subject_center=sample_subject.center
        )

        assert isinstance(score, CompositionScore)
        assert 0 <= score.total <= 100
        assert 0 <= score.rule_of_thirds <= 30
        assert 0 <= score.visual_balance <= 25
        assert 0 <= score.subject_prominence <= 25
        assert 0 <= score.breathing_room <= 20

    def test_score_without_subject(self, scorer, sample_image_file):
        """测试无主体的评分"""
        score = scorer.score(sample_image_file)

        assert isinstance(score, CompositionScore)
        assert 0 <= score.total <= 100

    def test_rule_of_thirds_scoring(self, scorer):
        """测试三分法则评分"""
        # 主体在三分点应该得分高
        center_at_third = (640 * 2 / 3, 480 * 2 / 3)
        # 这个测试依赖具体实现，这里只是示例

    def test_visual_balance_scoring(self, scorer):
        """测试视觉平衡评分"""
        # 主体在中心应该平衡
        center_subject = (320, 240)
        # 这个测试依赖具体实现

    def test_score_grading(self, scorer, sample_image_file):
        """测试评分等级"""
        score = scorer.score(sample_image_file)

        assert score.grade in ["S", "A", "B", "C", "D"]

        # 验证等级规则
        if score.total >= 90:
            assert score.grade == "S"
        elif score.total >= 80:
            assert score.grade == "A"
        elif score.total >= 70:
            assert score.grade == "B"
        elif score.total >= 60:
            assert score.grade == "C"
        else:
            assert score.grade == "D"


class TestCompositionScore:
    """构图评分测试类"""

    def test_score_total_calculation(self):
        """测试总分计算"""
        score = CompositionScore(
            rule_of_thirds=25,
            visual_balance=20,
            subject_prominence=20,
            breathing_room=15
        )

        assert score.total == 80

    def test_score_max_values(self):
        """测试最大值"""
        score = CompositionScore(
            rule_of_thirds=30,
            visual_balance=25,
            subject_prominence=25,
            breathing_room=20
        )

        assert score.total == 100
        assert score.grade == "S"

    def test_score_min_values(self):
        """测试最小值"""
        score = CompositionScore(
            rule_of_thirds=0,
            visual_balance=0,
            subject_prominence=0,
            breathing_room=0
        )

        assert score.total == 0
        assert score.grade == "D"

    def test_score_boundary_values(self):
        """测试边界值"""
        # S 级边界
        score_s = CompositionScore(30, 25, 25, 20)
        assert score_s.grade == "S"

        # A 级边界
        score_a = CompositionScore(28, 22, 22, 18)
        assert score_a.grade == "A"

        # B 级边界
        score_b = CompositionScore(20, 18, 18, 14)
        assert score_b.grade == "B"

        # C 级边界
        score_c = CompositionScore(15, 15, 15, 10)
        assert score_c.grade == "C"

        # D 级边界
        score_d = CompositionScore(10, 10, 10, 5)
        assert score_d.grade == "D"
