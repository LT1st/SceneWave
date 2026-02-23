"""
重构图器测试
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.core.reframer import (
    Reframer,
    AspectRatio,
    PaddingStrategy,
    ReframeResult
)


class TestReframer:
    """重构图器测试类"""

    def test_init(self, reframer):
        """测试初始化"""
        assert reframer is not None

    def test_reframe_square_ratio(self, reframer, sample_image_file):
        """测试正方形比例重构图"""
        result = reframer.reframe(
            sample_image_file,
            target_ratio=(1, 1),
            subject_center=(320, 240),
            padding=PaddingStrategy.NONE
        )

        assert isinstance(result, ReframeResult)
        assert isinstance(result.image, np.ndarray)
        assert result.new_size[0] == result.new_size[1]  # 正方形

    def test_reframe_portrait_ratio(self, reframer, sample_image_file):
        """测试竖图比例重构图"""
        result = reframer.reframe(
            sample_image_file,
            target_ratio=(4, 5),
            subject_center=(320, 240),
            padding=PaddingStrategy.NONE
        )

        assert isinstance(result, ReframeResult)
        # 竖图宽高比应该符合 4:5
        ratio = result.new_size[0] / result.new_size[1]
        assert abs(ratio - 0.8) < 0.01

    def test_reframe_with_padding_blur(self, reframer, sample_image_file):
        """测试模糊填充"""
        result = reframer.reframe(
            sample_image_file,
            target_ratio=(1, 1),
            subject_center=(320, 240),
            padding=PaddingStrategy.BLUR
        )

        assert result.padding_applied is True
        assert isinstance(result.image, np.ndarray)

    def test_reframe_with_padding_color(self, reframer, sample_image_file):
        """测试纯色填充"""
        result = reframer.reframe(
            sample_image_file,
            target_ratio=(1, 1),
            subject_center=(320, 240),
            padding=PaddingStrategy.COLOR,
            padding_color=(255, 255, 255)
        )

        assert isinstance(result.image, np.ndarray)

    def test_reframe_with_padding_mirror(self, reframer, sample_image_file):
        """测试镜像填充"""
        result = reframer.reframe(
            sample_image_file,
            target_ratio=(1, 1),
            subject_center=(320, 240),
            padding=PaddingStrategy.MIRROR
        )

        assert isinstance(result.image, np.ndarray)

    def test_reframe_multiple(self, reframer, sample_image_file):
        """测试批量重构图"""
        ratios = [(1, 1), (4, 5), (16, 9)]
        results = reframer.reframe_multiple(
            sample_image_file,
            ratios=ratios,
            subject_center=(320, 240),
            padding=PaddingStrategy.BLUR
        )

        assert len(results) == len(ratios)
        for result in results:
            assert isinstance(result, ReframeResult)

    def test_reframe_invalid_file(self, reframer):
        """测试无效文件"""
        with pytest.raises(ValueError):
            reframer.reframe("nonexistent.jpg", target_ratio=(1, 1))


class TestAspectRatio:
    """比例枚举测试类"""

    def test_aspect_ratios(self):
        """测试各种比例"""
        assert AspectRatio.SQUARE.value == (1, 1)
        assert AspectRatio.PORTRAIT_45.value == (4, 5)
        assert AspectRatio.LANDSCAPE_169.value == (16, 9)
        assert AspectRatio.STORY_916.value == (9, 16)

    def test_social_presets(self, reframer):
        """测试社交媒体预设"""
        assert "instagram_square" in Reframer.SOCIAL_PRESETS
        assert "instagram_portrait" in Reframer.SOCIAL_PRESETS
        assert "youtube_thumbnail" in Reframer.SOCIAL_PRESETS


class TestReframeResult:
    """重构图结果测试类"""

    def test_reframe_result_properties(self, sample_image):
        """测试结果属性"""
        result = ReframeResult(
            image=sample_image,
            original_size=(640, 480),
            new_size=(500, 500),
            crop_box=(70, 0, 570, 480),
            subject_center=(250.0, 240.0),
            padding_applied=True
        )

        assert result.original_size == (640, 480)
        assert result.new_size == (500, 500)
        assert result.padding_applied is True
        assert len(result.crop_box) == 4
        assert len(result.subject_center) == 2
