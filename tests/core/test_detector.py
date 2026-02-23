"""
主体检测器测试
"""

import pytest
import numpy as np
from pathlib import Path

from src.core.detector import SubjectDetector, SubjectInfo


class TestSubjectDetector:
    """主体检测器测试类"""

    def test_init(self, detector):
        """测试初始化"""
        assert detector is not None
        assert detector.model is not None

    def test_detect_valid_image(self, detector, sample_image_file):
        """测试检测有效图片"""
        subjects = detector.detect(sample_image_file)

        assert isinstance(subjects, list)
        # 检测结果可能是空的，也可能有结果

    def test_detect_invalid_file(self, detector):
        """测试检测无效文件"""
        with pytest.raises((FileNotFoundError, Exception)):
            detector.detect("nonexistent.jpg")

    def test_subject_info_creation(self):
        """测试主体信息创建"""
        subject = SubjectInfo(
            label="person",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
            center=(150.0, 150.0)
        )

        assert subject.label == "person"
        assert subject.confidence == 0.95
        assert subject.bbox == (100, 100, 200, 200)
        assert subject.center == (150.0, 150.0)

    def test_draw_detections_empty(self, detector, sample_image_file):
        """测试绘制空检测结果"""
        result = detector.draw_detections(sample_image_file, [])

        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # RGB

    def test_draw_detections_with_subjects(self, detector, sample_image_file, sample_subjects):
        """测试绘制检测结果"""
        result = detector.draw_detections(sample_image_file, sample_subjects)

        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # RGB


class TestSubjectInfo:
    """主体信息测试类"""

    def test_subject_info_properties(self, sample_subject):
        """测试主体信息属性"""
        assert sample_subject.label == "person"
        assert 0 <= sample_subject.confidence <= 1
        assert len(sample_subject.bbox) == 4
        assert len(sample_subject.center) == 2

    def test_subject_info_bbox_validation(self):
        """测试边界框验证"""
        # 有效边界框
        subject = SubjectInfo(
            label="test",
            confidence=0.9,
            bbox=(0, 0, 100, 100),
            center=(50, 50)
        )
        assert subject.bbox[2] > subject.bbox[0]  # x2 > x1
        assert subject.bbox[3] > subject.bbox[1]  # y2 > y1
