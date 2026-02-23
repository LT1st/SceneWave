"""
SceneWeave 测试配置
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import SubjectDetector, CompositionScorer, Reframer
from src.core.detector import SubjectInfo
from src.core.reframer import PaddingStrategy


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_image():
    """创建测试用的示例图片"""
    # 创建一个简单的测试图片 (640x480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加一些简单的图形
    cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)  # 中心白色圆
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # 蓝色方块
    cv2.rectangle(img, (400, 300), (550, 400), (0, 255, 0), -1)  # 绿色方块

    return img


@pytest.fixture
def sample_image_file(sample_image):
    """创建临时图片文件"""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, sample_image)
        yield f.name

    # 清理
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def detector():
    """检测器实例"""
    return SubjectDetector(model_size="n")


@pytest.fixture
def scorer():
    """评分器实例"""
    return CompositionScorer()


@pytest.fixture
def reframer():
    """重构图器实例"""
    return Reframer()


@pytest.fixture
def sample_subject():
    """示例主体"""
    return SubjectInfo(
        label="person",
        confidence=0.95,
        bbox=(100, 100, 200, 200),
        center=(150.0, 150.0)
    )


@pytest.fixture
def sample_subjects():
    """示例主体列表"""
    return [
        SubjectInfo(label="person", confidence=0.95, bbox=(100, 100, 200, 200), center=(150.0, 150.0)),
        SubjectInfo(label="car", confidence=0.88, bbox=(300, 250, 500, 400), center=(400.0, 325.0)),
    ]


# ============================================================================
# Pytest 配置
# ============================================================================

def pytest_configure(config):
    """Pytest 配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
