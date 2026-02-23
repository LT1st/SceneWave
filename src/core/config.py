"""
SceneWeave 配置管理
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import os


@dataclass
class ModelConfig:
    """模型配置"""
    yolo_model_size: str = "n"  # n, s, m, l, x
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45
    device: str = "cpu"  # cpu, cuda, mps


@dataclass
class ScoreConfig:
    """评分配置"""
    # 三分法则权重
    rule_of_thirds_weight: float = 30.0
    # 视觉平衡权重
    visual_balance_weight: float = 25.0
    # 主体突出度权重
    subject_prominence_weight: float = 25.0
    # 呼吸空间权重
    breathing_room_weight: float = 20.0

    # 评分阈值
    grade_s_threshold: float = 90.0
    grade_a_threshold: float = 80.0
    grade_b_threshold: float = 70.0
    grade_c_threshold: float = 60.0


@dataclass
class ReframeConfig:
    """重构图配置"""
    # 默认填充策略
    default_padding: str = "blur"
    # 默认填充颜色
    default_padding_color: tuple = (255, 255, 255)
    # 最小输出尺寸
    min_output_size: int = 480
    # 最大输出尺寸
    max_output_size: int = 2160


@dataclass
class APIConfig:
    """API 配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    # CORS
    allow_origins: list = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: list = field(default_factory=lambda: ["*"])
    allow_headers: list = field(default_factory=lambda: ["*"])
    # 限制
    max_upload_size: int = 10 * 1024 * 1024  # 10MB


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # 文件日志
    file_enabled: bool = False
    file_path: str = "logs/sceneweave.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """主配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    reframe: ReframeConfig = field(default_factory=ReframeConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # 环境标识
    environment: str = "production"  # development, production
    debug: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置"""
        config_file = Path(path)
        if not config_file.exists():
            return cls()

        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        config = cls()

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "score" in data:
            config.score = ScoreConfig(**data["score"])
        if "reframe" in data:
            config.reframe = ReframeConfig(**data["reframe"])
        if "api" in data:
            config.api = APIConfig(**data["api"])
        if "log" in data:
            config.log = LogConfig(**data["log"])
        if "environment" in data:
            config.environment = data["environment"]
        if "debug" in data:
            config.debug = data["debug"]

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model.__dict__,
            "score": self.score.__dict__,
            "reframe": self.reframe.__dict__,
            "api": self.api.__dict__,
            "log": self.log.__dict__,
            "environment": self.environment,
            "debug": self.debug,
        }


# ============================================================================
# 全局配置实例
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置

    Args:
        config_path: 配置文件路径，默认按环境查找

    Returns:
        Config: 配置实例
    """
    # 确定环境
    environment = os.getenv("SCENEWEAVE_ENV", "production")

    # 查找配置文件
    if config_path is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_path = config_dir / f"{environment}.yaml"

        # 如果环境配置不存在，使用默认配置
        if not config_path.exists():
            config_path = config_dir / "default.yaml"

        # 如果默认配置也不存在，返回空配置
        if not config_path.exists():
            return Config()

    config = Config.from_yaml(str(config_path))
    config.environment = environment

    # 覆盖 debug 设置
    if os.getenv("SCENEWEAVE_DEBUG"):
        config.debug = os.getenv("SCENEWEAVE_DEBUG").lower() == "true"

    return config


# 全局配置
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config):
    """设置全局配置"""
    global _config
    _config = config
