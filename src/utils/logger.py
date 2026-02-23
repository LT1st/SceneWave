"""
SceneWeave 日志系统
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.core.config import LogConfig, get_config


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # ANSI 颜色代码
    COLORS = {
        "DEBUG": "\033[36m",      # 青色
        "INFO": "\033[32m",       # 绿色
        "WARNING": "\033[33m",    # 黄色
        "ERROR": "\033[31m",      # 红色
        "CRITICAL": "\033[35m",   # 紫色
    }
    RESET = "\033[0m"

    def format(self, record):
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # 格式化消息
        result = super().format(record)

        # 恢复原始 levelname
        record.levelname = levelname

        return result


def setup_logger(
    name: str = "sceneweave",
    config: Optional[LogConfig] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        config: 日志配置
        log_file: 日志文件路径（覆盖配置）

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    if config is None:
        config = get_config().log

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.level.upper()))

    # 清除已有的处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(config.format)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # 使用彩色格式化器（仅在终端支持时）
    if sys.stdout.isatty():
        console_formatter = ColoredFormatter(config.format)
        console_handler.setFormatter(console_formatter)
    else:
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # 文件处理器
    if config.file_enabled or log_file:
        file_path = log_file or config.file_path

        # 创建日志目录
        log_file_path = Path(file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建文件处理器
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "sceneweave") -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logging.Logger: 日志记录器实例
    """
    logger = logging.getLogger(name)

    # 如果还没有配置，使用默认配置
    if not logger.handlers:
        return setup_logger(name)

    return logger


# ============================================================================
# 模块级日志记录器
# ============================================================================

# 核心模块日志
detector_logger = get_logger("sceneweave.detector")
scorer_logger = get_logger("sceneweave.scorer")
reframer_logger = get_logger("sceneweave.reframer")

# API 日志
api_logger = get_logger("sceneweave.api")

# UI 日志
ui_logger = get_logger("sceneweave.ui")
