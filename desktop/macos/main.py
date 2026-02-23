"""
SceneWeave macOS Desktop App
使用 PySide6 - 现代 Qt 框架
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
from concurrent.futures import QThreadPool, QRunnable, pyqtSignal

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit,
    QFileDialog, QGroupBox, QSlider, QTabWidget, QGridLayout,
    QFrame, QSizePolicy, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QFont, QIcon, QPalette, QColor
import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import SubjectDetector, CompositionScorer, Reframer
from src.core.reframer import PaddingStrategy, AspectRatio


# ============================================================================
# Worker 线程类
# ============================================================================

class AnalysisWorker(QThread):
    """构图分析工作线程"""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path: str, detector, scorer):
        super().__init__()
        self.image_path = image_path
        self.detector = detector
        self.scorer = scorer

    def run(self):
        try:
            # 检测主体
            subjects = self.detector.detect(self.image_path)

            # 计算评分
            if subjects:
                main_subject = subjects[0]
                score = self.scorer.score(
                    self.image_path,
                    subject_bbox=main_subject.bbox,
                    subject_center=main_subject.center
                )
            else:
                score = self.scorer.score(self.image_path)

            # 绘制结果
            result_img = self.detector.draw_detections(self.image_path, subjects)

            self.finished.emit({
                'subjects': subjects,
                'score': score,
                'result_image': result_img
            })

        except Exception as e:
            self.error.emit(str(e))


class ReframeWorker(QThread):
    """重构图工作线程"""
    finished = Signal(np.ndarray)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, image_path: str, ratio: tuple, padding: PaddingStrategy,
                 subject_center, subject_bbox, reframer):
        super().__init__()
        self.image_path = image_path
        self.ratio = ratio
        self.padding = padding
        self.subject_center = subject_center
        self.subject_bbox = subject_bbox
        self.reframer = reframer

    def run(self):
        try:
            self.progress.emit("正在重构图...")

            result = self.reframer.reframe(
                self.image_path,
                target_ratio=self.ratio,
                subject_center=self.subject_center,
                subject_bbox=self.subject_bbox,
                padding=self.padding
            )

            self.finished.emit(result.image)

        except Exception as e:
            self.error.emit(str(e))


# ============================================================================
# 自定义组件
# ============================================================================

class ImageDropLabel(QLabel):
    """支持拖拽的图片显示标签"""
    imageDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 10px;
                background-color: #2b2b2b;
                color: #888;
            }
            QLabel:hover {
                border-color: #0A84FF;
            }
        """)
        self.setText("拖拽图片到此处\n或点击选择")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                self.imageDropped.emit(file)
                break

    def mousePressEvent(self, event):
        # 点击触发文件选择
        self.parent().parent().select_file()

    def set_image(self, image: np.ndarray):
        """显示图片"""
        # 获取标签大小
        label_width = self.width()
        label_height = self.height()

        # 计算缩放
        h, w = image.shape[:2]
        scale = min(label_width / w, label_height / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放图片
        resized = cv2.resize(image, (new_w, new_h))

        # 转换格式
        if len(resized.shape) == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = resized.shape
            qt_image = QImage(resized.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qt_image)
        self.setPixmap(pixmap)
        self.setText("")


class ScoreDisplay(QWidget):
    """构图评分显示组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 总分大显示
        self.total_score = QLabel("0.0")
        self.total_score.setAlignment(Qt.AlignCenter)
        self.total_score.setStyleSheet("""
            QLabel {
                font-size: 72px;
                font-weight: bold;
                color: #0A84FF;
                background-color: #1c1c1e;
                border-radius: 20px;
                padding: 20px;
            }
        """)
        layout.addWidget(self.total_score)

        # 评级
        self.grade_label = QLabel("-")
        self.grade_label.setAlignment(Qt.AlignCenter)
        self.grade_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #30d158;
            }
        """)
        layout.addWidget(self.grade_label)

    def set_score(self, score):
        """设置评分"""
        from src.core import CompositionScore
        if isinstance(score, CompositionScore):
            total = score.total
            grade = score.grade
        else:
            total = float(score)
            grade = self._get_grade(total)

        self.total_score.setText(f"{total:.1f}")
        self.grade_label.setText(grade)

        # 根据分数改变颜色
        if total >= 80:
            color = "#30d158"  # 绿色
        elif total >= 60:
            color = "#ff9f0a"  # 橙色
        else:
            color = "#ff453a"  # 红色

        self.total_score.setStyleSheet(f"""
            QLabel {{
                font-size: 72px;
                font-weight: bold;
                color: {color};
                background-color: #1c1c1e;
                border-radius: 20px;
                padding: 20px;
            }}
        """)
        self.grade_label.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                font-weight: bold;
                color: {color};
            }}
        """)

    def _get_grade(self, score: float) -> str:
        if score >= 90:
            return "S - 完美"
        elif score >= 80:
            return "A - 优秀"
        elif score >= 70:
            return "B - 良好"
        elif score >= 60:
            return "C - 及格"
        else:
            return "D - 需改进"


# ============================================================================
# 主窗口
# ============================================================================

class SceneWeaveMacApp(QMainWindow):
    """SceneWeave macOS 主窗口"""

    def __init__(self):
        super().__init__()

        # 核心组件
        self.detector: Optional[SubjectDetector] = None
        self.reframer = Reframer()
        self.scorer = CompositionScorer()

        # 当前状态
        self.current_image_path: Optional[str] = None
        self.current_image: Optional[np.ndarray] = None
        self.subjects: List = []
        self.current_score = None
        self.result_image: Optional[np.ndarray] = None

        # 线程池
        self.thread_pool = QThreadPool()

        # 初始化 UI
        self.init_ui()
        self.apply_macos_style()

    def init_ui(self):
        """初始化 UI"""
        self.setWindowTitle("SceneWeave - AI 智能图片重构图")
        self.setMinimumSize(1200, 800)

        # 创建中心部件
        central = QWidget()
        self.setCentralWidget(central)

        # 主布局
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左侧 - 图片显示
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=2)

        # 右侧 - 控制面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def create_left_panel(self) -> QFrame:
        """创建左侧图片面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # 图片显示区域
        self.image_label = ImageDropLabel(self)
        self.image_label.imageDropped.connect(self._load_image)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        return panel

    def create_right_panel(self) -> QFrame:
        """创建右侧控制面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # 标题
        title = QLabel("SceneWeave")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #0A84FF;
            }
        """)
        layout.addWidget(title)

        subtitle = QLabel("AI 智能图片重构图工具")
        subtitle.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(subtitle)

        # 选项卡
        tabs = QTabWidget()
        tabs.addTab(self.create_analyze_tab(), "构图分析")
        tabs.addTab(self.create_reframe_tab(), "智能重构图")
        layout.addWidget(tabs)

        # 底部按钮
        layout.addStretch()

        save_btn = QPushButton("💾 保存结果")
        save_btn.clicked.connect(self.save_result)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #30d158;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #28cd41;
            }
            QPushButton:pressed {
                background-color: #1fb332;
            }
        """)
        layout.addWidget(save_btn)

        return panel

    def create_analyze_tab(self) -> QWidget:
        """创建构图分析选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 分析按钮
        self.analyze_btn = QPushButton("🔍 分析构图")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #0A84FF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #409cff;
            }
        """)
        layout.addWidget(self.analyze_btn)

        # 评分显示
        self.score_display = ScoreDisplay()
        layout.addWidget(self.score_display)

        # 详细评分
        details_group = QGroupBox("详细评分")
        details_layout = QGridLayout()

        self.detail_labels = {}
        metrics = [
            ("rule_of_thirds", "三分法则", 30),
            ("visual_balance", "视觉平衡", 25),
            ("subject_prominence", "主体突出", 25),
            ("breathing_room", "呼吸空间", 20),
        ]

        for i, (key, label, max_val) in enumerate(metrics):
            lbl = QLabel(f"{label}: 0 / {max_val}")
            lbl.setStyleSheet("color: #888; font-size: 13px;")
            details_layout.addWidget(lbl, i, 0)
            self.detail_labels[key] = lbl

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        # 检测结果
        self.subjects_label = QLabel("未检测到主体")
        self.subjects_label.setWordWrap(True)
        self.subjects_label.setStyleSheet("color: #888; font-size: 13px;")
        layout.addWidget(self.subjects_label)

        layout.addStretch()
        return widget

    def create_reframe_tab(self) -> QWidget:
        """创建重构图选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 比例选择
        ratio_group = QGroupBox("📐 目标比例")
        ratio_layout = QVBoxLayout()

        self.ratio_buttons = []
        ratios = [
            ("1:1 正方形 (Instagram)", (1, 1)),
            ("4:5 竖图 (Instagram/小红书)", (4, 5)),
            ("16:9 横屏 (YouTube)", (16, 9)),
            ("9:16 竖屏 (Story/抖音)", (9, 16)),
            ("2:3 封面 (小红书)", (2, 3)),
            ("3:1 Banner", (3, 1)),
        ]

        from src.core.reframe import RatioButton
        for label, ratio in ratios:
            btn = RatioButton(label, ratio)
            btn.clicked.connect(lambda checked, r=ratio: self._set_ratio(r))
            ratio_layout.addWidget(btn)
            self.ratio_buttons.append(btn)

        # 默认选中 4:5
        self.ratio_buttons[1].set_selected(True)
        self.selected_ratio = (4, 5)

        ratio_group.setLayout(ratio_layout)
        layout.addWidget(ratio_group)

        # 填充策略
        padding_group = QGroupBox("🎨 填充策略")
        padding_layout = QVBoxLayout()

        self.padding_buttons = []
        paddings = [
            ("模糊背景 (推荐)", PaddingStrategy.BLUR),
            ("不填充 (仅裁剪)", PaddingStrategy.NONE),
            ("纯色填充 (白色)", PaddingStrategy.COLOR),
            ("镜像填充", PaddingStrategy.MIRROR),
        ]

        from src.core.reframe import PaddingButton
        for label, padding in paddings:
            btn = PaddingButton(label, padding)
            btn.clicked.connect(lambda checked, p=padding: self._set_padding(p))
            padding_layout.addWidget(btn)
            self.padding_buttons.append(btn)

        # 默认选中模糊
        self.padding_buttons[0].set_selected(True)
        self.selected_padding = PaddingStrategy.BLUR

        padding_group.setLayout(padding_layout)
        layout.addWidget(padding_group)

        # 重构图按钮
        self.reframe_btn = QPushButton("✨ 开始重构图")
        self.reframe_btn.clicked.connect(self.reframe_image)
        self.reframe_btn.setStyleSheet("""
            QPushButton {
                background-color: #0A84FF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #409cff;
            }
        """)
        layout.addWidget(self.reframe_btn)

        layout.addStretch()
        return widget

    def apply_macos_style(self):
        """应用 macOS 风格"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QFrame {
                background-color: #252525;
                border-radius: 10px;
            }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #3a3a3c;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: none;
                background-color: transparent;
            }
            QTabBar::tab {
                background-color: #3a3a3c;
                color: #888;
                padding: 8px 16px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #0A84FF;
                color: white;
            }
            QComboBox {
                background-color: #3a3a3c;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)

    def _set_ratio(self, ratio: tuple):
        """设置比例"""
        self.selected_ratio = ratio
        for btn in self.ratio_buttons:
            btn.set_selected(btn.ratio == ratio)

    def _set_padding(self, padding: PaddingStrategy):
        """设置填充策略"""
        self.selected_padding = padding
        for btn in self.padding_buttons:
            btn.set_selected(btn.padding == padding)

    def select_file(self):
        """选择文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)"
        )

        if file_path:
            self._load_image(file_path)

    def _load_image(self, file_path: str):
        """加载图片"""
        self.current_image_path = file_path

        # 读取图片
        image = cv2.imread(file_path)
        if image is None:
            return

        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_label.set_image(self.current_image)

        self.setWindowTitle(f"SceneWeave - {Path(file_path).name}")

    def analyze_image(self):
        """分析构图"""
        if self.current_image_path is None:
            return

        # 初始化检测器
        if self.detector is None:
            self.detector = SubjectDetector(model_size="n")

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("⏳ 分析中...")

        # 创建工作线程
        worker = AnalysisWorker(self.current_image_path, self.detector, self.scorer)
        worker.finished.connect(self._on_analysis_finished)
        worker.error.connect(self._on_analysis_error)
        worker.start()

    def _on_analysis_finished(self, result):
        """分析完成"""
        self.subjects = result['subjects']
        self.current_score = result['score']

        # 显示结果图片
        self.image_label.set_image(result['result_image'])

        # 更新评分显示
        self.score_display.set_score(self.current_score)

        # 更新详细评分
        self.detail_labels['rule_of_thirds'].setText(
            f"三分法则: {self.current_score.rule_of_thirds:.1f} / 30"
        )
        self.detail_labels['visual_balance'].setText(
            f"视觉平衡: {self.current_score.visual_balance:.1f} / 25"
        )
        self.detail_labels['subject_prominence'].setText(
            f"主体突出: {self.current_score.subject_prominence:.1f} / 25"
        )
        self.detail_labels['breathing_room'].setText(
            f"呼吸空间: {self.current_score.breathing_room:.1f} / 20"
        )

        # 更新主体信息
        if self.subjects:
            main = self.subjects[0]
            self.subjects_label.setText(
                f"检测到 {len(self.subjects)} 个主体\n"
                f"主要: {main.label} (置信度: {main.confidence:.2f})"
            )
        else:
            self.subjects_label.setText("未检测到主体")

        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🔍 分析构图")

    def _on_analysis_error(self, error_msg):
        """分析错误"""
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🔍 分析构图")
        print(f"分析错误: {error_msg}")

    def reframe_image(self):
        """重构图"""
        if self.current_image_path is None:
            return

        # 获取主体信息
        subject_center = None
        subject_bbox = None
        if self.subjects:
            subject_center = self.subjects[0].center
            subject_bbox = self.subjects[0].bbox

        self.reframe_btn.setEnabled(False)
        self.reframe_btn.setText("⏳ 处理中...")

        # 创建工作线程
        worker = ReframeWorker(
            self.current_image_path,
            self.selected_ratio,
            self.selected_padding,
            subject_center,
            subject_bbox,
            self.reframer
        )
        worker.finished.connect(self._on_reframe_finished)
        worker.error.connect(self._on_reframe_error)
        worker.start()

    def _on_reframe_finished(self, image: np.ndarray):
        """重构图完成"""
        self.result_image = image
        self.image_label.set_image(image)
        self.reframe_btn.setEnabled(True)
        self.reframe_btn.setText("✨ 开始重构图")

    def _on_reframe_error(self, error_msg):
        """重构图错误"""
        self.reframe_btn.setEnabled(True)
        self.reframe_btn.setText("✨ 开始重构图")
        print(f"重构图错误: {error_msg}")

    def save_result(self):
        """保存结果"""
        if self.result_image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存结果",
            "sceneweave_result.png",
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg)"
        )

        if file_path:
            # 保存
            result_bgr = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, result_bgr)


# ============================================================================
# 自定义按钮组件
# ============================================================================

class RatioButton(QLabel):
    """比例选择按钮"""
    clicked = pyqtSignal()

    def __init__(self, label: str, ratio: tuple, parent=None):
        super().__init__(label, parent)
        self.ratio = ratio
        self._selected = False

        self.setStyleSheet("""
            QLabel {
                padding: 12px;
                border-radius: 8px;
                background-color: #3a3a3c;
                color: #ffffff;
            }
            QLabel:hover {
                background-color: #48484a;
            }
        """)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        self._selected = selected
        if selected:
            self.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    border-radius: 8px;
                    background-color: #0A84FF;
                    color: white;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    border-radius: 8px;
                    background-color: #3a3a3c;
                    color: #ffffff;
                }
                QLabel:hover {
                    background-color: #48484a;
                }
            """)

    def isSelected(self):
        return self._selected


class PaddingButton(QLabel):
    """填充策略选择按钮"""
    clicked = pyqtSignal()

    def __init__(self, label: str, padding: PaddingStrategy, parent=None):
        super().__init__(label, parent)
        self.padding = padding
        self._selected = False

        self.setStyleSheet("""
            QLabel {
                padding: 12px;
                border-radius: 8px;
                background-color: #3a3a3c;
                color: #ffffff;
            }
            QLabel:hover {
                background-color: #48484a;
            }
        """)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        self._selected = selected
        if selected:
            self.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    border-radius: 8px;
                    background-color: #0A84FF;
                    color: white;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    border-radius: 8px;
                    background-color: #3a3a3c;
                    color: #ffffff;
                }
                QLabel:hover {
                    background-color: #48484a;
                }
            """)


# ============================================================================
# 启动入口
# ============================================================================

def main():
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName("SceneWeave")
    app.setOrganizationName("SceneWeave")

    # macOS 特定设置
    if sys.platform == "darwin":
        app.setStyle("macos")

    window = SceneWeaveMacApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
