"""
SceneWeave Windows 桌面应用
使用 CustomTkinter - 现代化的 Tkinter UI
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
from threading import Thread

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import SubjectDetector, CompositionScorer, Reframer
from src.core.reframer import PaddingStrategy


class SceneWeaveApp(ctk.CTk):
    """SceneWeave 主窗口"""

    def __init__(self):
        super().__init__()

        # 窗口设置
        self.title("SceneWeave - AI 智能图片重构图")
        self.geometry("1000x700")
        self.minsize(900, 600)

        # 设置主题
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # 核心组件（延迟加载）
        self.detector = None
        self.reframer = Reframer()
        self.scorer = CompositionScorer()

        # 当前图片
        self.current_image_path = None
        self.current_image = None
        self.subjects = []
        self.current_score = None

        # 初始化 UI
        self._init_ui()

        # 状态栏
        self.status_var = ctk.StringVar(value="就绪")
        self._create_status_bar()

    def _init_ui(self):
        """初始化 UI"""
        # 创建主框架
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 顶部标题栏
        self._create_header(main_frame)

        # 主内容区域（左右分栏）
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, pady=(10, 0))

        # 左侧面板 - 图片预览
        left_panel = self._create_left_panel(content_frame)

        # 右侧面板 - 控制选项
        right_panel = self._create_right_panel(content_frame)

        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right_panel.pack(side="right", fill="y", padx=(0, 0))

    def _create_header(self, parent):
        """创建顶部标题"""
        header = ctk.CTkFrame(parent, height=60, fg_color=("gray90", "gray20"))
        header.pack(fill="x", pady=(0, 10))
        header.pack_propagate(False)

        # 标题
        title = ctk.CTkLabel(
            header,
            text="SceneWeave",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(side="left", padx=20, pady=15)

        # 副标题
        subtitle = ctk.CTkLabel(
            header,
            text="AI 智能图片重构图工具",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle.pack(side="left", padx=(0, 20))

        # 按钮
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=20)

        self.analyze_btn = ctk.CTkButton(
            btn_frame,
            text="分析构图",
            command=self.analyze_image,
            width=120
        )
        self.analyze_btn.pack(side="left", padx=5)

        self.reframe_btn = ctk.CTkButton(
            btn_frame,
            text="开始重构图",
            command=self.reframe_image,
            width=120,
            fg_color="#2CC985",
            hover_color="#22B077"
        )
        self.reframe_btn.pack(side="left", padx=5)

    def _create_left_panel(self, parent):
        """创建左侧图片预览面板"""
        panel = ctk.CTkFrame(parent)

        # 图片显示区域
        self.image_label = ctk.CTkLabel(
            panel,
            text="请选择图片\n或拖拽到此处",
            font=ctk.CTkFont(size=16),
            text_color="gray",
            width=600,
            height=500
        )
        self.image_label.pack(fill="both", expand=True, padx=20, pady=20)

        # 支持拖拽
        self.image_label.drop_target_register("DND_Files")
        self.image_label.dnd_bind("<<Drop>>", self._on_drop)

        return panel

    def _create_right_panel(self, parent):
        """创建右侧控制面板"""
        panel = ctk.CTkFrame(parent, width=350)
        panel.pack(fill="y")

        # 内部滚动区域
        scroll_frame = ctk.CTkScrollableFrame(panel, width=330, label_text="设置")
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 文件选择
        self._create_file_section(scroll_frame)

        # 分隔线
        ctk.CTkFrame(scroll_frame, height=2, fg_color="gray30").pack(fill="x", pady=15)

        # 比例选择
        self._create_ratio_section(scroll_frame)

        # 分隔线
        ctk.CTkFrame(scroll_frame, height=2, fg_color="gray30").pack(fill="x", pady=15)

        # 填充策略
        self._create_padding_section(scroll_frame)

        # 分隔线
        ctk.CTkFrame(scroll_frame, height=2, fg_color="gray30").pack(fill="x", pady=15)

        # 分析结果
        self._create_analysis_section(scroll_frame)

        # 底部按钮
        self._create_bottom_buttons(panel)

        return panel

    def _create_file_section(self, parent):
        """创建文件选择区域"""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            section,
            text="📁 选择图片",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))

        file_frame = ctk.CTkFrame(section)
        file_frame.pack(fill="x")

        self.file_label = ctk.CTkLabel(
            file_frame,
            text="未选择文件",
            anchor="w",
            height=35
        )
        self.file_label.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=5)

        ctk.CTkButton(
            file_frame,
            text="浏览",
            command=self.select_file,
            width=80
        ).pack(side="right", padx=(0, 10), pady=5)

    def _create_ratio_section(self, parent):
        """创建比例选择区域"""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            section,
            text="📐 目标比例",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))

        self.ratio_var = ctk.StringVar(value="4:5 竖图 (Instagram/小红书)")

        ratios = [
            "1:1 正方形 (Instagram)",
            "4:5 竖图 (Instagram/小红书)",
            "16:9 横屏 (YouTube)",
            "9:16 竖屏 (Story/抖音)",
            "2:3 封面 (小红书)",
            "3:1 Banner",
        ]

        for ratio in ratios:
            ctk.CTkRadioButton(
                section,
                text=ratio,
                variable=self.ratio_var,
                value=ratio
            ).pack(anchor="w", pady=5)

    def _create_padding_section(self, parent):
        """创建填充策略区域"""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            section,
            text="🎨 填充策略",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))

        self.padding_var = ctk.StringVar(value="blur")

        options = [
            ("模糊背景 (推荐)", "blur"),
            ("不填充 (仅裁剪)", "none"),
            ("纯色填充 (白色)", "color"),
            ("镜像填充", "mirror"),
        ]

        for label, value in options:
            ctk.CTkRadioButton(
                section,
                text=label,
                variable=self.padding_var,
                value=value
            ).pack(anchor="w", pady=5)

    def _create_analysis_section(self, parent):
        """创建分析结果区域"""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            section,
            text="📊 构图分析",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))

        self.analysis_text = ctk.CTkTextbox(
            section,
            height=150,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.analysis_text.pack(fill="x")
        self.analysis_text.insert("1.0", "分析结果将显示在这里...")
        self.analysis_text.configure(state="disabled")

    def _create_bottom_buttons(self, parent):
        """创建底部按钮"""
        btn_frame = ctk.CTkFrame(parent, height=60, fg_color="transparent")
        btn_frame.pack(fill="x", side="bottom", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="保存结果",
            command=self.save_result,
            height=40,
            fg_color="#6C63FF",
            hover_color="#5753D9"
        ).pack(fill="x", pady=5)

    def _create_status_bar(self):
        """创建状态栏"""
        status_bar = ctk.CTkFrame(self, height=30, fg_color=("gray80", "gray25"))
        status_bar.pack(side="bottom", fill="x")
        status_bar.pack_propagate(False)

        status_label = ctk.CTkLabel(
            status_bar,
            textvariable=self.status_var,
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        status_label.pack(side="left", padx=10)

    def select_file(self):
        """选择文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self._load_image(file_path)

    def _on_drop(self, event):
        """处理拖拽文件"""
        files = self.winfo_containing(
            self.winfo_pointerx(),
            self.winfo_pointery()
        ).split()

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                self._load_image(file)
                break

    def _load_image(self, file_path):
        """加载图片"""
        self.current_image_path = file_path
        self.file_label.configure(text=os.path.basename(file_path))

        # 读取并显示图片
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("错误", "无法读取图片")
            return

        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整大小显示
        self._display_image(self.current_image)

        self.status_var.set(f"已加载: {os.path.basename(file_path)}")

    def _display_image(self, image_array):
        """显示图片"""
        # 获取显示区域大小
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        # 计算缩放比例
        h, w = image_array.shape[:2]
        scale = min(label_width / w, label_height / h, 1.0)

        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放图片
        resized = cv2.resize(image_array, (new_w, new_h))

        # 转换为 PIL Image
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)

        self.image_label.configure(text="", image=photo)
        self.image_label.image = photo  # 保持引用

    def analyze_image(self):
        """分析构图"""
        if self.current_image_path is None:
            messagebox.showwarning("警告", "请先选择图片")
            return

        self.status_var.set("正在分析...")
        self.analyze_btn.configure(state="disabled")

        def _analyze():
            try:
                # 初始化检测器
                if self.detector is None:
                    self.detector = SubjectDetector(model_size="n")

                # 检测主体
                self.subjects = self.detector.detect(self.current_image_path)

                # 计算构图评分
                if self.subjects:
                    main_subject = self.subjects[0]
                    self.current_score = self.scorer.score(
                        self.current_image_path,
                        subject_bbox=main_subject.bbox,
                        subject_center=main_subject.center
                    )
                else:
                    self.current_score = self.scorer.score(self.current_image_path)

                # 绘制检测结果
                result_img = self.detector.draw_detections(
                    self.current_image_path,
                    self.subjects
                )

                # 更新 UI
                self.after(0, lambda: self._update_analysis_ui(result_img))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("错误", f"分析失败: {str(e)}"))
                self.after(0, lambda: self.status_var.set("分析失败"))
            finally:
                self.after(0, lambda: self.analyze_btn.configure(state="normal"))

        Thread(target=_analyze, daemon=True).start()

    def _update_analysis_ui(self, result_img):
        """更新分析结果 UI"""
        # 显示检测结果图片
        self._display_image(result_img)

        # 更新分析文本
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")

        score = self.current_score
        result_text = f"""构图评分: {score.total:.1f} / 100
评级: {score.grade}

┌─────────────────────────────┐
│ 三分法则     │ {score.rule_of_thirds:>5.1f} / 30 │
│ 视觉平衡     │ {score.visual_balance:>5.1f} / 25 │
│ 主体突出度    │ {score.subject_prominence:>5.1f} / 25 │
│ 呼吸空间     │ {score.breathing_room:>5.1f} / 20 │
└─────────────────────────────┘

检测到 {len(self.subjects)} 个主体
"""

        if self.subjects:
            main = self.subjects[0]
            result_text += f"\n主要主体: {main.label}\n置信度: {main.confidence:.2f}"

        self.analysis_text.insert("1.0", result_text)
        self.analysis_text.configure(state="disabled")

        self.status_var.set(f"分析完成 - 评分: {score.total:.1f}")

    def reframe_image(self):
        """重构图"""
        if self.current_image_path is None:
            messagebox.showwarning("警告", "请先选择图片")
            return

        self.status_var.set("正在重构图...")
        self.reframe_btn.configure(state="disabled")

        def _reframe():
            try:
                # 解析比例
                ratio_map = {
                    "1:1 正方形 (Instagram)": (1, 1),
                    "4:5 竖图 (Instagram/小红书)": (4, 5),
                    "16:9 横屏 (YouTube)": (16, 9),
                    "9:16 竖屏 (Story/抖音)": (9, 16),
                    "2:3 封面 (小红书)": (2, 3),
                    "3:1 Banner": (3, 1),
                }
                target_ratio = ratio_map[self.ratio_var.get()]

                # 解析填充策略
                padding_map = {
                    "blur": PaddingStrategy.BLUR,
                    "none": PaddingStrategy.NONE,
                    "color": PaddingStrategy.COLOR,
                    "mirror": PaddingStrategy.MIRROR,
                }
                padding = padding_map[self.padding_var.get()]

                # 获取主体信息
                subject_center = None
                subject_bbox = None
                if self.subjects:
                    subject_center = self.subjects[0].center
                    subject_bbox = self.subjects[0].bbox

                # 执行重构图
                result = self.reframer.reframe(
                    self.current_image_path,
                    target_ratio=target_ratio,
                    subject_center=subject_center,
                    subject_bbox=subject_bbox,
                    padding=padding
                )

                # 保存结果
                self.result_image = result.image

                # 更新显示
                self.after(0, lambda: self._display_image(result.image))
                self.after(0, lambda: self.status_var.set(
                    f"重构完成 - {result.original_size} -> {result.new_size}"
                ))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("错误", f"重构失败: {str(e)}"))
                self.after(0, lambda: self.status_var.set("重构失败"))
            finally:
                self.after(0, lambda: self.reframe_btn.configure(state="normal"))

        Thread(target=_reframe, daemon=True).start()

    def save_result(self):
        """保存结果"""
        if not hasattr(self, 'result_image'):
            messagebox.showwarning("警告", "请先进行重构图")
            return

        # 默认文件名
        default_name = f"sceneweave_result.png"

        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG 图片", "*.png"),
                ("JPEG 图片", "*.jpg"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            # 保存图片
            result_bgr = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, result_bgr)

            messagebox.showinfo("成功", f"图片已保存到:\n{file_path}")
            self.status_var.set(f"已保存: {os.path.basename(file_path)}")


def main():
    """主函数"""
    app = SceneWeaveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
