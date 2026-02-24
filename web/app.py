"""
SceneWeave Web UI - Gradio 界面
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np

from src.core import (
    SubjectDetector,
    CompositionScorer,
    Reframer,
    AIOutpainter,
    TraditionalOutpainter,
    OutpaintDirection
)
from src.core.reframer import PaddingStrategy


# 全局实例
detector = None
reframer = Reframer()
scorer = CompositionScorer()
ai_outpainter = None
traditional_outpainter = TraditionalOutpainter()


def get_ai_outpainter():
    """延迟加载 AI 扩图器"""
    global ai_outpainter
    if ai_outpainter is None:
        ai_outpainter = AIOutpainter(device="cpu")
    return ai_outpainter


def get_detector():
    """延迟加载检测器"""
    global detector
    if detector is None:
        detector = SubjectDetector(model_size="n")
    return detector


def analyze_image(image):
    """分析图片构图"""
    if image is None:
        return None, "请先上传图片"

    # 保存临时文件
    temp_path = "/tmp/sceneweave_input.jpg"
    import cv2
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 检测主体
    det = get_detector()
    subjects = det.detect(temp_path)

    # 分析构图
    main_subject = subjects[0] if subjects else None

    if main_subject:
        score = scorer.score(
            temp_path,
            subject_bbox=main_subject.bbox,
            subject_center=main_subject.center
        )
    else:
        score = scorer.score(temp_path)

    # 生成分析文本
    analysis = f"""
## 构图分析报告

### 检测结果
- 检测到 **{len(subjects)}** 个主体
- 主要主体: **{main_subject.label if main_subject else '未检测到'}**
- 主体位置: ({main_subject.center[0]:.0f}, {main_subject.center[1]:.0f})" if main_subject else ""

### 构图评分

| 维度 | 得分 | 满分 |
|------|------|------|
| 三分法则 | {score.rule_of_thirds:.1f} | 30 |
| 视觉平衡 | {score.visual_balance:.1f} | 25 |
| 主体突出 | {score.subject_prominence:.1f} | 25 |
| 呼吸空间 | {score.breathing_room:.1f} | 20 |
| **总分** | **{score.total:.1f}** | **100** |

### 评级: {score.grade}

### 改进建议
"""
    issues = scorer.get_issues()
    if issues:
        for issue in issues:
            analysis += f"\n- **{issue.description}**\n  - {issue.suggestion}"
    else:
        analysis += "\n\n✅ 构图良好, 无需改进!"

    # 绘制检测结果
    result_img = det.draw_detections(temp_path, subjects)

    return result_img, analysis


def reframe_image(image, ratio, padding):
    """重构图"""
    if image is None:
        return None, None, None, None

    # 保存临时文件
    temp_path = "/tmp/sceneweave_input.jpg"
    import cv2
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 解析比例
    ratio_map = {
        "1:1 正方形 (Instagram)": (1, 1),
        "4:5 竖图 (Instagram/小红书)": (4, 5),
        "16:9 横屏 (YouTube)": (16, 9),
        "9:16 竖屏 (Story/抖音)": (9, 16),
        "2:3 封面 (小红书)": (2, 3),
        "3:1 Banner": (3, 1),
    }
    target_ratio = ratio_map[ratio]

    # 解析填充策略
    padding_map = {
        "不填充 (裁剪)": PaddingStrategy.NONE,
        "模糊背景": PaddingStrategy.BLUR,
        "纯色填充 (白色)": PaddingStrategy.COLOR,
        "镜像填充": PaddingStrategy.MIRROR,
    }
    padding_strategy = padding_map[padding]

    # 检测主体
    det = get_detector()
    subjects = det.detect(temp_path)
    main_subject = subjects[0] if subjects else None

    # 重构图
    result = reframer.reframe(
        temp_path,
        target_ratio=target_ratio,
        subject_center=main_subject.center if main_subject else None,
        subject_bbox=main_subject.bbox if main_subject else None,
        padding=padding_strategy
    )

    # 生成所有预设版本
    all_results = []
    for r_name, r_val in ratio_map.items():
        r = reframer.reframe(
            temp_path,
            target_ratio=r_val,
            subject_center=main_subject.center if main_subject else None,
            subject_bbox=main_subject.bbox if main_subject else None,
            padding=padding_strategy
        )
        all_results.append((r_name, r.image))

    # 创建对比图
    h = max(image.shape[0], result.image.shape[0])
    orig_resized = cv2.resize(image, (int(image.shape[1] * h / image.shape[0]), h))
    result_resized = cv2.resize(result.image, (int(result.image.shape[1] * h / result.image.shape[0]), h))

    # 添加标签
    def add_label(img, label):
        labeled = np.zeros((img.shape[0] + 40, img.shape[1], 3), dtype=np.uint8)
        labeled[40:, :] = img
        cv2.putText(labeled, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return labeled

    orig_labeled = add_label(orig_resized, "Original")
    result_labeled = add_label(result_resized, "Reframed")
    comparison = np.hstack([orig_labeled, result_labeled])

    return result.image, comparison, all_results, f"尺寸: {result.original_size} -> {result.new_size}"


def outpaint_image(image, direction, expand_pixels, prompt, use_ai):
    """AI 扩图"""
    if image is None:
        return None, "请先上传图片"

    # 保存临时文件
    temp_path = "/tmp/sceneweave_input.jpg"
    import cv2
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 解析方向
    direction_map = {
        "向左扩展": OutpaintDirection.LEFT,
        "向右扩展": OutpaintDirection.RIGHT,
        "向上扩展": OutpaintDirection.TOP,
        "向下扩展": OutpaintDirection.BOTTOM,
        "四周扩展": OutpaintDirection.ALL,
    }
    direction_enum = direction_map[direction]

    try:
        if use_ai:
            # 使用 AI 扩图
            outpainter = get_ai_outpainter()
            result = outpainter.outpaint(
                temp_path,
                direction=direction_enum,
                expand_pixels=expand_pixels,
                prompt=prompt or "high quality, detailed, natural extension",
                num_inference_steps=30,  # 减少步数加快速度
                guidance_scale=7.5
            )
            info = f"AI 扩图完成 | 耗时: {result.generation_time:.2f}s | 尺寸: {result.original_size} -> {result.new_size}"
        else:
            # 使用传统方法
            outpainter = traditional_outpainter
            result_image = outpainter.extend(image, direction_enum, expand_pixels)

            from src.core.outpainter import OutpaintResult
            result = OutpaintResult(
                image=result_image,
                original_size=(image.shape[1], image.shape[0]),
                new_size=(result_image.shape[1], result_image.shape[0]),
                direction=direction_enum,
                expand_pixels=expand_pixels,
                mask=None,
                generation_time=0
            )
            info = f"传统扩图完成 | 尺寸: {result.original_size} -> {result.new_size}"

        return result.image, info

    except Exception as e:
        return None, f"扩图失败: {str(e)}"


# 创建 Gradio 界面
with gr.Blocks(
    title="SceneWeave - AI 智能重构图",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown("""
    # SceneWeave - AI 智能图片重构图工具
    让每一张照片都成为完美构图 | 支持 AI 智能扩图
    """)

    with gr.Tabs():
        # Tab 1: 构图分析
        with gr.Tab("构图分析"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="上传图片")
                    analyze_btn = gr.Button("分析构图", variant="primary")

                with gr.Column():
                    detection_result = gr.Image(label="检测结果")
                    analysis_result = gr.Markdown(label="分析报告")

            analyze_btn.click(
                analyze_image,
                inputs=[input_image],
                outputs=[detection_result, analysis_result]
            )

        # Tab 2: 智能重构图
        with gr.Tab("智能重构图"):
            with gr.Row():
                with gr.Column():
                    reframe_input = gr.Image(label="上传图片")
                    ratio_select = gr.Dropdown(
                        choices=[
                            "1:1 正方形 (Instagram)",
                            "4:5 竖图 (Instagram/小红书)",
                            "16:9 横屏 (YouTube)",
                            "9:16 竖屏 (Story/抖音)",
                            "2:3 封面 (小红书)",
                            "3:1 Banner",
                        ],
                        value="4:5 竖图 (Instagram/小红书)",
                        label="目标比例"
                    )
                    padding_select = gr.Dropdown(
                        choices=[
                            "不填充 (裁剪)",
                            "模糊背景",
                            "纯色填充 (白色)",
                            "镜像填充",
                        ],
                        value="模糊背景",
                        label="填充策略"
                    )
                    reframe_btn = gr.Button("开始重构图", variant="primary")

                with gr.Column():
                    reframe_output = gr.Image(label="重构图结果")
                    comparison_output = gr.Image(label="对比视图")
                    size_info = gr.Text(label="尺寸信息")

            reframe_btn.click(
                reframe_image,
                inputs=[reframe_input, ratio_select, padding_select],
                outputs=[reframe_output, comparison_output, gr.State(), size_info]
            )

        # Tab 3: 批量生成
        with gr.Tab("批量生成"):
            gr.Markdown("### 一键生成所有常用比例")

            with gr.Row():
                batch_input = gr.Image(label="上传图片")
                batch_padding = gr.Dropdown(
                    choices=[
                        "不填充 (裁剪)",
                        "模糊背景",
                        "纯色填充 (白色)",
                        "镜像填充",
                    ],
                    value="模糊背景",
                    label="填充策略"
                )
                batch_btn = gr.Button("生成所有版本", variant="primary")

            with gr.Row():
                batch_gallery = gr.Gallery(label="所有版本", columns=3)

            batch_btn.click(
                reframe_image,
                inputs=[batch_input, gr.State("4:5 竖图 (Instagram/小红书)"), batch_padding],
                outputs=[gr.State(), gr.State(), batch_gallery, gr.State()]
            )

        # Tab 4: AI 扩图
        with gr.Tab("AI 扩图"):
            gr.Markdown("""
            ### 🤖 AI 智能扩图
            使用 Stable Diffusion 进行智能图片扩展
            """)

            with gr.Row():
                with gr.Column():
                    outpaint_input = gr.Image(label="上传图片")
                    direction_select = gr.Dropdown(
                        choices=[
                            "向左扩展",
                            "向右扩展",
                            "向上扩展",
                            "向下扩展",
                            "四周扩展",
                        ],
                        value="四周扩展",
                        label="扩展方向"
                    )
                    expand_pixels = gr.Slider(
                        minimum=64,
                        maximum=512,
                        value=256,
                        step=64,
                        label="扩展像素数"
                    )
                    prompt_input = gr.Textbox(
                        label="提示词 (可选)",
                        placeholder="描述你想要扩展的内容，留空则自动生成"
                    )
                    use_ai = gr.Checkbox(
                        label="使用 AI 扩图 (取消则使用传统方法)",
                        value=True
                    )
                    outpaint_btn = gr.Button("开始扩图", variant="primary")

                with gr.Column():
                    outpaint_output = gr.Image(label="扩图结果")
                    outpaint_info = gr.Text(label="处理信息")

            outpaint_btn.click(
                outpaint_image,
                inputs=[outpaint_input, direction_select, expand_pixels, prompt_input, use_ai],
                outputs=[outpaint_output, outpaint_info]
            )

    gr.Markdown("""
    ---
    **使用说明:**
    1. 上传图片
    2. 选择目标比例和填充策略
    3. 点击按钮生成重构图

    **提示:** 模糊背景填充效果最好, 适合社交媒体发布
    """)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
