#!/usr/bin/env python3
"""
SceneWeave CLI - 命令行工具
"""

import sys
import os
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from src.core import SubjectDetector, CompositionScorer, Reframer
from src.core.reframer import PaddingStrategy, AspectRatio

app = typer.Typer(
    name="sceneweave",
    help="AI 智能图片重构图工具 - 让每一张照片都成为完美构图"
)
console = Console()


@app.command()
def analyze(
    image: str = typer.Argument(..., help="图片路径"),
    draw: bool = typer.Option(False, "--draw", "-d", help="绘制检测结果"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出路径")
):
    """分析图片构图"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 检测主体
        task = progress.add_task("正在检测主体...", total=None)
        detector = SubjectDetector(model_size="n")
        subjects = detector.detect(image)
        progress.remove_task(task)

    # 显示检测结果
    console.print(f"\n[bold green]检测到 {len(subjects)} 个主体[/bold green]")

    if subjects:
        table = Table(title="主体列表")
        table.add_column("序号", style="cyan", width=6)
        table.add_column("类别", style="green")
        table.add_column("置信度", style="yellow")
        table.add_column("位置", style="blue")

        for i, s in enumerate(subjects[:5]):  # 最多显示5个
            table.add_row(
                str(i + 1),
                s.label,
                f"{s.confidence:.2f}",
                f"({s.center[0]:.0f}, {s.center[1]:.0f})"
            )
        console.print(table)

    # 分析构图
    main_subject = subjects[0] if subjects else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("正在分析构图...", total=None)
        scorer = CompositionScorer()

        if main_subject:
            score = scorer.score(
                image,
                subject_bbox=main_subject.bbox,
                subject_center=main_subject.center
            )
        else:
            score = scorer.score(image)
        progress.remove_task(task)

    # 显示评分
    score_table = Table(title="构图评分")
    score_table.add_column("维度", style="cyan")
    score_table.add_column("得分", style="green")
    score_table.add_column("满分", style="dim")

    score_table.add_row("三分法则", f"{score.rule_of_thirds:.1f}", "30")
    score_table.add_row("视觉平衡", f"{score.visual_balance:.1f}", "25")
    score_table.add_row("主体突出", f"{score.subject_prominence:.1f}", "25")
    score_table.add_row("呼吸空间", f"{score.breathing_room:.1f}", "20")
    score_table.add_row("[bold]总分[/bold]", f"[bold]{score.total:.1f}[/bold]", "[dim]100[/dim]")

    console.print(score_table)

    # 评级
    grade_color = {
        "优秀": "green",
        "良好": "blue",
        "一般": "yellow",
        "待改进": "orange1",
        "需要重构": "red"
    }

    for key, color in grade_color.items():
        if key in score.grade:
            console.print(f"\n[{color}]评级: {score.grade}[/{color}]")
            break

    # 显示问题
    issues = scorer.get_issues()
    if issues:
        console.print("\n[bold yellow]检测到的问题:[/bold yellow]")
        for issue in issues:
            severity_color = {"high": "red", "medium": "yellow", "low": "blue"}
            console.print(
                f"  [{severity_color.get(issue.severity, 'white')}]{issue.description}[/{severity_color.get(issue.severity, 'white')}]"
            )
            console.print(f"    [dim]建议: {issue.suggestion}[/dim]")

    # 绘制检测结果
    if draw:
        output_path = output or "detection_result.jpg"
        import cv2
        result_img = detector.draw_detections(image, subjects, output_path)
        console.print(f"\n[green]检测结果已保存到: {output_path}[/green]")


@app.command()
def reframe(
    image: str = typer.Argument(..., help="图片路径"),
    ratio: str = typer.Option("4:5", "--ratio", "-r", help="目标比例 (1:1, 4:5, 16:9, 9:16, 3:1, 2:3)"),
    padding: str = typer.Option("none", "--padding", "-p", help="填充策略 (none, blur, color, mirror)"),
    output: str = typer.Option("output.jpg", "--output", "-o", help="输出路径"),
    all: bool = typer.Option(False, "--all", "-a", help="生成所有常用比例")
):
    """重构图 - 调整比例和构图"""

    # 解析比例
    ratio_map = {
        "1:1": (1, 1),
        "4:5": (4, 5),
        "16:9": (16, 9),
        "9:16": (9, 16),
        "3:1": (3, 1),
        "2:3": (2, 3),
    }

    target_ratio = ratio_map.get(ratio)
    if not target_ratio:
        console.print(f"[red]不支持的比例: {ratio}[/red]")
        console.print(f"支持的比例: {', '.join(ratio_map.keys())}")
        raise typer.Exit(1)

    # 解析填充策略
    padding_map = {
        "none": PaddingStrategy.NONE,
        "blur": PaddingStrategy.BLUR,
        "color": PaddingStrategy.COLOR,
        "mirror": PaddingStrategy.MIRROR,
    }

    padding_strategy = padding_map.get(padding, PaddingStrategy.NONE)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 检测主体
        task = progress.add_task("正在检测主体...", total=None)
        detector = SubjectDetector(model_size="n")
        subjects = detector.detect(image)
        main_subject = subjects[0] if subjects else None
        progress.remove_task(task)

        # 重构图
        task = progress.add_task("正在重构图...", total=None)
        reframer = Reframer()

        if all:
            # 生成所有比例
            ratios = [(1, 1), (4, 5), (16, 9), (9, 16), (2, 3)]
            results = reframer.reframe_multiple(
                image,
                ratios=ratios,
                subject_center=main_subject.center if main_subject else None,
                subject_bbox=main_subject.bbox if main_subject else None,
                padding=padding_strategy
            )
        else:
            result = reframer.reframe(
                image,
                target_ratio=target_ratio,
                subject_center=main_subject.center if main_subject else None,
                subject_bbox=main_subject.bbox if main_subject else None,
                padding=padding_strategy
            )
            results = [result]

        progress.remove_task(task)

    # 保存结果
    import cv2

    output_dir = Path(output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        if all:
            # 生成文件名
            stem = Path(image).stem
            ratio_str = f"{result.new_size[0]}x{result.new_size[1]}"
            out_path = output_dir / f"{stem}_{ratio_str}.jpg"
        else:
            out_path = Path(output)

        # 保存
        output_img = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), output_img)

        console.print(
            f"[green]✓[/green] {result.original_size[0]}x{result.original_size[1]} "
            f"-> {result.new_size[0]}x{result.new_size[1]} "
            f"[dim]{out_path}[/dim]"
        )

    console.print(f"\n[bold green]重构图完成![/bold green]")


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="输入目录"),
    output_dir: str = typer.Option("output", "--output", "-o", help="输出目录"),
    ratio: str = typer.Option("4:5", "--ratio", "-r", help="目标比例"),
    padding: str = typer.Option("blur", "--padding", "-p", help="填充策略")
):
    """批量处理图片"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    if not images:
        console.print(f"[yellow]未找到图片文件: {input_dir}[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]找到 {len(images)} 张图片[/bold]\n")

    # 处理每张图片
    with Progress(console=console) as progress:
        task = progress.add_task("处理中...", total=len(images))

        for img_path in images:
            try:
                # 使用 reframe 命令处理
                import subprocess
                result = subprocess.run(
                    ["python", __file__, "reframe", str(img_path),
                     "-r", ratio, "-p", padding,
                     "-o", str(output_path / f"{img_path.stem}_{ratio.replace(':', 'x')}.jpg")],
                    capture_output=True
                )
            except Exception as e:
                console.print(f"[red]处理失败: {img_path.name} - {e}[/red]")

            progress.advance(task)

    console.print(f"\n[bold green]批量处理完成! 输出目录: {output_dir}[/bold green]")


@app.command()
def web():
    """启动 Web UI"""
    console.print("[bold blue]启动 Web UI...[/bold blue]")
    import subprocess
    subprocess.run(["python", str(Path(__file__).parent.parent / "web" / "app.py")])


if __name__ == "__main__":
    app()
