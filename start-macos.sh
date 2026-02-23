#!/bin/bash
# SceneWeave macOS 应用启动脚本

echo "===================================="
echo "  SceneWeave macOS 应用启动中..."
echo "===================================="
echo ""

# 检查虚拟环境
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  警告: 未检测到虚拟环境"
    echo "建议先运行: source venv/bin/activate"
    echo ""
fi

# 启动应用
python desktop/macos/main.py
