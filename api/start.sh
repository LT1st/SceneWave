#!/bin/bash
# SceneWeave API 启动脚本

echo "===================================="
echo "  SceneWeave API 服务启动中..."
echo "===================================="
echo ""

# 检查虚拟环境
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  警告: 未检测到虚拟环境"
    echo "建议先运行: source venv/bin/activate"
    echo ""
fi

# 启动服务
python api/main.py
