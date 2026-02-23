#!/bin/bash

# SceneWeave Mobile 启动脚本

echo "🚀 SceneWeave Mobile 启动脚本"
echo "================================"

# 检查 Flutter 是否安装
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter 未安装，请先安装 Flutter"
    exit 1
fi

echo "✅ Flutter 版本："
flutter --version

# 检查设备
echo ""
echo "📱 可用设备："
flutter devices

# 安装依赖
echo ""
echo "📦 安装依赖..."
flutter pub get

# 运行应用
echo ""
echo "🎯 启动应用..."
flutter run
