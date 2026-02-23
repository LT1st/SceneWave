@echo off
chcp 65001 >nul
echo ====================================
echo   SceneWeave - 启动中...
echo ====================================
echo.

python app\main.py

if errorlevel 1 (
    echo.
    echo 启动失败！请确保已安装依赖：
    echo   pip install -r requirements.txt
    pause
)
