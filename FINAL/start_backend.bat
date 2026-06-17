@echo off
echo ==============================================
echo   ABSA 细粒度情感分析系统 — 启动脚本
echo ==============================================
echo.
echo [步骤 1/2] 启动模型 API 服务 (端口 5000)...

REM 启动模型API
REM 获取项目根目录（本脚本位于 FINAL\，根目录在上一级）
set "PROJECT_ROOT=%~dp0.."
start "ABSA-Model-API" cmd /c "cd /d %PROJECT_ROOT% && conda activate NLPhomework && python scripts\run_api.py --port 5000"

echo 等待模型服务启动...
timeout /t 8 /nobreak >nul

echo [步骤 2/2] 启动 Web 界面 (端口 5001)...
echo.
echo 浏览器打开: http://localhost:5001
echo.
echo 关闭本窗口即可停止所有服务
echo ==============================================

REM 启动Web界面
python app.py

echo.
pause
