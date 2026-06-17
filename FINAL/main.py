# -*- coding: utf-8 -*-
"""
一键启动 — 模型API (port 5000) + Web界面 (port 5001)
"""
import subprocess
import sys
import os
import time
import threading
import webbrowser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_API_SCRIPT = os.path.join(BASE_DIR, "scripts", "run_api.py")
WEB_APP_FILE = os.path.join(os.path.dirname(__file__), "app.py")


def _read_output(proc, prefix):
    """读取子进程输出"""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"  [{prefix}] {line.rstrip()}")


def start_model_api():
    """启动模型 API (端口 5000)"""
    print("[1/2] 启动模型 API 服务 (http://localhost:5000)...")
    proc = subprocess.Popen(
        [sys.executable, MODEL_API_SCRIPT, "--port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_read_output, args=(proc, "API"), daemon=True).start()
    return proc


def start_web_ui():
    """启动 Web 界面 (端口 5001)"""
    print("[2/2] 启动 Web 界面 (http://localhost:5001)...")
    proc = subprocess.Popen(
        [sys.executable, WEB_APP_FILE],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_read_output, args=(proc, "WEB"), daemon=True).start()
    return proc


def wait_for_api(url: str, timeout: int = 60):
    """等待 API 就绪"""
    import urllib.request
    for i in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    print("=" * 55)
    print("  ABSA 细粒度情感分析系统")
    print("  RoBERTa-wwm-ext · 18 维度")
    print("=" * 55)
    print()

    # 启动模型 API
    api_proc = start_model_api()

    # 等待 API 就绪
    print("  等待模型加载...", end="", flush=True)
    if not wait_for_api("http://127.0.0.1:5000/api/v1/health"):
        print(" 失败!")
        print("  模型 API 启动超时，请检查环境")
        api_proc.kill()
        sys.exit(1)
    print(" OK")

    # 启动 Web 界面
    web_proc = start_web_ui()

    # 等待 Web 就绪
    print("  等待 Web 界面就绪...", end="", flush=True)
    if not wait_for_api("http://127.0.0.1:5001/health"):
        print(" 失败!")
        api_proc.kill()
        web_proc.kill()
        sys.exit(1)
    print(" OK")

    print()
    print("=" * 55)
    print("  全部就绪!")
    print(f"  浏览器打开: http://localhost:5001")
    print("  按 Ctrl+C 停止所有服务")
    print("=" * 55)

    # 自动打开浏览器
    webbrowser.open("http://localhost:5001")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        api_proc.terminate()
        web_proc.terminate()
        print("已停止")


if __name__ == "__main__":
    main()
