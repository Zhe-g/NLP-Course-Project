# -*- coding: utf-8 -*-
"""
启动 Flask API 服务（懒加载版）
用法: python scripts/run_api.py --port 5000

路由定义统一在 src/api/app.py，此文件仅负责懒加载 + 启动，
避免冷启动时 PyTorch 模型阻塞 Flask 导入。
"""
import sys
import os
import time

print(f"[{time.strftime('%H:%M:%S')}] 初始化路径", flush=True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

print(f"[{time.strftime('%H:%M:%S')}] 导入 Flask 应用", flush=True)
from src.api.app import app, set_analyzer

print(f"[{time.strftime('%H:%M:%S')}] Flask 应用加载完成", flush=True)


def init_analyzer():
    """懒加载 ABSA 分析器（首次请求前完成，避免响应延迟）"""
    print(f"[{time.strftime('%H:%M:%S')}] 加载 ABSA 分析器...", flush=True)
    from src.pipeline.analyzer import ABSAAnalyzer
    analyzer = ABSAAnalyzer()
    set_analyzer(analyzer)
    print(f"[{time.strftime('%H:%M:%S')}] ABSA 分析器加载完成", flush=True)
    return analyzer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ABSA Flask API Server")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    # 启动前完成模型加载，确保服务就绪即可用
    init_analyzer()

    print("=" * 50)
    print("  ABSA 情感分析 API 服务")
    print(f"  http://{args.host}:{args.port}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
