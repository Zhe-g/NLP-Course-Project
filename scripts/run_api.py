# -*- coding: utf-8 -*-
"""
启动Flask API服务
用法: python scripts/run_api.py --port 5000
"""
import sys
import os

# 确保项目根目录在path中
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 导入 Flask 应用（模型会在导入时自动加载）
from src.api.app import app

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA Flask API Server")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("=" * 50)
    print("  ABSA 情感分析 API 服务")
    print("  Listening on http://{}:{}".format(args.host, args.port))
    print("=" * 50)

    # 多线程模式，避免并发请求时死锁
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
