# -*- coding: utf-8 -*-
"""
启动Flask API服务
用法: python scripts/run_api.py --port 5000
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA Flask API Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("=" * 50)
    print("  ABSA 情感分析 API 服务")
    print("  Listening on http://{}:{}".format(args.host, args.port))
    print("  API Docs:")
    print("    POST /api/v1/analyze     - 单条评论分析")
    print("    POST /api/v1/analyze/batch - 批量分析")
    print("    GET  /api/v1/taxonomy     - 分类体系")
    print("    GET  /api/v1/stats        - 统计报告")
    print("    GET  /api/v1/health       - 健康检查")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug)
