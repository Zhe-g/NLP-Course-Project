# -*- coding: utf-8 -*-
"""
启动Flask API服务
用法: python scripts/run_api.py --port 5000
"""
import sys
import os
import time

print(f"[{time.strftime('%H:%M:%S')}] 步骤1: 初始化路径", flush=True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

print(f"[{time.strftime('%H:%M:%S')}] 步骤2: 导入 Flask", flush=True)
from flask import Flask, request, jsonify

print(f"[{time.strftime('%H:%M:%S')}] 步骤3: 创建 Flask 应用", flush=True)
app = Flask(__name__)

_analyzer = None
_aggregator = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        print(f"[{time.strftime('%H:%M:%S')}] 加载 ABSA 分析器...", flush=True)
        from src.pipeline.analyzer import ABSAAnalyzer
        _analyzer = ABSAAnalyzer()
        print(f"[{time.strftime('%H:%M:%S')}] ABSA 分析器加载完成", flush=True)
    return _analyzer

def get_aggregator():
    global _aggregator
    if _aggregator is None:
        from src.analysis.aggregator import StatisticsAggregator
        _aggregator = StatisticsAggregator()
    return _aggregator

print(f"[{time.strftime('%H:%M:%S')}] 步骤4: 定义路由", flush=True)

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"code": 0, "message": "ABSA Service is running", "status": "healthy"})

@app.route("/api/v1/taxonomy", methods=["GET"])
def taxonomy():
    from src.taxonomy.categories import get_all_categories
    return jsonify({"code": 0, "data": get_all_categories()})

@app.route("/api/v1/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        if not data or "text" not in data:
            return jsonify({"code": -1, "message": "缺少text字段"}), 400
        text = data["text"]
        review_id = data.get("review_id")
        result = get_analyzer().analyze(text, review_id)
        get_aggregator().add_result(result)
        return jsonify({"code": 0, "data": result})
    except Exception as e:
        return jsonify({"code": -2, "message": str(e)}), 500

@app.route("/api/v1/analyze/batch", methods=["POST"])
def analyze_batch():
    try:
        data = request.get_json(force=True)
        if not data or "texts" not in data:
            return jsonify({"code": -1, "message": "缺少texts字段"}), 400
        texts = data["texts"]
        results = get_analyzer().analyze_batch(texts)
        get_aggregator().add_batch(results)
        return jsonify({"code": 0, "data": results})
    except Exception as e:
        return jsonify({"code": -2, "message": str(e)}), 500

@app.route("/api/v1/stats", methods=["GET"])
def stats():
    report = get_aggregator().compute_stats()
    return jsonify({"code": 0, "data": report})

@app.route("/api/v1/stats/reset", methods=["POST"])
def stats_reset():
    get_aggregator().reset()
    return jsonify({"code": 0, "message": "统计数据已重置"})

print(f"[{time.strftime('%H:%M:%S')}] 步骤5: 启动服务", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ABSA Flask API Server")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("=" * 50)
    print("  ABSA 情感分析 API 服务")
    print(f"  http://{args.host}:{args.port}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)