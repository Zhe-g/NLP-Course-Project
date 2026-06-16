# -*- coding: utf-8 -*-
"""
Flask API 应用入口和路由
"""
import sys
import os

# 确保项目根目录在path中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, request, jsonify
from src.pipeline.analyzer import ABSAAnalyzer
from src.taxonomy.categories import get_all_categories
from src.analysis.aggregator import StatisticsAggregator

# 创建Flask应用
app = Flask(__name__)

# 全局分析器实例（外部注入）
_analyzer = None
_aggregator = StatisticsAggregator()


def set_analyzer(analyzer):
    """设置分析器实例（供外部调用）"""
    global _analyzer
    _analyzer = analyzer


def get_analyzer():
    """获取分析器实例"""
    if _analyzer is None:
        raise RuntimeError("分析器未初始化，请先调用 set_analyzer()")
    return _analyzer


# ============ 路由定义 ============

@app.route("/api/v1/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "code": 0,
        "message": "ABSA Service is running",
        "status": "healthy",
    })


@app.route("/api/v1/taxonomy", methods=["GET"])
def taxonomy():
    """获取属性分类体系"""
    return jsonify({
        "code": 0,
        "data": get_all_categories(),
    })


@app.route("/api/v1/analyze", methods=["POST"])
def analyze():
    """
    单条评论分析

    Request:
        {"text": "评论内容", "review_id": "可选ID"}

    Response:
        {"code": 0, "data": {...}}
    """
    try:
        data = request.get_json(force=True)
        if not data or "text" not in data:
            return jsonify({"code": -1, "message": "缺少text字段"}), 400

        text = data["text"]
        review_id = data.get("review_id")

        analyzer = get_analyzer()
        result = analyzer.analyze(text, review_id)

        # 累积到统计分析器
        _aggregator.add_result(result)

        return jsonify({"code": 0, "data": result})

    except Exception as e:
        return jsonify({"code": -2, "message": str(e)}), 500


@app.route("/api/v1/analyze/batch", methods=["POST"])
def analyze_batch():
    """
    批量评论分析

    Request:
        {"texts": [{"text": "...", "review_id": "..."}, ...]}

    Response:
        {"code": 0, "data": [result1, result2, ...]}
    """
    try:
        data = request.get_json(force=True)
        if not data or "texts" not in data:
            return jsonify({"code": -1, "message": "缺少texts字段"}), 400

        texts = data["texts"]

        analyzer = get_analyzer()
        results = analyzer.analyze_batch(texts)

        # 累积到统计分析器
        _aggregator.add_batch(results)

        return jsonify({"code": 0, "data": results})

    except Exception as e:
        return jsonify({"code": -2, "message": str(e)}), 500


@app.route("/api/v1/stats", methods=["GET"])
def stats():
    """获取累积的统计分析报告"""
    report = _aggregator.compute_stats()
    return jsonify({"code": 0, "data": report})


@app.route("/api/v1/stats/reset", methods=["POST"])
def stats_reset():
    """重置统计数据"""
    _aggregator.reset()
    return jsonify({"code": 0, "message": "统计数据已重置"})


# ============ 错误处理 ============

@app.errorhandler(404)
def not_found(error):
    return jsonify({"code": -404, "message": "接口不存在"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"code": -500, "message": "服务器内部错误"}), 500


# ============ 入口 ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ABSA Flask API Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("=" * 50)
    print("  ABSA 情感分析 API 服务")
    print("  http://{}:{}".format(args.host, args.port))
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug)
