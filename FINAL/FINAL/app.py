# -*- coding: utf-8 -*-
"""
ABSA 情感分析 Web 应用 — Flask 入口 (v2)
前端界面 + 调用我们训练的 RoBERTa 模型 API
SnowNLP 已彻底移除，全部走模型
"""
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

analyzer = SentimentAnalyzer()


@app.route("/")
def index():
    """主页面"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """单文件上传 → 模型分析"""
    if "file" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename or "upload.tmp"))
    file.save(filepath)

    try:
        result = analyzer.analyze_file(filepath)
    finally:
        os.remove(filepath)

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/batch-upload", methods=["POST"])
def batch_upload():
    """批量文件上传 → 合并分析"""
    if "files" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    files = request.files.getlist("files")
    all_results = []
    for f in files:
        if not f.filename:
            continue
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename or "batch.tmp"))
        f.save(filepath)
        try:
            r = analyzer.analyze_file(filepath)
            if "error" not in r:
                all_results.append(r)
        finally:
            os.remove(filepath)

    if not all_results:
        return jsonify({"error": "所有文件分析失败"}), 400

    merged = _merge(all_results)
    return jsonify(merged)


@app.route("/input-text", methods=["POST"])
def input_text():
    """手动输入 → 模型分析"""
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "请输入评论内容"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "评论内容不能为空"}), 400

    result = analyzer.analyze_text(text)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """健康检查 + 模型状态"""
    model_ok = analyzer._check_health()
    return jsonify({
        "web_ui": "running",
        "model_api": "connected" if model_ok else "disconnected",
        "hint": "请先启动模型API: python scripts/run_api.py --port 5000" if not model_ok else "",
    })


def _merge(batches: list[dict]) -> dict:
    """合并多个批次结果"""
    from sentiment_analyzer import _compute_multi_dimension_summary
    all_r = []
    for b in batches:
        all_r.extend(b.get("results", []))
    return {
        "summary": _compute_multi_dimension_summary(all_r),
        "results": all_r,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  ABSA 情感分析 Web 界面")
    print("  浏览器打开: http://localhost:5001")
    print("  ")
    print("  依赖模型 API: http://localhost:5000")
    print("  如未启动, 先运行: python scripts/run_api.py --port 5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5001, debug=False)
