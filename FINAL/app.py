# -*- coding: utf-8 -*-
"""
ABSA 情感分析 Web 应用 — Flask 入口 (v2)
前端界面 + 调用我们训练的 RoBERTa 模型 API
SnowNLP 已彻底移除，全部走模型
"""
import os
import json
import uuid
import threading
import queue
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 清理上次运行遗留的未删除临时文件
for _old_file in os.listdir(UPLOAD_FOLDER):
    try:
        os.remove(os.path.join(UPLOAD_FOLDER, _old_file))
    except Exception:
        pass

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

    # 加唯一后缀避免同名文件并发冲突
    raw_name = secure_filename(file.filename or "upload.tmp")
    unique_name = f"{uuid.uuid4().hex[:8]}_{raw_name}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    try:
        result = analyzer.analyze_file(filepath)
    finally:
        _safe_remove(filepath)

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/batch-upload", methods=["POST"])
def batch_upload():
    """批量文件上传 → 合并分析（带进度）"""
    if "files" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    files = request.files.getlist("files")
    
    def generate():
        all_results = []
        total_files = len([f for f in files if f.filename])
        current_file = 0
        
        for f in files:
            if not f.filename:
                continue
            
            current_file += 1
            filename = f.filename
            yield f"data: {json.dumps({'stage': 'upload', 'current': current_file, 'total': total_files, 'file': filename, 'message': f'正在处理文件: {filename}'})}\n\n"
            
            raw_name = secure_filename(f.filename or "batch.tmp")
            unique_name = f"{uuid.uuid4().hex[:8]}_{raw_name}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_name)
            f.save(filepath)

            try:
                # 线程+队列：progress_callback 不再用 yield（生成器不会被 analyze_file 迭代）
                pq = queue.Queue()

                def progress_callback(current, total, message):
                    pq.put({
                        'stage': 'analyze',
                        'current': current,
                        'total': total,
                        'file': filename,
                        'progress': int((current_file - 1) / total_files * 100
                                        + current / total / total_files * 100),
                        'message': message,
                    })

                result_holder = [None]
                error_holder = [None]

                def _run():
                    try:
                        result_holder[0] = analyzer.analyze_file(
                            filepath, progress_callback=progress_callback
                        )
                    except Exception as exc:
                        error_holder[0] = str(exc)

                t = threading.Thread(target=_run, daemon=True)
                t.start()

                while t.is_alive():
                    try:
                        msg = pq.get(timeout=0.3)
                        yield f"data: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'stage': 'heartbeat'})}\n\n"

                t.join()

                while not pq.empty():
                    try:
                        msg = pq.get_nowait()
                        yield f"data: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        break

                if error_holder[0]:
                    yield f"data: {json.dumps({'stage': 'error', 'error': error_holder[0]})}\n\n"
                else:
                    r = result_holder[0]
                    if r and "error" not in r:
                        all_results.append(r)
            finally:
                _safe_remove(filepath)
        
        # 合并结果
        yield f"data: {json.dumps({'stage': 'merge', 'message': '正在合并分析结果...'})}\n\n"
        
        if not all_results:
            yield f"data: {json.dumps({'stage': 'error', 'error': '所有文件分析失败'})}\n\n"
            return
        
        merged = _merge(all_results)
        yield f"data: {json.dumps({'stage': 'complete', 'data': merged})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


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


def _safe_remove(filepath: str):
    """安全删除文件 — 文件不存在时静默忽略"""
    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass  # 已被其他进程删除，无需处理


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
