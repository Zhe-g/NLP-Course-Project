# -*- coding: utf-8 -*-
"""
情感分析核心模块 — 对接训练的 RoBERTa ABSA 模型 API
SnowNLP + 关键词匹配已全部移除，现在调用我们训练好的模型
"""
import requests
import pandas as pd
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

# 模型 API 地址（我们自己训练的 ABSA 服务）
MODEL_API_BASE = "http://127.0.0.1:5000/api/v1"


class SentimentAnalyzer:
    """ABSA 情感分析器 — 调用训练好的 RoBERTa 模型"""

    def __init__(self, api_base: str = MODEL_API_BASE):
        self.api_base = api_base

    def _check_health(self) -> bool:
        """检查模型服务是否在线"""
        try:
            resp = requests.get(f"{self.api_base}/health", timeout=5)
            return resp.json().get("code") == 0
        except Exception:
            return False

    def clean_text(self, text: str) -> str:
        """基础文本清洗"""
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\\n|\\t', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def analyze_text(self, text: str) -> dict:
        """
        单条评论分析 — 调用模型 API

        Returns:
            {
                'text': ...,
                'aspects': [{'category': 'Food#Taste', 'category_zh': '菜品口味',
                             'group': '菜品', 'sentiment': 'positive', ...}, ...],
                'summary': {positive_count, negative_count, neutral_count, overall_sentiment}
            }
        """
        text = self.clean_text(text)
        if not text:
            return {"text": "", "aspects": [], "summary": _empty_summary()}

        try:
            resp = requests.post(
                f"{self.api_base}/analyze",
                json={"text": text},
                timeout=30,
            )
            data = resp.json()
            if data.get("code") == 0:
                return data["data"]
            else:
                return {"text": text, "aspects": [], "summary": _empty_summary(),
                        "error": data.get("message", "API error")}
        except requests.exceptions.ConnectionError:
            return {"text": text, "aspects": [], "summary": _empty_summary(),
                    "error": "模型服务未启动，请先运行 python scripts/run_api.py --port 5000"}
        except Exception as e:
            return {"text": text, "aspects": [], "summary": _empty_summary(),
                    "error": str(e)}

    def analyze_file(self, filepath: str, progress_callback=None) -> dict:
        """
        批量文件分析 — 逐条调用模型 API

        支持: TXT (每行一条评论), CSV, XLSX
        
        Args:
            filepath: 文件路径
            progress_callback: 进度回调函数 (current, total, message)
        """
        # 读取文件
        ext = os.path.splitext(filepath)[1].lower()
        reviews = []

        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                reviews = [line.strip() for line in f if line.strip()]
        elif ext == '.csv':
            df = pd.read_csv(filepath)
            # 自动找评论文本列
            text_cols = [c for c in df.columns if c.lower() in ('review', 'text', 'reviewbody', 'content', 'comment')]
            col = text_cols[0] if text_cols else df.columns[0]
            reviews = df[col].dropna().astype(str).tolist()
        elif ext == '.xlsx':
            df = pd.read_excel(filepath)
            text_cols = [c for c in df.columns if c.lower() in ('review', 'text', 'reviewbody', 'content', 'comment')]
            col = text_cols[0] if text_cols else df.columns[0]
            reviews = df[col].dropna().astype(str).tolist()
        else:
            return {'error': f'不支持的文件格式: {ext}'}

        if not reviews:
            return {'error': '文件中没有找到评论文本'}

        # 逐条分析
        results = []
        total = len(reviews)
        if progress_callback:
            progress_callback(0, total, '开始分析...')
        
        for i, review in enumerate(reviews):
            result = self.analyze_text(review)
            result["index"] = i
            results.append(result)
            
            # 更新进度
            if progress_callback:
                progress = i + 1
                progress_callback(progress, total, f'正在分析第 {progress}/{total} 条评论...')

        # ===== 多维度汇总统计 =====
        summary = _compute_multi_dimension_summary(results)

        return {
            'summary': summary,
            'results': results,
        }

    def analyze_texts(self, texts: list[str], progress_callback=None) -> dict:
        """批量分析多条文本
        
        Args:
            texts: 文本列表
            progress_callback: 进度回调函数 (current, total, message)
        """
        results = []
        total = len(texts)
        if progress_callback:
            progress_callback(0, total, '开始分析...')
        
        for i, text in enumerate(texts):
            result = self.analyze_text(text)
            result["index"] = i
            results.append(result)
            
            if progress_callback:
                progress = i + 1
                progress_callback(progress, total, f'正在分析第 {progress}/{total} 条评论...')

        summary = _compute_multi_dimension_summary(results)
        return {
            'summary': summary,
            'results': results,
        }

    def get_taxonomy(self) -> list[dict]:
        """获取 18 维度分类体系"""
        try:
            resp = requests.get(f"{self.api_base}/taxonomy", timeout=5)
            data = resp.json()
            return data.get("data", [])
        except Exception:
            return []


# ===== 辅助函数 =====

def _empty_summary() -> dict:
    return {
        "positive_count": 0, "neutral_count": 0, "negative_count": 0,
        "overall_sentiment": "neutral", "total_aspects": 0,
    }


def _compute_multi_dimension_summary(results: list[dict]) -> dict:
    """
    多维度统计汇总（18 维度级别）：
    - 全局情感分布
    - 5 个一级维度（位置/服务/价格/环境/菜品）的正向/负向占比
    - 18 个具体维度的好评率排名
    """
    total_reviews = len(results)
    if total_reviews == 0:
        return {"total_reviews": 0, "error": "无有效分析结果"}

    # 收集所有 aspects
    all_aspects = []
    review_sentiments = []
    for r in results:
        aspects = r.get("aspects", [])
        all_aspects.extend(aspects)
        summary = r.get("summary", {})
        review_sentiments.append(summary.get("overall_sentiment", "neutral"))

    # 1. 整体评论情感分布
    review_dist = Counter(review_sentiments)

    # 2. 所有维度情感分布
    sentiment_counter = Counter(a["sentiment"] for a in all_aspects)
    total_aspects = len(all_aspects) if all_aspects else 1

    # 3. 按 5 个一级维度统计
    group_stats = defaultdict(lambda: {"total": 0, "positive": 0, "neutral": 0, "negative": 0})
    for a in all_aspects:
        g = a.get("group", "其他")
        group_stats[g]["total"] += 1
        group_stats[g][a["sentiment"]] += 1

    group_result = {}
    for g, s in group_stats.items():
        t = max(s["total"], 1)
        group_result[g] = {
            "total": s["total"],
            "positive": s["positive"],
            "neutral": s["neutral"],
            "negative": s["negative"],
            "positive_ratio": round(s["positive"] / t * 100, 1),
            "negative_ratio": round(s["negative"] / t * 100, 1),
        }

    # 4. 按 18 维度好评率排名
    cat_counter = defaultdict(lambda: {"total": 0, "positive": 0, "neutral": 0, "negative": 0})
    for a in all_aspects:
        cat = a.get("category", "unknown")
        cat_counter[cat]["total"] += 1
        cat_counter[cat][a["sentiment"]] += 1

    category_rankings = []
    for cat, s in cat_counter.items():
        t = max(s["total"], 1)
        # 从当前类别的aspect中获取正确的category_zh
        cat_zh = cat  # fallback
        for a in all_aspects:
            if a.get("category") == cat:
                cat_zh = a.get("category_zh", cat)
                break
        category_rankings.append({
            "category": cat,
            "category_zh": cat_zh,
            "total": s["total"],
            "positive_ratio": round(s["positive"] / t * 100, 1),
            "negative_ratio": round(s["negative"] / t * 100, 1),
        })
    category_rankings.sort(key=lambda x: x["positive_ratio"], reverse=True)

    # 5. 平均每条评论的维度数
    avg_aspects = round(total_aspects / total_reviews, 1) if total_reviews > 0 else 0

    return {
        "total_reviews": total_reviews,
        "total_aspects": total_aspects,
        "avg_aspects_per_review": avg_aspects,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # 整体评论情感
        "review_sentiment_dist": {
            "positive": review_dist.get("positive", 0),
            "neutral": review_dist.get("neutral", 0),
            "negative": review_dist.get("negative", 0),
            "positive_ratio": round(review_dist.get("positive", 0) / total_reviews * 100, 1),
            "negative_ratio": round(review_dist.get("negative", 0) / total_reviews * 100, 1),
        },

        # 所有维度情感分布
        "aspect_sentiment_dist": {
            "positive": sentiment_counter.get("positive", 0),
            "neutral": sentiment_counter.get("neutral", 0),
            "negative": sentiment_counter.get("negative", 0),
            "positive_ratio": round(sentiment_counter.get("positive", 0) / total_aspects * 100, 1),
            "negative_ratio": round(sentiment_counter.get("negative", 0) / total_aspects * 100, 1),
        },

        # 5 个一级维度统计
        "group_stats": group_result,

        # 18 维度排名
        "category_rankings": category_rankings,
    }
