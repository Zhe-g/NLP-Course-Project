# -*- coding: utf-8 -*-
"""
推理流水线 — 串联预处理 → 属性检测 → 情感分类 → 统计分析
"""
import time
from src.preprocessing.cleaner import clean_text
from src.aspect_detection.predict import AspectPredictor
from src.sentiment.predict import SentimentPredictor
from src.taxonomy.categories import get_category_zh, get_group
from src.config import ASAP_CATEGORIES, CATEGORY_ZH, SENTIMENT_MAP


class ABSAAnalyzer:
    """
    完整的ABSA分析流水线
    输入: 原始评论文本
    输出: 完整的属性-情感分析结果
    """

    def __init__(
        self,
        aspect_model_path: str = None,
        sentiment_model_path: str = None,
    ):
        print("=" * 50)
        print("初始化 ABSA 分析器...")
        print("=" * 50)

        # 加载模型
        print("[1/2] 加载属性检测模型...")
        self.aspect_predictor = AspectPredictor(
            model_path=aspect_model_path
        ) if aspect_model_path else AspectPredictor()

        print("[2/2] 加载情感分类模型...")
        self.sentiment_predictor = SentimentPredictor(
            model_path=sentiment_model_path
        ) if sentiment_model_path else SentimentPredictor()

        print("ABSA 分析器初始化完成！\n")

    def analyze(self, text: str, review_id: str = None) -> dict:
        """
        分析单条评论

        Args:
            text: 评论文本
            review_id: 可选，评论ID

        Returns:
            完整的分析结果
        """
        start_time = time.time()

        # Step 1: 文本清洗
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return _empty_result(text, review_id)

        # Step 2: 属性类别检测（多标签分类）
        aspect_result = self.aspect_predictor.predict(cleaned_text)
        categories = aspect_result["categories"]  # [{"name": ..., "name_zh": ..., "prob": ...}, ...]

        # Step 3: 对每个已提及的属性做情感分类
        aspects = []
        for cat_info in categories:
            cat_name = cat_info["name"]  # e.g. "dish_taste"
            sent_result = self.sentiment_predictor.predict(cleaned_text, cat_name)

            aspects.append({
                "category": cat_name,
                "category_zh": cat_info["name_zh"],
                "group": get_group(cat_name),
                "sentiment": sent_result["sentiment"],
                "sentiment_zh": sent_result["sentiment_zh"],
                "aspect_confidence": cat_info["prob"],
                "sentiment_confidence": sent_result["confidence"],
            })

        # Step 4: 汇总统计
        summary = _compute_summary(aspects)

        elapsed = round(time.time() - start_time, 3)

        return {
            "review_id": review_id,
            "text": cleaned_text,
            "aspects": aspects,
            "summary": summary,
            "elapsed_seconds": elapsed,
        }

    def analyze_batch(self, texts: list[dict]) -> list[dict]:
        """批量分析"""
        results = []
        for item in texts:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            review_id = item.get("review_id") if isinstance(item, dict) else None
            results.append(self.analyze(text, review_id))
        return results


def _compute_summary(aspects: list[dict]) -> dict:
    """计算情感汇总"""
    if not aspects:
        return {
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "overall_sentiment": "neutral",
            "total_aspects": 0,
        }

    positive_count = sum(1 for a in aspects if a["sentiment"] == "positive")
    neutral_count = sum(1 for a in aspects if a["sentiment"] == "neutral")
    negative_count = sum(1 for a in aspects if a["sentiment"] == "negative")

    # 整体情感判断
    if negative_count > positive_count:
        overall = "negative"
    elif positive_count > negative_count:
        overall = "positive"
    else:
        overall = "neutral"

    return {
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "overall_sentiment": overall,
        "total_aspects": len(aspects),
    }


def _empty_result(text: str, review_id: str = None) -> dict:
    """空文本结果"""
    return {
        "review_id": review_id,
        "text": text,
        "aspects": [],
        "summary": {
            "positive_count": 0, "neutral_count": 0, "negative_count": 0,
            "overall_sentiment": "neutral", "total_aspects": 0,
        },
        "elapsed_seconds": 0,
    }
