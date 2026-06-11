# -*- coding: utf-8 -*-
"""
多维度统计分析模块
"""
from collections import Counter
from src.config import CATEGORY_GROUPS, CATEGORY_ZH


class StatisticsAggregator:
    """统计分析器"""

    def __init__(self):
        self.results = []  # 累积分析结果

    def add_result(self, result: dict):
        """添加单条分析结果"""
        self.results.append(result)

    def add_batch(self, results: list[dict]):
        """批量添加"""
        self.results.extend(results)

    def compute_stats(self) -> dict:
        """
        计算多维度统计报告
        """
        if not self.results:
            return {"message": "暂无数据"}

        # 收集所有aspect信息
        all_aspects = []
        for r in self.results:
            for a in r.get("aspects", []):
                all_aspects.append(a)

        if not all_aspects:
            return {"message": "未提取到任何评价维度"}

        total_reviews = len(self.results)
        total_aspects = len(all_aspects)

        # 1. 全局情感分布
        sentiment_counter = Counter(a["sentiment"] for a in all_aspects)
        sentiment_dist = {
            "positive": sentiment_counter.get("positive", 0),
            "neutral": sentiment_counter.get("neutral", 0),
            "negative": sentiment_counter.get("negative", 0),
        }
        sentiment_ratio = {
            k: round(v / total_aspects * 100, 1) for k, v in sentiment_dist.items()
        }

        # 2. 按一级维度统计
        group_stats = {}
        for group_name, members in CATEGORY_GROUPS.items():
            group_aspects = [a for a in all_aspects if a["category"] in members]
            if not group_aspects:
                continue
            sc = Counter(a["sentiment"] for a in group_aspects)
            group_stats[group_name] = {
                "total": len(group_aspects),
                "positive": sc.get("positive", 0),
                "neutral": sc.get("neutral", 0),
                "negative": sc.get("negative", 0),
                "positive_ratio": round(sc.get("positive", 0) / len(group_aspects) * 100, 1),
                "negative_ratio": round(sc.get("negative", 0) / len(group_aspects) * 100, 1),
            }

        # 3. 按具体维度统计
        category_stats = {}
        for a in all_aspects:
            cat = a["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "positive": 0, "neutral": 0, "negative": 0}
            category_stats[cat]["total"] += 1
            category_stats[cat][a["sentiment"]] += 1

        # 计算好评率和差评率排名
        category_rankings = []
        for cat, stats in category_stats.items():
            category_rankings.append({
                "category": cat,
                "category_zh": CATEGORY_ZH.get(cat, cat),
                "total": stats["total"],
                "positive_ratio": round(stats["positive"] / stats["total"] * 100, 1),
                "negative_ratio": round(stats["negative"] / stats["total"] * 100, 1),
            })

        # 按好评率排序
        category_rankings.sort(key=lambda x: x["positive_ratio"], reverse=True)

        # 4. 整体情感汇总
        overall_sentiments = Counter(r["summary"]["overall_sentiment"] for r in self.results)

        return {
            "total_reviews": total_reviews,
            "total_aspects": total_aspects,
            "avg_aspects_per_review": round(total_aspects / total_reviews, 1),
            "overall_distribution": {
                "positive_ratio": round(overall_sentiments.get("positive", 0) / total_reviews * 100, 1),
                "neutral_ratio": round(overall_sentiments.get("neutral", 0) / total_reviews * 100, 1),
                "negative_ratio": round(overall_sentiments.get("negative", 0) / total_reviews * 100, 1),
            },
            "sentiment_distribution": {
                "counts": sentiment_dist,
                "ratios": sentiment_ratio,
            },
            "group_stats": group_stats,
            "category_rankings": category_rankings,
        }

    def reset(self):
        """清空累积数据"""
        self.results = []
