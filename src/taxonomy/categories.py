# -*- coding: utf-8 -*-
"""
属性分类体系定义
基于ASAP数据集的18个评价维度 + 5个一级维度
直接从config导入核心常量，这里提供便捷的查询接口
"""
from src.config import ASAP_CATEGORIES, CATEGORY_ZH, CATEGORY_GROUPS, SENTIMENT_MAP, SENTIMENT_ZH


def get_category_zh(category: str) -> str:
    """获取维度中文名"""
    return CATEGORY_ZH.get(category, category)


def get_group(category: str) -> str:
    """获取维度所属的一级维度"""
    for group_name, members in CATEGORY_GROUPS.items():
        if category in members:
            return group_name
    return "其他"


def get_group_zh(category: str, group: str = None) -> str:
    """获取一级维度中文名"""
    if group is None:
        group = get_group(category)
    return group


def get_all_categories() -> list[dict]:
    """获取完整的分类体系（供API返回）"""
    result = []
    for group_name, members in CATEGORY_GROUPS.items():
        group_info = {
            "group": group_name,
            "categories": []
        }
        for cat in members:
            group_info["categories"].append({
                "name": cat,
                "name_zh": CATEGORY_ZH.get(cat, cat),
            })
        result.append(group_info)
    return result


def get_sentiment_label(polarity: int) -> str:
    """数值极性→文本标签"""
    return SENTIMENT_MAP.get(polarity, "neutral")
