# -*- coding: utf-8 -*-
"""
中文分词接口（jieba封装）
注意：RoBERTa使用字符级tokenizer，jieba分词用于辅助统计
"""
import jieba


def tokenize(text: str) -> list[str]:
    """jieba精确模式分词"""
    return list(jieba.cut(text.strip()))


def tokenize_for_search(text: str) -> list[str]:
    """jieba搜索引擎模式分词"""
    return list(jieba.cut_for_search(text.strip()))


def extract_keywords(text: str, topk: int = 10) -> list[str]:
    """基于TF-IDF提取关键词（简化版，使用jieba内置）"""
    import jieba.analyse
    return jieba.analyse.extract_tags(text, topK=topk, withWeight=False)
