# -*- coding: utf-8 -*-
"""
文本清洗模块
"""
import re


def clean_text(text: str, max_length: int = 512) -> str:
    """
    清洗评论文本：去除HTML标签、URL、特殊字符，截断过长文本

    Args:
        text: 原始评论文本
        max_length: 最大保留长度（字符数）

    Returns:
        清洗后的文本
    """
    if not isinstance(text, str):
        return ""

    # 去除HTML标签
    text = re.sub(r"<[^>]+>", "", text)

    # 去除URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 去除微博@提及
    text = re.sub(r"@\S+", "", text)

    # 去除转义换行符
    text = text.replace("\\n", " ").replace("\\t", " ")

    # 全角转半角
    text = full_to_half(text)

    # 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()

    # 截断过长文本
    if len(text) > max_length:
        text = text[:max_length]

    return text


def full_to_half(text: str) -> str:
    """全角字符转半角"""
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        elif code == 0x3000:  # 全角空格
            code = 0x0020
        result.append(chr(code))
    return "".join(result)


def is_valid_review(text: str, min_length: int = 5) -> bool:
    """
    判断是否有效评论
    - 长度 >= min_length
    - 至少包含1个中文字符
    """
    if len(text) < min_length:
        return False
    # 检查是否包含中文
    if not re.search(r"[一-鿿]", text):
        return False
    return True
