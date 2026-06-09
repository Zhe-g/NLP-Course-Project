# -*- coding: utf-8 -*-
"""
中文停用词处理
"""
import os


# 内置常用中文停用词
_DEFAULT_STOPWORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些", "所", "为",
    "因为", "所以", "但是", "然而", "不过", "还是", "只是", "可以", "这个", "那个",
    "而且", "虽然", "如果", "当然", "因此", "于是", "然后", "接着", "之后", "之前",
    "已经", "正在", "将", "可能", "应该", "一定", "需要", "能够", "知道", "觉得",
    "认为", "让", "把", "被", "给", "对", "从", "以", "至", "与", "及", "或",
    "啊", "吧", "呢", "吗", "呀", "哦", "嗯", "哈", "呵", "哎", "嘛", "呗",
    "的", "得", "地", "之", "其", "某", "各", "每", "任何", "什么", "怎么",
    "怎样", "如何", "为何", "哪里", "哪", "谁", "几", "多", "少", "些", "点",
    "一个", "一种", "一样", "起来", "出来", "过来", "上去", "下去", "进来", "进去",
    "出", "来", "去", "做", "搞", "弄", "干", "用", "拿", "打", "开", "关",
    "请问", "谢谢", "麻烦", "您好", "各位", "大家", "本人", "真的", "感觉",
    "比较", "非常", "特别", "相当", "挺", "蛮", "够", "太", "极", "最",
    "还", "再", "又", "也", "就", "才", "刚", "正", "在", "已", "曾",
])


class Stopwords:
    """停用词管理"""

    def __init__(self, filepath: str = None):
        self._stopwords = set(_DEFAULT_STOPWORDS)
        if filepath and os.path.exists(filepath):
            self.load(filepath)

    def load(self, filepath: str):
        """从文件加载停用词（每行一个）"""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    self._stopwords.add(word)

    def save(self, filepath: str):
        """保存停用词到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            for word in sorted(self._stopwords):
                f.write(word + "\n")

    def is_stopword(self, word: str) -> bool:
        return word in self._stopwords

    def remove(self, tokens: list[str]) -> list[str]:
        """从token列表中移除停用词"""
        return [t for t in tokens if not self.is_stopword(t)]

    def add(self, word: str):
        self._stopwords.add(word)

    def __len__(self):
        return len(self._stopwords)
