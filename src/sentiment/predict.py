# -*- coding: utf-8 -*-
"""
情感极性推理
"""
import os
import torch

# 延迟导入
_transformers = None
_SentimentModel = None


def _get_transformers():
    global _transformers
    if _transformers is None:
        from transformers import AutoTokenizer
        _transformers = AutoTokenizer
    return _transformers


def _get_model_class():
    global _SentimentModel
    if _SentimentModel is None:
        from src.sentiment.model import SentimentModel
        _SentimentModel = SentimentModel
    return _SentimentModel


SENTIMENT_NAMES = ["neutral", "positive", "negative"]
SENTIMENT_ZH = {"neutral": "中性", "positive": "正向", "negative": "负向"}


class SentimentPredictor:
    """情感极性推理器"""

    def __init__(self, model_path: str = None):
        from src.config import PRETRAINED_MODEL, SENTIMENT_MODEL_PATH, DEVICE, SENTIMENT_CONFIG, CATEGORY_ZH
        self.device = DEVICE
        self.SENTIMENT_CONFIG = SENTIMENT_CONFIG
        self.CATEGORY_ZH = CATEGORY_ZH
        
        if model_path is None:
            model_path = SENTIMENT_MODEL_PATH

        AutoTokenizer = _get_transformers()
        SentimentModel = _get_model_class()

        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = SentimentModel(pretrained_model=model_path)
            classifier_path = os.path.join(model_path, "classifier.pt")
            if os.path.exists(classifier_path):
                self.model.classifier.load_state_dict(
                    torch.load(classifier_path, map_location=self.device)
                )
        else:
            print(f"警告: 未找到训练好的模型 {model_path}，使用预训练模型（未微调）")
            self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            self.model = SentimentModel(pretrained_model=PRETRAINED_MODEL)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str, category: str) -> dict:
        category_zh = self.CATEGORY_ZH.get(category, category)

        encoding = self.tokenizer(
            category_zh,
            text,
            max_length=self.SENTIMENT_CONFIG["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        preds, probs = self.model.predict(input_ids, attention_mask)
        pred_label = preds[0].item()
        confidence = probs[0][pred_label].item()

        sentiment = SENTIMENT_NAMES[pred_label]

        return {
            "sentiment": sentiment,
            "sentiment_zh": SENTIMENT_ZH.get(sentiment, sentiment),
            "confidence": round(confidence, 4),
        }

    @torch.no_grad()
    def predict_for_categories(self, text: str, categories: list[str]) -> list[dict]:
        """对一条文本的多个维度进行情感分析"""
        results = []
        for cat in categories:
            result = self.predict(text, cat)
            result["category"] = cat
            result["category_zh"] = self.CATEGORY_ZH.get(cat, cat)
            results.append(result)
        return results
