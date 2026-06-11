# -*- coding: utf-8 -*-
"""
情感极性推理
"""
import os
import torch
from transformers import AutoTokenizer
from src.config import PRETRAINED_MODEL, SENTIMENT_MODEL_PATH, SENTIMENT_CONFIG, DEVICE, CATEGORY_ZH
from src.sentiment.model import SentimentModel

SENTIMENT_NAMES = ["neutral", "positive", "negative"]
SENTIMENT_ZH = {"neutral": "中性", "positive": "正向", "negative": "负向"}


class SentimentPredictor:
    """情感极性推理器"""

    def __init__(self, model_path: str = SENTIMENT_MODEL_PATH):
        self.device = DEVICE

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
        """
        预测文本中特定维度的情感极性

        Args:
            text: 评论文本
            category: 维度名 (如 "dish_taste")

        Returns:
            {"sentiment": "positive", "sentiment_zh": "正向", "confidence": 0.93}
        """
        category_zh = CATEGORY_ZH.get(category, category)

        encoding = self.tokenizer(
            category_zh,
            text,
            max_length=SENTIMENT_CONFIG["max_length"],
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
            result["category_zh"] = CATEGORY_ZH.get(cat, cat)
            results.append(result)
        return results
