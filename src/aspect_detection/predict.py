# -*- coding: utf-8 -*-
"""
属性检测推理
"""
import os
import torch

# 延迟导入 transformers，避免模块加载时卡顿
_transformers = None
_AspectDetectionModel = None


def _get_transformers():
    global _transformers
    if _transformers is None:
        from transformers import AutoTokenizer
        _transformers = AutoTokenizer
    return _transformers


def _get_model_class():
    global _AspectDetectionModel
    if _AspectDetectionModel is None:
        from src.aspect_detection.model import AspectDetectionModel
        _AspectDetectionModel = AspectDetectionModel
    return _AspectDetectionModel


class AspectPredictor:
    """属性检测推理器"""

    def __init__(self, model_path: str = None):
        from src.config import PRETRAINED_MODEL, ASPECT_MODEL_PATH, DEVICE, ASPECT_CONFIG, ASAP_CATEGORIES
        self.device = DEVICE
        self.ASAP_CATEGORIES = ASAP_CATEGORIES
        self.ASPECT_CONFIG = ASPECT_CONFIG
        
        if model_path is None:
            model_path = ASPECT_MODEL_PATH

        AutoTokenizer = _get_transformers()
        AspectDetectionModel = _get_model_class()

        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AspectDetectionModel(pretrained_model=model_path)
            classifier_path = os.path.join(model_path, "classifier.pt")
            if os.path.exists(classifier_path):
                self.model.classifier.load_state_dict(
                    torch.load(classifier_path, map_location=self.device)
                )
        else:
            print(f"警告: 未找到训练好的模型 {model_path}，使用预训练模型（未微调）")
            self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            self.model = AspectDetectionModel(pretrained_model=PRETRAINED_MODEL)

        self.model.to(self.device)
        self.model.eval()
        self.threshold = 0.5

    @torch.no_grad()
    def predict(self, text: str, threshold: float = None) -> dict:
        if threshold is None:
            threshold = self.threshold

        encoding = self.tokenizer(
            text,
            max_length=self.ASPECT_CONFIG["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        _, probs = self.model.predict(input_ids, attention_mask, threshold)
        probs = probs[0].cpu().tolist()

        from src.taxonomy.categories import get_category_zh

        categories = []
        for i, (cat_name, prob) in enumerate(zip(self.ASAP_CATEGORIES, probs)):
            if prob >= threshold:
                categories.append({
                    "name": cat_name,
                    "name_zh": get_category_zh(cat_name),
                    "prob": round(prob, 4),
                })

        return {"categories": categories}

    @torch.no_grad()
    def predict_batch(self, texts: list[str], threshold: float = None) -> list[dict]:
        """批量预测"""
        results = []
        for text in texts:
            results.append(self.predict(text, threshold))
        return results
