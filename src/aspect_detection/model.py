# -*- coding: utf-8 -*-
"""
属性检测模型 — RoBERTa-wwm-ext + 多标签分类头
"""
import torch
import torch.nn as nn

# 延迟导入
_AutoModel = None
_AutoConfig = None


def _get_auto_model():
    global _AutoModel
    if _AutoModel is None:
        from transformers import AutoModel
        _AutoModel = AutoModel
    return _AutoModel


def _get_auto_config():
    global _AutoConfig
    if _AutoConfig is None:
        from transformers import AutoConfig
        _AutoConfig = AutoConfig
    return _AutoConfig


class AspectDetectionModel(nn.Module):
    """
    RoBERTa-wwm-ext Encoder → [CLS] → Dropout → Linear(768,256) → GELU → Linear(256,18) → Sigmoid
    """

    def __init__(self, pretrained_model: str = None, num_labels: int = None):
        super().__init__()
        
        from src.config import PRETRAINED_MODEL, ASPECT_CONFIG
        if pretrained_model is None:
            pretrained_model = PRETRAINED_MODEL
        if num_labels is None:
            num_labels = ASPECT_CONFIG["num_labels"]

        AutoConfig = _get_auto_config()
        AutoModel = _get_auto_model()
        
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.encoder = AutoModel.from_pretrained(pretrained_model)

        hidden_size = self.config.hidden_size  # 768 for RoBERTa-base
        mid_dim = ASPECT_CONFIG["hidden_dim"]
        dropout = ASPECT_CONFIG["dropout"]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_labels),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] token hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_embedding)  # (batch, 18)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def predict(self, input_ids, attention_mask, threshold: float = 0.5):
        """
        推理：返回哪些属性被提及

        Returns:
            list[list[int]]: 每样本的18维0/1预测
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"])
            predictions = (probs > threshold).int()
        return predictions, probs
