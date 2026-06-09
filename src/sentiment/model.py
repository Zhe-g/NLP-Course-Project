# -*- coding: utf-8 -*-
"""
情感极性分类模型 — RoBERTa-wwm-ext + 3分类头
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from src.config import PRETRAINED_MODEL, SENTIMENT_CONFIG


class SentimentModel(nn.Module):
    """
    RoBERTa-wwm-ext Encoder → [CLS] → Dropout → Linear(768, 3) → Softmax
    """

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL, num_labels: int = None):
        super().__init__()
        if num_labels is None:
            num_labels = SENTIMENT_CONFIG["num_labels"]

        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.encoder = AutoModel.from_pretrained(pretrained_model)

        hidden_size = self.config.hidden_size  # 768
        dropout = SENTIMENT_CONFIG["dropout"]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] token hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_embedding)  # (batch, 3)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        """
        推理：返回情感分类结果

        Returns:
            (predicted_label, probabilities)
        """
        self.eval()
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs
