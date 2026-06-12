# -*- coding: utf-8 -*-
"""
模型评估脚本 — Precision / Recall / F1 on test set
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    accuracy_score
)
import torch
from transformers import AutoTokenizer
from src.config import (
    PRETRAINED_MODEL, ASAP_CATEGORIES, CATEGORY_ZH,
    SENTIMENT_MODEL_PATH, ASPECT_MODEL_PATH,
    TEST_CSV, TEST_CLEAN_CSV, SENTIMENT_CONFIG,
)
from src.aspect_detection.model import AspectDetectionModel
from src.sentiment.model import SentimentModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTIMENT_NAMES = ["neutral", "positive", "negative"]
SENTIMENT_ZH = {"neutral": "中性", "positive": "正向", "negative": "负向"}


def evaluate_aspect_detection():
    """评估属性检测模型 — 18维多标签分类"""
    print("=" * 60)
    print("  属性检测模型评估 (Multi-Label)")
    print("=" * 60)

    # 加载数据
    import pandas as pd
    test_path = TEST_CLEAN_CSV if os.path.exists(TEST_CLEAN_CSV) else TEST_CSV
    df = pd.read_csv(test_path)
    texts = df["review"].fillna("").astype(str).tolist()

    # Ground truth labels
    y_true = []
    for _, row in df.iterrows():
        vec = [0 if int(row.get(c, -2)) == -2 else 1 for c in ASAP_CATEGORIES]
        y_true.append(vec)
    y_true = np.array(y_true)

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(ASPECT_MODEL_PATH)
    model = AspectDetectionModel(pretrained_model=ASPECT_MODEL_PATH, num_labels=18)
    classifier_path = os.path.join(ASPECT_MODEL_PATH, "classifier.pt")
    if os.path.exists(classifier_path):
        model.classifier.load_state_dict(
            torch.load(classifier_path, map_location=DEVICE, weights_only=True)
        )
    model.to(DEVICE)
    model.eval()

    # 推理
    y_pred_prob = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, max_length=256, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE),
                          enc["attention_mask"].to(DEVICE))["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
        y_pred_prob.append(probs)

    y_pred_prob = np.concatenate(y_pred_prob, axis=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Per-category metrics
    print(f"\n{'维度':<16} {'Precision':>9} {'Recall':>9} {'F1':>9} {'支持数':>7}")
    print("-" * 55)
    for i, cat in enumerate(ASAP_CATEGORIES):
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        s = y_true[:, i].sum()
        print(f"{CATEGORY_ZH.get(cat, cat):<16} {p:>9.4f} {r:>9.4f} {f:>9.4f} {int(s):>7}")

    # Overall
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    print("-" * 55)
    print(f"{'微平均(Micro)':<16} {micro_p:>9.4f} {micro_r:>9.4f} {micro_f1:>9.4f}")
    print(f"{'宏平均(Macro)':<16} {'-':>9} {'-':>9} {macro_f1:>9.4f}")
    print(f"\n  测试样本数: {len(texts)}")


def evaluate_sentiment():
    """评估情感分类模型 — 3分类"""
    print("\n" + "=" * 60)
    print("  情感分类模型评估 (3-Class)")
    print("=" * 60)

    import pandas as pd
    test_path = TEST_CLEAN_CSV if os.path.exists(TEST_CLEAN_CSV) else TEST_CSV
    df = pd.read_csv(test_path)

    # Build test samples
    samples = []
    for _, row in df.iterrows():
        text = str(row["review"]).strip()
        for cat in ASAP_CATEGORIES:
            val = int(row.get(cat, -2))
            if val != -2:
                label = 1 if val == 1 else (0 if val == 0 else 2)
                samples.append((text, cat, label))

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
    model = SentimentModel(pretrained_model=SENTIMENT_MODEL_PATH, num_labels=3)
    classifier_path = os.path.join(SENTIMENT_MODEL_PATH, "classifier.pt")
    if os.path.exists(classifier_path):
        model.classifier.load_state_dict(
            torch.load(classifier_path, map_location=DEVICE, weights_only=True)
        )
    model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []
    batch_size = 64
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        texts_zh = [(CATEGORY_ZH.get(cat, cat), text) for text, cat, _ in batch]
        enc = tokenizer([t[0] for t in texts_zh], [t[1] for t in texts_zh],
                        max_length=256, padding="max_length", truncation=True,
                        return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE),
                          enc["attention_mask"].to(DEVICE))["logits"]
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        y_true.extend([l for _, _, l in batch])
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n{'类别':<10} {'Precision':>9} {'Recall':>9} {'F1':>9} {'支持数':>7}")
    print("-" * 50)
    for i, name in enumerate(SENTIMENT_NAMES):
        mask = y_true == i
        p = precision_score(y_true == i, y_pred == i, zero_division=0)
        r = recall_score(y_true == i, y_pred == i, zero_division=0)
        f = f1_score(y_true == i, y_pred == i, zero_division=0)
        s = mask.sum()
        print(f"{SENTIMENT_ZH[name]:<10} {p:>9.4f} {r:>9.4f} {f:>9.4f} {int(s):>7}")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print("-" * 50)
    print(f"{'准确率':<10} {acc:>9.4f}")
    print(f"{'Macro F1':<10} {macro_f1:>9.4f}")
    print(f"{'Weighted F1':<10} {weighted_f1:>9.4f}")
    print(f"\n  测试样本数: {len(samples)}")


if __name__ == "__main__":
    evaluate_aspect_detection()
    evaluate_sentiment()
