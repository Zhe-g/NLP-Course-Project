# -*- coding: utf-8 -*-
"""
属性检测数据集 — 多标签分类格式
将ASAP CSV转换为：text → 18-dim binary label vector
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.config import ASAP_CATEGORIES, PRETRAINED_MODEL, MAX_LENGTH
from src.preprocessing.cleaner import clean_text


class AspectDetectionDataset(Dataset):
    """
    属性检测数据集
    X: 评论文本
    y: 18维多标签binary vector（1=已提及, 0=未提及）
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: AutoTokenizer = None,
        max_length: int = MAX_LENGTH,
    ):
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        self.max_length = max_length

        # 加载CSV
        df = pd.read_csv(csv_path, header=0)
        self.texts = [clean_text(t) for t in df["review"].fillna("").astype(str).tolist()]
        self.stars = df["star"].fillna(3).astype(int).tolist()

        # 构建多标签矩阵: shape = (n_samples, 18)
        self.labels = []
        for _, row in df.iterrows():
            label_vec = []
            for cat in ASAP_CATEGORIES:
                val = int(row.get(cat, -2))
                # -2 → 0 (未提及), 其他(-1/0/1) → 1 (已提及)
                label_vec.append(0 if val == -2 else 1)
            self.labels.append(label_vec)
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }

    @property
    def num_labels(self) -> int:
        return len(ASAP_CATEGORIES)


def load_asap_aspect_data(tokenizer=None):
    """加载ASAP的训练/验证/测试数据集（优先使用清洗后数据）"""
    import os
    from src.config import (
        TRAIN_CSV, DEV_CSV, TEST_CSV,
        TRAIN_CLEAN_CSV, DEV_CLEAN_CSV, TEST_CLEAN_CSV,
    )

    train_path = TRAIN_CLEAN_CSV if os.path.exists(TRAIN_CLEAN_CSV) else TRAIN_CSV
    dev_path = DEV_CLEAN_CSV if os.path.exists(DEV_CLEAN_CSV) else DEV_CSV
    test_path = TEST_CLEAN_CSV if os.path.exists(TEST_CLEAN_CSV) else TEST_CSV

    print(f"  训练数据: {os.path.basename(train_path)}")
    print(f"  验证数据: {os.path.basename(dev_path)}")

    train_dataset = AspectDetectionDataset(train_path, tokenizer)
    dev_dataset = AspectDetectionDataset(dev_path, train_dataset.tokenizer)
    test_dataset = AspectDetectionDataset(test_path, train_dataset.tokenizer)

    return train_dataset, dev_dataset, test_dataset
