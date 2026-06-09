# -*- coding: utf-8 -*-
"""
情感极性分类数据集
将ASAP CSV转换为：(text, category, polarity) 三元组
只保留已提及的维度（标签 != -2）
"""
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.config import ASAP_CATEGORIES, PRETRAINED_MODEL, MAX_LENGTH, CATEGORY_ZH
from src.preprocessing.cleaner import clean_text


class SentimentDataset(Dataset):
    """
    情感分类数据集
    输入: [CLS] [维度中文名] [SEP] 评论文本 [SEP]
    输出: sentiment label (0=neutral, 1=positive, 2=negative)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: AutoTokenizer = None,
        max_length: int = MAX_LENGTH,
    ):
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        self.max_length = max_length

        # 展开数据集：每条评论×18维度 → 只保留已提及的
        df = pd.read_csv(csv_path, header=0)

        self.samples = []  # list of (text, category_name, polarity)
        for _, row in df.iterrows():
            text = clean_text(str(row["review"]))
            if not text or text == "nan":
                continue
            for cat in ASAP_CATEGORIES:
                val = int(row.get(cat, -2))
                if val != -2:  # 已提及该维度
                    # 映射: 1→1(positive), 0→0(neutral), -1→2(negative)
                    label = 1 if val == 1 else (0 if val == 0 else 2)
                    self.samples.append((text, cat, label))

        print(f"  从 {csv_path} 加载 {len(self.samples)} 个情感样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, category, label = self.samples[idx]
        category_zh = CATEGORY_ZH.get(category, category)

        # 构建输入: [CLS] 维度名 [SEP] 评论文本 [SEP]
        encoding = self.tokenizer(
            category_zh,
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", encoding["attention_mask"]).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_asap_sentiment_data(tokenizer=None):
    """加载ASAP的情感分类数据集（优先使用清洗后数据）"""
    import os
    from src.config import (
        TRAIN_CSV, DEV_CSV, TEST_CSV,
        TRAIN_CLEAN_CSV, DEV_CLEAN_CSV, TEST_CLEAN_CSV,
    )

    train_path = TRAIN_CLEAN_CSV if os.path.exists(TRAIN_CLEAN_CSV) else TRAIN_CSV
    dev_path = DEV_CLEAN_CSV if os.path.exists(DEV_CLEAN_CSV) else DEV_CSV
    test_path = TEST_CLEAN_CSV if os.path.exists(TEST_CLEAN_CSV) else TEST_CSV

    train_dataset = SentimentDataset(train_path, tokenizer)
    dev_dataset = SentimentDataset(dev_path, train_dataset.tokenizer)
    test_dataset = SentimentDataset(test_path, train_dataset.tokenizer)

    return train_dataset, dev_dataset, test_dataset
