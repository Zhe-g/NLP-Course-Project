# -*- coding: utf-8 -*-
"""
全局配置文件
"""
import os
import torch

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASAP_DIR = os.path.join(DATA_DIR, "asap")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ASAP 数据文件路径（原始）
TRAIN_CSV = os.path.join(ASAP_DIR, "train.csv")
DEV_CSV = os.path.join(ASAP_DIR, "dev.csv")
TEST_CSV = os.path.join(ASAP_DIR, "test.csv")

# 清洗后的数据路径（预处理脚本输出）
TRAIN_CLEAN_CSV = os.path.join(PROCESSED_DIR, "train_clean.csv")
DEV_CLEAN_CSV = os.path.join(PROCESSED_DIR, "dev_clean.csv")
TEST_CLEAN_CSV = os.path.join(PROCESSED_DIR, "test_clean.csv")

# 模型保存路径
ASPECT_MODEL_PATH = os.path.join(MODEL_DIR, "aspect_detection")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment")

# --- 设备配置 ---
def get_device():
    """获取设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设备常量（导入时初始化）
DEVICE = get_device()

# --- 预训练模型配置 ---
PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
MAX_LENGTH = 256  # 最大序列长度（餐饮评论通常较短）

# --- 属性检测模型超参 ---
ASPECT_CONFIG = {
    "num_labels": 18,  # 18个ASAP维度
    "batch_size": 8,
    "gradient_accumulation_steps": 4,  # 有效batch = 32
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "max_length": MAX_LENGTH,
    "fp16": True,
    "gradient_checkpointing": True,
    "eval_steps": 200,
    "save_steps": 400,
    "dropout": 0.1,
    "hidden_dim": 256,  # 中间层维度
}

# --- 情感分类模型超参 ---
SENTIMENT_CONFIG = {
    "num_labels": 3,  # positive, neutral, negative
    "batch_size": 16,
    "gradient_accumulation_steps": 2,  # 有效batch = 32
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "max_length": MAX_LENGTH,
    "fp16": True,
    "gradient_checkpointing": True,
    "eval_steps": 300,
    "save_steps": 600,
    "dropout": 0.1,
}

# --- 18个ASAP评价维度 (匹配CSV列名) ---
ASAP_CATEGORIES = [
    "Location#Transportation",
    "Location#Downtown",
    "Location#Easy_to_find",
    "Service#Queue",
    "Service#Hospitality",
    "Service#Parking",
    "Service#Timely",
    "Price#Level",
    "Price#Cost_effective",
    "Price#Discount",
    "Ambience#Decoration",
    "Ambience#Noise",
    "Ambience#Space",
    "Ambience#Sanitary",
    "Food#Portion",
    "Food#Taste",
    "Food#Appearance",
    "Food#Recommend",
]

# 维度中文名映射
CATEGORY_ZH = {
    "Location#Transportation": "交通便利",
    "Location#Downtown": "是否在商圈",
    "Location#Easy_to_find": "是否好找",
    "Service#Queue": "排队时间",
    "Service#Hospitality": "服务态度",
    "Service#Parking": "停车便利",
    "Service#Timely": "上菜速度",
    "Price#Level": "价格水平",
    "Price#Cost_effective": "性价比",
    "Price#Discount": "优惠活动",
    "Ambience#Decoration": "装修风格",
    "Ambience#Noise": "噪音水平",
    "Ambience#Space": "空间大小",
    "Ambience#Sanitary": "卫生状况",
    "Food#Portion": "菜品分量",
    "Food#Taste": "菜品口味",
    "Food#Appearance": "菜品外观",
    "Food#Recommend": "是否推荐",
}

# 5个一级维度分组
CATEGORY_GROUPS = {
    "位置": [
        "Location#Transportation",
        "Location#Downtown",
        "Location#Easy_to_find",
    ],
    "服务": [
        "Service#Queue",
        "Service#Hospitality",
        "Service#Parking",
        "Service#Timely",
    ],
    "价格": [
        "Price#Level",
        "Price#Cost_effective",
        "Price#Discount",
    ],
    "环境": [
        "Ambience#Decoration",
        "Ambience#Noise",
        "Ambience#Space",
        "Ambience#Sanitary",
    ],
    "菜品": [
        "Food#Portion",
        "Food#Taste",
        "Food#Appearance",
        "Food#Recommend",
    ],
}

# 情感标签映射
SENTIMENT_MAP = {1: "positive", 0: "neutral", -1: "negative"}
SENTIMENT_MAP_REVERSE = {"positive": 1, "neutral": 0, "negative": -1}

# 情感标签中文
SENTIMENT_ZH = {"positive": "正向", "neutral": "中性", "negative": "负向"}
