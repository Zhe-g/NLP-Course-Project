# -*- coding: utf-8 -*-
"""
数据预处理脚本：清洗ASAP原始CSV，输出清洗后的CSV
用法: python scripts/preprocess_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.preprocessing.cleaner import clean_text, is_valid_review
from src.config import ASAP_CATEGORIES, TRAIN_CSV, DEV_CSV, TEST_CSV


def preprocess_csv(input_path: str, output_path: str):
    """清洗单个CSV文件"""
    print(f"\n处理: {os.path.basename(input_path)}")
    df = pd.read_csv(input_path, header=0)

    original_count = len(df)

    # 清洗文本
    df["review"] = df["review"].fillna("").astype(str).apply(clean_text)

    # 过滤无效评论
    valid_mask = df["review"].apply(is_valid_review)
    df = df[valid_mask].copy()

    removed = original_count - len(df)
    print(f"  原始: {original_count} 条")
    print(f"  清洗后: {len(df)} 条")
    print(f"  移除无效: {removed} 条")

    # 验证标签列完整性
    for cat in ASAP_CATEGORIES:
        if cat not in df.columns:
            print(f"  警告: 缺少列 {cat}")
        else:
            # 确保标签为整数
            df[cat] = df[cat].astype(int)

    # 保存
    df.to_csv(output_path, index=False)
    print(f"  已保存: {os.path.basename(output_path)}")


def main():
    print("=" * 50)
    print("  ASAP 数据预处理")
    print("=" * 50)

    # 检查原始文件
    for path in [TRAIN_CSV, DEV_CSV, TEST_CSV]:
        if not os.path.exists(path):
            print(f"错误: 找不到 {path}")
            sys.exit(1)

    # 输出到 processed 目录
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "processed")

    preprocess_csv(TRAIN_CSV, os.path.join(out_dir, "train_clean.csv"))
    preprocess_csv(DEV_CSV, os.path.join(out_dir, "dev_clean.csv"))
    preprocess_csv(TEST_CSV, os.path.join(out_dir, "test_clean.csv"))

    print("\n" + "=" * 50)
    print("  预处理完成！")
    print(f"  输出目录: {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
