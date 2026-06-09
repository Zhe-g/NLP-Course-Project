# -*- coding: utf-8 -*-
"""
一键训练两个模型
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aspect_detection.train import train as train_aspect
from src.sentiment.train import train as train_sentiment


def main():
    print("=" * 60)
    print("  ABSA 模型训练 - 全流程")
    print("  GPU: RTX 4060 Laptop (8GB)")
    print("=" * 60)

    # Step 1: 训练属性检测模型
    print("\n\n")
    print("#" * 60)
    print("# 步骤 1/2: 训练属性检测模型")
    print("#" * 60)
    train_aspect()

    # Step 2: 训练情感分类模型
    print("\n\n")
    print("#" * 60)
    print("# 步骤 2/2: 训练情感分类模型")
    print("#" * 60)
    train_sentiment()

    print("\n\n" + "=" * 60)
    print("  全部训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
