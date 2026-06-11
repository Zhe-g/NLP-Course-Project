#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试脚本 - 验证产品评论情感分析系统
"""

import requests
import json
import time
import os

def test_system():
    """测试系统功能"""
    print("=" * 60)
    print("产品评论情感分析系统测试")
    print("=" * 60)

    # 测试1：单条评论分析
    print("\n1. 测试单条评论分析...")
    test_reviews = [
        "产品质量非常好，我很喜欢！",  # 正面
        "产品质量太差了，不值这个价格。",  # 负面
        "产品还可以，没什么特别之处。",  # 中性
    ]

    for review in test_reviews:
        response = requests.post('http://localhost:5000/input-text',
                              json={'text': review})

        if response.status_code == 200:
            result = response.json()
            print(f"评论: {review}")
            print(f"情感: {result['sentiment']} (得分: {result['score']})")
            print("-" * 40)

    # 测试2：文件分析
    print("\n2. 测试文件分析...")

    # 创建测试文件
    test_file = "test_reviews.txt"
    test_reviews = [
        "这个产品很棒，强烈推荐！",
        "质量一般，价格偏贵。",
        "非常满意，会回购的。",
        "服务态度不好。",
        "性价比很高，值得购买。"
    ]

    with open(test_file, 'w', encoding='utf-8') as f:
        for review in test_reviews:
            f.write(review + '\n')

    # 上传文件
    with open(test_file, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/upload', files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"文件分析结果:")
        print(f"总评论数: {result['results']['summary']['total_reviews']}")
        print(f"正面评论: {result['results']['summary']['positive_reviews']} ({result['results']['summary']['positive_ratio']}%)")
        print(f"负面评论: {result['results']['summary']['negative_reviews']} ({result['results']['summary']['negative_ratio']}%)")
        print(f"中性评论: {result['results']['summary']['neutral_reviews']} ({result['results']['summary']['neutral_ratio']}%)")
        print(f"平均情感得分: {result['results']['summary']['average_score']}")
        print("-" * 40)

        # 显示每条评论的分析结果
        print("详细分析结果:")
        for i, review in enumerate(result['results']['results']):
            print(f"{i+1}. {review['text'][:30]}...")
            print(f"   情感: {review['sentiment']} (得分: {review['score']})")
    else:
        print(f"错误: {response.status_code} - {response.text}")

    # 测试3：CSV文件分析
    print("\n3. 测试CSV文件分析...")

    # 创建CSV测试文件
    csv_file = "test_reviews.csv"
    import pandas as pd

    df = pd.DataFrame({'review': test_reviews})
    df.to_csv(csv_file, index=False, encoding='utf-8')

    # 上传CSV文件
    with open(csv_file, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/upload', files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"CSV文件分析结果:")
        print(f"总评论数: {result['results']['summary']['total_reviews']}")
        print(f"情感分布 - 正面: {result['results']['summary']['positive_reviews']}, "
              f"负面: {result['results']['summary']['negative_reviews']}, "
              f"中性: {result['results']['summary']['neutral_reviews']}")
    else:
        print(f"错误: {response.status_code} - {response.text}")

    # 清理测试文件
    for file in [test_file, csv_file]:
        if os.path.exists(file):
            os.remove(file)

    # 测试4：错误处理
    print("\n4. 测试错误处理...")

    # 测试空文本
    response = requests.post('http://localhost:5000/input-text',
                          json={'text': ''})
    if response.status_code == 400:
        print("✓ 空文本检测正常")

    # 测试无文件上传
    response = requests.post('http://localhost:5000/upload', data={})
    if response.status_code == 400:
        print("✓ 无文件上传检测正常")

    print("\n" + "=" * 60)
    print("测试完成！系统运行正常。")
    print("=" * 60)

if __name__ == "__main__":
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(3)

    try:
        test_system()
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到服务器，请确保Flask应用正在运行（http://localhost:5000）")
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")