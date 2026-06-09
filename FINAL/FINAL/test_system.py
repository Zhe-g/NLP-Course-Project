#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证产品评论情感分析系统
"""

import requests
import json
import time

def test_text_analysis():
    """测试文本分析功能"""
    print("=" * 50)
    print("测试1：单条评论情感分析")
    print("=" * 50)

    # 正面评论
    positive_text = "这个产品质量非常好，我很喜欢！"
    response = requests.post('http://localhost:5000/input-text',
                          json={'text': positive_text})

    if response.status_code == 200:
        result = response.json()
        print(f"评论: {positive_text}")
        print(f"情感: {result['sentiment']}")
        print(f"得分: {result['score']}")
    else:
        print(f"错误: {response.status_code} - {response.text}")

    # 负面评论
    negative_text = "产品质量太差了，完全不值这个价格。"
    response = requests.post('http://localhost:5000/input-text',
                          json={'text': negative_text})

    if response.status_code == 200:
        result = response.json()
        print(f"\n评论: {negative_text}")
        print(f"情感: {result['sentiment']}")
        print(f"得分: {result['score']}")
    else:
        print(f"错误: {response.status_code} - {response.text}")

def test_file_analysis():
    """测试文件分析功能"""
    print("\n" + "=" * 50)
    print("测试2：文件情感分析")
    print("=" * 50)

    # 准备测试文件
    test_reviews = [
        "产品质量很好，值得购买！",
        "物流速度快，包装精美。",
        "价格有点贵，但是质量不错。",
        "客服态度不好，不太满意。"
    ]

    # 创建临时测试文件
    test_file = "test_reviews.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        for review in test_reviews:
            f.write(review + '\n')

    # 上传文件分析
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
    else:
        print(f"错误: {response.status_code} - {response.text}")

    # 清理测试文件
    os.remove(test_file)

def test_batch_upload():
    """测试批量上传功能"""
    print("\n" + "=" * 50)
    print("测试3：批量文件分析")
    print("=" * 50)

    # 创建多个测试文件
    test_files = []
    reviews_sets = [
        ["产品很好，强烈推荐！", "性价比很高。"],
        ["质量一般，价格偏贵。", "包装有点简陋。"],
        ["非常满意，会回购的！", "服务一流。"]
    ]

    for i, reviews in enumerate(reviews_sets):
        filename = f"batch_test_{i+1}.txt"
        test_files.append(filename)

        with open(filename, 'w', encoding='utf-8') as f:
            for review in reviews:
                f.write(review + '\n')

    # 准备批量上传
    files_list = []
    for filename in test_files:
        with open(filename, 'rb') as f:
            files_list.append(('files', (filename, f, 'text/plain')))

    # 批量上传
    response = requests.post('http://localhost:5000/batch-upload', files=files_list)

    if response.status_code == 200:
        result = response.json()
        print(f"批量分析结果:")
        print(f"处理文件数: {len(result['results'])}")

        # 计算总体统计
        total_reviews = sum(batch['summary']['total_reviews'] for batch in result['results'])
        total_positive = sum(batch['summary']['positive_reviews'] for batch in result['results'])
        total_negative = sum(batch['summary']['negative_reviews'] for batch in result['results'])
        total_neutral = sum(batch['summary']['neutral_reviews'] for batch in result['results'])

        print(f"\n总体统计:")
        print(f"总评论数: {total_reviews}")
        print(f"正面评论: {total_positive} ({total_positive/total_reviews*100:.1f}%)")
        print(f"负面评论: {total_negative} ({total_negative/total_reviews*100:.1f}%)")
        print(f"中性评论: {total_neutral} ({total_neutral/total_reviews*100:.1f}%)")
    else:
        print(f"错误: {response.status_code} - {response.text}")

    # 清理测试文件
    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)

def test_invalid_file():
    """测试无效文件处理"""
    print("\n" + "=" * 50)
    print("测试4：无效文件处理")
    print("=" * 50)

    # 创建无效文件
    invalid_file = "test_invalid.txt"
    with open(invalid_file, 'w', encoding='utf-8') as f:
        f.write("This is not a Chinese text.")

    # 上传无效文件
    with open(invalid_file, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/upload', files=files)

    if response.status_code == 400:
        print("成功拦截无效文件")
    else:
        print(f"响应: {response.status_code} - {response.text}")

    # 清理
    os.remove(invalid_file)

if __name__ == "__main__":
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(2)

    # 导入os模块
    import os

    # 运行测试
    try:
        test_text_analysis()
        test_file_analysis()
        test_batch_upload()
        test_invalid_file()

        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("错误：无法连接到服务器，请确保Flask应用正在运行（http://localhost:5000）")
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")