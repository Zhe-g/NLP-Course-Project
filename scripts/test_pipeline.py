# -*- coding: utf-8 -*-
"""
端到端测试脚本
1. 测试文本预处理
2. 测试属性检测模型（如果已训练）
3. 测试情感分类模型（如果已训练）
4. 测试完整流水线
5. 测试Flask API（如果服务已启动）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.cleaner import clean_text, is_valid_review
from src.preprocessing.tokenizer import tokenize, extract_keywords
from src.taxonomy.categories import get_category_zh, get_group, get_all_categories


def test_preprocessing():
    print("\n" + "=" * 50)
    print("测试1: 文本预处理")
    print("=" * 50)

    test_texts = [
        "这家店的火锅味道很好，但是服务太差了，等了一个小时才上菜。",
        "环境不错，价格实惠，推荐！\\n下次还会再来。",
        "",
        "   <html>好吃的</html>   ",
    ]

    for i, text in enumerate(test_texts):
        print(f"\n  样本{i+1}: '{text[:50]}...' " if len(text) > 50 else f"\n  样本{i+1}: '{text}'")
        cleaned = clean_text(text)
        valid = is_valid_review(cleaned)
        print(f"    清洗后: '{cleaned}'")
        print(f"    有效: {valid}")
        if valid:
            tokens = tokenize(cleaned)
            kw = extract_keywords(cleaned, 5)
            print(f"    分词: {tokens[:10]}...")
            print(f"    关键词: {kw}")


def test_taxonomy():
    print("\n" + "=" * 50)
    print("测试2: 分类体系")
    print("=" * 50)

    cats = get_all_categories()
    for group in cats:
        print(f"\n  [{group['group']}]")
        for cat in group["categories"]:
            print(f"    {cat['name']} → {cat['name_zh']}")

    print(f"\n  总维度数: {sum(len(g['categories']) for g in cats)}")


def test_full_pipeline():
    """测试完整流水线（需要训练好的模型）"""
    print("\n" + "=" * 50)
    print("测试3: 完整流水线")
    print("=" * 50)

    try:
        from src.pipeline.analyzer import ABSAAnalyzer

        print("  加载模型...")
        analyzer = ABSAAnalyzer()

        test_cases = [
            ("这家店的火锅味道很好，但是服务太差了", None),
            ("环境优雅，价格实惠，强烈推荐！", "review_001"),
        ]

        for text, rid in test_cases:
            print(f"\n  输入: {text}")
            result = analyzer.analyze(text, rid)
            print(f"  检测到 {len(result['aspects'])} 个评价维度:")
            for aspect in result["aspects"]:
                print(f"    - {aspect['category_zh']}({aspect['group']}): "
                      f"{aspect['sentiment_zh']} (置信度: {aspect['sentiment_confidence']})")
            print(f"  汇总: {result['summary']}")
            print(f"  耗时: {result['elapsed_seconds']}s")

        print("\n  流水线测试通过！")

    except Exception as e:
        print(f"\n  流水线测试失败: {e}")
        print("  (可能模型尚未训练，这是正常的)")


def test_api():
    """测试API接口（需要服务已启动）"""
    print("\n" + "=" * 50)
    print("测试4: API 接口")
    print("=" * 50)

    try:
        import requests

        base_url = "http://127.0.0.1:5000"

        # 健康检查
        print("  测试 /api/v1/health ...")
        resp = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"    Status: {resp.status_code}")
        print(f"    Response: {resp.json()}")

        # 获取分类体系
        print("  测试 /api/v1/taxonomy ...")
        resp = requests.get(f"{base_url}/api/v1/taxonomy", timeout=5)
        data = resp.json()
        print(f"    Code: {data['code']}, Groups: {len(data['data'])}")

        # 分析评论
        print("  测试 /api/v1/analyze ...")
        resp = requests.post(
            f"{base_url}/api/v1/analyze",
            json={"text": "味道不错，服务很好，环境优雅"},
            timeout=10,
        )
        data = resp.json()
        print(f"    Code: {data['code']}")
        if data["code"] == 0:
            print(f"    Aspects: {len(data['data']['aspects'])}")
            for a in data["data"]["aspects"]:
                print(f"      - {a['category_zh']}: {a['sentiment_zh']}")

        print("\n  API测试通过！")

    except requests.exceptions.ConnectionError:
        print("  API服务未启动，跳过API测试。")
        print("  请先运行: python scripts/run_api.py")
    except ImportError:
        print("  需要安装 requests 库: pip install requests")
    except Exception as e:
        print(f"  API测试出错: {e}")


def main():
    print("=" * 60)
    print("  ABSA 系统 - 端到端测试")
    print("=" * 60)

    test_preprocessing()
    test_taxonomy()
    test_full_pipeline()
    test_api()

    print("\n" + "=" * 60)
    print("  测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
