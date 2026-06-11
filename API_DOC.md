# ABSA 情感分析系统 — 接口文档

## 概述

本系统提供电商评论细粒度情感分析（ABSA）能力，基于 RoBERTa-wwm-ext 预训练模型 + ASAP 数据集训练。输入餐饮评论文本，自动识别涉及的18个评价维度及其情感极性（正向/中性/负向），并支持多维度统计分析。

- 基础地址: `http://{host}:{port}/api/v1`
- 默认端口: `5000`
- Content-Type: `application/json`

---

## 1. 健康检查

检查服务是否正常运行。

**GET** `/api/v1/health`

#### 响应示例
```json
{
  "code": 0,
  "message": "ABSA Service is running",
  "status": "healthy"
}
```

---

## 2. 获取分类体系

获取完整的18个评价维度及5个一级维度分组。

**GET** `/api/v1/taxonomy`

#### 响应示例
```json
{
  "code": 0,
  "data": [
    {
      "group": "位置",
      "categories": [
        {"name": "Location#Transportation", "name_zh": "交通便利"},
        {"name": "Location#Downtown", "name_zh": "是否在商圈"},
        {"name": "Location#Easy_to_find", "name_zh": "是否好找"}
      ]
    },
    {
      "group": "服务",
      "categories": [
        {"name": "Service#Queue", "name_zh": "排队时间"},
        {"name": "Service#Hospitality", "name_zh": "服务态度"},
        {"name": "Service#Parking", "name_zh": "停车便利"},
        {"name": "Service#Timely", "name_zh": "上菜速度"}
      ]
    },
    {
      "group": "价格",
      "categories": [
        {"name": "Price#Level", "name_zh": "价格水平"},
        {"name": "Price#Cost_effective", "name_zh": "性价比"},
        {"name": "Price#Discount", "name_zh": "优惠活动"}
      ]
    },
    {
      "group": "环境",
      "categories": [
        {"name": "Ambience#Decoration", "name_zh": "装修风格"},
        {"name": "Ambience#Noise", "name_zh": "噪音水平"},
        {"name": "Ambience#Space", "name_zh": "空间大小"},
        {"name": "Ambience#Sanitary", "name_zh": "卫生状况"}
      ]
    },
    {
      "group": "菜品",
      "categories": [
        {"name": "Food#Portion", "name_zh": "菜品分量"},
        {"name": "Food#Taste", "name_zh": "菜品口味"},
        {"name": "Food#Appearance", "name_zh": "菜品外观"},
        {"name": "Food#Recommend", "name_zh": "是否推荐"}
      ]
    }
  ]
}
```

---

## 3. 单条评论分析

输入一条评论文本，返回所有被提及的评价维度及其情感极性。

**POST** `/api/v1/analyze`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | 是 | 评论文本（建议长度 10-500 字） |
| review_id | string | 否 | 评论 ID，用于追踪 |

#### 请求示例
```json
{
  "text": "这家店的火锅味道很好，但是服务太差了，等了一个小时才上菜，环境还不错。",
  "review_id": "rev_001"
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 0=成功, -1=参数错误, -2=服务器错误 |
| data | object | 分析结果 |
| data.text | string | 清洗后的评论文本 |
| data.review_id | string | 评论 ID（与请求一致） |
| data.aspects | array | 检测到的评价维度列表 |
| data.aspects[].category | string | 维度英文名（如 `Food#Taste`） |
| data.aspects[].category_zh | string | 维度中文名（如 `菜品口味`） |
| data.aspects[].group | string | 所属一级维度（位置/服务/价格/环境/菜品） |
| data.aspects[].sentiment | string | 情感极性: `positive` / `neutral` / `negative` |
| data.aspects[].sentiment_zh | string | 中文极性: `正向` / `中性` / `负向` |
| data.aspects[].aspect_confidence | float | 属性检测置信度 (0-1) |
| data.aspects[].sentiment_confidence | float | 情感分类置信度 (0-1) |
| data.summary | object | 汇总统计 |
| data.summary.positive_count | int | 正向维度数量 |
| data.summary.negative_count | int | 负向维度数量 |
| data.summary.neutral_count | int | 中性维度数量 |
| data.summary.overall_sentiment | string | 整体情感倾向 |
| data.summary.total_aspects | int | 总维度数量 |
| data.elapsed_seconds | float | 分析耗时（秒） |

#### 响应示例
```json
{
  "code": 0,
  "data": {
    "review_id": "rev_001",
    "text": "这家店的火锅味道很好,但是服务太差了,等了一个小时才上菜,环境还不错。",
    "aspects": [
      {
        "category": "Service#Hospitality",
        "category_zh": "服务态度",
        "group": "服务",
        "sentiment": "negative",
        "sentiment_zh": "负向",
        "aspect_confidence": 0.98,
        "sentiment_confidence": 0.95
      },
      {
        "category": "Service#Timely",
        "category_zh": "上菜速度",
        "group": "服务",
        "sentiment": "negative",
        "sentiment_zh": "负向",
        "aspect_confidence": 0.97,
        "sentiment_confidence": 0.97
      },
      {
        "category": "Food#Taste",
        "category_zh": "菜品口味",
        "group": "菜品",
        "sentiment": "positive",
        "sentiment_zh": "正向",
        "aspect_confidence": 0.91,
        "sentiment_confidence": 0.97
      },
      {
        "category": "Ambience#Decoration",
        "category_zh": "装修风格",
        "group": "环境",
        "sentiment": "positive",
        "sentiment_zh": "正向",
        "aspect_confidence": 0.85,
        "sentiment_confidence": 0.99
      }
    ],
    "summary": {
      "positive_count": 2,
      "neutral_count": 0,
      "negative_count": 2,
      "overall_sentiment": "neutral",
      "total_aspects": 4
    },
    "elapsed_seconds": 0.25
  }
}
```

#### 调用示例

**cURL:**
```bash
curl -X POST http://127.0.0.1:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"味道不错，服务很好","review_id":"rev_001"}'
```

**Python:**
```python
import requests

resp = requests.post(
    "http://127.0.0.1:5000/api/v1/analyze",
    json={"text": "味道不错，服务很好，环境优雅"}
)
print(resp.json())
```

**JavaScript (fetch):**
```javascript
fetch("http://127.0.0.1:5000/api/v1/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "味道不错，服务很好" })
})
.then(res => res.json())
.then(data => console.log(data));
```

**Java (OkHttp):**
```java
OkHttpClient client = new OkHttpClient();
String json = "{\"text\":\"味道不错，服务很好\"}";
RequestBody body = RequestBody.create(json, MediaType.parse("application/json"));
Request request = new Request.Builder()
    .url("http://127.0.0.1:5000/api/v1/analyze")
    .post(body)
    .build();
Response response = client.newCall(request).execute();
System.out.println(response.body().string());
```

---

## 4. 批量评论分析

一次分析多条评论。

**POST** `/api/v1/analyze/batch`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| texts | array | 是 | 评论对象数组 |
| texts[].text | string | 是 | 评论文本 |
| texts[].review_id | string | 否 | 评论 ID |

#### 请求示例
```json
{
  "texts": [
    {"text": "味道不错，服务很好，环境优雅", "review_id": "r1"},
    {"text": "太难吃了，上菜慢得要命，再也不会去了", "review_id": "r2"},
    {"text": "价格实惠，性价比高，就是停车不太方便"}
  ]
}
```

#### 响应示例
```json
{
  "code": 0,
  "data": [
    {
      "review_id": "r1",
      "text": "味道不错,服务很好,环境优雅",
      "aspects": [...],
      "summary": {...},
      "elapsed_seconds": 0.23
    },
    {
      "review_id": "r2",
      "text": "太难吃了,上菜慢得要命,再也不会去了",
      "aspects": [...],
      "summary": {...},
      "elapsed_seconds": 0.21
    },
    {
      "review_id": null,
      "text": "价格实惠,性价比高,就是停车不太方便",
      "aspects": [...],
      "summary": {...},
      "elapsed_seconds": 0.22
    }
  ]
}
```

> **注意**: 批量请求时每条评论顺序处理，耗时约为 `数量 × 0.25s`。大批量建议分批调用，每批 50-100 条。

---

## 5. 统计分析

获取自服务启动以来所有分析的累积统计报告。

**GET** `/api/v1/stats`

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| total_reviews | int | 累计分析评论数 |
| total_aspects | int | 累计检测到的评价维度总数 |
| avg_aspects_per_review | float | 平均每条评论的维度数 |
| overall_distribution | object | 整体评论情感偏向分布 |
| sentiment_distribution | object | 所有维度情感分布（计数+占比） |
| group_stats | object | 按一级维度（位置/服务/价格/环境/菜品）的统计 |
| category_rankings | array | 18维度按好评率排序 |

#### 响应示例
```json
{
  "code": 0,
  "data": {
    "total_reviews": 100,
    "total_aspects": 564,
    "avg_aspects_per_review": 5.6,
    "overall_distribution": {
      "positive_ratio": 45.0,
      "neutral_ratio": 20.0,
      "negative_ratio": 35.0
    },
    "sentiment_distribution": {
      "counts": {"positive": 298, "neutral": 89, "negative": 177},
      "ratios": {"positive": 52.8, "neutral": 15.8, "negative": 31.4}
    },
    "group_stats": {
      "菜品": {
        "total": 180,
        "positive": 120, "neutral": 30, "negative": 30,
        "positive_ratio": 66.7, "negative_ratio": 16.7
      },
      "服务": {
        "total": 142,
        "positive": 70, "neutral": 22, "negative": 50,
        "positive_ratio": 49.3, "negative_ratio": 35.2
      }
    },
    "category_rankings": [
      {
        "category": "Food#Taste",
        "category_zh": "菜品口味",
        "total": 95,
        "positive_ratio": 72.6,
        "negative_ratio": 11.6
      }
    ]
  }
}
```

---

## 6. 重置统计数据

清空累积的统计数据。

**POST** `/api/v1/stats/reset`

#### 响应示例
```json
{
  "code": 0,
  "message": "统计数据已重置"
}
```

---

## 错误码

| code | 说明 |
|------|------|
| 0 | 成功 |
| -1 | 请求参数错误（缺少 text 字段等） |
| -2 | 服务器内部错误 |
| -404 | 接口不存在 |

---

## 启动服务

```bash
# 1. 激活环境
conda activate NLPhomework

# 2. 进入项目目录
cd ABSA_NLPHomework

# 3. 启动服务
python scripts/run_api.py --host 0.0.0.0 --port 5000

# 可选参数:
#   --host  监听地址，默认 0.0.0.0（允许外部访问）
#   --port  监听端口，默认 5000
#   --debug 调试模式（开发用）
```

启动后访问 `http://127.0.0.1:5000/api/v1/health` 确认服务正常。

---

## 评价维度速查表

| 一级维度 | 二级维度 key | 中文名 |
|----------|-------------|--------|
| 位置 | `Location#Transportation` | 交通便利 |
| 位置 | `Location#Downtown` | 是否在商圈 |
| 位置 | `Location#Easy_to_find` | 是否好找 |
| 服务 | `Service#Queue` | 排队时间 |
| 服务 | `Service#Hospitality` | 服务态度 |
| 服务 | `Service#Parking` | 停车便利 |
| 服务 | `Service#Timely` | 上菜速度 |
| 价格 | `Price#Level` | 价格水平 |
| 价格 | `Price#Cost_effective` | 性价比 |
| 价格 | `Price#Discount` | 优惠活动 |
| 环境 | `Ambience#Decoration` | 装修风格 |
| 环境 | `Ambience#Noise` | 噪音水平 |
| 环境 | `Ambience#Space` | 空间大小 |
| 环境 | `Ambience#Sanitary` | 卫生状况 |
| 菜品 | `Food#Portion` | 菜品分量 |
| 菜品 | `Food#Taste` | 菜品口味 |
| 菜品 | `Food#Appearance` | 菜品外观 |
| 菜品 | `Food#Recommend` | 是否推荐 |
