# ABSA 细粒度情感分析系统

基于 **RoBERTa-wwm-ext** 预训练模型的餐饮评论细粒度情感分析（ABSA）系统。自动识别评论文本中涉及的 **18 个评价维度**（菜品口味、服务态度、装修风格等），并对每个维度进行**正向/中性/负向**情感极性判别。

---

## 功能特性

- **18 维度属性级情感分析** — 不是简单的正/负分类，而是精确到具体评价维度
- **双模型架构** — 属性检测（多标签分类）+ 情感判别（多分类），两个独立模型
- **Web 可视化界面** — 批量导入评论、多维图表统计、18 维度好评率排名
- **REST API** — Flask 接口，供后端或第三方调用
- **GPU 推理优化** — fp16 推理，单条评论分析 < 0.3 秒

---

## 系统架构

```
┌──────────────────────────────────────────────────┐
│                   Web UI (:5001)                  │
│              FINAL/app.py + HTML/JS               │
│         批量导入 · 多维图表 · 好评率排名           │
└─────────────────────┬────────────────────────────┘
                      │ HTTP
┌─────────────────────▼────────────────────────────┐
│              模型 API (:5000)                     │
│              scripts/run_api.py                   │
│         文本清洗 → 属性检测 → 情感判别             │
└───────┬─────────────────────────┬────────────────┘
        │                         │
┌───────▼──────────┐    ┌─────────▼───────────┐
│ 属性检测模型      │    │ 情感判别模型         │
│ RoBERTa + 18维   │    │ RoBERTa + 3分类     │
│ 多标签 Sigmoid   │    │ Softmax             │
└──────────────────┘    └─────────────────────┘
```

**推理流水线**：
```
评论文本 → 清洗 → 属性检测(18维) → 筛选已提及维度 → 逐个维度情感分类 → 汇总输出
```

---

## 18 维度分类体系

| 一级维度 | 二级维度 | 中文名 |
|----------|---------|--------|
| 位置 | Location#Transportation | 交通便利 |
| 位置 | Location#Downtown | 是否在商圈 |
| 位置 | Location#Easy_to_find | 是否好找 |
| 服务 | Service#Queue | 排队时间 |
| 服务 | Service#Hospitality | 服务态度 |
| 服务 | Service#Parking | 停车便利 |
| 服务 | Service#Timely | 上菜速度 |
| 价格 | Price#Level | 价格水平 |
| 价格 | Price#Cost_effective | 性价比 |
| 价格 | Price#Discount | 优惠活动 |
| 环境 | Ambience#Decoration | 装修风格 |
| 环境 | Ambience#Noise | 噪音水平 |
| 环境 | Ambience#Space | 空间大小 |
| 环境 | Ambience#Sanitary | 卫生状况 |
| 菜品 | Food#Portion | 菜品分量 |
| 菜品 | Food#Taste | 菜品口味 |
| 菜品 | Food#Appearance | 菜品外观 |
| 菜品 | Food#Recommend | 是否推荐 |

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.x + RTX 4060 (8GB) 或同等 GPU
- conda (推荐)

### 1. 创建环境

```bash
conda create -n NLPhomework python=3.10 -y
conda activate NLPhomework
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 数据清洗（可选，训练时自动处理）

```bash
python scripts/preprocess_data.py
```

### 3. 训练模型

```bash
python scripts/train_all.py
```

训练分两步：属性检测模型 → 情感分类模型。单个模型约 2-3 小时（RTX 4060 Laptop）。

### 4. 启动服务

**方式一：一键启动（推荐）**

```bash
python FINAL/main.py
# 自动启动 API(:5000) + Web界面(:5001) + 打开浏览器
```

**方式二：手动分别启动**

```bash
# 终端 1：模型 API
python scripts/run_api.py --port 5000

# 终端 2：Web 界面
cd FINAL && python app.py
# 浏览器打开 http://localhost:5001
```

### 5. 测试

```bash
# 端到端测试
python scripts/test_pipeline.py
```

---

## API 接口

完整文档见 [API_DOC.md](API_DOC.md)

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/taxonomy` | 获取 18 维分类体系 |
| POST | `/api/v1/analyze` | 单条评论分析 |
| POST | `/api/v1/analyze/batch` | 批量评论分析 |
| GET | `/api/v1/stats` | 累积统计报告 |
| POST | `/api/v1/stats/reset` | 重置统计 |

**示例请求**：

```python
import requests

resp = requests.post("http://127.0.0.1:5000/api/v1/analyze",
    json={"text": "味道不错，服务很好，环境优雅，就是上菜慢了点"})
print(resp.json())
```

**示例响应**：

```json
{
  "code": 0,
  "data": {
    "text": "味道不错,服务很好,环境优雅,就是上菜慢了点",
    "aspects": [
      {"category": "Food#Taste", "category_zh": "菜品口味", "group": "菜品",
       "sentiment": "positive", "sentiment_zh": "正向", "sentiment_confidence": 0.95},
      {"category": "Service#Hospitality", "category_zh": "服务态度", "group": "服务",
       "sentiment": "positive", "sentiment_zh": "正向", "sentiment_confidence": 0.92},
      {"category": "Service#Timely", "category_zh": "上菜速度", "group": "服务",
       "sentiment": "negative", "sentiment_zh": "负向", "sentiment_confidence": 0.88}
    ],
    "summary": {
      "positive_count": 2, "neutral_count": 0, "negative_count": 1,
      "overall_sentiment": "positive", "total_aspects": 3
    },
    "elapsed_seconds": 0.25
  }
}
```

---

## 项目结构

```
NLPHomework/
├── src/                          # 核心源码
│   ├── config.py                 # 全局配置（路径、超参、维度定义）
│   ├── preprocessing/            # 文本预处理
│   │   ├── cleaner.py            #   文本清洗（HTML/URL/全角半角）
│   │   ├── tokenizer.py          #   jieba 分词
│   │   └── stopwords.py          #   停用词管理
│   ├── taxonomy/                 # 分类体系
│   │   └── categories.py         #   18维 + 5大类查询
│   ├── aspect_detection/         # 属性检测模型
│   │   ├── dataset.py            #   多标签数据集构建
│   │   ├── model.py              #   RoBERTa + 18维多标签分类头
│   │   ├── train.py              #   训练脚本（fp16优化）
│   │   └── predict.py            #   推理
│   ├── sentiment/                # 情感分类模型
│   │   ├── dataset.py            #   (文本, 维度) → 情感 数据集
│   │   ├── model.py              #   RoBERTa + 3分类头
│   │   ├── train.py              #   训练脚本
│   │   └── predict.py            #   推理
│   ├── pipeline/                 # 推理流水线
│   │   └── analyzer.py           #   串联清洗→属性检测→情感分类
│   ├── analysis/                 # 统计分析
│   │   └── aggregator.py         #   多维度统计汇总
│   └── api/                      # Flask 模型API
│       └── app.py                #   应用入口 + 路由
│
├── scripts/                      # 一键脚本
│   ├── preprocess_data.py        # 数据清洗
│   ├── train_all.py              # 训练全部模型
│   ├── run_api.py                # 启动模型 API
│   └── test_pipeline.py          # 端到端测试
│
├── FINAL/                        # Web 可视化界面
│   ├── main.py                   #   一键启动双服务
│   ├── app.py                    #   Flask Web 应用
│   ├── sentiment_analyzer.py     #   模型 API 客户端
│   ├── templates/index.html      #   前端页面
│   └── static/                   #   CSS + JS (Chart.js)
│
├── data/
│   ├── asap/                     # 原始 ASAP 数据集
│   └── processed/                # 清洗后数据集
│
├── models/                       # 训练好的模型
│   ├── aspect_detection/         #   属性检测模型
│   └── sentiment/                #   情感分类模型
│
├── API_DOC.md                    # 接口文档
├── WORK_LOG.md                   # 开发工作日志
├── requirements.txt              # 依赖清单
└── README.md                     # 本文件
```

---

## 训练数据

使用 **ASAP**（NAACL 2021）数据集：

- 来源：美团点评 46,730 条真实用户评论
- 标注：18 个评价维度 × 情感极性（1 正向 / 0 中性 / -1 负向 / -2 未提及）
- 划分：训练 36,850 / 验证 4,940 / 测试 4,940

引用：
> Bu et al., "ASAP: A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction", NAACL 2021

---

## 技术栈

| 层 | 技术 |
|------|------|
| 预训练模型 | `hfl/chinese-roberta-wwm-ext` |
| 深度学习 | PyTorch 2.x + HuggingFace Transformers |
| GPU 优化 | fp16 混合精度 + 梯度累积 |
| 分词 | jieba |
| 模型 API | Flask |
| Web 界面 | Flask + Chart.js |
| 数据 | ASAP 数据集 (NAACL 2021) |

---

## 模型详情

### 属性检测模型

| 项目 | 值 |
|------|-----|
| 任务 | 18维多标签二分类 |
| 架构 | RoBERTa → [CLS] → Linear(768,256) → GELU → Linear(256,18) → Sigmoid |
| 损失 | BCEWithLogitsLoss |
| 训练配置 | batch=8, grad_accum=4, lr=2e-5, epochs=5, fp16 |

### 情感分类模型

| 项目 | 值 |
|------|-----|
| 任务 | 3分类 (正向/中性/负向) |
| 架构 | RoBERTa → [CLS] → Linear(768,3) → Softmax |
| 输入 | `[CLS] 维度中文名 [SEP] 评论文本 [SEP]` |
| 损失 | CrossEntropyLoss |
| 训练配置 | batch=16, grad_accum=2, lr=2e-5, epochs=5, fp16 |

---

## 许可

本项目仅用于学习和研究目的。
