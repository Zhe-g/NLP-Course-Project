# ABSA 情感分析系统 — 开发工作日志

## 项目概述

构建一个完整的电商评论细粒度情感分析（ABSA）系统，覆盖：
> 电商评论数据爬取与清洗过滤 → 去除停用词完成文本预处理 → 搭建完整产品评价属性分类体系 → 利用预训练模型实现属性与观点联合抽取 → 完成观点情感极性二次判别 → 多维度用户评价数据统计分析 → API接口

---

## 1. 需求分析与技术选型

### 1.1 项目背景

workspace 中有两套现有代码：

| 项目 | 说明 |
|------|------|
| `ABSA_system-master` | 基于 SemEval-2014 的英文 ABSA 系统，技术栈：Word2Vec + CRF + SVM，依赖 nltk + Stanford Parser + pycrfsuite，**技术栈陈旧，不支持中文，仅作参考** |
| `asap-master` | NAACL 2021 论文的 ASAP 数据集：46,730 条中文餐饮评论，18 个评价维度，类别级标注 |

### 1.2 硬件约束

- **GPU**: RTX 4060 Laptop，8GB VRAM
- **环境管理**: conda
- **API 框架**: Flask（后端同学技术栈）

### 1.3 技术选型决策

| 决策项 | 选项 | 最终选择 | 理由 |
|--------|------|----------|------|
| 数据源 | 爬虫 vs ASAP 数据集 | **ASAP 数据集** | 已有高质量人工标注，爬虫后置为可选 |
| 预训练模型 | BERT-base vs RoBERTa-wwm | **RoBERTa-wwm-ext** | 哈工大发布，全词掩码，中文效果更好 |
| 目标领域 | 餐饮 vs 通用电商 | **餐饮外卖** | 与 ASAP 数据集完美匹配 |
| API 框架 | FastAPI vs Flask | **Flask** | 后端同学技术栈 |
| 任务建模 | 序列标注(BIO) vs 多标签分类 | **多标签分类** | ASAP 无 token 级标注，只有类别级标注 |

### 1.4 关键技术决策：为什么用多标签分类而非序列标注？

计划初期设想用 BIO 序列标注做属性词抽取，但仔细检查 ASAP 数据集后发现：

- ASAP 的标注是**类别级**的（每条评论 × 18 维度 = 1/0/-1/-2），没有 token 级（span）标注
- 无法构造 "B-ASPECT / I-ASPECT / O" 的训练数据
- 因此改用**多标签二分类**：18 个独立 sigmoid 输出，预测每个维度是否被提及

---

## 2. 环境搭建

### 2.1 Conda 环境

```bash
conda create -n NLPhomework python=3.10 -y
conda activate NLPhomework
```

用户已提前安装 PyTorch 2.12.0 + CUDA 12.6。

### 2.2 依赖安装

**requirements.txt**:
```
torch>=2.0.0
transformers
datasets
accelerate
jieba
flask
pandas
scikit-learn
tqdm
numpy
matplotlib
zhconv
```

### 2.3 环境信息

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 |
| PyTorch | 2.12.0+cu126 |
| CUDA | 12.6 |
| Transformers | 最终升级至 5.10.2 |
| Accelerate | 1.13.0 |
| GPU | RTX 4060 Laptop (8GB) |

---

## 3. 项目目录结构

```
NLPHomework/
├── data/
│   ├── asap/                    # 原始 ASAP CSV 数据
│   │   ├── train.csv            # 36,850 条
│   │   ├── dev.csv              # 4,940 条
│   │   └── test.csv             # 4,940 条
│   ├── processed/               # 清洗后的数据
│   │   ├── train_clean.csv      # 36,844 条 (移除6条无效)
│   │   ├── dev_clean.csv        # 4,940 条
│   │   └── test_clean.csv       # 4,938 条 (移除2条无效)
│   └── stopwords/
├── models/                      # 训练好的模型
│   ├── aspect_detection/        # 属性检测模型（18维多标签分类）
│   └── sentiment/               # 情感分类模型（3分类）
├── src/
│   ├── config.py                # 全局配置
│   ├── preprocessing/           # 文本清洗 + jieba分词 + 停用词
│   ├── taxonomy/                # 18维度分类体系定义
│   ├── aspect_detection/        # 属性检测模块
│   │   ├── dataset.py           # 多标签数据集构建
│   │   ├── model.py             # RoBERTa + 多标签分类头
│   │   ├── train.py             # 训练脚本
│   │   └── predict.py           # 推理
│   ├── sentiment/               # 情感分类模块
│   │   ├── dataset.py           # 情感分类数据集
│   │   ├── model.py             # RoBERTa + 3分类头
│   │   ├── train.py             # 训练脚本
│   │   └── predict.py           # 推理
│   ├── pipeline/                # 推理流水线
│   │   └── analyzer.py          # 串联所有模块
│   ├── analysis/                # 统计分析
│   │   └── aggregator.py        # 多维度统计汇总
│   └── api/                     # Flask API
│       ├── app.py               # 应用入口 + 路由
│       └── routes.py            # 路由扩展点
├── scripts/
│   ├── preprocess_data.py       # 数据清洗脚本
│   ├── train_aspect.py          # 单独训练属性检测
│   ├── train_sentiment.py       # 单独训练情感分类
│   ├── train_all.py             # 一键训练全部模型
│   ├── run_api.py               # 启动 Flask 服务
│   └── test_pipeline.py         # 端到端测试
├── API_DOC.md                   # 接口文档
├── WORK_LOG.md                  # 本文件：工作日志
├── environment.yml
└── requirements.txt
```

---

## 4. 数据探索与预处理

### 4.1 ASAP 数据集格式

通过读取 CSV 文件发现实际列名与初始假设不同：

**实际列名**：
```
id, review, star,
Location#Transportation, Location#Downtown, Location#Easy_to_find,
Service#Queue, Service#Hospitality, Service#Parking, Service#Timely,
Price#Level, Price#Cost_effective, Price#Discount,
Ambience#Decoration, Ambience#Noise, Ambience#Space, Ambience#Sanitary,
Food#Portion, Food#Taste, Food#Appearance, Food#Recommend
```

**标签含义**：
| 值 | 含义 |
|----|------|
| 1 | 正向 |
| 0 | 中性 |
| -1 | 负向 |
| -2 | 未提及 |

### 4.2 数据清洗流程

`src/preprocessing/cleaner.py` 实现：

1. 去除 HTML 标签：`<[^>]+>`
2. 去除 URL：`https?://\S+`
3. 去除 @ 提及
4. 转义换行符 `\\n` → 空格
5. 全角字符转半角（`０-９` → `0-9`，`Ａ-Ｚ` → `A-Z`）
6. 去除多余空白
7. 截断 > 512 字符
8. 有效性检查：长度 ≥ 5 且包含中文

**清洗结果**：

| 数据集 | 原始 | 清洗后 | 移除无效 |
|--------|------|--------|----------|
| train | 36,850 | 36,844 | 6 |
| dev | 4,940 | 4,940 | 0 |
| test | 4,940 | 4,938 | 2 |

### 4.3 数据集构建

**属性检测数据集**（`AspectDetectionDataset`）：
- 每条评论 → 18 维 binary 标签向量
- `-2`（未提及）→ 0，其他（1/0/-1）→ 1
- 训练集 36,844 样本

**情感分类数据集**（`SentimentDataset`）：
- 每条评论 × 18 维度 → 只保留非 `-2` 的样本
- 标签映射：1→0(positive), 0→1(neutral), -1→2(negative)
- 训练集 213,371 样本（展开后）

---

## 5. 模型设计与训练

### 5.1 属性检测模型

**任务**: 多标签二分类（18 个独立 sigmoid）

**架构**:
```
RoBERTa-wwm-ext Encoder → [CLS] → Dropout(0.1) → Linear(768, 256) → GELU → Linear(256, 18) → Sigmoid
```

**损失函数**: BCEWithLogitsLoss

**训练配置**:
| 参数 | 值 |
|------|-----|
| batch_size | 8 |
| gradient_accumulation | 4（有效 batch=32） |
| learning_rate | 2e-5 |
| warmup_ratio | 0.1 |
| epochs | 5 |
| max_length | 256 |
| fp16 | True |
| 优化器 | AdamW |

### 5.2 情感分类模型

**任务**: 3 分类（positive/neutral/negative）

**输入格式**: `[CLS] 维度中文名 [SEP] 评论文本 [SEP]`

**架构**:
```
RoBERTa-wwm-ext Encoder → [CLS] → Dropout(0.1) → Linear(768, 3) → Softmax
```

**训练配置**:
| 参数 | 值 |
|------|-----|
| batch_size | 16 |
| gradient_accumulation | 2（有效 batch=32） |
| learning_rate | 2e-5 |
| epochs | 5 |
| fp16 | True |

### 5.3 遇到的训练问题与解决

#### 问题 1: `eval_strategy` 参数不存在
- **原因**: transformers 4.36 使用 `evaluation_strategy`，但 5.x 改回 `eval_strategy`
- **解决**: 先改为 `evaluation_strategy`，升级 transformers 5.x 后又改回 `eval_strategy`

#### 问题 2: `overwrite_output_dir` 参数不存在
- **原因**: transformers 5.x 移除了此参数
- **解决**: 直接删除该参数行

#### 问题 3: safetensors 保存 non-contiguous tensor 报错
- **原因**: transformers 5.x 默认用 safetensors 格式保存，safetensors 要求 tensor 连续
- **解决**: 在自定义 Trainer 中覆盖 `_save` 方法，手动 `.contiguous()` 后用 `torch.save` 保存

```python
def _save(self, output_dir=None, state_dict=None):
    import torch
    os.makedirs(output_dir, exist_ok=True)
    if state_dict is None:
        state_dict = self.model.state_dict()
    for k in state_dict:
        if hasattr(state_dict[k], 'contiguous'):
            state_dict[k] = state_dict[k].contiguous()
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
```

#### 问题 4: `gradient_checkpointing_enable` 属性不存在
- **原因**: 自定义模型包装了 encoder，Trainer 在模型对象上调用此方法
- **解决**: 在 `AspectDetectionModel` 和 `SentimentModel` 中添加代理方法

```python
def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
```

#### 问题 5: `dispatch_batches` 参数报错
- **原因**: transformers 4.36 与 accelerate 1.13 不兼容
- **解决**: 升级 transformers 至 5.10.2

### 5.4 训练结果

模型保存至：
- `models/aspect_detection/` — 属性检测模型
- `models/sentiment/` — 情感分类模型

---

## 6. 推理流水线

`src/pipeline/analyzer.py` 中的 `ABSAAnalyzer` 类实现端到端分析：

```
用户输入文本
     │
     ▼
文本清洗 (cleaner.py: 去HTML/URL、全角半角、空白规范化)
     │
     ▼
属性检测模型 → 18维 sigmoid 输出 → 阈值0.5筛选已提及维度
     │
     ▼
对每个已提及维度: (text, category) → 情感分类模型 → positive/neutral/negative
     │
     ▼
统计分析汇总 (aggregator.py)
     │
     ▼
返回 JSON 结果
```

**推理速度**: 单条评论约 0.25 秒（RTX 4060 Laptop）

---

## 7. Flask API 接口

### 7.1 端点列表

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/taxonomy` | 获取18维分类体系 |
| POST | `/api/v1/analyze` | 单条评论分析 |
| POST | `/api/v1/analyze/batch` | 批量评论分析 |
| GET | `/api/v1/stats` | 累积统计报告 |
| POST | `/api/v1/stats/reset` | 重置统计数据 |

### 7.2 设计要点

- **懒加载模型**: Flask 应用启动时不加载模型，首次请求时才加载，避免启动慢
- **累积统计**: 所有分析结果自动累积到 `StatisticsAggregator`，支持随时查询
- **统一响应格式**: `{"code": 0, "data": {...}}` 或 `{"code": -1, "message": "..."}`

### 7.3 启动命令

```bash
conda activate NLPhomework
cd ABSA_NLPHomework
python scripts/run_api.py --host 0.0.0.0 --port 5000
```

---

## 8. 18 维度分类体系

### 5 大一级维度 × 18 二级维度

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

---

## 9. 验证结果

### 9.1 基础模块测试

- 文本清洗：HTML 标签移除、全角半角转换、空白规范化 ✅
- jieba 分词：精确模式 + 关键词提取 ✅
- 分类体系：18 维度 × 5 大类查询 ✅
- 数据加载：多标签数据集 + 情感数据集正确构建 ✅

### 9.2 端到端测试

**输入**:
```
这家店的火锅味道很好，但是服务太差了，等了一个小时才上菜，环境还不错。
```

**输出**:
```
检测到 7 个评价维度:
  服务态度(服务): 负向 (0.95)
  上菜速度(服务): 负向 (0.97)
  装修风格(环境): 正向 (0.99)
  空间大小(环境): 正向 (0.99)
  卫生状况(环境): 正向 (1.00)
  价格水平(价格): 正向 (0.99)
  菜品口味(菜品): 正向 (0.97)
耗时: 0.25s
```

正确识别了 "服务太差" → 服务态度负向、"等了一个小时" → 上菜速度负向、"味道很好" → 菜品口味正向。

### 9.3 API 测试

```python
# 健康检查
GET /api/v1/health → {"code":0,"status":"healthy"}  ✅

# 单条分析
POST /api/v1/analyze {"text":"味道不错，服务很好，环境优雅"}
→ 正确返回6个维度及其情感极性  ✅
```

---

## 10. 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/config.py` | 148 | 全局配置（路径、超参、维度定义） |
| `src/preprocessing/cleaner.py` | 56 | 文本清洗 |
| `src/preprocessing/tokenizer.py` | 22 | jieba分词封装 |
| `src/preprocessing/stopwords.py` | 58 | 停用词管理 |
| `src/taxonomy/categories.py` | 43 | 分类体系查询接口 |
| `src/aspect_detection/dataset.py` | 88 | 多标签数据集 |
| `src/aspect_detection/model.py` | 70 | 属性检测模型 |
| `src/aspect_detection/train.py` | 148 | 训练脚本 |
| `src/aspect_detection/predict.py` | 89 | 推理 |
| `src/sentiment/dataset.py` | 88 | 情感分类数据集 |
| `src/sentiment/model.py` | 67 | 情感分类模型 |
| `src/sentiment/train.py` | 148 | 训练脚本 |
| `src/sentiment/predict.py` | 88 | 推理 |
| `src/pipeline/analyzer.py` | 121 | 推理流水线 |
| `src/analysis/aggregator.py` | 97 | 统计分析 |
| `src/api/app.py` | 137 | Flask API + 路由 |
| `scripts/preprocess_data.py` | 72 | 数据清洗脚本 |
| `scripts/train_all.py` | 40 | 一键训练 |
| `scripts/run_api.py` | 28 | 启动API |
| `scripts/test_pipeline.py` | 138 | 端到端测试 |
| `API_DOC.md` | - | 接口文档 |
| `WORK_LOG.md` | - | 本文件 |
