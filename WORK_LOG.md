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
│       └── app.py               # 应用入口 + 路由
├── scripts/
│   ├── preprocess_data.py       # 数据清洗脚本
│   ├── train_all.py             # 一键训练全部模型
│   ├── run_api.py               # 启动 Flask 服务（懒加载版）
│   ├── evaluate.py              # 模型评估（Precision/Recall/F1）
│   └── test_pipeline.py         # 端到端测试
├── FINAL/                       # Web 可视化界面
│   ├── main.py                  # 一键启动双服务
│   ├── app.py                   # Flask Web 应用 (port 5001)
│   ├── sentiment_analyzer.py    # 模型 API HTTP 客户端
│   ├── start_backend.bat        # Windows 启动脚本
│   ├── templates/index.html     # 前端页面
│   └── static/                  # CSS + JS + Chart.js
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
| `src/analysis/aggregator.py` | 114 | 统计分析 |
| `src/api/app.py` | 160+ | Flask API 入口 + 路由（唯一路由定义处） |
| `scripts/preprocess_data.py` | 72 | 数据清洗脚本 |
| `scripts/train_all.py` | 40 | 一键训练全部模型 |
| `scripts/evaluate.py` | 166 | 模型评估（Precision/Recall/F1） |
| `scripts/run_api.py` | 50 | API 懒加载启动器（路由来自 src.api.app） |
| `scripts/test_pipeline.py` | 138 | 端到端测试 |
| `FINAL/app.py` | 160+ | Web UI 入口 (port 5001) |
| `FINAL/sentiment_analyzer.py` | 283 | 模型 API HTTP 客户端 + 统计汇总 |
| `FINAL/main.py` | 117 | 一键启动双服务 |
| `API_DOC.md` | - | 接口文档 |
| `WORK_LOG.md` | - | 本文件 |

---

## 11. 后端 FINAL 项目评估 (2026-06-09 ~ 2026-06-10)

### 11.1 评估 FINAL 完成度

后端同学提交的 FINAL 项目位于 `FINAL/`，包含一个独立的 Web UI。分析了其代码后，评估如下：

**已完成的（约 40%）**：
- UI 界面框架：HTML + CSS + Chart.js，美观度不错
- 评论批量导入：支持单文件/多文件上传、拖拽、手动输入，TXT/CSV/XLSX
- 基础情感分类：SnowNLP + jieba 关键词匹配做正/负/中三分类

**未完成/问题**：
- **没有对接我们的模型**：用的是 SnowNLP 规则方法，不是 RoBERTa 深度学习模型
- **没有 18 维多维度**：只做正/负/中三分类，没有按 18 个维度分析
- **`app.py` 缺失**（但后来发现有旧版）
- **`init_db.py` 缺失**
- **`requirements.txt` 缺失**
- 启动脚本 `start_backend.bat` 引用不存在的文件

**决策**: 保留前端框架，干掉 SnowNLP，全部走我们训练的模型 API。

### 11.2 FINAL 改造 — SnowNLP → 模型 API

改造了 5 个文件：

**1. `sentiment_analyzer.py`** — 全量重写
- 删除所有 SnowNLP、jieba 关键词词表（~70 行规则代码）
- 改为 HTTP 调用我们的模型 API：`GET /api/v1/taxonomy`、`POST /api/v1/analyze`
- 新增 `_compute_multi_dimension_summary()` — 18 维多维度汇总：
  - 整体评论情感分布
  - 5 个一级维度（位置/服务/价格/环境/菜品）统计
  - 18 个具体维度好评率排名
- 新依赖：`requests`（调用模型 API）

**2. `app.py`** — Flask 入口重写
- 端口从 5000（与模型 API 冲突）改为 5001
- 移除 `flask_cors` 依赖
- 简化路由：只保留 `/upload`、`/batch-upload`、`/input-text`、`/health`
- 上传文件分析后自动清理临时文件
- 旧版中保存 JSON 到磁盘的逻辑全部移除

**3. `templates/index.html`** — 前端页面改造
- 新增模型状态指示器（绿色/红色圆点显示 API 连接状态）
- 新增 5 个概览卡片：总评论数、总维度数、平均维度/条、正向评论、负向评论
- 新增 4 个图表区域：
  - 评论整体情感饼图
  - 评价维度情感饼图
  - 5 个一级维度分组堆叠柱状图
  - 18 维度好评率排名水平柱状图
- 新增评论详情列表，每条评论显示所有检测到的维度标签

**4. `static/js/main.js`** — 前端逻辑重写
- 新增 `checkModelHealth()` — 页面加载时检测模型 API 连接状态
- `handleTextAnalysis()` — 调用 `/input-text`，将单条结果包装成批量格式统一渲染
- `updateSummary()` — 更新 5 张概览卡片
- `updateCharts()` — 渲染 4 个 Chart.js 图表：饼图×2、堆叠柱状图、水平排名图
- `updateGroupDetail()` — 渲染 5 个一级维度的统计卡片
- `updateReviewsList()` — 每条评论显示维度标签（as aspect-tag badges）
- `_buildGroupStats()` / `_buildCategoryRankings()` — 单条评论输入时在前端计算统计

**5. `static/css/style.css`** — 新增样式
- `.model-status` — 模型状态指示器
- `.group-detail` / `.group-card` — 一级维度统计卡片
- `.aspect-tag` — 维度标签（绿色正向/红色负向/蓝色中性）
- `.chart-wrapper.full` — 全宽图表

### 11.3 新增 main.py — 一键启动双服务

`FINAL/main.py`：
- 自动启动模型 API (:5000) + Web 界面 (:5001)
- 等待 API 就绪后再启动 Web，避免空请求
- 自动打开浏览器到 `http://localhost:5001`
- Ctrl+C 一键停止两个服务
- 备选：双击 `start_backend.bat` 也可启动

### 11.4 FINAL 冗余文件清理

删除了 FINAL 中的旧文件：
- `sentiment_analyzer_advanced.py` — 旧版分析器（含 api_key 占位符）
- `api_examples.py` — 旧版 API 示例
- `config_template.json` — 旧配置模板
- `本地模型使用指南.md` — 过时文档
- `测试报告.md` — 过时文档

---

## 12. 项目清理 (2026-06-10)

### 12.1 删除冗余大目录

| 删除项 | 大小 | 原因 |
|--------|------|------|
| `ABSA_system-master/` | ~几十MB | 旧英文项目，不参与运行 |
| `asap-master/` | ~几十MB | 数据已复制到 `data/asap/` |
| `FINAL/FINAL/.venv/` | ~几百MB | 虚拟环境，conda 已够用 |
| `models/*/checkpoint-*/` | ~400MB×2组 | 训练中间产物，最终模型已保存 |
| `scripts/train_aspect.py` | 重复 | `train_all.py` 覆盖 |
| `scripts/train_sentiment.py` | 重复 | `train_all.py` 覆盖 |

清理后总大小：~2GB → **867MB**（其中 782MB 是两个 RoBERTa 模型权重，代码+数据 85MB）

---

## 13. README.md 撰写 (2026-06-10)

写了完整的 README.md，包含：
- 项目概述 + ASCII 系统架构图
- 18 维度速查表（中英文对照）
- 快速开始（5 步：环境 → 清洗 → 训练 → 启动 → 测试）
- API 接口表 + 请求/响应 JSON 示例
- 完整项目结构树
- 训练数据说明（ASAP 论文引用）
- 技术栈总览 + 两个模型的详细配置表

---

## 14. GitHub 推送 (2026-06-10)

### 14.1 问题：模型文件超过 GitHub 100MB 限制

- `models/aspect_detection/model.safetensors` — 390MB
- `models/sentiment/model.safetensors` — 390MB
- GitHub pre-receive hook 拒绝推送

### 14.2 解决步骤

1. 创建 `.gitignore`，排除 `data/`、`models/`、`__pycache__/` 等
2. 用 `git rm --cached` 移除已追踪的大文件
3. 由于大文件在历史 commit 中，GitHub 仍然拒绝
4. 删除 `.git` 重新初始化仓库
5. 只 add 源码文件（57 个文件），排除模型和数据
6. 创建干净 commit，force push 到 main

### 14.3 安全审查

推送前扫描了敏感信息：
- **绝对路径**：`start_backend.bat` 中 `E:\programingCodeFile\...` 改为相对路径
- **文档路径**：`API_DOC.md`、`WORK_LOG.md` 中本地路径替换为通用路径
- **旧文件**：含 `api_key` 占位符的旧文件已删除
- **敏感信息**：全项目 grep `password`、`api_key`、`secret` — 无泄露

---

## 15. sentint_analyzer.py Bug 修复 — category_zh 全显示同一个维度 (2026-06-10)

### 15.1 表现

Web 界面的"18 维度好评率排名"图表中，所有维度的中文名都显示为"装修风格"。

### 15.2 根因

`_compute_multi_dimension_summary()` 第 212 行：

```python
# 错误: 对每个 cat 都使用了 all_aspects[0] 的 category_zh
"category_zh": all_aspects[0].get("category_zh", cat) if all_aspects else cat,
```

`all_aspects[0]` 写死了第一个 aspect 的中文名，导致所有 18 个维度都被标记为第一个维度的名称（如"装修风格"）。

### 15.3 修复

```python
# 正确: 按当前 cat 查找对应的 category_zh
cat_zh = cat  # fallback
for a in all_aspects:
    if a.get("category") == cat:
        cat_zh = a.get("category_zh", cat)
        break
```

---

## 16. 模型评估实验 (2026-06-10)

### 16.1 新增 evaluate.py

`scripts/evaluate.py` — 在测试集上计算 Precision、Recall、F1。

### 16.2 属性检测模型结果（18维多标签，测试集 4,938 条）

| 指标 | 值 |
|------|-----|
| Micro-Precision | 0.8555 |
| Micro-Recall | 0.8094 |
| **Micro-F1** | **0.8318** |
| **Macro-F1** | **0.7965** |

Top 5 维度（F1）：
1. 菜品口味: 0.9741
2. 装修风格: 0.8834
3. 停车便利: 0.8719
4. 服务态度: 0.8682
5. 交通便利: 0.8611

Bottom 3 维度（F1）：
1. 菜品外观: 0.5602（Recall 低，容易被遗漏）
2. 是否推荐: 0.6446
3. 性价比: 0.6985

### 16.3 情感分类模型结果（3分类，测试集 28,356 条）

| 指标 | 值 |
|------|-----|
| **Accuracy** | **0.8089** |
| **Macro-F1** | **0.7503** |
| **Weighted-F1** | **0.8041** |

各类别 F1：
- 正向: 0.8870（多数类，表现最好）
- 负向: 0.7223
- 中性: 0.6418（少数类，表现最差）

---

## 17. 其他修复 (2026-06-10)

### 17.1 requirements.txt 重写
- 移除 `peft`（未使用 LoRA）
- 移除 `transformers==4.36.0` 版本锁定
- 新增 `requests`（FINAL Web 界面需要）

### 17.2 API 懒加载优化（用户自行修改）
- `run_api.py` 改为懒加载模式：先导入 Flask → 再加载模型
- `src/api/app.py` 拆分为路由定义 + `set_analyzer()` 外部注入
- `src/aspect_detection/predict.py` 和 `src/sentiment/predict.py` 改为延迟导入 transformers

### 17.3 LF/CRLF 警告说明
Git 在 Windows 上默认 `core.autocrlf=true`，会将 LF 转为 CRLF。"LF will be replaced by CRLF" 警告不影响代码运行，是 Git 的格式统一行为。

### 17.4 多线程使用情况
- Flask `threaded=True` — 处理并发 HTTP 请求
- `main.py` `threading.Thread` — 后台读取子进程输出
- PyTorch `dataloader_num_workers=2` — 训练时多进程数据加载
- 推理流水线是单线程串行，不存在竞态条件

---

## 18. 前端页面优化 (2026-06-16)

> Author: hxy · Commit: `29955be` · 12 files changed, +2693/-170 lines

### 18.1 改动概览

前端同学对 Web UI 进行了大幅功能增强，涉及 12 个文件：

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `FINAL/FINAL/static/js/main.js` | +1370/-? | 前端逻辑从 ~200 行暴增到 ~1400 行 |
| `FINAL/FINAL/static/css/style.css` | +949/-? | 样式从 ~200 行扩展到 ~1100 行 |
| `FINAL/FINAL/templates/index.html` | +164/-? | 新增大量 UI 组件 |
| `FINAL/FINAL/app.py` | +79/-? | 后端适配新前端功能 |
| `FINAL/FINAL/sentiment_analyzer.py` | +32/-? | 增强返回数据结构 |
| `scripts/run_api.py` | +85/-? | 后端配合 |
| `src/api/app.py` | +15/-? | 后端配合 |
| `src/aspect_detection/model.py` | +31/-? | 模型层配合 |
| `src/aspect_detection/predict.py` | +49/-? | 推理配合 |
| `src/sentiment/model.py` | +31/-? | 模型层配合 |
| `src/sentiment/predict.py` | +51/-? | 推理配合 |
| `src/config.py` | +7/-? | 配置配合 |

### 18.2 前端新增功能

#### 18.2.1 模式切换
- 新增"手动输入"和"文件上传"两个选项卡，用户可自由切换
- 默认显示手动输入面板，文件上传面板隐藏

#### 18.2.2 文件管理增强
- 支持拖拽上传 + 点击选择文件（TXT/CSV/XLSX）
- 文件列表可视化显示：文件名、文件大小、文件图标
- 支持逐项删除和"清空"按钮
- `formatFileSize()` 格式化文件大小（B/KB/MB）

#### 18.2.3 上传进度条
- 批量上传使用 XHR + SSE 流式进度事件
- `uploadMultipleWithProgress()` 实时解析 `data:` 事件更新进度条
- 进度条显示百分比 + 文字描述（"正在处理文件: xxx"）
- 后端 `batch_upload` 改为 `stream_with_context` + SSE 格式响应

#### 18.2.4 累计统计面板
- 使用 `localStorage` 持久化历史分析累计数据
- 显示：分析次数、累计评论数、累计维度数、累计正向数、累计负向数
- 数据持久化到浏览器本地存储，跨次分析累加

#### 18.2.5 图表布局重构
- 原来单一的柱状图 → 改为 **雷达图 + 柱状图** 并排布局
- `drawRadarChart()` — 5 个一级维度好评率雷达图
- 柱状图展示各维度情感分布（正/中/负堆叠）

#### 18.2.6 筛选与搜索
- 关键词搜索输入框，实时过滤评论列表
- 按情感标签筛选：全部 / 正向 / 中性 / 负向
- 筛选按钮高亮当前激活的过滤状态
- 显示"显示 X 条"计数

#### 18.2.7 评论列表分页
- 分页变量：`currentPage=1`, `pageSize=10`
- 分页控件：上一页 / 第 X 页 / 下一页
- 自动根据筛选后的结果重新计算页数

#### 18.2.8 暂存列表
- "暂存结果"按钮将当前分析结果保存到 `localStorage`
- 模态框展示已暂存的记录列表（编号、文件名、维度数、评论数）
- 支持逐项删除暂存记录
- 关闭模态框（点击遮罩层、×按钮、ESC键）

#### 18.2.9 导出功能
- 新增导出下拉菜单（"导出 Excel" / "导出 CSV"）
- 引入 SheetJS (xlsx@0.18.5) CDN
- `exportToExcel()` — 导出 .xlsx 格式，含详情 Sheet + 汇总 Sheet
- `exportToCSV()` — 导出 .csv 格式

#### 18.2.10 后端适配
- `batch_upload()` 改为 SSE 流式响应（`text/event-stream`），每个文件处理完成推送进度事件
- `analyze_file()` 返回值新增 `sentiment_counts`、`dimension_stats` 等字段
- `_compute_multi_dimension_summary()` 增强，新增 `average_aspects`、`review_level_sentiment` 等统计

---

## 19. 项目冗余清理 (2026-06-17)

> AI 辅助清理，共删除 234 行，新增 37 行，净减 ~200 行

### 19.1 API 路由重复消除

**问题**：`scripts/run_api.py` (101 行) 和 `src/api/app.py` (161 行) 各自独立定义了相同的 6 个 API 路由（health/taxonomy/analyze/batch/stats/reset），任何 API 改动需要同步两个文件。

**修复内容**：

1. **重写 `scripts/run_api.py`** (101→50 行)：
   - 改为从 `src/api.app` 导入 `app` 和 `set_analyzer`
   - 仅保留懒加载 + 启动逻辑，路由定义不再重复
   - `init_analyzer()` 在启动前完成模型加载，调用 `set_analyzer()` 注入

2. **修复 `src/api/app.py`**：
   - 移除未使用的 `from src.pipeline.analyzer import ABSAAnalyzer` 导入（analyse 类从未在此文件中实例化，由 `set_analyzer()` 外部注入）
   - 修复 `__main__` 入口：新增懒加载初始化（之前直接 `app.run()` 会因为 `_analyzer is None` 在首次请求报错）

### 19.2 死代码删除

| 删除文件 | 行数 | 原因 |
|----------|------|------|
| `src/api/routes.py` | 10 | 空壳扩展点，全项目零 import |
| `FINAL/FINAL/simple_test.py` | 134 | 与 `test_system.py` 功能重复，且使用旧接口路径 `/input-text` |

对应更新了 `README.md` 和 `WORK_LOG.md` 中的目录结构文档。

### 19.3 空目录和死配置清理

- **删除 `data/stopwords/`**：空目录，git 未跟踪任何文件
- **删除 `config.py` 中的 `STOPWORDS_DIR`**：全项目零引用

### 19.4 FINAL/FINAL/ 双层嵌套扁平化

**问题**：`FINAL/FINAL/` 外层 `FINAL/` 只是空壳容器，所有文件在内层 `FINAL/FINAL/`，增加了不必要的目录深度。

**修复**：`FINAL/FINAL/*` → `FINAL/*`，共 21 个文件。

**连带修复的路径引用**：

| 文件 | 改动 |
|------|------|
| `FINAL/main.py:12` | `BASE_DIR` 从 3 层 `dirname` 改为 2 层 |
| `FINAL/start_backend.bat:10` | `PROJECT_ROOT` 从 `%~dp0..\..` 改为 `%~dp0..` |
| `README.md:107,118` | `FINAL/FINAL/main.py` → `FINAL/main.py` |
| `README.md:215-221` | 项目结构树中移除嵌套层 |

### 19.5 清理统计

```
 FINAL/FINAL/simple_test.py | 134 行删除
 FINAL/main.py              |   2 行修改
 FINAL/start_backend.bat    |   4 行修改
 README.md                  |  18 行修改
 WORK_LOG.md                |   3 行修改
 scripts/run_api.py         |  93 行重写 (101→50)
 src/api/app.py             |   6 行修改
 src/api/routes.py          |  10 行删除
 src/config.py              |   1 行删除
 ─────────────────────────────────
 9 files, +37 / -234, 净减 ~200 行
```

### 19.6 未处理但已知的冗余

以下问题已知但未在本次处理，留给后续决策：

- **统计逻辑重复**：`FINAL/sentiment_analyzer.py::_compute_multi_dimension_summary()` 与 `src/analysis/aggregator.py::compute_stats()` 实现相同功能，Web UI 应通过 API 获取统计而非本地重算
- **文件读取列检测重复**：CSV/XLSX review 列名匹配逻辑在 `FINAL/app.py` 和 `FINAL/sentiment_analyzer.py` 中各有一份
- **`FINAL/uploads/` 中的临时文件**：测试上传残留，可安全清理
- `FINAL/FINAL/` 在 git 历史中仍作为 `git mv` rename 记录存在

---


## 20. Web 前端 Bug 修复与功能迭代 (2026-06-20)

> AI 辅助调试，共涉及 5 个独立问题，覆盖 CSS 层叠冲突、事件传播、localStorage 配额、生成器特性、线程通信等知识点。

### 20.1 点击上传无效 — CSS 层叠冲突 + JS 缺失

**表现**：点击"点击上传"区域无任何反应，文件选择框不弹出。

**根因分析**：

1. **CSS 冲突**：`#fileInput` 被定义两次——
   - 旧规则 `#fileInput { display: none; }`（第 81 行）来自旧版代码
   - 新规则 `#fileInput { position: absolute; opacity: 0; ... }`（第 616 行）是覆盖层方案
   - CSS 规则层叠时，`opacity: 0` **不会覆盖** `display: none`——两个属性独立生效
   - 结果：`input` 既透明又 `display: none`，完全不渲染，无法交互

2. **JS 缺失**：`initDragDrop()` 只注册了拖拽事件（dragenter/dragover/dragleave/drop），没有 click 事件。重构时丢失了旧版 `dropZone.addEventListener('click', () => fileInput.click())`。

**知识点 — CSS 层叠规则**：
- `display: none` 使元素完全脱离渲染树，不占空间
- `opacity: 0` 只是视觉透明，元素仍在渲染树中，占据空间且可交互
- 自定义文件上传的标准模式：`opacity: 0` + `position: absolute` 覆盖层，利用浏览器原生行为
- 不同属性不会互相覆盖，需显式清除旧规则

```css
/* 错误：两条规则叠加，display:none 未被清除 */
#fileInput { display: none; }           /* 旧 */
#fileInput { opacity: 0; position:... } /* 新 → display 仍然是 none */

/* 正确：显式覆盖 */
#fileInput { display: block; opacity: 0; position: absolute; ... }
```

**修复**：删除旧 `#fileInput { display: none; }` 规则，在新规则中加 `display: block`；在 `initDragDrop()` 中新增 click 监听。

---

### 20.2 点击一次打开两次文件对话框 — 事件冒泡

**表现**：修复上传后，点击上传区域需要选两次文件。

**根因**：`#fileInput` 作为绝对定位覆盖层铺满 dropZone（CSS 层），点击事件到达 `#fileInput` 触发原生文件对话框，事件冒泡到 `dropZone`，JS click handler 再次调用 `fileInput.click()`，又弹一次对话框。

**知识点 — DOM 事件传播**：
- 事件传播三阶段：捕获（capture）→ 目标（target）→ 冒泡（bubble）
- `addEventListener` 默认注册在冒泡阶段
- `e.target` 指向实际被点击的元素（事件源头），而非绑定监听器的元素
- `e.stopPropagation()` 阻止冒泡，但不阻止同一元素上的其他监听器

```
用户点击 → 命中 #fileInput（opacity:0覆盖层）
  ├─ 原生行为触发文件选择 ①
  └─ 事件冒泡到 dropZone
      └─ JS: fileInput.click() 再次触发 ②  ← BUG
```

**修复**：在 dropZone click handler 中判断 `e.target === fileInput`，如果是则 `return`（已被原生处理）：

```javascript
dropZone.addEventListener('click', (e) => {
    if (e.target === fileInput || fileInput.contains(e.target)) {
        return;  // 原生已处理，避免重复
    }
    fileInput.click();  // 仅兜底边缘情况（如点击 padding）
});
```

---

### 20.3 批量分析进度条一直 0% — 生成器特性与线程通信

**表现**：批量上传时进度条始终停在 0%，分析完成后直接跳到结束。

**根因**：`app.py` 中 `progress_callback` 函数体内用了 `yield`，它是**生成器函数**。调用方 `analyzer.analyze_file(filepath, progress_callback=progress_callback)` 只是**普通函数调用**——传入了一个生成器对象，但从不迭代它。

**知识点 — Python 生成器（Generator）**：
- 包含 `yield` 的函数是生成器函数，调用它**不会执行函数体**，而是返回一个生成器对象
- 只有对生成器对象调用 `next()` 或在 `for` 循环中迭代时，函数体才开始执行
- 生成器在每次 `yield` 处暂停，交出值，等待下一次迭代

```python
# 错误示范
def progress_callback(current, total, message):
    yield f"data: {json.dumps({...})}\n\n"  # ← 生成器函数

r = analyzer.analyze_file(filepath, progress_callback=progress_callback)
# ↑ analyzer.analyze_file 内部只是 callback(1, 100, "...")
#   返回生成器对象，但从不 next() → yield 从未执行 → SSE 从未发送
```

**知识点 — 多线程 + 队列实现实时通信**：
- `threading.Thread` 在 Flask 请求上下文中创建子线程执行耗时操作
- `queue.Queue()` 是线程安全的生产者-消费者队列，`put()` 和 `get()` 内部有锁保护
- `queue.get(timeout=0.3)` 阻塞等待，超时抛 `queue.Empty`
- `stream_with_context` 下的生成器在主线程中 `yield` SSE 消息
- 心跳包（heartbeat）防止代理/浏览器因长时间无数据而断开连接

```
主线程（SSE生成器）                    子线程（分析）
    │                                    │
    ├─ pq = queue.Queue() ──────────────┤
    ├─ t = Thread(target=_run) ────────→│
    │                                    ├─ analyze_file()
    │                                    ├─ callback → pq.put(msg)
    │  while t.is_alive():               │   ...
    │    try: pq.get(0.3) → yield       │   循环逐条评论
    │    except Empty: yield heartbeat   │
    │                                    └─ 完成
    ├─ t.join()                          │
    ├─ 排空残余消息                       │
    └─ yield complete/error              │
```

```python
def _run():
    result_holder[0] = analyzer.analyze_file(filepath, progress_callback=progress_callback)

t = threading.Thread(target=_run, daemon=True)
t.start()

while t.is_alive():
    try:
        msg = pq.get(timeout=0.3)       # 阻塞等消息
        yield f"data: {json.dumps(msg)}\n\n"
    except queue.Empty:
        yield f"data: {json.dumps({'stage': 'heartbeat'})}\n\n"
```

**后续**：用户觉得进度条太麻烦，最终决定删除整个进度条组件，只保留 spinner + "正在分析中…" 文字。

---

### 20.4 侧栏历史报告 + localStorage 配额溢出

**功能**：新增左侧历史报告侧栏，自动保存每次分析结果。

**知识点 — Web Storage API**：
- `localStorage`：浏览器键值存储，同源共享，数据持久化（除非手动清除）
- 每个域名配额 **~5MB**，超出抛 `QuotaExceededError`
- `JSON.stringify()/JSON.parse()` 做序列化，大型对象（如批量评论结果）可达数 MB
- `localStorage.setItem()` 是同步操作，可能阻塞 UI 线程

**实现**：
- 侧栏 HTML：固定定位 + `transition: left 0.3s` 实现滑入/滑出动画
- 侧栏 CSS：`.sidebar { left: -360px }` → `.sidebar.open { left: 0 }`
- 遮罩层：`.sidebar-overlay` 配合 `opacity` 过渡
- 数据流：`displayResults()` → `saveToHistory()` → `localStorage.setItem()`
- `viewHistoryReport()`：从 `summarySnapshot` 重建 minimal `currentData`，重渲染图表

**遇到的关键 Bug — QuotaExceededError**：

批量分析 100+ 条评论时，`saveToHistory` 将完整 `resultData`（含每条评论的 aspects 详情）存入 localStorage，单条记录可达 3-8MB，超过 5MB 配额。

**修复策略 — 数据瘦身**：
- 不保存 `results` 数组（逐条评论明细），只保存 `summarySnapshot`（4 个汇总对象）
- 汇总对象只含 `{review_sentiment_dist, aspect_sentiment_dist, group_stats, category_rankings}`
- 单条记录从 **~3MB → ~3KB**，缩小 1000 倍
- 加 `try-catch` 保护，配额超限时砍掉一半旧记录后重试

```javascript
// 错误：存储整个 resultData
const record = { data: JSON.parse(JSON.stringify(resultData)) };  // 3MB+

// 正确：只存汇总
const record = {
    totalReviews, totalAspects, positiveRatio, ...
    summarySnapshot: {
        review_sentiment_dist: summary.review_sentiment_dist,  // ~200B
        group_stats: summary.group_stats,                      // ~1KB
        category_rankings: summary.category_rankings,           // ~1KB
    }
};
```

---

### 20.5 单文件上传进度 — fetch API 的局限性

**现象**：单文件上传使用 `fetch('/upload')`，无进度事件，进度条无法更新。

**知识点 — fetch vs XHR**：
- `fetch()` 返回 Promise，`resp.json()` 等整个响应结束后才解析——**不支持流式进度**
- `XMLHttpRequest` 有 `xhr.onprogress` 事件，可读取部分响应（配合 SSE）
- 因此批量上传使用 XHR + SSE 流式处理，单文件上传只能用 spinner 等待

**后续**：删除进度条后，单文件/批量都统一用 spinner + 文字提示。

---

### 20.6 文件上传安全性 — 竞态条件 + 资源清理

**Bug**：`os.remove(filepath)` 在 `finally` 块中无保护，同名文件并发上传时报 `FileNotFoundError` → Flask 500 → 返回 HTML 错误页 → 前端 JSON 解析失败。

**知识点**：
- `finally` 块无论 `try` 是否异常都会执行，适合做资源清理
- 但 `finally` 块自身的异常**会覆盖** `try` 块中的原始异常
- 文件清理应使用 `try-except` 包裹，容忍文件已不存在的情况
- `uuid.uuid4().hex[:8]` 生成唯一前缀，避免并发写同一路径

**修复**：
```python
def _safe_remove(filepath):
    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass  # 已被其他进程删除，忽略

# 文件名加唯一前缀
unique_name = f"{uuid.uuid4().hex[:8]}_{secure_filename(filename)}"
```

同时在启动时清理 `uploads/` 中上次异常退出遗留的临时文件。

---

### 20.7 CSS 主题切换 — 紫→白蓝

**修改**：全局替换 CSS 配色变量，涉及 40+ 处。

**知识点 — CSS 颜色体系**：
- 原主题：主色 `#667eea`（紫蓝 Indigo），辅色 `#764ba2`（深紫），阴影 `rgba(102,126,234,*)`
- 新主题：主色 `#2563eb`（蓝 Blue-600），辅色 `#1d4ed8`（深蓝 Blue-700），阴影 `rgba(37,99,235,*)`
- Header 从紫色渐变背景改为纯白 + 细边框 + 深色文字
- 累计统计卡片从粉红渐变改为蓝色渐变

| 元素 | 旧 | 新 |
|------|----|----|
| 主按钮/链接/边框 | `#667eea` 紫蓝 | `#2563eb` 蓝 |
| 悬停态 | `#5a6fd6` 深紫 | `#1d4ed8` 深蓝 |
| Header 背景 | 紫蓝渐变 + 白字 | 白色 + 深字 + 边框 |
| 统计卡片 | 粉红渐变 | 蓝色渐变 |
| 盒子阴影 | `rgba(102,126,234,*)` | `rgba(37,99,235,*)` |

---

### 20.8 今日改动汇总

| # | 问题 | 涉及知识点 | 文件 |
|---|------|-----------|------|
| 20.1 | 点击上传无效 | CSS 层叠、`display:none` vs `opacity:0` | CSS + JS |
| 20.2 | 双击打开文件对话框 | DOM 事件冒泡、`e.target` | JS |
| 20.3 | 批量进度条不动 | Python 生成器 yield、多线程+队列 | Python |
| 20.4 | 侧栏历史报告 | localStorage、Web Storage API、配额限制 | HTML/CSS/JS |
| 20.5 | localStorage 溢出 | QuotaExceededError、数据瘦身 | JS |
| 20.6 | 文件删除崩溃 | try-finally 异常覆盖、并发竞态、UUID | Python |
| 20.7 | 进度条删除 | 组件精简 | HTML/CSS/JS |
| 20.8 | 紫→白蓝主题 | CSS 颜色体系替换 | CSS |
