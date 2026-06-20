# ABSA 细粒度情感分析系统 — 技术报告

> **课题**: 面向电商评论的属性-观点细粒度信息抽取系统
> **技术栈**: RoBERTa-wwm-ext + 多标签分类 + Flask + Chart.js
> **数据集**: ASAP (NAACL 2021) — 46,730 条中文餐饮评论，18 个评价维度

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [数据预处理](#3-数据预处理)
4. [属性分类体系](#4-属性分类体系)
5. [模型设计](#5-模型设计)
6. [模型训练](#6-模型训练)
7. [推理流水线](#7-推理流水线)
8. [API 服务](#8-api-服务)
9. [Web 可视化界面](#9-web-可视化界面)
10. [实验评测](#10-实验评测)
11. [关键技术问题与解决方案](#11-关键技术问题与解决方案)
12. [系统运行指南](#12-系统运行指南)

---

## 1. 项目概述

### 1.1 问题定义

传统情感分析仅判断评论的**整体**情绪倾向（好评/差评），无法回答：

- 用户具体夸了产品的**哪个方面**？
- 用户具体吐槽了**什么属性**？
- 不同评价维度的**情感分布**如何？

**细粒度情感分析（ABSA, Aspect-Based Sentiment Analysis）** 解决的就是这个问题：从评论文本中自动识别 `(评价维度, 情感极性)` 结构化信息。

### 1.2 技术选型

| 决策项 | 选项 | 选择 | 理由 |
|--------|------|------|------|
| 预训练模型 | BERT vs RoBERTa | **RoBERTa-wwm-ext** | 哈工大发布，全词掩码(WWM)，中文 NLU 效果优于 BERT-base |
| 任务建模 | 序列标注 vs 多标签分类 | **多标签分类** | ASAP 数据集是类别级标注（无 token 级 span） |
| API 框架 | FastAPI vs Flask | **Flask** | 团队统一技术栈 |
| 数据 | 爬虫 vs 公开数据集 | **ASAP 数据集** | NAACL 2021 论文发布，46,730 条人工标注 |

### 1.3 为什么用多标签分类而不是序列标注？

ASAP 数据集的标注格式是 **类别级**的：

```
评论: "这家店味道很好，但服务太差"
标注: Food#Taste=1(正向), Service#Hospitality=-1(负向), 其余16维=-2(未提及)
```

它标注了**整个维度**的正/负/中，但没有标注具体的**观点词 span**（如"很好"在文本的哪个位置）。因此无法构造 BIO 序列标注数据，只能建模为：

> 给定评论文本 → 预测 18 个维度各自是否被提及（二分类）+ 被提及维度的情感极性（三分类）

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Web UI (FINAL/)                       │
│              Flask :5001 + Chart.js + HTML/CSS            │
│   sentiment_analyzer.py ── HTTP 客户端，调用模型 API      │
└──────────────────────┬──────────────────────────────────┘
                       │  HTTP POST/GET
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Model API (src/api/app.py)                   │
│                    Flask :5000                            │
│   /api/v1/analyze   单条分析                              │
│   /api/v1/analyze/batch  批量分析                         │
│   /api/v1/taxonomy  分类体系查询                          │
│   /api/v1/stats     统计报告                              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           推理流水线 (src/pipeline/analyzer.py)            │
│                                                          │
│  文本清洗 → 属性检测(18维多标签) → 情感分类(每维度3分类)    │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐   ┌──────────────────┐
│ AspectDetection  │   │ SentimentModel   │
│ RoBERTa + 18-sig │   │ RoBERTa + 3-soft │
└──────────────────┘   └──────────────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              models/ (训练好的权重)                        │
│   aspect_detection/  +  sentiment/                       │
└─────────────────────────────────────────────────────────┘
```

**架构说明**：系统采用 **两阶段流水线（Pipeline）** 设计，而非端到端联合模型。

- **阶段一**：属性检测 — 判断评论提到了哪些评价维度
- **阶段二**：情感分类 — 对已提及的每个维度独立判断情感极性

选择流水线而非联合模型的原因：
1. 两个任务可以独立优化、独立评估
2. 推理时只需对"已提及"的维度做情感分类，跳过未提及的维度，提升效率
3. 便于后续替换或升级单个模块

---

## 3. 数据预处理

### 3.1 数据集概述

ASAP (Aspect-based Sentiment Analysis for Pegged reviews) 是 NAACL 2021 发布的中文餐饮评论数据集：

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| train.csv | 36,850 | 训练集 |
| dev.csv | 4,940 | 验证集 |
| test.csv | 4,940 | 测试集 |

数据格式：

```
id, review, star,
Location#Transportation, Location#Downtown, ..., Food#Recommend
1, 这家店味道好位置方便, 5, 1, 1, ..., -2
```

**标签含义**：

| 值 | 含义 |
|----|------|
| `1` | 正向评价 |
| `0` | 中性评价 |
| `-1` | 负向评价 |
| `-2` | **未提及该维度** |

### 3.2 文本清洗

`src/preprocessing/cleaner.py` — 8 步清洗流水线：

```python
def clean_text(text: str, max_length: int = 512) -> str:
    # 1. 去除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)

    # 2. 去除 URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. 去除 @ 提及
    text = re.sub(r"@\S+", "", text)

    # 4. 转义换行符
    text = text.replace("\\n", " ").replace("\\t", " ")

    # 5. 全角转半角 (０→0, Ａ→A, ...)
    text = full_to_half(text)

    # 6. 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()

    # 7. 截断过长文本 (>512字符)
    if len(text) > max_length:
        text = text[:max_length]

    return text
```

**全角转半角实现**：

```python
def full_to_half(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0       # 全角→半角偏移
        elif code == 0x3000:      # 全角空格
            code = 0x0020
        result.append(chr(code))
    return "".join(result)
```

**有效性检查**：

```python
def is_valid_review(text: str, min_length: int = 5) -> bool:
    if len(text) < min_length:
        return False
    # 至少包含1个中文字符
    if not re.search(r"[一-鿿]", text):
        return False
    return True
```

**清洗效果**：

| 数据集 | 原始 | 清洗后 | 移除无效 |
|--------|------|--------|----------|
| train | 36,850 | 36,844 | 6 条（无中文/过短） |
| dev | 4,940 | 4,940 | 0 |
| test | 4,940 | 4,938 | 2 条 |

### 3.3 数据集构建

核心技巧：**同一个 CSV 数据被构建成两种不同格式的数据集**。

#### 属性检测数据集（AspectDetectionDataset）

一条评论 → 一个 18 维 binary 向量：

```python
class AspectDetectionDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        df = pd.read_csv(csv_path)
        self.texts = [clean_text(t) for t in df["review"]]

        # 构建多标签矩阵: (n_samples, 18)
        self.labels = []
        for _, row in df.iterrows():
            label_vec = []
            for cat in ASAP_CATEGORIES:
                val = int(row.get(cat, -2))
                # -2(未提及)→0, 其他→1
                label_vec.append(0 if val == -2 else 1)
            self.labels.append(label_vec)
        self.labels = torch.tensor(self.labels, dtype=torch.float)
```

#### 情感分类数据集（SentimentDataset）

一条评论 × 18 维度 → 展开为多条样本，只保留**已提及**的维度：

```python
class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        self.samples = []  # list of (text, category_name, polarity)
        for _, row in df.iterrows():
            text = clean_text(str(row["review"]))
            for cat in ASAP_CATEGORIES:
                val = int(row.get(cat, -2))
                if val != -2:  # 只保留已提及的维度
                    # 映射: 1→1(positive), 0→0(neutral), -1→2(negative)
                    label = 1 if val == 1 else (0 if val == 0 else 2)
                    self.samples.append((text, cat, label))
```

展开后训练集从 36,844 条 → **213,371 条**样本，极大扩充了情感分类的训练数据。

情感分类的关键设计：**输入不只是评论文本，而是 `[维度名, 评论文本]` 拼接**——这告诉模型"请从××维度的角度判断情感"：

```python
def __getitem__(self, idx):
    text, category, label = self.samples[idx]
    category_zh = CATEGORY_ZH.get(category, category)

    # 构建输入: [CLS] 维度名 [SEP] 评论文本 [SEP]
    encoding = self.tokenizer(
        category_zh,   # 第一个序列：维度中文名，如 "菜品口味"
        text,           # 第二个序列：评论文本
        max_length=256,
        padding="max_length",
        truncation=True,
    )
```

---

## 4. 属性分类体系

### 4.1 18 个评价维度 × 5 个一级维度

定义在 `src/config.py` 中：

```python
# 18 个评价维度（匹配 ASAP CSV 列名）
ASAP_CATEGORIES = [
    # 位置 (3维)
    "Location#Transportation",   # 交通便利
    "Location#Downtown",         # 是否在商圈
    "Location#Easy_to_find",     # 是否好找
    # 服务 (4维)
    "Service#Queue",             # 排队时间
    "Service#Hospitality",       # 服务态度
    "Service#Parking",           # 停车便利
    "Service#Timely",            # 上菜速度
    # 价格 (3维)
    "Price#Level",               # 价格水平
    "Price#Cost_effective",      # 性价比
    "Price#Discount",            # 优惠活动
    # 环境 (4维)
    "Ambience#Decoration",       # 装修风格
    "Ambience#Noise",            # 噪音水平
    "Ambience#Space",            # 空间大小
    "Ambience#Sanitary",         # 卫生状况
    # 菜品 (4维)
    "Food#Portion",              # 菜品分量
    "Food#Taste",                # 菜品口味
    "Food#Appearance",           # 菜品外观
    "Food#Recommend",            # 是否推荐
]

# 5 个一级维度分组
CATEGORY_GROUPS = {
    "位置": ["Location#Transportation", "Location#Downtown",
             "Location#Easy_to_find"],
    "服务": ["Service#Queue", "Service#Hospitality",
             "Service#Parking", "Service#Timely"],
    "价格": ["Price#Level", "Price#Cost_effective", "Price#Discount"],
    "环境": ["Ambience#Decoration", "Ambience#Noise",
             "Ambience#Space", "Ambience#Sanitary"],
    "菜品": ["Food#Portion", "Food#Taste",
             "Food#Appearance", "Food#Recommend"],
}
```

### 4.2 分类体系查询接口

`src/taxonomy/categories.py` 提供便捷查询：

```python
def get_category_zh(category: str) -> str:
    """获取维度中文名，如 'Food#Taste' → '菜品口味'"""
    return CATEGORY_ZH.get(category, category)

def get_group(category: str) -> str:
    """获取维度所属的一级维度"""
    for group_name, members in CATEGORY_GROUPS.items():
        if category in members:
            return group_name
    return "其他"
```

---

## 5. 模型设计

系统包含**两个独立模型**，共享 RoBERTa-wwm-ext 预训练权重但各自微调为不同的分类任务。

### 5.1 属性检测模型（AspectDetectionModel）

**任务**: 多标签二分类 — 给定评论文本，判断 18 个维度各自是否被提及。

**架构**:

```
RoBERTa-wwm-ext Encoder (12层Transformer, 768维hidden)
         │
    [CLS] token (聚合整句语义)
         │
    Dropout(0.1)
         │
    Linear(768, 256) → GELU 激活
         │
    Dropout(0.1)
         │
    Linear(256, 18) → Sigmoid (独立概率)
         │
    输出: 18维向量，每维 ∈ [0,1]，>0.5 表示提及
```

**核心代码**：

```python
class AspectDetectionModel(nn.Module):
    def __init__(self, pretrained_model=None, num_labels=None):
        super().__init__()
        # 加载 RoBERTa-wwm-ext 预训练权重
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.config.hidden_size  # 768

        # 分类头：768 → 256 → 18
        mid_dim = ASPECT_CONFIG["hidden_dim"]  # 256
        dropout = ASPECT_CONFIG["dropout"]      # 0.1

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_labels),  # 18
        )
        # BCEWithLogitsLoss = Sigmoid + BCE，数值更稳定
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask)
        # 取 [CLS] token 的 hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        logits = self.classifier(cls_embedding)  # (batch, 18)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)  # BCEWithLogitsLoss

        return {"loss": loss, "logits": logits}
```

**为什么用 BCEWithLogitsLoss？** 多标签分类中每个标签是独立的二分类。`BCEWithLogitsLoss = Sigmoid + BinaryCrossEntropy`，将 sigmoid 和 loss 合并计算，避免数值溢出，比分开写更稳定。

### 5.2 情感分类模型（SentimentModel）

**任务**: 三分类 — 给定（维度名, 评论文本），判断该维度的情感极性（positive/neutral/negative）。

**架构**:

```
[CLS] 维度中文名 [SEP] 评论文本 [SEP]
         │
RoBERTa-wwm-ext Encoder
         │
    [CLS] token
         │
    Dropout(0.1)
         │
    Linear(768, 3) → Softmax
         │
    输出: 3维概率分布
```

**核心代码**：

```python
class SentimentModel(nn.Module):
    def __init__(self, pretrained_model=None, num_labels=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.config.hidden_size  # 768

        # 分类头：768 → 3（正/中/负）
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),  # 3
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        logits = self.classifier(cls_embedding)  # (batch, 3)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
```

**关键设计点**：情感分类模型的输入并非单纯评论文本，而是 `[维度中文名] + [SEP] + [评论文本]` 的拼接。这引导模型学习**视角相关（aspect-aware）的情感判断**——同一条评论"味道好但服务差"，对"菜品口味"维度输出 positive，对"服务态度"维度输出 negative。

### 5.3 两个模型对比

| 特性 | 属性检测 | 情感分类 |
|------|----------|----------|
| 任务类型 | 多标签二分类 | 多分类 |
| 输出维度 | 18 (每维独立) | 3 (互斥) |
| 激活函数 | Sigmoid | Softmax |
| 损失函数 | BCEWithLogitsLoss | CrossEntropyLoss |
| 中间层 | 有(768→256) | 无(直接768→3) |
| 输入 | 纯文本 | 维度名+文本 |
| 有效 batch | 8×4=32 | 16×2=32 |

---

## 6. 模型训练

### 6.1 训练配置

针对 **RTX 4060 Laptop (8GB VRAM)** 的显存限制进行了专门优化：

```python
# 属性检测训练配置 (src/config.py)
ASPECT_CONFIG = {
    "num_labels": 18,
    "batch_size": 8,                    # 小批量
    "gradient_accumulation_steps": 4,   # 梯度累积 → 有效batch=32
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_ratio": 0.1,               # 前10%步数线性预热
    "fp16": True,                       # 混合精度训练，省一半显存
    "gradient_checkpointing": True,      # 用计算换显存
    "dropout": 0.1,
    "hidden_dim": 256,
}
```

**显存优化策略**：

| 策略 | 原理 | 效果 |
|------|------|------|
| `fp16=True` | 前向/反向用半精度浮点 | 显存减半 |
| `gradient_accumulation_steps=4` | 累积 4 个小 batch 的梯度再更新 | 等效 batch=32，但显存按 batch=8 计算 |
| `gradient_checkpointing=True` | 不保存中间激活，反向时重新计算 | 显存进一步降低 |

### 6.2 自定义 Trainer

Hugging Face 的 `Trainer` 默认用 `safetensors` 格式保存模型，但 safetensors 要求 tensor 连续（contiguous），自定义模型中部分权重可能不连续导致保存失败。通过覆盖 `_save` 方法解决：

```python
class AspectDetectionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        """用 torch.save 替代 safetensors，避免 non-contiguous 报错"""
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        for k in state_dict:
            if hasattr(state_dict[k], 'contiguous'):
                state_dict[k] = state_dict[k].contiguous()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
```

### 6.3 评估指标

训练过程中实时计算多标签分类指标：

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (n, 18), labels: (n, 18)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)

    micro_f1 = f1_score(labels, preds, average="micro")
    macro_f1 = f1_score(labels, preds, average="macro")
    subset_acc = accuracy_score(labels, preds)  # 完全匹配率

    # 每个维度的独立 F1
    per_category = {}
    for i, cat in enumerate(ASAP_CATEGORIES):
        per_category[f"f1_{cat}"] = f1_score(
            labels[:, i], preds[:, i], zero_division=0
        )
    return {"micro_f1": micro_f1, "macro_f1": macro_f1, **per_category}
```

### 6.4 模型保存策略

分类头与 Encoder 分开保存：

```python
# 保存 RoBERTa encoder + tokenizer（HuggingFace 标准格式）
model.encoder.save_pretrained(ASPECT_MODEL_PATH)
tokenizer.save_pretrained(ASPECT_MODEL_PATH)

# 单独保存分类头权重
torch.save(model.classifier.state_dict(),
           os.path.join(ASPECT_MODEL_PATH, "classifier.pt"))
```

分开保存的好处：分类头只有几 MB，Encoder 约 390MB。推理时先加载 encoder，再加载分类头权重覆盖。

---

## 7. 推理流水线

### 7.1 整体流程

`src/pipeline/analyzer.py` 中的 `ABSAAnalyzer` 类实现端到端推理：

```
用户输入文本
     │
     ▼
[Step 1] 文本清洗 (clean_text)
     │
     ▼
[Step 2] 属性检测 (AspectPredictor)
     │  18 维 sigmoid → 阈值 0.5 筛选
     │  输出: ["Food#Taste", "Service#Hospitality", ...]
     │
     ▼
[Step 3] 对每个已提及维度做情感分类 (SentimentPredictor)
     │  输入: "菜品口味 [SEP] 味道很好但服务太差"
     │  输出: positive/neutral/negative
     │
     ▼
[Step 4] 汇总统计 (_compute_summary)
     │  正/中/负计数，整体情感倾向
     │
     ▼
完整分析结果 JSON
```

### 7.2 核心代码

```python
class ABSAAnalyzer:
    def __init__(self):
        print("[1/2] 加载属性检测模型...")
        self.aspect_predictor = AspectPredictor()
        print("[2/2] 加载情感分类模型...")
        self.sentiment_predictor = SentimentPredictor()

    def analyze(self, text: str, review_id: str = None) -> dict:
        start_time = time.time()

        # Step 1: 文本清洗
        cleaned_text = clean_text(text)

        # Step 2: 属性检测 — 18维多标签推理
        aspect_result = self.aspect_predictor.predict(cleaned_text)
        categories = aspect_result["categories"]

        # Step 3: 对每个已提及的维度独立做情感分类
        aspects = []
        for cat_info in categories:
            sent_result = self.sentiment_predictor.predict(
                cleaned_text, cat_info["name"]
            )
            aspects.append({
                "category": cat_info["name"],
                "category_zh": cat_info["name_zh"],
                "group": get_group(cat_info["name"]),
                "sentiment": sent_result["sentiment"],
                "sentiment_zh": sent_result["sentiment_zh"],
                "aspect_confidence": cat_info["prob"],
                "sentiment_confidence": sent_result["confidence"],
            })

        # Step 4: 汇总统计
        summary = _compute_summary(aspects)

        return {
            "text": cleaned_text,
            "aspects": aspects,
            "summary": summary,
            "elapsed_seconds": round(time.time() - start_time, 3),
        }
```

### 7.3 属性检测推理器

```python
class AspectPredictor:
    @torch.no_grad()
    def predict(self, text: str, threshold: float = 0.5) -> dict:
        # Tokenize
        encoding = self.tokenizer(text, max_length=256,
                                  padding="max_length", truncation=True)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 推理
        _, probs = self.model.predict(input_ids, attention_mask, threshold)
        probs = probs[0].cpu().tolist()

        # 筛选概率 > 阈值的维度
        categories = []
        for i, (cat_name, prob) in enumerate(zip(ASAP_CATEGORIES, probs)):
            if prob >= threshold:
                categories.append({
                    "name": cat_name,
                    "name_zh": get_category_zh(cat_name),
                    "prob": round(prob, 4),
                })
        return {"categories": categories}
```

### 7.4 情感推理器

```python
class SentimentPredictor:
    @torch.no_grad()
    def predict(self, text: str, category: str) -> dict:
        category_zh = CATEGORY_ZH.get(category, category)

        # 关键：输入是 "维度名 + 文本" 的拼接
        encoding = self.tokenizer(
            category_zh, text,    # tokenizer(category, text) → [CLS]cat[SEP]text[SEP]
            max_length=256, padding="max_length", truncation=True
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        preds, probs = self.model.predict(input_ids, attention_mask)
        pred_label = preds[0].item()          # 0=neutral, 1=positive, 2=negative
        confidence = probs[0][pred_label].item()  # 预测置信度

        sentiment = ["neutral", "positive", "negative"][pred_label]

        return {
            "sentiment": sentiment,
            "sentiment_zh": SENTIMENT_ZH[sentiment],
            "confidence": round(confidence, 4),
        }
```

### 7.5 延迟导入（Lazy Import）

两个推理器的 `__init__` 和模型文件都使用了**延迟导入**模式，避免模块加载时立即导入大型依赖（transformers）：

```python
# 模块级变量，初始为 None
_transformers = None
_SentimentModel = None

def _get_transformers():
    global _transformers
    if _transformers is None:
        from transformers import AutoTokenizer  # 首次调用时才导入
        _transformers = AutoTokenizer
    return _transformers
```

这对 Flask 冷启动至关重要——能先返回健康检查响应，再在后台加载 400MB 的模型权重。

---

## 8. API 服务

### 8.1 API 路由定义

`src/api/app.py` 提供 6 个 RESTful 端点：

```python
app = Flask(__name__)
_analyzer = None       # 由外部注入（set_analyzer）
_aggregator = StatisticsAggregator()  # 统计累加器

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"code": 0, "status": "healthy"})

@app.route("/api/v1/taxonomy", methods=["GET"])
def taxonomy():
    """获取 18 维度分类体系（供前端渲染）"""
    return jsonify({"code": 0, "data": get_all_categories()})

@app.route("/api/v1/analyze", methods=["POST"])
def analyze():
    """单条评论分析"""
    data = request.get_json(force=True)
    text = data["text"]
    result = get_analyzer().analyze(text, data.get("review_id"))
    _aggregator.add_result(result)  # 累积到统计
    return jsonify({"code": 0, "data": result})

@app.route("/api/v1/analyze/batch", methods=["POST"])
def analyze_batch():
    """批量分析 → {"texts": [{"text": "..."}, ...]}"""
    data = request.get_json(force=True)
    results = get_analyzer().analyze_batch(data["texts"])
    _aggregator.add_batch(results)
    return jsonify({"code": 0, "data": results})

@app.route("/api/v1/stats", methods=["GET"])
def stats():
    """获取累积统计报告"""
    return jsonify({"code": 0, "data": _aggregator.compute_stats()})

@app.route("/api/v1/stats/reset", methods=["POST"])
def stats_reset():
    """重置统计"""
    _aggregator.reset()
    return jsonify({"code": 0, "message": "统计数据已重置"})
```

### 8.2 懒加载启动器

`scripts/run_api.py` 作为精简的启动入口：

```python
from src.api.app import app, set_analyzer

def init_analyzer():
    """启动前加载模型，确保首个请求无需等待"""
    from src.pipeline.analyzer import ABSAAnalyzer
    set_analyzer(ABSAAnalyzer())

if __name__ == "__main__":
    # 先加载模型，再启动服务
    init_analyzer()
    app.run(host="127.0.0.1", port=5000, threaded=True)
```

### 8.3 统计分析器

`src/analysis/aggregator.py` 累积所有分析结果，提供多维度统计：

```python
class StatisticsAggregator:
    def compute_stats(self) -> dict:
        # 收集所有 aspect
        all_aspects = []
        for r in self.results:
            all_aspects.extend(r.get("aspects", []))

        # 按一级维度统计
        group_stats = {}
        for group_name, members in CATEGORY_GROUPS.items():
            group_aspects = [a for a in all_aspects
                           if a["category"] in members]
            sc = Counter(a["sentiment"] for a in group_aspects)
            group_stats[group_name] = {
                "total": len(group_aspects),
                "positive_ratio": round(
                    sc.get("positive", 0) / len(group_aspects) * 100, 1
                ),
            }

        # 按具体维度排名（好评率从高到低）
        category_rankings = []
        for cat, stats in category_stats.items():
            category_rankings.append({
                "category_zh": CATEGORY_ZH.get(cat, cat),
                "positive_ratio": round(
                    stats["positive"] / stats["total"] * 100, 1
                ),
            })
        category_rankings.sort(key=lambda x: x["positive_ratio"],
                               reverse=True)

        return {
            "total_reviews": len(self.results),
            "total_aspects": len(all_aspects),
            "avg_aspects_per_review": round(
                len(all_aspects) / len(self.results), 1
            ),
            "group_stats": group_stats,
            "category_rankings": category_rankings,
        }
```

---

## 9. Web 可视化界面

### 9.1 前端架构

```
FINAL/
├── app.py                   # Flask Web 应用 (port 5001)
├── sentiment_analyzer.py    # HTTP 客户端，调用模型 API
├── main.py                  # 一键启动双服务
├── templates/index.html     # 前端页面 (~200 行)
├── static/
│   ├── css/style.css        # 样式 (~1100 行)
│   └── js/main.js           # 交互逻辑 (~1400 行)
```

### 9.2 双服务架构

```
main.py
  ├─ 启动 Model API (:5000)  — python scripts/run_api.py
  ├─ 等待 API 健康检查通过
  └─ 启动 Web UI (:5001)    — python FINAL/app.py
```

```python
# FINAL/main.py 核心逻辑
def main():
    # 启动模型 API
    api_proc = subprocess.Popen(
        [sys.executable, "scripts/run_api.py", "--port", "5000"]
    )

    # 轮询等待模型就绪
    if not wait_for_api("http://127.0.0.1:5000/api/v1/health"):
        print("模型启动失败!")
        sys.exit(1)

    # 启动 Web 界面
    web_proc = subprocess.Popen(
        [sys.executable, "FINAL/app.py"]
    )

    # 自动打开浏览器
    webbrowser.open("http://localhost:5001")
```

### 9.3 前端功能清单

| 功能 | 实现 |
|------|------|
| **模式切换** | 手动输入 / 文件上传两个选项卡 |
| **文件上传** | 拖拽 + 点击选择，支持 TXT/CSV/XLSX，多文件 |
| **进度条** | XHR + SSE 流式进度，实时显示百分比 |
| **概览卡片** | 总评论数、总维度数、平均维度/条、正向/负向评论 |
| **雷达图** | 5 个一级维度好评率雷达图 (Chart.js) |
| **柱状图** | 各维度正/中/负情感堆叠分布 |
| **饼图** | 整体情感比例 + 维度情感比例 |
| **评论列表** | 分页 + 关键词搜索 + 情感筛选 |
| **累计统计** | localStorage 持久化，跨会话累加 |
| **暂存列表** | 保存/加载/删除分析结果 |
| **导出** | Excel (.xlsx) + CSV，支持批量暂存导出 |

### 9.4 前端调用链

```
用户操作 (main.js)
    │
    ├─ handleTextAnalysis()  → POST /input-text
    │                          → 结果标准化 → displayResults()
    │
    ├─ handleFileUpload()    → POST /upload (单文件)
    │                         或 POST /batch-upload (多文件+SSE)
    │                          → displayResults()
    │
    └─ displayResults(data)
        ├─ updateSummary()      → 5张概览卡片
        ├─ updateCharts()       → Chart.js 渲染图表
        ├─ updateGroupDetail()  → 5个一级维度统计卡片
        └─ updateReviewsList()  → 分页评论列表
```

---

## 10. 实验评测

### 10.1 属性检测模型（18 维多标签分类）

测试集 4,938 条评论：

| 指标 | 值 |
|------|-----|
| **Micro-Precision** | 0.8555 |
| **Micro-Recall** | 0.8094 |
| **Micro-F1** | **0.8318** |
| **Macro-F1** | **0.7965** |

**Top 5 维度（F1 最高）**：

| 排名 | 维度 | F1 |
|------|------|-----|
| 1 | 菜品口味 | 0.9741 |
| 2 | 装修风格 | 0.8834 |
| 3 | 停车便利 | 0.8719 |
| 4 | 服务态度 | 0.8682 |
| 5 | 交通便利 | 0.8611 |

**Bottom 3 维度（F1 最低）**：

| 排名 | 维度 | F1 | 原因分析 |
|------|------|-----|----------|
| 16 | 性价比 | 0.6985 | 隐含评价多，显式提及少 |
| 17 | 是否推荐 | 0.6446 | 表达方式多样，难以统一建模 |
| 18 | 菜品外观 | 0.5602 | Recall 低 (0.41)，容易被遗漏 |

### 10.2 情感分类模型（3 分类）

测试集 28,356 条（已提及维度展开后）：

| 指标 | 值 |
|------|-----|
| **Accuracy** | **0.8089** |
| **Macro-F1** | **0.7503** |
| **Weighted-F1** | **0.8041** |

**各类别 F1**：

| 类别 | F1 | 支持数 | 分析 |
|------|-----|--------|------|
| 正向 | 0.8870 | 18,432 | 多数类，表现最好 |
| 负向 | 0.7223 | 7,201 | 可接受 |
| 中性 | 0.6418 | 2,723 | 少数类（仅 9.6%），表现最弱 |

### 10.3 推理性能

| 指标 | 值 |
|------|-----|
| 单条推理耗时 | ~0.25 秒 |
| 模型大小 | ~780MB (2×390MB) |
| GPU 显存占用 | ~3.5GB (推理) / ~7GB (训练) |
| 单条吞吐量 | ~4 条/秒 |

---

## 11. 关键技术问题与解决方案

### 问题 1: safetensors 保存 non-contiguous tensor 报错

**表现**: transformers 5.x 默认使用 safetensors 格式保存模型，safetensors 要求 tensor 内存连续，但自定义模型中的部分参数可能不连续。

**解决**: 覆盖 `Trainer._save()` 方法，手动 `.contiguous()` 后用 `torch.save` 保存：

```python
def _save(self, output_dir=None, state_dict=None):
    os.makedirs(output_dir, exist_ok=True)
    if state_dict is None:
        state_dict = self.model.state_dict()
    for k in state_dict:
        if hasattr(state_dict[k], 'contiguous'):
            state_dict[k] = state_dict[k].contiguous()
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
```

### 问题 2: gradient_checkpointing_enable 属性不存在

**表现**: HuggingFace Trainer 调用 `model.gradient_checkpointing_enable()`，但自定义模型包装了 encoder，该方法在模型类上不存在。

**解决**: 在自定义模型类中添加代理方法：

```python
class AspectDetectionModel(nn.Module):
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)
```

### 问题 3: 18 维度全部显示"装修风格"

**表现**: Web 界面的"18 维度好评率排名"图表中，所有维度的中文名都显示为"装修风格"。

**根因**: `sentiment_analyzer.py` 第 212 行写死了 `all_aspects[0].get("category_zh")`，导致所有维度的中文名都被取成了第一个维度的名称。

**修复**: 按当前 category 名称查找对应的 category_zh：

```python
# 错误: 所有维度都用第一个 aspect 的中文名
"category_zh": all_aspects[0].get("category_zh", cat) if all_aspects else cat

# 正确: 按当前 cat 查找对应的中文名
cat_zh = cat
for a in all_aspects:
    if a.get("category") == cat:
        cat_zh = a.get("category_zh", cat)
        break
```

### 问题 4: Flask 冷启动慢

**表现**: 首次请求需要等待 10+ 秒，因为模型加载阻塞了 Flask 的启动。

**解决**: 懒加载模式。将 `scripts/run_api.py` 分为两步：
1. 先导入 Flask，启动 HTTP 服务（秒级）
2. 再加载 PyTorch 模型（后台进行）

```python
# 路由文件只定义路由，不导入模型
from src.api.app import app, set_analyzer

# 启动前完成模型加载
def init_analyzer():
    from src.pipeline.analyzer import ABSAAnalyzer
    set_analyzer(ABSAAnalyzer())
```

### 问题 5: GitHub 100MB 文件限制

**表现**: 两个模型权重的 `.safetensors` 文件各约 390MB，GitHub push 被拒绝。

**解决**: 通过 `.gitignore` 排除 `models/` 和 `data/` 目录，模型通过 HuggingFace Hub 或本地路径加载。仓库只保留源码（~57 个文件）。

---

## 12. 系统运行指南

### 12.1 环境准备

```bash
# 创建 conda 环境
conda create -n NLPhomework python=3.10 -y
conda activate NLPhomework

# 安装依赖
pip install torch transformers datasets accelerate
pip install jieba flask pandas scikit-learn tqdm openpyxl
```

### 12.2 数据预处理

```bash
python scripts/preprocess_data.py
```

### 12.3 模型训练

```bash
# 一键训练两个模型
python scripts/train_all.py

# 或分别训练
python src/aspect_detection/train.py
python src/sentiment/train.py
```

### 12.4 模型评估

```bash
python scripts/evaluate.py
```

### 12.5 启动服务

```bash
# 方式一：一键启动（推荐）
python FINAL/main.py

# 方式二：分别启动
python scripts/run_api.py --port 5000    # 终端1：模型API
cd FINAL && python app.py                # 终端2：Web界面
```

### 12.6 测试 API

```bash
# 健康检查
curl http://127.0.0.1:5000/api/v1/health

# 单条分析
curl -X POST http://127.0.0.1:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "味道不错，服务态度很好，环境优雅"}'

# 批量分析
curl -X POST http://127.0.0.1:5000/api/v1/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": [{"text": "很好吃！"}, {"text": "服务太差了"}]}'
```

---

## 附录：项目文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/config.py` | 148 | 全局配置（路径、超参、18维度定义） |
| `src/preprocessing/cleaner.py` | 72 | 文本清洗（HTML/URL/全角半角） |
| `src/taxonomy/categories.py` | 50 | 分类体系查询接口 |
| `src/aspect_detection/dataset.py` | 91 | 多标签数据集构建 |
| `src/aspect_detection/model.py` | 95 | 属性检测模型（RoBERTa + 18-sigmoid） |
| `src/aspect_detection/train.py` | 167 | 训练脚本（含自定义Trainer） |
| `src/aspect_detection/predict.py` | 100 | 推理器（含延迟导入） |
| `src/sentiment/dataset.py` | 91 | 情感分类数据集（维度×评论展开） |
| `src/sentiment/model.py` | 92 | 情感分类模型（RoBERTa + 3-softmax） |
| `src/sentiment/train.py` | 167 | 训练脚本 |
| `src/sentiment/predict.py` | 102 | 推理器（维度名+文本拼接输入） |
| `src/pipeline/analyzer.py` | 147 | 端到端推理流水线 |
| `src/analysis/aggregator.py` | 115 | 多维度统计汇总 |
| `src/api/app.py` | 165 | Flask API 路由定义 |
| `scripts/run_api.py` | 50 | 懒加载启动器 |
| `scripts/evaluate.py` | 167 | 模型评测（P/R/F1） |
| `FINAL/app.py` | 160+ | Web UI 入口 (:5001) |
| `FINAL/main.py` | 117 | 一键启动双服务 |

---

> **报告日期**: 2026-06-17
> **项目仓库**: [github.com/Zhe-g/NLP-Course-Project](https://github.com/Zhe-g/NLP-Course-Project)
