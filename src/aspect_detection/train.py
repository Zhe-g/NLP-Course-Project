# -*- coding: utf-8 -*-
"""
属性检测模型训练脚本
适用于 RTX 4060 Laptop (8GB VRAM)
"""
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    PRETRAINED_MODEL,
    ASPECT_CONFIG,
    ASPECT_MODEL_PATH,
    ASAP_CATEGORIES,
)
from src.aspect_detection.dataset import load_asap_aspect_data
from src.aspect_detection.model import AspectDetectionModel


def compute_metrics(eval_pred):
    """计算多标签分类评估指标"""
    logits, labels = eval_pred
    # logits: (n, 18), labels: (n, 18)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)

    # Micro F1
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    # Macro F1
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    # Subset Accuracy (exact match)
    subset_acc = accuracy_score(labels, preds)

    # Per-category F1
    per_category = {}
    for i, cat in enumerate(ASAP_CATEGORIES):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        per_category[f"f1_{cat}"] = f1

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "subset_accuracy": subset_acc,
        **per_category,
    }


class AspectDetectionTrainer(Trainer):
    """自定义Trainer，处理model返回的dict格式输出，并用torch.save绕过safetensors"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        """用torch.save代替safetensors，避免non-contiguous报错"""
        import torch
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        # 确保所有tensor连续
        for k in state_dict:
            if hasattr(state_dict[k], 'contiguous'):
                state_dict[k] = state_dict[k].contiguous()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


def train():
    print("=" * 60)
    print("属性检测模型训练 (Multi-Label Aspect Detection)")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    train_dataset, dev_dataset, test_dataset = load_asap_aspect_data(tokenizer)
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(dev_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 创建模型
    print("\n[2/4] 创建模型...")
    model = AspectDetectionModel(pretrained_model=PRETRAINED_MODEL)
    print(f"  预训练模型: {PRETRAINED_MODEL}")
    print(f"  分类头: 18维多标签 (Sigmoid)")

    # 训练参数（已针对8GB显存优化）
    print("\n[3/4] 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=ASPECT_MODEL_PATH,
        per_device_train_batch_size=ASPECT_CONFIG["batch_size"],
        per_device_eval_batch_size=ASPECT_CONFIG["batch_size"],
        gradient_accumulation_steps=ASPECT_CONFIG["gradient_accumulation_steps"],
        num_train_epochs=ASPECT_CONFIG["num_epochs"],
        learning_rate=ASPECT_CONFIG["learning_rate"],
        warmup_ratio=ASPECT_CONFIG["warmup_ratio"],
        fp16=ASPECT_CONFIG["fp16"],
        gradient_checkpointing=False,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=ASPECT_CONFIG["eval_steps"],
        save_steps=ASPECT_CONFIG["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        report_to="none",
        save_only_model=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,  # 保留labels字段
    )

    # 打印训练配置
    effective_batch = ASPECT_CONFIG["batch_size"] * ASPECT_CONFIG["gradient_accumulation_steps"]
    print(f"  有效batch size: {effective_batch}")
    print(f"  混合精度(FP16): {ASPECT_CONFIG['fp16']}")
    print(f"  梯度检查点: {ASPECT_CONFIG['gradient_checkpointing']}")
    print(f"  学习率: {ASPECT_CONFIG['learning_rate']}")
    print(f"  训练轮数: {ASPECT_CONFIG['num_epochs']}")

    # 训练
    print("\n[4/4] 开始训练...")
    trainer = AspectDetectionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # 保存最终模型
    print("\n保存模型...")
    model.encoder.save_pretrained(ASPECT_MODEL_PATH)
    tokenizer.save_pretrained(ASPECT_MODEL_PATH)
    # 保存分类头权重
    torch.save(model.classifier.state_dict(), os.path.join(ASPECT_MODEL_PATH, "classifier.pt"))

    # 在测试集上评估
    print("\n测试集评估...")
    test_results = trainer.evaluate(test_dataset)
    print(f"  Micro F1: {test_results.get('eval_micro_f1', 'N/A'):.4f}")
    print(f"  Macro F1: {test_results.get('eval_macro_f1', 'N/A'):.4f}")
    print(f"  Subset Accuracy: {test_results.get('eval_subset_accuracy', 'N/A'):.4f}")

    print("\n训练完成！模型保存到:", ASPECT_MODEL_PATH)
    return trainer, test_results


if __name__ == "__main__":
    train()
