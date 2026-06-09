# -*- coding: utf-8 -*-
"""
情感极性分类模型训练脚本
"""
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    PRETRAINED_MODEL,
    SENTIMENT_CONFIG,
    SENTIMENT_MODEL_PATH,
)
from src.sentiment.dataset import load_asap_sentiment_data
from src.sentiment.model import SentimentModel

# 情感标签
SENTIMENT_NAMES = ["neutral", "positive", "negative"]


def compute_metrics(eval_pred):
    """计算情感分类评估指标"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


class SentimentTrainer(Trainer):
    """自定义Trainer，处理model返回的dict格式，并用torch.save绕过safetensors"""

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
        for k in state_dict:
            if hasattr(state_dict[k], 'contiguous'):
                state_dict[k] = state_dict[k].contiguous()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


def train():
    print("=" * 60)
    print("情感极性分类模型训练 (Sentiment Classification)")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    train_dataset, dev_dataset, test_dataset = load_asap_sentiment_data(tokenizer)
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(dev_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 创建模型
    print("\n[2/4] 创建模型...")
    model = SentimentModel(pretrained_model=PRETRAINED_MODEL)
    print(f"  预训练模型: {PRETRAINED_MODEL}")
    print(f"  分类头: 3分类 (neutral/positive/negative)")

    # 训练参数
    print("\n[3/4] 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=SENTIMENT_MODEL_PATH,
        per_device_train_batch_size=SENTIMENT_CONFIG["batch_size"],
        per_device_eval_batch_size=SENTIMENT_CONFIG["batch_size"],
        gradient_accumulation_steps=SENTIMENT_CONFIG["gradient_accumulation_steps"],
        num_train_epochs=SENTIMENT_CONFIG["num_epochs"],
        learning_rate=SENTIMENT_CONFIG["learning_rate"],
        warmup_ratio=SENTIMENT_CONFIG["warmup_ratio"],
        fp16=SENTIMENT_CONFIG["fp16"],
        gradient_checkpointing=False,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=SENTIMENT_CONFIG["eval_steps"],
        save_steps=SENTIMENT_CONFIG["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        save_only_model=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    effective_batch = SENTIMENT_CONFIG["batch_size"] * SENTIMENT_CONFIG["gradient_accumulation_steps"]
    print(f"  有效batch size: {effective_batch}")
    print(f"  混合精度(FP16): {SENTIMENT_CONFIG['fp16']}")
    print(f"  梯度检查点: {SENTIMENT_CONFIG['gradient_checkpointing']}")

    # 训练
    print("\n[4/4] 开始训练...")
    trainer = SentimentTrainer(
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
    model.encoder.save_pretrained(SENTIMENT_MODEL_PATH)
    tokenizer.save_pretrained(SENTIMENT_MODEL_PATH)
    torch.save(model.classifier.state_dict(), os.path.join(SENTIMENT_MODEL_PATH, "classifier.pt"))

    # 在测试集上评估
    print("\n测试集评估...")
    test_results = trainer.evaluate(test_dataset)
    print(f"  Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"  Macro F1: {test_results.get('eval_macro_f1', 'N/A'):.4f}")
    print(f"  Weighted F1: {test_results.get('eval_weighted_f1', 'N/A'):.4f}")

    print("\n训练完成！模型保存到:", SENTIMENT_MODEL_PATH)
    return trainer, test_results


if __name__ == "__main__":
    train()
