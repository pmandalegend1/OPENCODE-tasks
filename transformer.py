from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def fine_tune_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str | Path,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> None:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    import evaluate

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]].rename(columns={"label": "labels"}))
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]].rename(columns={"label": "labels"}))
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"],
        }

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def evaluate_transformer_model(
    model_dir: str | Path,
    test_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    threshold: float = 0.40,
) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    probas = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        proba = torch.softmax(logits, dim=-1)[:, 1].numpy()
        probas.extend(proba.tolist())

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.array(labels)
    probas_arr = np.array(probas)
    y_pred = (probas_arr >= threshold).astype(int)

    metrics = {
        "model": str(model_dir),
        "threshold": threshold,
        "precision_injection": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_injection": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_injection": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probas_arr)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
