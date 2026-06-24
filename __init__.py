from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from prompt_injection_detector.data import CATEGORIES


def _build_pipeline(model_name: str) -> Pipeline:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=30000,
        sublinear_tf=True,
        analyzer="word",
        strip_accents="unicode",
    )
    char_vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        max_features=20000,
        analyzer="char_wb",
        sublinear_tf=True,
    )

    from sklearn.pipeline import FeatureUnion

    features = FeatureUnion([
        ("word", vectorizer),
        ("char", char_vectorizer),
    ])

    if model_name == "lr":
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs")
    elif model_name == "svm":
        clf = CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=False))
    else:
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)

    return Pipeline([("features", features), ("clf", clf)])


def _evaluate_pipeline(
    pipeline: Pipeline,
    X_test: list[str],
    y_test: list[int],
    threshold: float = 0.5,
    label: str = "",
) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "model": label,
        "precision_injection": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_injection": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_injection": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "false_negatives": int(((y_test == 1) & (y_pred == 0)).sum()),
        "false_positives": int(((y_test == 0) & (y_pred == 1)).sum()),
        "threshold": threshold,
    }


def train(
    dataset_path: str | Path,
    model_out: str | Path,
    decision_threshold: float = 0.40,
    random_seed: int = 42,
) -> dict:
    df = pd.read_csv(dataset_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = np.array(test_df["label"].tolist())

    comparison: list[dict] = []
    best_model: Optional[Pipeline] = None
    best_recall = -1.0
    best_name = ""

    for name in ["lr", "svm", "rf"]:
        pipeline = _build_pipeline(name)
        pipeline.fit(X_train, y_train)

        val_metrics = _evaluate_pipeline(pipeline, X_val, y_val, decision_threshold, label=name)
        comparison.append(val_metrics)

        if val_metrics["recall_injection"] > best_recall or (
            val_metrics["recall_injection"] == best_recall
            and val_metrics["f1_injection"] > (best_model and comparison[-2].get("f1_injection", 0) or 0)
        ):
            best_recall = val_metrics["recall_injection"]
            best_model = pipeline
            best_name = name

    best_model.fit(X_train + X_val, y_train + y_val)

    proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (proba_test >= decision_threshold).astype(int)

    per_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        mask = test_df["category"] == cat
        if mask.sum() == 0:
            continue
        cat_proba = proba_test[mask.values]
        cat_pred = (cat_proba >= decision_threshold).astype(int)
        cat_true = y_test[mask.values]
        per_category[cat] = {
            "detection_rate": float((cat_pred == cat_true).mean()),
            "recall": float(recall_score(cat_true, cat_pred, zero_division=0)),
            "count": int(mask.sum()),
        }

    cm = confusion_matrix(y_test, y_pred_test).tolist()
    report = classification_report(y_test, y_pred_test, output_dict=True)

    metrics = {
        "selected_model": best_name,
        "threshold": decision_threshold,
        "precision_injection": float(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)),
        "recall_injection": float(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)),
        "f1_injection": float(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "confusion_matrix": cm,
        "classification_report": report,
        "per_category": per_category,
        "model_comparison": comparison,
    }

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_model, "threshold": decision_threshold}, model_out)

    return metrics


class Detector:
    def __init__(self, model_path: str | Path):
        payload = joblib.load(model_path)
        self.pipeline: Pipeline = payload["pipeline"]
        self.threshold: float = payload["threshold"]

    def predict(self, text: str) -> dict:
        proba = float(self.pipeline.predict_proba([text])[0, 1])
        is_injection = proba >= self.threshold
        category = self._infer_category(text) if is_injection else "clean"
        top_features = self._top_features(text)
        return {
            "is_injection": is_injection,
            "confidence": proba,
            "category": category,
            "top_features": top_features,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        probas = self.pipeline.predict_proba(texts)[:, 1]
        results = []
        for text, proba in zip(texts, probas):
            is_injection = proba >= self.threshold
            results.append({
                "text": text,
                "is_injection": bool(is_injection),
                "confidence": float(proba),
                "category": self._infer_category(text) if is_injection else "clean",
            })
        return results

    def _infer_category(self, text: str) -> str:
        from prompt_injection_detector.data import _infer_category
        return _infer_category(text)

    def _top_features(self, text: str, top_n: int = 10) -> list[str]:
        try:
            features_step = self.pipeline.named_steps["features"]
            last_step = self.pipeline.named_steps["clf"]

            transformed = features_step.transform([text])
            if hasattr(last_step, "coef_"):
                coefs = last_step.coef_[0]
                feature_names: list[str] = []
                for _, transformer in features_step.transformer_list:
                    feature_names.extend(transformer.get_feature_names_out().tolist())
                indices = np.argsort(np.abs(coefs))[::-1][:top_n]
                return [feature_names[i] for i in indices if i < len(feature_names)]
        except Exception:
            pass
        return []
