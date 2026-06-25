"""Adversarial loop: detector and red-team generator evolve against each other.

For `iterations` rounds:
  1. Run the test set through the current detector, collect false negatives.
  2. Pass each false negative to the red-team generator to craft harder variants.
  3. Add successful evasions to the training set as hard negatives.
  4. Retrain the detector on the expanded dataset.
  5. Track attack success rate, detector F1/recall, and evasion-strategy diversity.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from prompt_injection_detector.model import Detector, _build_pipeline, _evaluate_pipeline
from prompt_injection_detector.redteam import run_redteam_batch


def _save_round_model(pipeline, threshold: float, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, path)


def run_adversarial_loop(
    dataset_path: str | Path,
    model_out: str | Path,
    iterations: int = 3,
    min_evasion_variants: int = 5,
    decision_threshold: float = 0.40,
    strategies: Optional[list[str]] = None,
    random_seed: int = 42,
    report_out: Optional[str | Path] = None,
) -> dict:
    df = pd.read_csv(dataset_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    history: list[dict] = []

    # Round 0: train an initial detector.
    pipeline = _build_pipeline("rf")
    pipeline.fit(train_df["text"].tolist(), train_df["label"].tolist())
    _save_round_model(pipeline, decision_threshold, model_out)
    detector = Detector(model_out)

    for it in range(1, iterations + 1):
        X_test = test_df["text"].tolist()
        y_test = test_df["label"].tolist()

        preds = detector.predict_batch(X_test)
        false_negatives = [
            text
            for text, label, pred in zip(X_test, y_test, preds)
            if label == 1 and not pred["is_injection"]
        ]

        attack_success_rate = (
            len(false_negatives) / max(1, sum(1 for l in y_test if l == 1))
        )

        # Red-team the missed injections to find harder variants.
        redteam_results = run_redteam_batch(
            false_negatives,
            detector,
            strategies=strategies,
            min_variants=min_evasion_variants,
            random_seed=random_seed + it,
        )

        hard_negatives: list[str] = []
        strategy_counter: Counter = Counter()
        for result in redteam_results:
            for variant in result.successful_evasions:
                hard_negatives.append(variant.text)
                strategy_counter[variant.strategy] += 1

        # Add hard negatives (still label=1, they are injections that evaded us).
        if hard_negatives:
            hard_df = pd.DataFrame(
                {
                    "text": hard_negatives,
                    "label": 1,
                    "category": "adversarial",
                    "split": "train",
                }
            )
            train_df = pd.concat([train_df, hard_df], ignore_index=True)

        # Retrain on expanded training set.
        pipeline = _build_pipeline("rf")
        pipeline.fit(train_df["text"].tolist(), train_df["label"].tolist())
        _save_round_model(pipeline, decision_threshold, model_out)
        detector = Detector(model_out)

        val_metrics = _evaluate_pipeline(
            pipeline,
            val_df["text"].tolist(),
            val_df["label"].tolist(),
            decision_threshold,
            label=f"round_{it}",
        )

        round_record = {
            "iteration": it,
            "attack_success_rate": attack_success_rate,
            "false_negatives_found": len(false_negatives),
            "hard_negatives_added": len(hard_negatives),
            "evasion_strategy_counts": dict(strategy_counter),
            "detector_f1_val": val_metrics["f1_injection"],
            "detector_recall_val": val_metrics["recall_injection"],
            "train_set_size": len(train_df),
        }
        history.append(round_record)

    final_test_metrics = _evaluate_pipeline(
        pipeline, test_df["text"].tolist(), test_df["label"].tolist(), decision_threshold, label="final"
    )

    report = {
        "iterations": iterations,
        "history": history,
        "final_test_metrics": final_test_metrics,
    }

    if report_out:
        Path(report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(report_out, "w") as f:
            json.dump(report, f, indent=2)

    return report


def plot_adversarial_history(history: list[dict], output_path: str | Path) -> None:
    """Plot attack success rate and detector F1/recall across iterations."""
    import matplotlib.pyplot as plt

    iterations = [h["iteration"] for h in history]
    asr = [h["attack_success_rate"] for h in history]
    f1 = [h["detector_f1_val"] for h in history]
    recall = [h["detector_recall_val"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(iterations, asr, marker="o", label="Attack success rate")
    ax.plot(iterations, f1, marker="s", label="Detector F1 (val)")
    ax.plot(iterations, recall, marker="^", label="Detector recall (val)")
    ax.set_xlabel("Adversarial loop iteration")
    ax.set_ylabel("Score")
    ax.set_title("Adversarial Loop: Detector vs Red-Team Generator")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
