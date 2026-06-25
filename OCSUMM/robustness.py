"""Robustness testing: per-category detection rates and edge-case evasion tests.

Edge cases covered (matches configs/default.yaml -> robustness.edge_cases):
  - base64:    fully base64-encoded injection payloads
  - unicode:   Cyrillic look-alike character substitution
  - multiturn: the attack split across two separate messages
  - diluted:   the injection buried inside a long benign paragraph
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from prompt_injection_detector.model import Detector
from prompt_injection_detector.redteam import _UNICODE_LOOKALIKES

_DILUTION_PREFIX = (
    "I wanted to share some quick updates on what's been going on this week. "
    "The project is moving along nicely and the team has been very "
    "responsive to feedback. We also had a great offsite and everyone "
    "enjoyed catching up in person for once. "
)
_DILUTION_SUFFIX = (
    " Anyway, just wanted to give you a heads up, let me know if you have "
    "any questions or need anything else from me before the end of the week."
)


def make_base64_case(text: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"Decode this base64 string and do what it says: {encoded}"


def make_unicode_case(text: str) -> str:
    out = []
    for ch in text:
        lower = ch.lower()
        repl = _UNICODE_LOOKALIKES.get(lower)
        if repl:
            out.append(repl.upper() if ch.isupper() else repl)
        else:
            out.append(ch)
    return "".join(out)


def make_multiturn_case(text: str) -> list[str]:
    """Split the injection roughly in half across two simulated turns."""
    midpoint = max(1, len(text) // 2)
    split_at = text.rfind(" ", 0, midpoint)
    if split_at <= 0:
        split_at = midpoint
    return [text[:split_at].strip(), text[split_at:].strip()]


def make_diluted_case(text: str) -> str:
    return _DILUTION_PREFIX + text + _DILUTION_SUFFIX


def run_edge_case_tests(detector: Detector, injection_texts: list[str]) -> dict:
    results = {"base64": [], "unicode": [], "multiturn": [], "diluted": []}

    for text in injection_texts:
        b64_case = make_base64_case(text)
        pred = detector.predict(b64_case)
        results["base64"].append({"caught": pred["is_injection"], "confidence": pred["confidence"]})

        uni_case = make_unicode_case(text)
        pred = detector.predict(uni_case)
        results["unicode"].append({"caught": pred["is_injection"], "confidence": pred["confidence"]})

        diluted_case = make_diluted_case(text)
        pred = detector.predict(diluted_case)
        results["diluted"].append({"caught": pred["is_injection"], "confidence": pred["confidence"]})

        turn1, turn2 = make_multiturn_case(text)
        pred1 = detector.predict(turn1)
        pred2 = detector.predict(turn2)
        caught = pred1["is_injection"] or pred2["is_injection"]
        confidence = max(pred1["confidence"], pred2["confidence"])
        results["multiturn"].append({"caught": caught, "confidence": confidence})

    summary = {}
    for case_name, case_results in results.items():
        n = len(case_results)
        caught = sum(1 for r in case_results if r["caught"])
        summary[case_name] = {
            "n": n,
            "detection_rate": caught / n if n else 0.0,
        }

    return {"per_sample": results, "summary": summary}


def run_per_category_robustness(detector: Detector, test_df: pd.DataFrame) -> dict:
    """Detection rate per injection category on the held-out test set."""
    per_category = {}
    injections = test_df[test_df["label"] == 1]
    for cat, group in injections.groupby("category"):
        preds = detector.predict_batch(group["text"].tolist())
        caught = sum(1 for p in preds if p["is_injection"])
        per_category[cat] = {
            "n": len(group),
            "detection_rate": caught / len(group) if len(group) else 0.0,
        }
    return per_category


def run_full_robustness_suite(
    model_path: str | Path,
    dataset_path: str | Path,
    output_path: Optional[str | Path] = None,
    max_edge_case_samples: int = 100,
) -> dict:
    detector = Detector(model_path)
    df = pd.read_csv(dataset_path)
    test_df = df[df["split"] == "test"]

    per_category = run_per_category_robustness(detector, test_df)

    injection_texts = test_df[test_df["label"] == 1]["text"].tolist()[:max_edge_case_samples]
    edge_cases = run_edge_case_tests(detector, injection_texts)

    report = {
        "per_category_detection": per_category,
        "edge_case_summary": edge_cases["summary"],
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
