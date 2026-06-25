from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from prompt_injection_detector.config import load_config
from prompt_injection_detector.data import build_dataset
from prompt_injection_detector.model import Detector, train as train_model
from prompt_injection_detector.adversarial import run_adversarial_loop, plot_adversarial_history
from prompt_injection_detector.robustness import run_full_robustness_suite
from prompt_injection_detector.redteam import run_redteam

app = typer.Typer(help="Prompt Injection Detector: build, train, attack, and evaluate.")
console = Console()


@app.command()
def build_data(
    config: Optional[str] = typer.Option(None, help="Path to a YAML config file."),
    output: Optional[str] = typer.Option(None, help="Output CSV path."),
    include_public: bool = typer.Option(False, help="Also pull in deepset/prompt-injections from HF."),
):
    """Generate the synthetic + (optionally) public labeled dataset."""
    cfg = load_config(config)
    out_path = output or cfg["paths"]["data_processed"]
    df = build_dataset(
        output=out_path,
        injection_samples=cfg["dataset"]["injection_samples"],
        clean_samples=cfg["dataset"]["clean_samples"],
        random_seed=cfg["dataset"]["random_seed"],
        include_public=include_public,
    )
    console.print(f"[green]Wrote {len(df)} rows to {out_path}[/green]")


@app.command()
def train(
    config: Optional[str] = typer.Option(None, help="Path to a YAML config file."),
    dataset: Optional[str] = typer.Option(None, help="Path to processed dataset CSV."),
    model_out: Optional[str] = typer.Option(None, help="Where to save the trained model (.joblib)."),
):
    """Train and compare classical ML models, save the best one."""
    cfg = load_config(config)
    dataset_path = dataset or cfg["paths"]["data_processed"]
    out_path = model_out or str(Path(cfg["paths"]["artifacts_dir"]) / "detector.joblib")

    metrics = train_model(
        dataset_path=dataset_path,
        model_out=out_path,
        decision_threshold=cfg["training"]["decision_threshold"],
        random_seed=cfg["training"]["random_seed"],
    )

    table = Table(title="Model comparison (validation)")
    table.add_column("Model")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("F1")
    table.add_column("ROC-AUC")
    for row in metrics["model_comparison"]:
        table.add_row(
            row["model"],
            f"{row['precision_injection']:.3f}",
            f"{row['recall_injection']:.3f}",
            f"{row['f1_injection']:.3f}",
            f"{row['roc_auc']:.3f}",
        )
    console.print(table)
    console.print(f"[green]Selected model: {metrics['selected_model']}[/green]")
    console.print(f"[green]Test recall: {metrics['recall_injection']:.3f}  "
                  f"Test F1: {metrics['f1_injection']:.3f}[/green]")
    console.print(f"Saved to {out_path}")


@app.command()
def redteam(
    text: str = typer.Argument(..., help="The injection text to generate evasions for."),
    model: Optional[str] = typer.Option(None, help="Path to trained model .joblib"),
    config: Optional[str] = typer.Option(None, help="Path to a YAML config file."),
    min_variants: Optional[int] = typer.Option(None, help="Minimum number of evasion variants."),
):
    """Generate evasion variants for a single piece of text and score them."""
    cfg = load_config(config)
    model_path = model or str(Path(cfg["paths"]["artifacts_dir"]) / "detector.joblib")
    detector = Detector(model_path)
    n_variants = min_variants or cfg["adversarial_loop"]["min_evasion_variants"]

    result = run_redteam(text, detector, min_variants=n_variants)

    table = Table(title="Red-team variants")
    table.add_column("Strategy")
    table.add_column("Bypassed?")
    table.add_column("Confidence")
    table.add_column("Variant text", overflow="fold")
    for v in result.variants:
        table.add_row(
            v.strategy,
            "[red]YES[/red]" if v.bypassed else "no",
            f"{v.confidence:.3f}",
            v.text[:100] + ("..." if len(v.text) > 100 else ""),
        )
    console.print(table)

    best = result.best_evasion
    if best:
        console.print(f"[yellow]Best evasion strategy: {best.strategy} "
                       f"(confidence dropped to {best.confidence:.3f})[/yellow]")
    else:
        console.print("[green]No variant bypassed the detector.[/green]")


@app.command()
def adversarial_loop(
    config: Optional[str] = typer.Option(None, help="Path to a YAML config file."),
    dataset: Optional[str] = typer.Option(None, help="Path to processed dataset CSV."),
    model_out: Optional[str] = typer.Option(None, help="Where to save the final model."),
    plot: bool = typer.Option(True, help="Save a plot of attack success rate / F1 / recall."),
):
    """Run the adversarial training loop: detector vs red-team generator."""
    cfg = load_config(config)
    dataset_path = dataset or cfg["paths"]["data_processed"]
    out_path = model_out or str(Path(cfg["paths"]["artifacts_dir"]) / "detector_adversarial.joblib")
    report_path = str(Path(cfg["paths"]["reports_dir"]) / "adversarial_loop.json")

    report = run_adversarial_loop(
        dataset_path=dataset_path,
        model_out=out_path,
        iterations=cfg["adversarial_loop"]["iterations"],
        min_evasion_variants=cfg["adversarial_loop"]["min_evasion_variants"],
        decision_threshold=cfg["training"]["decision_threshold"],
        strategies=cfg["redteam"]["strategies"],
        random_seed=cfg["training"]["random_seed"],
        report_out=report_path,
    )

    table = Table(title="Adversarial loop history")
    table.add_column("Iter")
    table.add_column("Attack success rate")
    table.add_column("Detector F1 (val)")
    table.add_column("Detector recall (val)")
    table.add_column("Hard negatives added")
    for row in report["history"]:
        table.add_row(
            str(row["iteration"]),
            f"{row['attack_success_rate']:.3f}",
            f"{row['detector_f1_val']:.3f}",
            f"{row['detector_recall_val']:.3f}",
            str(row["hard_negatives_added"]),
        )
    console.print(table)
    console.print(f"Saved report to {report_path}")

    if plot:
        plot_path = str(Path(cfg["paths"]["reports_dir"]) / "adversarial_loop.png")
        plot_adversarial_history(report["history"], plot_path)
        console.print(f"Saved plot to {plot_path}")


@app.command()
def robustness(
    config: Optional[str] = typer.Option(None, help="Path to a YAML config file."),
    dataset: Optional[str] = typer.Option(None, help="Path to processed dataset CSV."),
    model: Optional[str] = typer.Option(None, help="Path to trained model .joblib"),
):
    """Run the full robustness test suite (per-category + edge cases)."""
    cfg = load_config(config)
    dataset_path = dataset or cfg["paths"]["data_processed"]
    model_path = model or str(Path(cfg["paths"]["artifacts_dir"]) / "detector.joblib")
    report_path = str(Path(cfg["paths"]["reports_dir"]) / "robustness.json")

    report = run_full_robustness_suite(model_path, dataset_path, output_path=report_path)

    table = Table(title="Per-category detection rate")
    table.add_column("Category")
    table.add_column("N")
    table.add_column("Detection rate")
    for cat, stats in report["per_category_detection"].items():
        table.add_row(cat, str(stats["n"]), f"{stats['detection_rate']:.3f}")
    console.print(table)

    table2 = Table(title="Edge case detection rate")
    table2.add_column("Edge case")
    table2.add_column("N")
    table2.add_column("Detection rate")
    for case, stats in report["edge_case_summary"].items():
        table2.add_row(case, str(stats["n"]), f"{stats['detection_rate']:.3f}")
    console.print(table2)
    console.print(f"Saved report to {report_path}")


@app.command()
def predict(
    text: str = typer.Argument(..., help="Text to classify."),
    model: Optional[str] = typer.Option(None, help="Path to trained model .joblib"),
    config: Optional[str] = typer.Option(None, help="Path to a YAML config file."),
):
    """Classify a single piece of text as injection or clean."""
    cfg = load_config(config)
    model_path = model or str(Path(cfg["paths"]["artifacts_dir"]) / "detector.joblib")
    detector = Detector(model_path)
    result = detector.predict(text)
    console.print_json(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
