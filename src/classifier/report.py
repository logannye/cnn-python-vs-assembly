"""Build a retained, quantitative report from completed benchmark seed runs."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METRICS = {
    "accuracy": ("Top-1 accuracy", True),
    "balanced_accuracy": ("Balanced accuracy", True),
    "macro_precision": ("Macro precision", True),
    "macro_recall": ("Macro recall", True),
    "macro_f1": ("Macro F1", True),
    "weighted_f1": ("Weighted F1", True),
    "top3_accuracy": ("Top-3 accuracy", True),
    "top5_accuracy": ("Top-5 accuracy", True),
    "negative_log_likelihood": ("Negative log likelihood", False),
    "brier_score": ("Multiclass Brier score", False),
    "ece": ("Expected calibration error", True),
    "mce": ("Maximum calibration error", True),
    "mean_confidence": ("Mean confidence", True),
    "confidence_correct": ("Confidence when correct", True),
    "confidence_incorrect": ("Confidence when incorrect", True),
    "multiclass_roc_auc_ovr_macro": ("Macro one-vs-rest ROC AUC", False),
    "average_precision_macro": ("Macro average precision", False),
}


def _read(path: Path):
    return json.loads(path.read_text())


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _flatten_scalars(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten_scalars(item, name))
        elif _numeric(item):
            result[name] = item
    return result


def _aggregate(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def _format(value, percent: bool = False) -> str:
    if not _numeric(value):
        return "—"
    return f"{100 * value:.2f}%" if percent else f"{value:.4f}"


def _mib(value) -> str:
    return _format(value / 2**20) if _numeric(value) else "—"


def _ci(value: dict | None, percent: bool = True) -> str:
    if not value:
        return "—"
    return f"[{_format(value.get('lower'), percent)}, {_format(value.get('upper'), percent)}]"


def _load_runs(output_dir: Path) -> list[dict]:
    runs = []
    for directory in output_dir.glob("seed-*"):
        if not directory.is_dir() or not directory.name[5:].isdigit():
            continue
        required = [
            directory / name for name in ("metrics.json", "performance.json", "history.json")
        ]
        if not all(path.is_file() for path in required):
            continue
        metrics, performance, history = map(_read, required)
        if "test" not in metrics or not history:
            continue
        runs.append(
            {
                "seed": int(directory.name[5:]),
                "directory": directory,
                "metrics": metrics,
                "performance": performance,
                "history": history,
            }
        )
    return sorted(runs, key=lambda run: run["seed"])


def _save_figure(fig, directory: Path, name: str) -> str:
    fig.savefig(directory / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"


def _learning_curves(runs: list[dict], directory: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for index, run in enumerate(runs):
        history = run["history"]
        epochs = [row["epoch"] for row in history]
        color = colors[index % len(colors)]
        for ax, metric, scale in zip(axes, ("loss", "accuracy"), (1, 100), strict=True):
            ax.plot(
                epochs,
                [scale * row[f"train_{metric}"] for row in history],
                color=color,
                linestyle="--",
                label=f"Seed {run['seed']} train",
            )
            ax.plot(
                epochs,
                [scale * row[f"val_{metric}"] for row in history],
                color=color,
                label=f"Seed {run['seed']} validation",
            )
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_ylabel("Accuracy (%)")
    axes[0].set_title("Loss during training")
    axes[1].set_title("Accuracy during training")
    axes[1].legend(fontsize=8)
    return _save_figure(fig, directory, "learning_curves.png")


def _confusion_plot(test: dict, directory: Path, seed: int) -> str:
    matrix = np.asarray(test["confusion_matrix"], dtype=float)
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix), where=support != 0)
    names = test["class_names"]
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    display = ax.imshow(100 * normalized, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Test confusion matrix · seed {seed}\nRows normalized to 100%")
    for row in range(len(names)):
        for column in range(len(names)):
            value = 100 * normalized[row, column]
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 55 else "black",
            )
    fig.colorbar(display, ax=ax, label="Percentage of actual class")
    return _save_figure(fig, directory, "confusion_matrix.png")


def _reliability_plot(test: dict, directory: Path, seed: int) -> str:
    bins = test["calibration_bins"]
    nonempty = [item for item in bins if item["count"] > 0]
    fig, axes = plt.subplots(
        2, 1, figsize=(6.5, 7), sharex=True, height_ratios=[3, 1], layout="constrained"
    )
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", label="Perfect calibration")
    axes[0].plot(
        [item["confidence"] for item in nonempty],
        [item["accuracy"] for item in nonempty],
        "o-",
        color="#176b91",
        label="Observed test accuracy",
    )
    axes[0].set(xlim=(0, 1), ylim=(0, 1), ylabel="Fraction correct")
    axes[0].set_title(f"Test reliability · seed {seed}")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)
    axes[1].bar(
        [(item["lower"] + item["upper"]) / 2 for item in bins],
        [item["count"] for item in bins],
        width=[0.9 * (item["upper"] - item["lower"]) for item in bins],
        color="#176b91",
    )
    axes[1].set(xlabel="Predicted top-class probability", ylabel="Images")
    return _save_figure(fig, directory, "reliability.png")


def _class_f1_plot(runs: list[dict], directory: Path) -> str:
    names = runs[0]["metrics"]["test"]["class_names"]
    values = np.asarray(
        [[row["f1"] for row in run["metrics"]["test"]["per_class"]] for run in runs]
    )
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    x = np.arange(len(names))
    errors = 100 * values.std(axis=0, ddof=1) if len(runs) > 1 else None
    ax.bar(x, 100 * values.mean(axis=0), yerr=errors, capsize=4, color="#176b91", alpha=0.8)
    if len(runs) > 1:
        for row in values:
            ax.scatter(x, 100 * row, s=18, color="#d97706", zorder=3)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set(ylabel="F1 (%)", ylim=(0, 100))
    ax.set_title(
        "Test per-class F1 · mean ± seed sample SD" if len(runs) > 1 else "Test per-class F1"
    )
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(fig, directory, "per_class_f1.png")


def _seed_plot(runs: list[dict], directory: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    x = np.arange(len(runs))
    for offset, split, color in ((-0.18, "validation", "#176b91"), (0.18, "test", "#d97706")):
        ax.bar(
            x + offset,
            [100 * run["metrics"][split]["accuracy"] for run in runs],
            width=0.34,
            label=split.title(),
            color=color,
        )
    ax.set_xticks(x, [str(run["seed"]) for run in runs])
    ax.set(xlabel="Training seed", ylabel="Accuracy (%)", ylim=(0, 100))
    ax.set_title("Selected-checkpoint accuracy by seed")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    return _save_figure(fig, directory, "accuracy_seed_comparison.png")


def _summary(runs: list[dict], representative_seed: int, figures: list[str]) -> dict:
    per_seed = []
    for run in runs:
        best = max(run["history"], key=lambda row: (row["val_accuracy"], -row["val_loss"]))
        per_seed.append(
            {
                "seed": run["seed"],
                "selected_epoch": run["performance"].get("best_epoch", best["epoch"]),
                **_flatten_scalars(run["metrics"]),
                **_flatten_scalars(run["performance"], "performance"),
            }
        )
    keys = sorted(
        key
        for key in set().union(*(row.keys() for row in per_seed))
        if not any(part.endswith("_ci95") for part in key.split("."))
        and key.rsplit(".", 1)[-1] not in {"seed", "resamples", "confidence_level"}
    )
    aggregates = {key: _aggregate([row[key] for row in per_seed if key in row]) for key in keys}
    return {
        "schema_version": 1,
        "seed_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "representative_seed": representative_seed,
        "aggregate": aggregates,
        "per_seed": per_seed,
        "artifacts": {"report": "report.md", "figures": figures},
    }


def _metric_table(summary: dict, split: str) -> list[str]:
    lines = ["| Metric | Mean | Sample SD | Minimum | Maximum |", "|---|---:|---:|---:|---:|"]
    for key, (label, percent) in METRICS.items():
        stats = summary["aggregate"].get(f"{split}.{key}")
        if stats is None:
            continue
        values = " | ".join(
            _format(stats[name], percent) for name in ("mean", "sample_std", "min", "max")
        )
        lines.append(f"| {label} | {values} |")
    return lines


def _report_lines(
    manifest: dict, runs: list[dict], representative: dict, summary: dict
) -> list[str]:
    seed = representative["seed"]
    test = representative["metrics"]["test"]
    protocol = manifest.get("protocol", {})
    expected = protocol.get("seeds", summary["seeds"])
    lines = [
        "# Simple CNN · CIFAR-10 baseline",
        "",
        f"This report contains **{len(runs)} completed training seeds**: "
        + ", ".join(str(run["seed"]) for run in runs)
        + ". "
        + f"Detailed example-level diagnostics use representative seed **{seed}** "
        + "(seed 42 when available; otherwise the lowest completed seed).",
        "",
    ]
    missing = sorted(set(expected) - set(summary["seeds"]))
    if missing:
        lines += [
            f"**Incomplete run set:** planned seeds {missing} are missing from this report.",
            "",
        ]
    lines += [
        "## Evaluation protocol",
        "",
        "The CNN is trained from randomly initialized weights, with no pretraining. "
        "The official CIFAR-10 training set is split into training and validation subsets; "
        "the official test set remains separate. "
        "The same saved split is used across training seeds. "
        "Within each seed, the checkpoint is selected by highest validation accuracy, "
        "breaking ties by lower validation loss. Test results do not select checkpoints or seeds.",
        "",
        "Final train, validation, and test metrics below evaluate the selected checkpoint with "
        "evaluation preprocessing. Epoch training metrics use augmented images and mix "
        "successive model states, so they need not equal final training metrics.",
        "",
        "| Split | Images (representative seed) |",
        "|---|---:|",
    ]
    for split in ("train", "validation", "test"):
        lines.append(f"| {split.title()} | {representative['metrics'][split]['num_examples']:,} |")
    lines += [
        "",
        "Uniform random top-1 guessing has expected accuracy **10%** for this 10-class benchmark. "
        "This is a reference point, not a trained competing model.",
        "",
        "## Test performance across seeds",
        "",
        "Every seed uses the same test images. Means and sample standard deviations describe "
        "variation across training seeds; repeated test predictions are not independent test sets. "
        "Standard deviations of percentage metrics are shown in percentage points. "
        "A small number of seeds provides only a preliminary estimate of training variability.",
        "",
        *_metric_table(summary, "test"),
        "",
        "Lower negative log likelihood, Brier score, ECE, and MCE are better. "
        "Brier score sums squared probability error over classes, then averages over images. "
        "ECE is the count-weighted absolute accuracy–confidence gap over 15 equal-width bins; "
        "MCE is the largest nonempty-bin gap. Both depend on the binning. ROC AUC and average "
        "precision use one-vs-rest scores averaged equally over classes. "
        "Undefined metrics appear as —.",
        "",
        "## Individual runs and uncertainty",
        "",
        "| Seed | Selected epoch | Validation accuracy | Test accuracy | "
        "Accuracy 95% CI | Macro F1 | Macro F1 95% CI |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for run, row in zip(runs, summary["per_seed"], strict=True):
        metrics = run["metrics"]
        result = metrics["test"]
        lines.append(
            f"| {run['seed']} | {row['selected_epoch']} | "
            f"{_format(metrics['validation']['accuracy'], True)} | "
            f"{_format(result['accuracy'], True)} | {_ci(result.get('accuracy_ci95'))} | "
            f"{_format(result['macro_f1'], True)} | {_ci(result.get('macro_f1_ci95'))} |"
        )
    bootstrap = test.get("macro_f1_ci95", {})
    lines += [
        "",
        "Accuracy intervals use the 95% Wilson method. Macro F1 intervals use a percentile "
        f"bootstrap over test examples ({bootstrap.get('resamples', 'recorded')} resamples; "
        "the random seed and exact method are saved in each metrics.json). These intervals "
        "describe test-sample uncertainty conditional on a fitted model and an independent, "
        "representative sampling assumption; they do not include training-seed variability or "
        "distribution shift. They are not intervals for the cross-seed mean.",
        "",
        "## Fit and learning curves",
        "",
        "| Seed | Final train accuracy | Validation accuracy | Test accuracy | "
        "Train − validation gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        metrics = run["metrics"]
        gap = metrics["train"]["accuracy"] - metrics["validation"]["accuracy"]
        lines.append(
            f"| {run['seed']} | {_format(metrics['train']['accuracy'], True)} | "
            f"{_format(metrics['validation']['accuracy'], True)} | "
            f"{_format(metrics['test']['accuracy'], True)} | {100 * gap:.2f} pp |"
        )
    lines += ["", "![Loss and accuracy by epoch](figures/learning_curves.png)", ""]
    if len(runs) > 1:
        lines += [
            "![Validation and test accuracy by seed](figures/accuracy_seed_comparison.png)",
            "",
        ]
    lines += [
        "## Class-level performance",
        "",
        f"The table reports seed {seed}; the figure summarizes all completed seeds.",
        "",
        "| Class | Support | Precision | Recall | F1 | ROC AUC | Average precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test["per_class"]:
        lines.append(
            f"| {row['class']} | {row['support']} | {_format(row['precision'], True)} | "
            f"{_format(row['recall'], True)} | {_format(row['f1'], True)} | "
            f"{_format(row.get('roc_auc'))} | {_format(row.get('average_precision'))} |"
        )
    lines += [
        "",
        "![Per-class F1](figures/per_class_f1.png)",
        "",
        "![Normalized confusion matrix](figures/confusion_matrix.png)",
        "",
        f"Most frequent directed errors for seed {seed}:",
        "",
        "| Actual → predicted | Images | Percentage of actual class |",
        "|---|---:|---:|",
    ]
    matrix = np.asarray(test["confusion_matrix"])
    pairs = sorted(
        (
            (int(matrix[i, j]), i, j)
            for i in range(len(matrix))
            for j in range(len(matrix))
            if i != j
        ),
        reverse=True,
    )
    names = test["class_names"]
    for count, actual, predicted in pairs[:10]:
        if count > 0:
            lines.append(
                f"| {names[actual]} → {names[predicted]} | {count} | "
                f"{100 * count / matrix[actual].sum():.2f}% |"
            )
    lines += [
        "",
        "## Confidence and calibration",
        "",
        f"For seed {seed}, mean top-class confidence is "
        f"{_format(test.get('mean_confidence'), True)}. "
        f"It is {_format(test.get('confidence_correct'), True)} on correct predictions and "
        f"{_format(test.get('confidence_incorrect'), True)} on incorrect predictions. "
        "Confidence is a model probability, not a guarantee of correctness. "
        "The reliability plot shows observed accuracy against mean confidence "
        "in each nonempty bin; the lower panel shows how many images fall in each bin. "
        "No post-hoc calibration is fitted.",
        "",
        "![Reliability plot and confidence distribution](figures/reliability.png)",
        "",
        "## Runtime and resource use",
        "",
        "| Seed | Train (s) | Final evaluation (s) | Parameters | "
        "Checkpoint (MiB) | Peak process RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        perf = run["performance"]
        lines.append(
            f"| {run['seed']} | {_format(perf.get('training_seconds'))} | "
            f"{_format(perf.get('final_evaluation_seconds'))} | "
            f"{perf.get('parameter_count', '—')} | "
            f"{_mib(perf.get('checkpoint_bytes'))} | "
            f"{_mib(perf.get('peak_rss_bytes'))} |"
        )
    lines += [
        "",
        "| Seed | Batch size | Median batch latency (ms) | P95 batch latency (ms) | Images/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        for batch, perf in run["performance"].get("inference", {}).items():
            lines.append(
                f"| {run['seed']} | {batch.removeprefix('batch')} | "
                f"{_format(perf.get('median_ms'))} | {_format(perf.get('p95_ms'))} | "
                f"{_format(perf.get('images_per_second'))} |"
            )
    lines += [
        "",
        "Timing is specific to the recorded host, device, thread count, batch size, warmup, and "
        "measurement procedure. Hosted runners can differ between seeds; consult each seed's "
        "environment in manifest.json before comparing speed. Latency measurements cover the "
        "model forward pass described in performance.json, "
        "not an end-to-end image-serving pipeline. "
        "Peak RSS is the process high-water mark, including data and evaluation allocations; "
        "it is not an isolated model-memory measurement. Per-epoch training and validation timings "
        "are retained in history.json.",
        "",
        "## Retained evidence and future comparisons",
        "",
        "- [Machine-readable summary](summary.json): "
        "per-seed scalar values and cross-seed aggregates.",
        "- [Manifest](manifest.json): source revision, protocol, configuration, package versions, "
        "and hardware/software environment.",
        "- [Split indices](split_indices.npz): exact training and validation memberships.",
    ]
    for run in runs:
        prefix = run["directory"].name
        lines += [
            f"- Seed {run['seed']}: [metrics]({prefix}/metrics.json), "
            f"[history]({prefix}/history.json), [performance]({prefix}/performance.json), "
            f"[checkpoint]({prefix}/best.pt), "
            f"[train predictions]({prefix}/predictions_train.npz), "
            f"[validation predictions]({prefix}/predictions_validation.npz), "
            f"[test predictions]({prefix}/predictions_test.npz).",
        ]
    lines += [
        "",
        "Prediction archives retain labels, class probabilities, and example indices. "
        "A future model can use these same splits, seeds, evaluation code, and indexed test "
        "examples for paired comparisons. Fix the next experiment's protocol before examining "
        "its test results, select checkpoints using validation only, and report both paired "
        "test-example uncertainty and variability across training seeds. Repeated experiment "
        "selection based on this test set will turn it into development data.",
        "",
        f"This is a small CNN baseline at a fixed {protocol.get('epochs', 'recorded')}-epoch "
        "training budget, not an estimate of the architecture's maximum attainable accuracy. "
        "It does not establish "
        "optimal hyperparameters, convergence, state-of-the-art performance, robustness to "
        "distribution shift, or production suitability.",
        "",
        "## Recorded protocol",
        "",
        "```json",
        json.dumps(protocol, indent=2, sort_keys=True),
        "```",
        "",
        "## Source provenance",
        "",
        "```json",
        json.dumps(manifest.get("source", {}), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return lines


def build_report(output_dir: Path) -> None:
    """Write report.md, summary.json, and plots using only retained run results."""
    output_dir = Path(output_dir)
    manifest = _read(output_dir / "manifest.json")
    runs = _load_runs(output_dir)
    if not runs:
        raise ValueError(
            "No completed seed runs found: expected metrics, performance, and history."
        )
    representative = next((run for run in runs if run["seed"] == 42), runs[0])
    directory = output_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    test = representative["metrics"]["test"]
    figures = [
        _learning_curves(runs, directory),
        _confusion_plot(test, directory, representative["seed"]),
        _reliability_plot(test, directory, representative["seed"]),
        _class_f1_plot(runs, directory),
    ]
    if len(runs) > 1:
        figures.append(_seed_plot(runs, directory))
    summary = _summary(runs, representative["seed"], figures)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = _report_lines(manifest, runs, representative, summary)
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.output_dir)


if __name__ == "__main__":
    main()
