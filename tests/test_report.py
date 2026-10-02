"""Report artifact checks with small, explicitly synthetic seed results."""

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from classifier.report import build_report  # noqa: E402


def _write_seed(directory: Path, seed: int, correct_per_class: int) -> None:
    directory.mkdir()
    names = [f"class-{index}" for index in range(10)]
    accuracy = correct_per_class / 10
    confusion = [[0] * 10 for _ in range(10)]
    for index in range(10):
        confusion[index][index] = correct_per_class
        confusion[index][(index + 1) % 10] = 10 - correct_per_class
    result = {
        "num_examples": 100,
        "num_classes": 10,
        "class_names": names,
        "accuracy": accuracy,
        "macro_f1": accuracy,
        "confidence_incorrect": None,
        "confusion_matrix": confusion,
        "accuracy_ci95": {"lower": 0.3, "upper": 0.9, "method": "wilson"},
        "macro_f1_ci95": {
            "lower": 0.2,
            "upper": 0.9,
            "method": "percentile_multinomial_confusion_bootstrap",
            "resamples": 1000,
            "seed": seed,
        },
        "per_class": [
            {
                "class": name,
                "support": 10,
                "precision": accuracy,
                "recall": accuracy,
                "f1": accuracy,
                "roc_auc": None,
                "average_precision": None,
            }
            for name in names
        ],
        "calibration_bins": [
            {"lower": 0, "upper": 0.5, "count": 0, "accuracy": None, "confidence": None},
            {"lower": 0.5, "upper": 1, "count": 100, "accuracy": accuracy, "confidence": 0.7},
        ],
    }
    (directory / "metrics.json").write_text(
        json.dumps({split: result for split in ("train", "validation", "test")})
    )
    (directory / "performance.json").write_text(
        json.dumps(
            {
                "training_seconds": 10,
                "final_evaluation_seconds": 2,
                "parameter_count": 100,
                "checkpoint_bytes": 1024,
                "peak_rss_bytes": 1048576,
                "inference": {"batch1": {"median_ms": 1, "p95_ms": 2, "images_per_second": 900}},
            }
        )
    )
    (directory / "history.json").write_text(
        json.dumps(
            [
                {
                    "epoch": epoch,
                    "train_loss": 1 / epoch,
                    "train_accuracy": accuracy,
                    "val_loss": 0.9 / epoch,
                    "val_accuracy": accuracy,
                }
                for epoch in (1, 2)
            ]
        )
    )


@pytest.mark.parametrize("seeds", [(42,), (42, 43)])
def test_report_retains_results_and_distinguishes_seed_variability(tmp_path, seeds):
    (tmp_path / "manifest.json").write_text(json.dumps({"protocol": {"seeds": [42, 43, 44]}}))
    for index, seed in enumerate(seeds):
        _write_seed(tmp_path / f"seed-{seed}", seed, 5 + 2 * index)
    # An interrupted run must not count as a completed seed.
    (tmp_path / "seed-44").mkdir()
    build_report(tmp_path)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["seeds"] == list(seeds)
    assert summary["representative_seed"] == 42
    assert all(row["selected_epoch"] == 2 for row in summary["per_seed"])
    accuracy = summary["aggregate"]["test.accuracy"]
    assert accuracy["mean"] == pytest.approx(0.5 if len(seeds) == 1 else 0.6)
    assert accuracy["sample_std"] == (None if len(seeds) == 1 else pytest.approx(0.2 / 2**0.5))
    report = (tmp_path / "report.md").read_text()
    assert "Incomplete run set" in report
    assert "not intervals for the cross-seed mean" in report
    assert "[train predictions](seed-42/predictions_train.npz)" in report
    assert len(summary["artifacts"]["figures"]) == (4 if len(seeds) == 1 else 5)
    for name in summary["artifacts"]["figures"]:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_report_rejects_empty_results(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="No completed seed runs"):
        build_report(tmp_path)
