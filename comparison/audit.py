"""Independent NumPy-only audit; run after all measured children and the collector."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (42, 43, 44)
IMPLEMENTATIONS = ("python", "assembly")
COUNTS = {"train": 45000, "validation": 5000, "test": 10000}
OUTPUTS = ("best.bin", "last.bin", *(f"predictions_{s}.bin" for s in COUNTS))
METRICS = ("accuracy", "macro_f1", "negative_log_likelihood", "brier_score", "ece")
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "selected_epoch",
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(args):
    errors, checks, computed, hashes, times, histories = [], 0, {}, {}, {}, {}

    def check(ok, name):
        nonlocal checks
        checks += 1
        if not bool(ok):
            errors.append(name)

    def close(actual, expected, name):
        check(np.allclose(actual, expected, rtol=0, atol=1e-12), name)

    result = {"schema_version": 1, "auditor_sha256": sha(Path(__file__).resolve())}
    try:
        plan, summary = read(args.results / "plan.json"), read(args.report / "summary.json")
        result.update(
            source_revision=plan["source_revision"],
            plan_sha256=sha(args.results / "plan.json"),
            report_summary_sha256=sha(args.report / "summary.json"),
        )
        check(read(args.results / "completion.json")["status"] == "completed", "completion")
        check(summary["source_revision"] == plan["source_revision"], "source_revision")
        check(summary["audit"]["status"] == "passed", "collector_status")
        for location in (args.inputs / "manifest.json", args.results / "prepared-manifest.json"):
            check(sha(location) == plan["prepared_manifest_sha256"], f"manifest:{location.name}")
        check(sha(args.results / "cnn-assembly") == plan["binary_sha256"], "assembly_binary")
        for name, expected in plan["source_hashes"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                check(False, "unsafe_source_path")
                continue
            check(sha(args.results / "source" / path) == expected, f"source:{name}")
        expected_runs = {(i, s, r) for i in IMPLEMENTATIONS for s in SEEDS for r in (1, 2)}
        actual_runs = {(e["implementation"], e["seed"], e["repeat"]) for e in plan["runs"]}
        check(len(plan["runs"]) == 12 and actual_runs == expected_runs, "twelve_matched_runs")
        for entry in plan["runs"]:
            run_id = entry["id"]
            if Path(run_id).name != run_id or run_id in (".", ".."):
                check(False, "unsafe_run_id")
                continue
            folder, inputs = args.results / "raw" / run_id, args.inputs / f"seed-{entry['seed']}"
            run, perf = read(folder / "run.json"), read(folder / "performance.json")
            saved = read(args.report / "metrics" / f"{run_id}.json")
            key = entry["implementation"], entry["seed"], entry["repeat"]
            check(
                all(
                    run[k] == entry[k] for k in ("id", "implementation", "seed", "repeat", "order")
                ),
                f"identity:{run_id}",
            )
            check(
                run["returncode"] == 0 and run["source_revision"] == plan["source_revision"],
                f"success:{run_id}",
            )
            check(
                perf["epochs"] == 25
                and perf["train_limit"] == 45000
                and perf["evaluation_counts"] == COUNTS,
                f"full_protocol:{run_id}",
            )
            for boundary in ("before", "after"):
                check("AC Power" in str(run[boundary]["power"]), f"power:{run_id}:{boundary}")
            for name in ("labels.bin", "train_indices.bin", "val_indices.bin"):
                check(
                    sha(inputs / name) == plan["input_hashes"][str(entry["seed"])][name],
                    f"input:{run_id}:{name}",
                )
            labels = np.fromfile(inputs / "labels.bin", dtype="<u4")
            check(labels.shape == (60000,) and (labels < 10).all(), f"labels:{run_id}")
            indices = {
                "train": np.fromfile(inputs / "train_indices.bin", dtype="<u4"),
                "validation": np.fromfile(inputs / "val_indices.bin", dtype="<u4"),
                "test": np.arange(50000, 60000),
            }
            computed[key], hashes[key] = {}, {name: sha(folder / name) for name in OUTPUTS}
            times[key] = float(perf["training_seconds"])
            with (folder / "history.csv").open(newline="") as stream:
                histories[key] = [
                    tuple(float(row[name]) for name in HISTORY_FIELDS)
                    for row in csv.DictReader(stream)
                ]
            check(math.isfinite(times[key]) and times[key] > 0, f"training_time:{run_id}")
            for split, count in COUNTS.items():
                check(len(indices[split]) == count, f"split_count:{run_id}:{split}")
                p = (
                    np.fromfile(folder / f"predictions_{split}.bin", dtype="<f4")
                    .reshape(count, 10)
                    .astype(np.float64)
                )
                valid = np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                check(
                    valid and np.allclose(p.sum(1), 1, rtol=0, atol=1e-6),
                    f"probabilities:{run_id}:{split}",
                )
                p /= p.sum(1, keepdims=True)
                y, predicted = labels[indices[split]], p.argmax(1)
                confusion = np.zeros((10, 10), dtype=np.int64)
                np.add.at(confusion, (y, predicted), 1)
                denominator = confusion.sum(0) + confusion.sum(1)
                f1 = np.divide(
                    2 * confusion.diagonal(), denominator, out=np.zeros(10), where=denominator != 0
                )
                one_hot = np.eye(10)[y]
                confidence, correct = p.max(1), predicted == y
                bins, ece = np.minimum((confidence * 15).astype(int), 14), 0.0
                for index in range(15):
                    mask = bins == index
                    if mask.any():
                        ece += abs(correct[mask].mean() - confidence[mask].mean()) * mask.mean()
                values = {
                    "accuracy": float(correct.mean()),
                    "macro_f1": float(f1.mean()),
                    "negative_log_likelihood": float(
                        -np.log(np.clip(p[np.arange(count), y], np.finfo(float).eps, 1)).mean()
                    ),
                    "brier_score": float(np.square(p - one_hot).sum(1).mean()),
                    "ece": float(ece),
                }
                check(
                    np.array_equal(confusion, saved[split]["confusion_matrix"]),
                    f"confusion:{run_id}:{split}",
                )
                check(saved[split]["num_examples"] == count, f"metric_count:{run_id}:{split}")
                for name, value in values.items():
                    close(value, saved[split][name], f"metric:{run_id}:{split}:{name}")
                computed[key][split] = values
        repeat_checks = []
        for implementation in IMPLEMENTATIONS:
            for seed in SEEDS:
                same = all(
                    hashes[implementation, seed, 1][n] == hashes[implementation, seed, 2][n]
                    for n in OUTPUTS
                )
                recorded = next(
                    x
                    for x in summary["audit"]["repeatability"]
                    if x["implementation"] == implementation and x["seed"] == seed
                )
                check(
                    all(
                        recorded["binary_outputs_equal"][n]
                        == (
                            hashes[implementation, seed, 1][n] == hashes[implementation, seed, 2][n]
                        )
                        for n in OUTPUTS
                    ),
                    f"repeat_hash_status:{implementation}:{seed}",
                )
                history_same = (
                    histories[implementation, seed, 1] == histories[implementation, seed, 2]
                )
                check(
                    history_same == recorded["non_timing_history_equal"]
                    and (same and history_same) == recorded["identical_numerical_outputs"],
                    f"repeat_status:{implementation}:{seed}",
                )
                repeat_checks.append(
                    {
                        "implementation": implementation,
                        "seed": seed,
                        "binary_outputs_equal": same,
                        "history_equal": history_same,
                    }
                )
            for split in COUNTS:
                for metric in METRICS:
                    values = [
                        np.mean([computed[implementation, seed, r][split][metric] for r in (1, 2)])
                        for seed in SEEDS
                    ]
                    aggregate = summary["implementations"][implementation]["quality"][split][metric]
                    close(
                        np.mean(values),
                        aggregate["mean"],
                        f"quality_mean:{implementation}:{split}:{metric}",
                    )
                    close(
                        np.std(values, ddof=1),
                        aggregate["sample_std"],
                        f"quality_sd:{implementation}:{split}:{metric}",
                    )
            values = [np.mean([times[implementation, seed, r] for r in (1, 2)]) for seed in SEEDS]
            aggregate = summary["implementations"][implementation]["performance"][
                "training_seconds"
            ]["seed_means"]
            close(np.mean(values), aggregate["mean"], f"timing_mean:{implementation}")
            close(np.std(values, ddof=1), aggregate["sample_std"], f"timing_sd:{implementation}")
        ratio = math.exp(
            sum(
                math.log(times["python", s, r] / times["assembly", s, r])
                for s in SEEDS
                for r in (1, 2)
            )
            / 6
        )
        close(
            ratio,
            summary["paired_performance"]["training_seconds"]["geometric_mean_paired_ratio"],
            "paired_training_geometric_ratio",
        )
        check(
            summary["audit"]["all_repeats_identical"]
            == all(r["binary_outputs_equal"] and r["history_equal"] for r in repeat_checks),
            "all_repeat_status",
        )
        check(
            summary["audit"]["prediction_rows_verified"] == 12 * sum(COUNTS.values()),
            "prediction_row_count",
        )
        result.update(
            prediction_rows_recomputed=12 * sum(COUNTS.values()),
            unique_test_images=10000,
            repeat_checks=repeat_checks,
            paired_training_geometric_ratio=ratio,
            source_files_verified=len(plan["source_hashes"]),
        )
    except Exception as exc:
        errors.append(f"Audit could not complete: {type(exc).__name__}: {exc}")
    result.update(
        status="passed" if not errors else "failed",
        checks=checks,
        errors=errors,
        scope=(
            "Independent NumPy recomputation: all 36 prediction arrays; five scalar metrics "
            "and confusion; seed/repeat aggregation; timing ratio; repeat binary hashes and "
            "retained provenance. No full image reads, fitting, framework imports, or "
            "CI/bootstrap verification."
        ),
        numeric_comparison={"atol": 1e-12, "rtol": 0},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "errors": len(errors),
                "output": str(args.output),
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "inputs", "report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    raise SystemExit(audit(parser.parse_args()))
