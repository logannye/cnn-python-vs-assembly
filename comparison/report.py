"""Audit completed matched runs and generate reproducible comparison tables/figures.

Run only after comparison/run.py has completed every scheduled process. This
collector performs inference audits, never fitting, and does not edit raw runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from classifier.metrics import classification_metrics
from classifier.model import SimpleCNN
from classifier.report import METRICS

SEEDS = (42, 43, 44)
IMPLEMENTATIONS = ("python", "assembly")
COUNTS = {"train": 45000, "validation": 5000, "test": 10000}
NUMERICAL_HISTORY = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "selected_epoch",
)
BINARY_OUTPUTS = (
    "best.bin",
    "last.bin",
    "predictions_train.bin",
    "predictions_validation.bin",
    "predictions_test.bin",
)
COLORS = {"python": "#3274A1", "assembly": "#D57C27"}
LABELS = {"python": "Python / PyTorch", "assembly": "ARM64 assembly"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_file(path, dtype, shape):
    path = Path(path)
    require(
        path.stat().st_size == math.prod(shape) * np.dtype(dtype).itemsize,
        f"Wrong byte count: {path}",
    )
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    if value.dtype.kind == "f":
        require(np.isfinite(value).all(), f"Nonfinite values: {path}")
    return value


def aggregate(values):
    values = np.asarray(values, dtype=np.float64)
    require(len(values) > 0 and np.isfinite(values).all(), "Invalid aggregation values")
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def table(path, rows):
    require(bool(rows), f"Empty table: {path}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(root):
    manifest = read_json(root / "manifest.json")
    require(manifest["protocol"]["epochs"] == 25, "Inputs are not a 25-epoch protocol")
    require(manifest["protocol"]["seeds"] == list(SEEDS), "Unexpected input seeds")
    cache, hashes = {}, {}
    for seed in SEEDS:
        folder = root / f"seed-{seed}"
        metadata = read_json(folder / "inputs.json")
        require(metadata == manifest["seeds"][str(seed)], "Input manifest disagreement")
        require(
            metadata["epochs"] == 25
            and metadata["batch_size"] == 128
            and metadata["parameters"] == 5418,
            "Wrong input protocol",
        )
        require(
            set(metadata["hashes"])
            == {
                "images.bin",
                "labels.bin",
                "train_indices.bin",
                "val_indices.bin",
                "init.bin",
                "schedule.bin",
            },
            "Incomplete input fingerprints",
        )
        for name, expected in metadata["hashes"].items():
            path = folder / name
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in cache:
                cache[key] = digest(path)
            require(cache[key] == expected, f"Input hash mismatch: {path}")
            hashes[f"seed-{seed}/{name}"] = expected
            if name not in {"init.bin", "schedule.bin"}:
                require(
                    expected == manifest["seeds"]["42"]["hashes"][name],
                    f"Shared input differs for seed {seed}: {name}",
                )
        array_file(folder / "init.bin", "<f4", (5418,))
        schedule = array_file(folder / "schedule.bin", "<u4", (25, 45000, 2))
        train = array_file(folder / "train_indices.bin", "<u4", (45000,))
        require((schedule[:, :, 1] <= 1).all(), "Nonbinary flip flags")
        for epoch in schedule:
            require(np.array_equal(np.sort(epoch[:, 0]), train), "Invalid shuffle permutation")
    shared = root / "seed-42"
    labels = array_file(shared / "labels.bin", "<u4", (60000,)).astype(np.int64)
    require((labels < 10).all(), "Labels outside class range")
    indices = {
        "train": array_file(shared / "train_indices.bin", "<u4", (45000,)).astype(np.int64),
        "validation": array_file(shared / "val_indices.bin", "<u4", (5000,)).astype(np.int64),
        "test": np.arange(50000, 60000),
    }
    require(
        np.array_equal(np.sort(np.r_[indices["train"], indices["validation"]]), np.arange(50000)),
        "Splits do not partition training set",
    )
    require((shared / "images.bin").stat().st_size == 60000 * 32 * 32 * 3, "Wrong image data size")
    images = np.memmap(shared / "images.bin", dtype="u1", mode="r", shape=(60000, 32, 32, 3))
    return manifest, labels, indices, images, hashes


def load_run(root, entry, revision):
    run_id = entry["id"]
    require(
        isinstance(run_id, str) and Path(run_id).name == run_id and run_id not in {".", ".."},
        "Unsafe run id",
    )
    folder = root / "raw" / run_id
    run = read_json(folder / "run.json")
    require(run["returncode"] == 0, f"Run did not finish: {run_id}")
    require(run["source_revision"] == revision, f"Source changed: {run_id}")
    with (folder / "history.csv").open(newline="") as stream:
        history = [
            {
                key: int(value) if key in {"epoch", "selected_epoch"} else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]
    require(
        [row["epoch"] for row in history] == list(range(1, 26)), f"Not 25 complete epochs: {run_id}"
    )
    for offset, row in enumerate(history):
        require(all(math.isfinite(value) for value in row.values()), "Nonfinite history")
        require(
            0 <= row["train_accuracy"] <= 1 and 0 <= row["val_accuracy"] <= 1, "Invalid accuracy"
        )
        for name in ("train_loss", "val_loss", "train_seconds", "val_seconds", "epoch_seconds"):
            require(row[name] >= 0, f"Invalid {name}")
        best = max(
            history[: offset + 1], key=lambda item: (item["val_accuracy"], -item["val_loss"])
        )
        require(row["selected_epoch"] == best["epoch"], f"Wrong checkpoint selection: {run_id}")
        require(
            row["epoch_seconds"] + 1e-6 >= row["train_seconds"] + row["val_seconds"],
            "Epoch wall scope is too short",
        )
    perf = read_json(folder / "performance.json")
    for key, value in {
        "epochs": 25,
        "train_limit": 45000,
        "threads": 2,
        "parameter_count": 5418,
        "evaluation_counts": COUNTS,
    }.items():
        require(perf[key] == value, f"Incorrect full-run protocol {key}: {run_id}")
    require(perf["best_epoch"] == best["epoch"], "Incorrect best epoch")
    measurements = {
        key: float(perf[key])
        for key in (
            "training_seconds",
            "training_cpu_seconds",
            "peak_rss_loaded_bytes",
            "peak_rss_training_bytes",
            "peak_rss_bytes",
            "final_evaluation_seconds",
        )
    }
    measurements["process_wall_seconds"] = float(run["process_wall_seconds"])
    measurements["process_cpu_seconds"] = (
        perf["process_user_cpu_seconds"] + perf["process_system_cpu_seconds"]
    )
    measurements["train_phase_seconds"] = sum(row["train_seconds"] for row in history)
    measurements["validation_phase_seconds"] = sum(row["val_seconds"] for row in history)
    measurements["training_average_cores"] = perf["training_cpu_seconds"] / perf["training_seconds"]
    measurements["training_images_per_second"] = 25 * 45000 / measurements["train_phase_seconds"]
    for batch in (1, 128):
        samples = array_file(folder / f"timing_batch{batch}.bin", "<f8", (100,))
        require((samples > 0).all(), "Nonpositive latency sample")
        for percentile in (50, 95, 99):
            measurements[f"batch{batch}_p{percentile}_ms"] = float(
                np.percentile(samples, percentile)
            )
        measurements[f"batch{batch}_mean_ms"] = float(samples.mean())
        measurements[f"batch{batch}_images_per_second"] = float(batch * 1000 / samples.mean())
    require(
        all(math.isfinite(value) and value > 0 for value in measurements.values()),
        f"Invalid performance measurement: {run_id}",
    )
    require(
        perf["peak_rss_loaded_bytes"] <= perf["peak_rss_training_bytes"] <= perf["peak_rss_bytes"],
        "RSS high-water decreased",
    )
    require(
        perf["training_cpu_seconds"] <= measurements["process_cpu_seconds"] + 1e-6,
        "Training CPU exceeds lifetime CPU",
    )
    require(
        perf["training_seconds"] + 1e-5 >= sum(row["epoch_seconds"] for row in history),
        "Training wall scope excludes epoch durations",
    )
    require(
        measurements["process_wall_seconds"] >= perf["training_seconds"],
        "Child process wall time excludes training",
    )
    for name in ("best.bin", "last.bin"):
        array_file(folder / name, "<f4", (5418,))
    hashes = {path.name: digest(path) for path in sorted(folder.iterdir()) if path.is_file()}
    return {
        **entry,
        "performance": measurements,
        "best_epoch": perf["best_epoch"],
        "history": history,
        "hashes": hashes,
        "runtime": run,
    }


def checkpoint_audit(folder, images, probabilities):
    flat = array_file(folder / "best.bin", "<f4", (5418,))
    model = SimpleCNN(10).cpu().eval()
    cursor = 0
    with torch.inference_mode():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(torch.from_numpy(flat[cursor : cursor + count]).view_as(parameter))
            cursor += count
        require(cursor == 5418, "Unexpected parameter layout")
        batch = torch.from_numpy(np.asarray(images).copy()).permute(0, 3, 1, 2).contiguous().float()
        batch.div_(255).sub_(0.5).div_(0.5)
        replay = model(batch).softmax(1).numpy()
    np.testing.assert_allclose(
        probabilities[:128],
        replay,
        atol=3e-5,
        rtol=1e-4,
        err_msg=f"Checkpoint inference replay: {folder.name}",
    )
    return {
        "status": "passed",
        "test_images": 128,
        "atol": 3e-5,
        "rtol": 1e-4,
        "max_absolute_probability_error": float(np.abs(probabilities[:128] - replay).max()),
    }


def paired_test(seed, repeat, python, assembly, labels):
    py = python.argmax(1)
    asm = assembly.argmax(1)
    py_correct, asm_correct = py == labels, asm == labels
    delta = asm_correct.astype(int) - py_correct.astype(int)
    counts = np.bincount(delta + 1, minlength=3)
    samples = np.random.default_rng(20261002 + seed).multinomial(
        len(labels), counts / len(labels), size=10000
    )
    low, high = np.quantile((samples[:, 2] - samples[:, 0]) / len(labels), [0.025, 0.975])
    return {
        "seed": seed,
        "repeat": repeat,
        "test_examples": len(labels),
        "python_accuracy": float(py_correct.mean()),
        "assembly_accuracy": float(asm_correct.mean()),
        "accuracy_difference": float(delta.mean()),
        "accuracy_difference_ci95": {
            "lower": float(low),
            "upper": float(high),
            "confidence_level": 0.95,
            "method": "paired iid image percentile bootstrap of discordance counts",
            "resamples": 10000,
            "seed": 20261002 + seed,
            "scope": "conditional on the fitted pair; excludes training-seed uncertainty",
        },
        "prediction_disagreements": int((py != asm).sum()),
        "prediction_disagreement_rate": float((py != asm).mean()),
        "assembly_only_correct": int(counts[2]),
        "python_only_correct": int(counts[0]),
        "both_correct": int((py_correct & asm_correct).sum()),
        "both_incorrect": int((~py_correct & ~asm_correct).sum()),
        "mean_absolute_probability_difference": float(
            np.abs(python.astype(float) - assembly).mean()
        ),
        "max_absolute_probability_difference": float(np.abs(python.astype(float) - assembly).max()),
    }


def summarize(runs, paired, repeats, revision, input_hashes):
    implementations = {}
    for implementation in IMPLEMENTATIONS:
        selected = [run for run in runs if run["implementation"] == implementation]
        quality = {}
        for split in COUNTS:
            quality[split] = {}
            for name in METRICS:
                values = [
                    np.mean(
                        [run["metrics"][split][name] for run in selected if run["seed"] == seed]
                    )
                    for seed in SEEDS
                ]
                quality[split][name] = aggregate(values)
        performance = {}
        for name in selected[0]["performance"]:
            means = [
                {
                    "seed": seed,
                    "value": float(
                        np.mean(
                            [run["performance"][name] for run in selected if run["seed"] == seed]
                        )
                    ),
                }
                for seed in SEEDS
            ]
            performance[name] = {
                "all_runs": aggregate([run["performance"][name] for run in selected]),
                "seed_means": aggregate([item["value"] for item in means]),
                "per_seed_means": means,
            }
        implementations[implementation] = {"quality": quality, "performance": performance}
    paired_performance = {}
    indexed = {(run["implementation"], run["seed"], run["repeat"]): run for run in runs}
    for name in runs[0]["performance"]:
        values = [
            {
                "seed": seed,
                "repeat": repeat,
                "python_over_assembly": (
                    indexed["python", seed, repeat]["performance"][name]
                    / indexed["assembly", seed, repeat]["performance"][name]
                ),
            }
            for seed in SEEDS
            for repeat in (1, 2)
        ]
        seed_ratios = [
            implementations["python"]["performance"][name]["per_seed_means"][i]["value"]
            / implementations["assembly"]["performance"][name]["per_seed_means"][i]["value"]
            for i in range(len(SEEDS))
        ]
        paired_performance[name] = {
            "ratio": "python / assembly; >1 means Python quantity is larger",
            "pairs": values,
            "geometric_mean_paired_ratio": float(
                np.exp(np.mean(np.log([value["python_over_assembly"] for value in values])))
            ),
            "geometric_mean_seed_mean_ratio": float(np.exp(np.mean(np.log(seed_ratios)))),
            "seed_mean_ratios": dict(zip(map(str, SEEDS), seed_ratios, strict=True)),
        }
    per_seed = []
    for seed in SEEDS:
        selected = [row for row in paired if row["seed"] == seed]
        exact = all(row["identical_numerical_outputs"] for row in repeats if row["seed"] == seed)
        row = {key: value for key, value in selected[0].items() if key != "repeat"}
        for key in (
            "python_accuracy",
            "assembly_accuracy",
            "accuracy_difference",
            "prediction_disagreement_rate",
            "prediction_disagreements",
            "mean_absolute_probability_difference",
            "max_absolute_probability_difference",
        ):
            row[key] = float(np.mean([item[key] for item in selected]))
        row["identical_repeats"] = exact
        if not exact:
            row["accuracy_difference_ci95"] = None
            for key in (
                "assembly_only_correct",
                "python_only_correct",
                "both_correct",
                "both_incorrect",
            ):
                row[key] = None
        per_seed.append(row)
    return {
        "schema_version": 1,
        "source_revision": revision,
        "experiment": "CIFAR-10, matched Python/PyTorch vs ARM64 assembly, 25 epochs",
        "seeds": list(SEEDS),
        "repeats_per_seed": 2,
        "runs_per_implementation": 6,
        "quality_statistics": "Mean and sample SD across 3 seeds, averaging repeats within seed. "
        "Identical repeats are not independent fits; all seeds share the same test images.",
        "performance_statistics": "All 6 individual runs retained; headline mean and sample SD "
        "across 3 seed means after averaging the 2 repeats. Paired ratios match seed and repeat. "
        "No timing confidence interval or machine-general claim is inferred from 3 seeds.",
        "measurement_scopes": {
            "training": "Transforms, forward/backward, Adam, validation, checkpoint/report writes; "
            "excludes shared input preparation, process startup and final evaluation.",
            "process_wall": "Child launch through exit, including runtime startup, input loading, "
            "training, evaluation, output writing and 20+100 forward timing iterations.",
            "inference": "Fixed preprocessed first test images, forward only, 20 warmups then "
            "100 samples per run/batch; p50/p95/p99 computed within each run.",
            "memory": "Lifetime process RSS high-water sampled after setup, after training, "
            "and after evaluation/timing; includes runtime and input buffers, not model memory. "
            "Differences between phase peaks do not measure allocated training memory.",
        },
        "audit": {
            "status": "passed",
            "full_runs": len(runs),
            "input_files_hashed": len(input_hashes),
            "prediction_rows_verified": len(runs) * sum(COUNTS.values()),
            "unique_test_images": 10000,
            "repeatability": repeats,
            "all_repeats_identical": all(row["identical_numerical_outputs"] for row in repeats),
            "checkpoint_replays": {run["id"]: run["checkpoint_replay"] for run in runs},
        },
        "implementations": implementations,
        "paired_performance": paired_performance,
        "paired_test_pairs": paired,
        "per_seed_paired_test": per_seed,
        "paired_accuracy_difference": aggregate([row["accuracy_difference"] for row in per_seed]),
        "input_sha256": input_hashes,
        "raw_sha256": {run["id"]: run["hashes"] for run in runs},
        "limitations": [
            "Matched conditions compare these implementations, not programming languages alone. "
            "PyTorch convolution and autograd execute native kernels.",
            "Batch layout, scheduling, allocation, native kernels and reduction order remain "
            "implementation differences. Thread count does not fix CPU affinity or clock speed.",
            "Float32 models are mathematically matched; implementations are not bitwise identical.",
            "Repeat equality is checked from checkpoints, probabilities and non-timing history.",
            "The paired interval is not an equivalence test. Three seeds do not establish "
            "statistical or hardware-general equivalence.",
            "Thermal and background-load variation are observed externally; no energy consumption "
            "or exclusive machine control is claimed.",
        ],
    }


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": "#DDE2E7",
            "grid.alpha": 0.65,
            "svg.fonttype": "none",
            "svg.hashsalt": "cnn-python-vs-assembly-matched-v1",
        }
    )


def save_figure(fig, output, name, caption):
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8, color="#53606C")
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    fig.savefig(
        output / f"{name}.png", facecolor="white", metadata={"Software": "comparison/report.py"}
    )
    fig.savefig(output / f"{name}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def figures(output, summary, runs):
    style()
    # Quality plots use seed means, retaining repeat aggregation even if repeatability failed.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Same CNN, matched inputs: predictive quality", fontsize=16, weight="bold")
    for implementation in IMPLEMENTATIONS:
        accuracy = [
            np.mean(
                [
                    run["metrics"]["test"]["accuracy"]
                    for run in runs
                    if run["implementation"] == implementation and run["seed"] == seed
                ]
            )
            for seed in SEEDS
        ]
        offset = -0.08 if implementation == "python" else 0.08
        axes[0].plot(
            np.arange(3) + offset,
            np.array(accuracy) * 100,
            "o-",
            markersize=7,
            label=LABELS[implementation],
            color=COLORS[implementation],
            linewidth=1.6,
        )
        classes = runs[0]["metrics"]["test"]["class_names"]
        f1 = np.mean(
            [
                [row["f1"] for row in run["metrics"]["test"]["per_class"]]
                for run in runs
                if run["implementation"] == implementation
            ],
            axis=0,
        )
        width = 0.37
        axes[1].bar(
            np.arange(10) + (-width / 2 if implementation == "python" else width / 2),
            f1,
            width,
            color=COLORS[implementation],
            label=LABELS[implementation],
        )
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=[str(seed) for seed in SEEDS],
        xlabel="Training seed",
        ylabel="Test accuracy (%)",
        title="10,000 held-out images",
    )
    # Explicitly show the full accuracy scale; tiny differences are tabulated precisely in CSV/JSON.
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, loc="lower center")
    axes[1].set(
        xticks=np.arange(10),
        xticklabels=classes,
        ylabel="Per-class F1",
        title="Class-level agreement",
        ylim=(0, 1),
    )
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.grid(axis="y")
    save_figure(
        fig,
        output,
        "quality",
        "3 training seeds; 2 repeats averaged within each seed. "
        "Repeated fits and test-image overlap do not add independent samples.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Compute cost under the same experimental protocol", fontsize=16, weight="bold")
    for ax, metric, title in zip(
        axes,
        ("training_seconds", "training_cpu_seconds", "process_wall_seconds"),
        ("Training + validation", "Training CPU time", "Whole child process"),
        strict=True,
    ):
        for index, implementation in enumerate(IMPLEMENTATIONS):
            perf = summary["implementations"][implementation]["performance"][metric]
            mean, std = perf["seed_means"]["mean"], perf["seed_means"]["sample_std"]
            ax.bar(
                index,
                mean,
                yerr=std,
                capsize=4,
                width=0.57,
                color=COLORS[implementation],
                alpha=0.85,
            )
            values = [
                run["performance"][metric]
                for run in runs
                if run["implementation"] == implementation
            ]
            ax.scatter(
                index + np.linspace(-0.10, 0.10, len(values)),
                values,
                s=24,
                color="#243746",
                zorder=3,
            )
            ax.text(
                index, mean + std + max(mean * 0.04, 1), f"{mean:.1f}", ha="center", fontsize=10
            )
        ax.set(xticks=(0, 1), xticklabels=("Python", "Assembly"), ylabel="Seconds", title=title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.grid(axis="y")
    save_figure(
        fig,
        output,
        "runtime",
        "Bars: mean of 3 seed means; error bars: seed sample SD. "
        "Dots: all 6 runs per implementation. CPU seconds include both workers.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Forward-pass latency", fontsize=16, weight="bold")
    for ax, batch in zip(axes, (1, 128), strict=True):
        for implementation in IMPLEMENTATIONS:
            metrics = [
                summary["implementations"][implementation]["performance"][f"batch{batch}_p{p}_ms"]
                for p in (50, 95, 99)
            ]
            means = [item["seed_means"]["mean"] for item in metrics]
            stds = [item["seed_means"]["sample_std"] for item in metrics]
            width = 0.35
            x = np.arange(3) + (-width / 2 if implementation == "python" else width / 2)
            bars = ax.bar(
                x,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=COLORS[implementation],
                label=LABELS[implementation],
            )
            ax.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=4, fontsize=8)
        ax.set(
            xticks=np.arange(3),
            xticklabels=("Median", "95th percentile", "99th percentile"),
            ylabel="Milliseconds per forward pass",
            title=f"Batch size {batch}",
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.grid(axis="y")
    axes[0].legend(frameon=False, fontsize=9)
    save_figure(
        fig,
        output,
        "inference",
        "20 warmups + 100 samples per run. Bars average within-run "
        "percentiles across seed means; error bars are seed SD, not latency intervals.",
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle("Process memory through each measurement boundary", fontsize=16, weight="bold")
    memory_fields = ("peak_rss_loaded_bytes", "peak_rss_training_bytes", "peak_rss_bytes")
    for implementation in IMPLEMENTATIONS:
        stats = [
            summary["implementations"][implementation]["performance"][name]["seed_means"]
            for name in memory_fields
        ]
        width = 0.35
        x = np.arange(3) + (-width / 2 if implementation == "python" else width / 2)
        means, stds = (
            [row["mean"] / 2**20 for row in stats],
            [row["sample_std"] / 2**20 for row in stats],
        )
        bars = ax.bar(
            x,
            means,
            width,
            yerr=stds,
            capsize=4,
            color=COLORS[implementation],
            label=LABELS[implementation],
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=4)
    ax.set(
        xticks=np.arange(3),
        xticklabels=("After loading + setup", "After training", "Whole process"),
        ylabel="Lifetime peak resident memory (MiB)",
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.20)
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save_figure(
        fig,
        output,
        "memory",
        "Process high-water marks include runtime and raw input buffers. "
        "These are not model allocations or incremental phase memory. Error bars: seed SD.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Learning trajectories with identical initialization and image schedules",
        fontsize=15,
        weight="bold",
    )
    for ax, field, title, factor in zip(
        axes,
        ("val_accuracy", "val_loss"),
        ("Validation accuracy (%)", "Validation cross-entropy"),
        (100, 1),
        strict=True,
    ):
        for implementation in IMPLEMENTATIONS:
            values = (
                np.array(
                    [
                        np.mean(
                            [
                                [row[field] for row in run["history"]]
                                for run in runs
                                if run["implementation"] == implementation and run["seed"] == seed
                            ],
                            axis=0,
                        )
                        for seed in SEEDS
                    ]
                )
                * factor
            )
            epochs = np.arange(1, 26)
            mean, std = values.mean(0), values.std(0, ddof=1)
            ax.fill_between(
                epochs, mean - std, mean + std, color=COLORS[implementation], alpha=0.14
            )
            ax.plot(
                epochs,
                mean,
                color=COLORS[implementation],
                label=LABELS[implementation],
                linewidth=2,
            )
        ax.set(xlabel="Epoch", ylabel=title, xlim=(1, 25), xticks=(1, 5, 10, 15, 20, 25))
        ax.grid()
    axes[0].legend(frameon=False)
    save_figure(
        fig,
        output,
        "learning",
        "Mean validation trajectory across 3 seed means; shaded region: "
        "seed sample SD. Checkpoints selected using validation only.",
    )
    return [
        f"figures/{name}.{extension}"
        for name in ("quality", "runtime", "inference", "memory", "learning")
        for extension in ("png", "svg")
    ]


def collect(results, inputs, output):
    require(not output.exists() or not any(output.iterdir()), "Refusing nonempty report output")
    plan = read_json(results / "plan.json")
    entries = plan["runs"]
    expected = {
        (impl, seed, repeat) for impl in IMPLEMENTATIONS for seed in SEEDS for repeat in (1, 2)
    }
    actual = {(row["implementation"], row["seed"], row["repeat"]) for row in entries}
    require(
        len(entries) == 12 and actual == expected, "Plan must contain exactly the 12 matched runs"
    )
    require(len({row["id"] for row in entries}) == 12, "Duplicate run IDs")
    require(len({row["order"] for row in entries}) == 12, "Duplicate run orders")
    revision = plan["source_revision"]
    require(
        len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), "Invalid revision"
    )
    manifest, labels, indices, images, input_hashes = validate_inputs(inputs)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    runs = [
        load_run(results, entry, revision)
        for entry in sorted(entries, key=lambda row: row["order"])
    ]
    repeats = []
    for impl in IMPLEMENTATIONS:
        for seed in SEEDS:
            pair = sorted(
                [run for run in runs if run["implementation"] == impl and run["seed"] == seed],
                key=lambda run: run["repeat"],
            )
            binary_equal = {
                name: pair[0]["hashes"][name] == pair[1]["hashes"][name] for name in BINARY_OUTPUTS
            }
            history_equal = all(
                a[key] == b[key]
                for a, b in zip(pair[0]["history"], pair[1]["history"], strict=True)
                for key in NUMERICAL_HISTORY
            )
            repeats.append(
                {
                    "implementation": impl,
                    "seed": seed,
                    "binary_outputs_equal": binary_equal,
                    "non_timing_history_equal": history_equal,
                    "identical_numerical_outputs": all(binary_equal.values()) and history_equal,
                }
            )
    cache, test_probabilities = {}, {}
    for run in runs:
        folder = results / "raw" / run["id"]
        key = tuple(run["hashes"][name] for name in BINARY_OUTPUTS)
        if key not in cache:
            metrics = {}
            for split, count in COUNTS.items():
                probabilities = array_file(folder / f"predictions_{split}.bin", "<f4", (count, 10))
                metrics[split] = classification_metrics(
                    labels[indices[split]],
                    probabilities,
                    manifest["protocol"]["class_names"],
                    run["seed"],
                )
                if split == "test":
                    test = probabilities
            replay = checkpoint_audit(folder, images[indices["test"][:128]], test)
            cache[key] = metrics, replay, test
        run["metrics"], run["checkpoint_replay"], test = cache[key]
        test_probabilities[run["implementation"], run["seed"], run["repeat"]] = test
    paired = [
        paired_test(
            seed,
            repeat,
            test_probabilities["python", seed, repeat],
            test_probabilities["assembly", seed, repeat],
            labels[indices["test"]],
        )
        for seed in SEEDS
        for repeat in (1, 2)
    ]
    summary = summarize(runs, paired, repeats, revision, input_hashes)
    summary["audit"]["distinct_numerical_result_sets"] = len(cache)
    summary["protocol"] = manifest["protocol"]
    summary["run_order"] = entries
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir()
    (output / "figures").mkdir()
    for run in runs:
        write_json(output / "metrics" / f"{run['id']}.json", run["metrics"])
    table(
        output / "runs.csv",
        [
            {
                key: run[key]
                for key in ("id", "order", "implementation", "seed", "repeat", "best_epoch")
            }
            | run["performance"]
            for run in runs
        ],
    )
    table(
        output / "models.csv",
        [
            {key: run[key] for key in ("id", "implementation", "seed", "repeat", "best_epoch")}
            | {
                f"{split}_{metric}": run["metrics"][split][metric]
                for split in COUNTS
                for metric in METRICS
            }
            for run in runs
        ],
    )
    table(
        output / "paired_test.csv",
        [
            {key: value for key, value in row.items() if key != "accuracy_difference_ci95"}
            | {
                f"accuracy_difference_ci95_{bound}": row["accuracy_difference_ci95"][bound]
                for bound in ("lower", "upper")
            }
            for row in paired
        ],
    )
    summary["figures"] = figures(output / "figures", summary, runs)
    write_json(output / "summary.json", summary)
    write_json(
        output / "checksums.json",
        {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": "passed",
                "runs": len(runs),
                "all_repeats_identical": summary["audit"]["all_repeats_identical"],
            },
            indent=2,
        )
    )
    if not summary["audit"]["all_repeats_identical"]:
        print(
            "WARNING: numerical repeats differ. Inspect summary.audit.repeatability; "
            "quality summaries average repeats within seed."
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collect(args.results, args.inputs, args.output)


if __name__ == "__main__":
    main()
