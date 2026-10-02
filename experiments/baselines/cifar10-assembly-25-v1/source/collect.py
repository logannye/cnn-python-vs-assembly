"""Audit and report the complete handwritten-assembly experiment, without fitting."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from prepare import digest, flat_parameters, load_parameters

from classifier.metrics import classification_metrics
from classifier.model import SimpleCNN
from classifier.report import METRICS, build_report

SEEDS = (42, 43, 44)
COUNTS = {"train": 45000, "validation": 5000, "test": 10000}
HISTORY_FIELDS = {
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "train_seconds",
    "val_seconds",
    "epoch_seconds",
    "selected_epoch",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_file(path, dtype, shape):
    path = Path(path)
    require(
        path.stat().st_size == np.prod(shape) * np.dtype(dtype).itemsize,
        f"Wrong byte count: {path}",
    )
    array = np.fromfile(path, dtype=dtype).reshape(shape)
    if array.dtype.kind == "f":
        require(np.isfinite(array).all(), f"Nonfinite values: {path}")
    return array


def aggregate(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def history_and_performance(folder):
    with (folder / "history.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(reader.fieldnames or ()) == HISTORY_FIELDS, "Unexpected history fields")
        history = [
            {
                key: int(value) if key in {"epoch", "selected_epoch"} else float(value)
                for key, value in row.items()
            }
            for row in reader
        ]
    require(
        [row["epoch"] for row in history] == list(range(1, 26)),
        f"Not exactly 25 complete epochs: {folder}",
    )
    for index, row in enumerate(history):
        require(all(math.isfinite(value) for value in row.values()), "Nonfinite history")
        for name in ("train_accuracy", "val_accuracy"):
            require(0 <= row[name] <= 1, f"Out-of-range {name}")
        for name in ("train_loss", "val_loss", "train_seconds", "val_seconds", "epoch_seconds"):
            require(row[name] >= 0, f"Negative {name}")
        require(
            row["epoch_seconds"] + 1e-6 >= row["train_seconds"] + row["val_seconds"],
            "Epoch time excludes a measured phase",
        )
        best = max(history[: index + 1], key=lambda item: (item["val_accuracy"], -item["val_loss"]))
        require(row["selected_epoch"] == best["epoch"], "Wrong running checkpoint selection")
    perf = read_json(folder / "performance.json")
    expected = {"parameter_count": 5418, "threads": 2, "epochs": 25, "train_limit": 45000}
    for name, value in expected.items():
        require(perf.get(name) == value, f"Wrong {name}: {folder}")
    require(perf.get("evaluation_counts") == COUNTS, "Incomplete evaluation sample counts")
    require(perf["best_epoch"] == best["epoch"], "Wrong selected checkpoint epoch")
    for name in (
        "training_seconds",
        "final_evaluation_seconds",
        "peak_rss_bytes",
        "process_user_cpu_seconds",
        "process_system_cpu_seconds",
    ):
        require(math.isfinite(perf[name]) and perf[name] >= 0, f"Invalid {name}")
    require(
        perf["training_seconds"] + 1e-5 >= sum(row["epoch_seconds"] for row in history),
        "Training duration shorter than sum of epoch durations",
    )
    require(set(perf["prediction_seconds"]) == set(COUNTS), "Missing prediction durations")
    require(
        all(math.isfinite(value) and value > 0 for value in perf["prediction_seconds"].values()),
        "Invalid prediction duration",
    )
    return history, perf


def verify_inputs(prepared, baseline):
    manifest = read_json(prepared / "manifest.json")
    old = read_json(baseline / "manifest.json")
    require(manifest["baseline_data"] == old["data"], "Prepared dataset differs from baseline")
    require(manifest["protocol"] == old["protocol"], "Prepared protocol differs from baseline")
    require(manifest["protocol"]["epochs"] == 25, "Protocol is not 25 epochs")
    require(manifest["protocol"]["seeds"] == list(SEEDS), "Wrong seed set")
    shared = prepared / "shared"
    require((shared / "images.bin").stat().st_size == 60000 * 32 * 32 * 3, "Wrong image byte count")
    images = np.memmap(shared / "images.bin", dtype=np.uint8, mode="r", shape=(60000, 32, 32, 3))
    labels = array_file(shared / "labels.bin", "<u4", (60000,)).astype(np.int64)
    require(np.all(labels < 10), "Out-of-range labels")
    indices = {
        "train": array_file(shared / "train_indices.bin", "<u4", (45000,)).astype(np.int64),
        "validation": array_file(shared / "val_indices.bin", "<u4", (5000,)).astype(np.int64),
        "test": np.arange(10000, dtype=np.int64),
    }
    with np.load(baseline / "split_indices.npz") as old_indices:
        for split, values in indices.items():
            np.testing.assert_array_equal(values, old_indices[split])
    np.testing.assert_array_equal(
        np.sort(np.concatenate((indices["train"], indices["validation"]))), np.arange(50000)
    )
    for key, value in (
        ("training_images_sha256", images[:50000]),
        ("test_images_sha256", images[50000:]),
        ("training_labels_sha256", labels[:50000].astype("<i8")),
        ("test_labels_sha256", labels[50000:].astype("<i8")),
    ):
        actual = hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()
        require(actual == old["data"][key], f"Dataset fingerprint mismatch: {key}")
    hash_cache, verified = {}, 0
    for seed in SEEDS:
        folder = prepared / f"seed-{seed}"
        info = read_json(folder / "inputs.json")
        require(info == manifest["seeds"][str(seed)], f"Input metadata mismatch for {seed}")
        require(
            info["seed"] == seed
            and info["epochs"] == 25
            and info["batch_size"] == 128
            and info["parameters"] == 5418
            and info["schedule_records"] == 25 * 45000,
            f"Incorrect prepared protocol for {seed}",
        )
        require(
            info["actual_baseline_loader_images_verified_bitwise"] >= 256,
            "Missing actual DataLoader schedule verification",
        )
        require(
            set(info["hashes"])
            == {
                "images.bin",
                "labels.bin",
                "train_indices.bin",
                "val_indices.bin",
                "init.bin",
                "schedule.bin",
            },
            "Incomplete input hashes",
        )
        for name, expected in info["hashes"].items():
            path = folder / name
            stat = path.stat()
            key = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
            if key not in hash_cache:
                hash_cache[key] = digest(path)
            require(hash_cache[key] == expected, f"Input hash mismatch: {path}")
            if name in {"images.bin", "labels.bin", "train_indices.bin", "val_indices.bin"}:
                require(
                    expected == manifest["seeds"]["42"]["hashes"][name],
                    f"Shared dataset differs between seeds: {name}",
                )
            verified += 1
        initialization = array_file(folder / "init.bin", "<f4", (5418,))
        torch.manual_seed(seed)
        np.testing.assert_array_equal(initialization, flat_parameters(SimpleCNN(10)))
        schedule = array_file(folder / "schedule.bin", "<u4", (25, 45000, 2))
        require(np.all(schedule[:, :, 1] <= 1), "Invalid horizontal flip flags")
        for epoch in schedule:
            np.testing.assert_array_equal(np.sort(epoch[:, 0]), indices["train"])
    return manifest, images, labels, indices, verified


def verify_source(run_manifest):
    source = run_manifest["source"]
    revision = source["git_revision"]
    require(
        len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
        "Missing fixed Git source revision",
    )
    require(source.get("working_tree_clean") is True, "Measured source tree was not clean")
    require(source.get("binary_sha256"), "Missing measured executable hash")
    require(source.get("source_hashes"), "Missing measured assembly source hashes")
    repo = Path(__file__).resolve().parent.parent
    require(
        digest(repo / "assembly/build/cnn-assembly") == source["binary_sha256"],
        "Current executable differs from measured executable",
    )
    source_hashes = source["source_hashes"]
    required = {"assembly/model.S", "assembly/convolution.S", "assembly/runtime.S"}
    require(required.issubset(source_hashes), "Missing numerical source hashes")
    for name, expected in source_hashes.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe source reference")
        require(digest(repo / path) == expected, f"Measured source changed: {name}")
    seed_runs = run_manifest.get("seed_runs", {})
    require(set(seed_runs) == {str(seed) for seed in SEEDS}, "Missing per-seed run provenance")
    for seed in SEEDS:
        recorded_source = seed_runs[str(seed)]["source"]
        for name in ("git_revision", "binary_sha256", "source_hashes"):
            require(recorded_source[name] == source[name], f"Seed {seed} source mismatch: {name}")
        require(recorded_source.get("working_tree_clean") is True, f"Seed {seed} dirty source")
        require(seed_runs[str(seed)].get("completed_at_utc"), f"Seed {seed} not complete")
    return revision


def checkpoint_audit(weights, raw_images, saved_probabilities):
    """Inference only: independently replay 128 real images in PyTorch."""
    model = load_parameters(SimpleCNN(10), weights).eval()
    inputs = torch.from_numpy(raw_images.copy()).permute(0, 3, 1, 2).float()
    inputs.div_(255.0).sub_(0.5).div_(0.5)
    with torch.inference_mode():
        expected = model(inputs.contiguous()).softmax(dim=1).numpy()
    actual = saved_probabilities[:128]
    np.testing.assert_allclose(
        actual,
        expected,
        atol=3e-5,
        rtol=1e-4,
        err_msg="Assembly checkpoint/prediction independent replay",
    )
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "status": "passed",
        "images": 128,
        "split": "first 128 sorted training indices",
        "operation": "PyTorch inference from saved assembly checkpoint; no fitting",
        "atol": 3e-5,
        "rtol": 1e-4,
        "max_absolute_probability_error": float(difference.max()),
        "max_tolerance_ratio": float(np.max(difference / (3e-5 + 1e-4 * np.abs(expected)))),
        "predicted_class_disagreements": int(
            np.count_nonzero(actual.argmax(1) != expected.argmax(1))
        ),
    }


def adapt_performance(folder, raw, history, seed):
    performance = {
        **raw,
        "seed": seed,
        "trainable_parameter_count": 5418,
        "best_val_accuracy": history[raw["best_epoch"] - 1]["val_accuracy"],
        "checkpoint_bytes": (folder / "best.bin").stat().st_size,
        "checkpoint_sha256": digest(folder / "best.bin"),
        "training_images_per_second": 25 * 45000 / sum(row["train_seconds"] for row in history),
        "training_images_per_second_scope": "Training only; excludes validation/checkpointing",
        "prediction_seconds_by_split": raw["prediction_seconds"],
        "training_seconds_scope": "25 train/validation/checkpoint epochs plus epoch reporting",
        "final_evaluation_scope": "Selected-checkpoint inference, softmax and binary writes; "
        "excludes Python metrics/audit",
        "process_cpu_seconds": raw["process_user_cpu_seconds"] + raw["process_system_cpu_seconds"],
        "process_cpu_scope": "Whole executable lifetime, including setup, train, evaluation, "
        "and timing; not training-only CPU",
        "peak_rss_scope": "Whole executable lifetime; includes image data; excludes "
        "separate Python preparation/report process",
        "inference": {},
    }
    for batch in (1, 128):
        samples = array_file(folder / f"timing_batch{batch}.bin", "<f8", (100,))
        require(np.all(samples > 0), "Nonpositive latency sample")
        performance["inference"][f"batch{batch}"] = {
            "actual_batch_size": batch,
            "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
            "images_per_second": float(batch * 1000 / samples.mean()),
            "samples_ms": samples.tolist(),
            "warmup_iterations": 20,
            "measured_iterations": 100,
            "scope": "CPU forward only; preprocessed fixed test inputs; excludes I/O, "
            "transform and softmax; "
            + ("calling thread" if batch == 1 else "two-worker dispatch overhead included"),
        }
    return performance


def paired_comparison(seed, new_folder, baseline):
    with (
        np.load(new_folder / "predictions_test.npz") as new,
        np.load(baseline / f"seed-{seed}/predictions_test.npz") as old,
    ):
        np.testing.assert_array_equal(new["indices"], old["indices"])
        np.testing.assert_array_equal(new["labels"], old["labels"])
        a, b = new["probabilities"].argmax(1), old["probabilities"].argmax(1)
        correct_a, correct_b = a == new["labels"], b == old["labels"]
        delta = correct_a.astype(int) - correct_b.astype(int)
        counts = np.bincount(delta + 1, minlength=3)
        samples = np.random.default_rng(20261002 + seed).multinomial(
            len(delta),
            counts / len(delta),
            size=10000,
        )
        differences = (samples[:, 2] - samples[:, 0]) / len(delta)
        lower, upper = np.quantile(differences, [0.025, 0.975])
        baseline_metric = read_json(baseline / f"seed-{seed}/metrics.json")["test"]["accuracy"]
        require(abs(correct_b.mean() - baseline_metric) < 1e-12, "Baseline prediction audit failed")
        return {
            "seed": seed,
            "test_examples": len(delta),
            "assembly_accuracy": float(correct_a.mean()),
            "pytorch_accuracy": float(correct_b.mean()),
            "accuracy_difference": float(delta.mean()),
            "accuracy_difference_ci95": {
                "lower": float(lower),
                "upper": float(upper),
                "confidence_level": 0.95,
                "method": "paired iid test-image percentile bootstrap via multinomial "
                "{-1,0,+1} counts",
                "resamples": 10000,
                "seed": 20261002 + seed,
                "interpretation": "Conditional on these two fitted models; excludes training-seed "
                "uncertainty",
            },
            "prediction_disagreements": int(np.count_nonzero(a != b)),
            "prediction_disagreement_rate": float(np.mean(a != b)),
            "assembly_only_correct": int(counts[2]),
            "pytorch_only_correct": int(counts[0]),
            "both_correct": int(np.count_nonzero(correct_a & correct_b)),
            "both_incorrect": int(np.count_nonzero(~correct_a & ~correct_b)),
            "mean_absolute_probability_difference": float(
                np.abs(new["probabilities"].astype(np.float64) - old["probabilities"]).mean()
            ),
        }


def build_comparison(output, baseline):
    summary, old_summary = read_json(output / "summary.json"), read_json(baseline / "summary.json")
    rows = [paired_comparison(seed, output / f"seed-{seed}", baseline) for seed in SEEDS]
    metrics = {}
    for name in METRICS:
        key = f"test.{name}"
        if key in summary["aggregate"] and key in old_summary["aggregate"]:
            a, b = summary["aggregate"][key], old_summary["aggregate"][key]
            metrics[name] = {"assembly": a, "pytorch": b, "mean_difference": a["mean"] - b["mean"]}
    timings = {}
    for name in (
        "training_seconds",
        "training_images_per_second",
        "peak_rss_bytes",
        "inference.batch1.median_ms",
        "inference.batch128.median_ms",
    ):
        key = f"performance.{name}"
        a, b = summary["aggregate"][key], old_summary["aggregate"][key]
        timings[name] = {
            "assembly": a,
            "pytorch": b,
            "ratio_assembly_over_pytorch": a["mean"] / b["mean"],
        }
    result = {
        "baseline": "cifar10-mac-cpu-25-v1",
        "seeds": list(SEEDS),
        "per_seed_paired_test": rows,
        "test_metrics": metrics,
        "paired_accuracy_differences": aggregate([row["accuracy_difference"] for row in rows]),
        "performance": timings,
        "interpretation": [
            "Same mathematical architecture, exact initial weights, image "
            "order, flip schedule, splits, batch size, optimizer constants "
            "and 25 epochs; floating-point reduction order differs.",
            "Results do not establish bitwise identity or statistical "
            "equivalence. A paired interval covering zero is not an "
            "equivalence test.",
            "These are two CPU implementations on the same Mac with two "
            "compute workers/threads; PyTorch executes optimized native C++ "
            "kernels, not Python loops for convolution.",
            "Assembly shuffle order and flip RNG decisions are precomputed outside its "
            "training timer; the PyTorch DataLoader performs shuffle/flip RNG work inside "
            "its timer. Pixel normalization and applying horizontal flips remain inside "
            "both training timers. End-to-end preparation is not included in the ratio.",
            "Timing differences include kernel layout, batching, allocation "
            "and worker scheduling; they do not isolate programming-language "
            "overhead.",
            "Runs were sequential, not randomized/interleaved controlled "
            "thermal trials; system load and frequency may differ.",
            "Assembly final evaluation excludes separate Python metric "
            "computation; baseline final evaluation includes it, so that "
            "duration is not directly compared.",
            "Assembly process CPU is whole-lifetime; baseline "
            "training_cpu_seconds is training-only, so those quantities are "
            "not directly compared.",
            "RSS is whole-process high-water memory for different runtimes, "
            "not isolated model memory; assembly excludes its separate "
            "preparation/report processes.",
            "All three seeds share the same 10000 test images. Paired "
            "bootstrap intervals condition on each fitted model pair; seed "
            "sample SD is descriptive with only three seeds.",
        ],
    }
    write_json(output / "comparison.json", result)
    return result


def write_assessment(output, comparison):
    accuracy = comparison["test_metrics"]["accuracy"]
    macro = comparison["test_metrics"]["macro_f1"]
    runtime = comparison["performance"]["training_seconds"]
    lines = [
        "# Handwritten ARM64 CNN: 25-epoch comparison",
        "",
        "Three seeds (42, 43, 44) completed the full CIFAR-10 protocol: 45,000 training, "
        "5,000 validation and 10,000 test images; 25 epochs; batch 128; Adam; two CPU workers.",
        "",
        f"Assembly test accuracy is **{100 * accuracy['assembly']['mean']:.2f}% ± "
        f"{100 * accuracy['assembly']['sample_std']:.2f} percentage points** (seed sample SD), "
        f"compared with **{100 * accuracy['pytorch']['mean']:.2f}% ± "
        f"{100 * accuracy['pytorch']['sample_std']:.2f}** for the original PyTorch run. "
        f"The mean difference is **{100 * accuracy['mean_difference']:+.2f} percentage points**. "
        f"Macro F1 is {macro['assembly']['mean']:.4f} versus {macro['pytorch']['mean']:.4f}.",
        "",
        "| Seed | Assembly accuracy | PyTorch accuracy | Difference (pp) "
        "| Paired 95% interval (pp) | Prediction disagreement |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in comparison["per_seed_paired_test"]:
        ci = row["accuracy_difference_ci95"]
        lines.append(
            f"| {row['seed']} | {100 * row['assembly_accuracy']:.2f}% | "
            f"{100 * row['pytorch_accuracy']:.2f}% | {100 * row['accuracy_difference']:+.2f} | "
            f"[{100 * ci['lower']:+.2f}, {100 * ci['upper']:+.2f}] | "
            f"{100 * row['prediction_disagreement_rate']:.2f}% |"
        )
    lines += [
        "",
        "The paired intervals resample test images for each fixed model pair. They do not "
        "measure training-seed uncertainty and are not equivalence tests.",
        "",
        f"Mean training/validation/checkpoint time was {runtime['assembly']['mean']:.2f} seconds "
        f"for assembly and {runtime['pytorch']['mean']:.2f} seconds for PyTorch; assembly/PyTorch "
        f"duration ratio = {runtime['ratio_assembly_over_pytorch']:.3f}. "
        "Both ran on this Mac's CPU. "
        "PyTorch dispatches optimized native kernels; this is an implementation comparison, "
        "not a comparison against convolution implemented as interpreted Python loops. "
        "Assembly receives precomputed shuffle order and flip RNG decisions outside its "
        "training timer, whereas the baseline DataLoader performs that work inside its timer. "
        "Pixel normalization and applying the flips remain inside both training timers. "
        "The ratio therefore excludes assembly preparation and is not end-to-end time.",
        "",
        "The replica reproduces the mathematical CNN and fixed training protocol. It does not "
        "promise bitwise-identical floating-point computations: different reductions can change "
        "the optimization trajectory over many updates. Exact initial weights, permutations, "
        "flip decisions and dataset memberships are retained. Kernel/gradient/optimizer parity "
        "is checked separately; saved final checkpoints are independently replayed on 128 real "
        "training images per seed using PyTorch inference only. All reported full-dataset metrics "
        "use probabilities produced by the assembly executable.",
        "",
        "Timing scopes differ for final evaluation and CPU time; see "
        "comparison.json. Whole-process "
        "peak RSS includes different runtime overheads and excludes assembly's separate Python "
        "preparation/reporting process. Runs were sequential and can experience different thermal "
        "and background-load conditions.",
        "",
        "- [Detailed metrics and learning curves](report.md)",
        "- [Paired comparison and scope notes](comparison.json)",
        "- [Independent audit](audit.json)",
        "- [Source, protocol and environment](manifest.json)",
        "- [File checksums](checksums.json)",
        "",
        "Raw executable outputs are retained under each seed's raw/ directory. "
        "prepared/ retains initialization, schedules, labels and split indices; the 184 MB "
        "image tensor is referenced by hashes and reproducible from "
        "CIFAR-10, rather than duplicated. "
        "No test result selected the checkpoint or discarded a seed.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def collect(args):
    collector_start = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not args.output.exists() or not any(args.output.iterdir()), "Refusing existing output")
    run_manifest = read_json(args.inputs / "manifest.json")
    revision = verify_source(run_manifest)
    prepared, images, labels, indices, hash_count = verify_inputs(args.prepared, args.baseline)
    require(run_manifest["protocol"] == prepared["protocol"], "Run protocol differs from inputs")
    # Validate all completion records before making any output directory.
    histories = {seed: history_and_performance(args.inputs / f"seed-{seed}") for seed in SEEDS}
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.inputs / "manifest.json", args.output / "run_manifest.json")
    for name in ("numerical-parity.json", "runtime-parity.json", "runtime-safety.json"):
        if (args.inputs / name).is_file():
            shutil.copy2(args.inputs / name, args.output / name)
    if (args.inputs / "source").is_dir():
        shutil.copytree(args.inputs / "source", args.output / "source")
    if (args.inputs / "cnn-assembly").is_file():
        require(
            digest(args.inputs / "cnn-assembly") == run_manifest["source"]["binary_sha256"],
            "Retained executable does not match measured executable",
        )
        shutil.copy2(args.inputs / "cnn-assembly", args.output / "cnn-assembly")
    prepared_out = args.output / "prepared"
    prepared_out.mkdir()
    shutil.copy2(args.prepared / "manifest.json", prepared_out / "manifest.json")
    for name in ("labels.bin", "train_indices.bin", "val_indices.bin"):
        shutil.copy2(args.prepared / "shared" / name, prepared_out / name)
    audits = []
    classes = prepared["protocol"]["class_names"]
    for seed in SEEDS:
        raw_folder = args.inputs / f"seed-{seed}"
        folder = args.output / f"seed-{seed}"
        shutil.copytree(raw_folder, folder / "raw")
        root_log = args.inputs / f"seed-{seed}-console.log"
        if root_log.is_file():
            log_target = folder / "raw/console.log"
            require(
                not log_target.exists() or digest(root_log) == digest(log_target),
                "Conflicting per-seed console logs",
            )
            shutil.copy2(root_log, log_target)
        prepared_seed = prepared_out / f"seed-{seed}"
        prepared_seed.mkdir()
        for name in ("init.bin", "schedule.bin", "inputs.json"):
            shutil.copy2(args.prepared / f"seed-{seed}" / name, prepared_seed / name)
        weights = array_file(raw_folder / "best.bin", "<f4", (5418,))
        array_file(raw_folder / "last.bin", "<f4", (5418,))
        history, raw_perf = histories[seed]
        metrics, checkpoint_check = {}, None
        for split, count in COUNTS.items():
            probabilities = array_file(raw_folder / f"predictions_{split}.bin", "<f4", (count, 10))
            require(
                np.all((probabilities >= 0) & (probabilities <= 1)), "Invalid probability range"
            )
            require(
                np.allclose(probabilities.astype(np.float64).sum(1), 1, rtol=0, atol=1e-6),
                "Probability rows do not sum to one",
            )
            example_indices = indices[split]
            global_indices = example_indices + (50000 if split == "test" else 0)
            truth = labels[global_indices]
            np.savez_compressed(
                folder / f"predictions_{split}.npz",
                labels=truth,
                probabilities=probabilities,
                indices=example_indices,
            )
            metrics[split] = classification_metrics(truth, probabilities, classes, seed=seed)
            metrics[split]["interval_interpretation"] = (
                "Conditional on this fitted model; iid held-out test examples; "
                "excludes seed variation"
                if split == "test"
                else "Descriptive only: fitted/selected on these examples; no "
                "generalization inference"
            )
            if split == "train":
                checkpoint_check = checkpoint_audit(
                    weights, images[global_indices[:128]], probabilities
                )
        selected = history[raw_perf["best_epoch"] - 1]
        require(
            abs(metrics["validation"]["accuracy"] - selected["val_accuracy"]) < 1e-10,
            "Saved selected checkpoint validation accuracy mismatch",
        )
        # Runtime loss is stable logsumexp; metric NLL clips extreme probabilities.
        # At normal logits they should agree to float32 rounding. A huge mismatch is invalid.
        nll = metrics["validation"]["negative_log_likelihood"]
        require(
            abs(nll - selected["val_loss"]) < 1e-4,
            "Selected checkpoint validation loss disagrees with probabilities",
        )
        write_json(folder / "history.json", history)
        write_json(folder / "metrics.json", metrics)
        write_json(
            folder / "performance.json", adapt_performance(raw_folder, raw_perf, history, seed)
        )
        audits.append(
            {
                "seed": seed,
                "selected_epoch": raw_perf["best_epoch"],
                "test_accuracy": metrics["test"]["accuracy"],
                "checkpoint_replay": checkpoint_check,
            }
        )
    np.savez_compressed(args.output / "split_indices.npz", **indices)
    manifest = {
        **run_manifest,
        "protocol": prepared["protocol"],
        "data": prepared["baseline_data"],
        "implementation": "Handwritten Apple ARM64 assembly; all fitting in standalone executable",
        "prepared_images_reference": {
            "filename": "images.bin",
            "shape": [60000, 32, 32, 3],
            "dtype": "uint8 NHWC",
            "bytes": 60000 * 32 * 32 * 3,
            "sha256": prepared["seeds"]["42"]["hashes"]["images.bin"],
            "rebuild": "Run assembly/prepare.py against the recorded CIFAR-10 data and baseline",
        },
        "collector": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "source_sha256": digest(Path(__file__)),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(
        args.output / "audit.json",
        {
            "status": "passed",
            "source_revision": revision,
            "prepared_files_verified_by_sha256": hash_count,
            "prediction_rows_verified": 3 * sum(COUNTS.values()),
            "unique_test_examples": 10000,
            "checkpoint_replay_images": 3 * 128,
            "results": audits,
            "checks": [
                "fixed source/executable hashes and three complete seed records",
                "exact dataset and split fingerprints against retained Mac baseline",
                "exact seed initializations and 25 valid permutations/flip schedules",
                "25 full epochs, 45000 training examples, two workers, 5418 parameters",
                "validation-only checkpoint selection at every epoch",
                "finite normalized predictions for all 180000 rows",
                "independent PyTorch checkpoint replay, inference only",
                "validation metrics agree with selected history checkpoint",
            ],
        },
    )
    build_report(args.output)
    report = (args.output / "report.md").read_text()
    report = report.replace(
        "# Simple CNN · CIFAR-10 baseline", "# Handwritten ARM64 CNN · CIFAR-10"
    )
    report = report.replace("/best.pt)", "/raw/best.bin)")
    report = report.replace(
        "## Runtime and resource use\n",
        "## Runtime and resource use\n\n"
        "Assembly final evaluation excludes Python metrics and audit. Process CPU "
        "and peak RSS cover the complete executable; see comparison.json for scope "
        "differences from the PyTorch baseline.\n",
    )
    (args.output / "report.md").write_text(report)
    comparison = build_comparison(args.output, args.baseline)
    write_assessment(args.output, comparison)
    manifest["collector"]["wall_seconds_before_final_checksums"] = (
        time.perf_counter() - collector_start
    )
    manifest["collector"]["wall_seconds_scope"] = (
        "Separate Python collection, source/data audits, checkpoint replay, metrics, "
        "paired comparisons and report/plot generation; excludes assembly runtime and "
        "final checksum manifest serialization"
    )
    manifest["collector"]["finished_at_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output / "manifest.json", manifest)
    write_json(
        args.output / "checksums.json",
        {
            str(path.relative_to(args.output)): digest(path)
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output),
                "source_revision": revision,
                "test_accuracy": comparison["test_metrics"]["accuracy"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("results/assembly-runs-v1"))
    parser.add_argument("--prepared", type=Path, default=Path("results/assembly-inputs-v1"))
    parser.add_argument("--baseline", type=Path, default=Path("../cifar10-mac-cpu-25-v1"))
    parser.add_argument("--output", type=Path, default=Path("../cifar10-assembly-25-v1"))
    collect(parser.parse_args())
