"""Reproducible CIFAR-10 baseline with retained predictions and measurements."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from classifier.checkpoint import load_checkpoint
from classifier.data import build_transform
from classifier.evaluate import score
from classifier.metrics import classification_metrics
from classifier.model import SimpleCNN


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stratified_split(
    targets: np.ndarray, per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, disjoint train/validation indices into official training data."""
    if per_class <= 0:
        raise ValueError("Validation examples per class must be positive.")
    generator = np.random.default_rng(seed)
    training, validation = [], []
    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        if len(indices) <= per_class:
            raise ValueError("Each class must retain at least one training example.")
        shuffled = generator.permutation(indices)
        validation.extend(shuffled[:per_class])
        training.extend(shuffled[per_class:])
    return np.sort(training).astype(np.int64), np.sort(validation).astype(np.int64)


def loader(dataset, batch_size: int, seed: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def collect_predictions(model: nn.Module, batches: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    for images, targets in batches:
        labels.append(targets.numpy())
        probabilities.append(model(images).softmax(dim=1).numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


@torch.inference_mode()
def inference_timing(model: nn.Module, images: torch.Tensor) -> dict:
    """CPU forward-only timings on preprocessed inputs; exclude loading and softmax."""
    model.eval()
    result = {}
    for batch_size in (1, 128):
        batch = images[:batch_size].contiguous()
        for _ in range(20):
            model(batch)
        durations = []
        for _ in range(100):
            start = time.perf_counter()
            model(batch)
            durations.append(time.perf_counter() - start)
        result[f"batch{batch_size}"] = {
            "actual_batch_size": len(batch),
            "median_ms": float(np.median(durations) * 1000),
            "p95_ms": float(np.percentile(durations, 95) * 1000),
            "images_per_second": float(len(batch) / np.mean(durations)),
            "samples_ms": [duration * 1000 for duration in durations],
            "warmup_iterations": 20,
            "measured_iterations": 100,
            "scope": "CPU forward only; preprocessed input; excludes I/O, transforms, softmax",
        }
    return result


def environment() -> dict:
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "total_memory_bytes": psutil.virtual_memory().total,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "scikit-learn",
                "matplotlib",
                "Pillow",
                "psutil",
            )
        },
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def run_seed(args, seed: int, datasets: dict, split_indices: dict, class_names: list[str]) -> None:
    destination = args.output / f"seed-{seed}"
    destination.mkdir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SimpleCNN(num_classes=len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = loader(datasets["augmented_training"], args.batch_size, seed, shuffle=True)
    val_loader = loader(datasets["validation"], args.batch_size, seed)
    history = []
    best_rank = (-1.0, float("-inf"))
    best_epoch = 0
    process = psutil.Process()
    cpu_start = process.cpu_times()
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, correct, examples = 0.0, 0, 0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            examples += len(targets)
        train_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        validation = score(model, val_loader, torch.device("cpu"), len(class_names))
        val_seconds = time.perf_counter() - validation_start
        if not np.isfinite(validation["loss"]):
            raise ValueError("Nonfinite validation loss.")
        rank = (validation["accuracy"], -validation["loss"])
        if rank > best_rank:
            best_rank, best_epoch = rank, epoch
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "class_to_idx": {name: i for i, name in enumerate(class_names)},
                    "preprocessing": {
                        "image_size": 32,
                        "mean": [0.5] * 3,
                        "std": [0.5] * 3,
                    },
                    "config": {
                        "dataset": "CIFAR10",
                        "seed": seed,
                        "split_seed": args.split_seed,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "device": "cpu",
                    },
                    "epoch": epoch,
                    "val_metrics": validation,
                },
                destination / "best.pt",
            )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / examples,
            "train_accuracy": correct / examples,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "train_images_per_second": examples / train_seconds,
        }
        history.append(record)
        write_json(destination / "history.json", history)
        print(json.dumps({"seed": seed, **record}), flush=True)
    training_seconds = time.perf_counter() - training_start
    cpu_end = process.cpu_times()

    model, checkpoint = load_checkpoint(destination / "best.pt")
    evaluation_start = time.perf_counter()
    metrics, evaluation_seconds = {}, {}
    for split in ("train", "validation", "test"):
        start = time.perf_counter()
        labels, probabilities = collect_predictions(
            model, loader(datasets[split], args.batch_size, seed)
        )
        evaluation_seconds[split] = time.perf_counter() - start
        np.savez_compressed(
            destination / f"predictions_{split}.npz",
            labels=labels,
            probabilities=probabilities,
            indices=split_indices[split],
        )
        metrics[split] = classification_metrics(labels, probabilities, class_names, seed=seed)
        metrics[split]["interval_interpretation"] = (
            "Conditional on this fitted model; iid held-out test examples; excludes seed variation"
            if split == "test"
            else "Descriptive only: fitted/selected on these examples; no generalization inference"
        )
    final_evaluation_seconds = time.perf_counter() - evaluation_start
    write_json(destination / "metrics.json", metrics)

    timing_images, _ = next(iter(loader(datasets["test"], 128, seed)))
    timing = inference_timing(model, timing_images)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_rss *= 1024
    performance = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": checkpoint["val_metrics"]["accuracy"],
        "training_seconds": training_seconds,
        "training_cpu_seconds": cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system,
        "training_images_per_second": len(datasets["train"])
        * args.epochs
        / sum(row["train_seconds"] for row in history),
        "final_evaluation_seconds": final_evaluation_seconds,
        "prediction_seconds_by_split": evaluation_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "checkpoint_sha256": sha256(destination / "best.pt"),
        "peak_rss_bytes": peak_rss,
        "peak_rss_scope": "Whole process lifetime, including dataset, training, and metrics",
        "inference": timing,
    }
    write_json(destination / "performance.json", performance)
    print(
        json.dumps(
            {
                "seed": seed,
                "test_accuracy": metrics["test"]["accuracy"],
                "training_seconds": training_seconds,
            }
        ),
        flush=True,
    )


def run(args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty results: {args.output}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("Epochs, batch size, and threads must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = datetime.now(UTC).isoformat()
    download_start = time.perf_counter()
    train_data = CIFAR10(
        args.data_dir, train=True, download=True, transform=build_transform(32, augment=True)
    )
    evaluation_data = copy.copy(train_data)
    evaluation_data.transform = build_transform(32)
    test_data = CIFAR10(args.data_dir, train=False, download=False, transform=build_transform(32))
    download_seconds = time.perf_counter() - download_start
    targets = np.asarray(train_data.targets)
    training, validation = stratified_split(targets, 500, args.split_seed)
    split_indices = {
        "train": training,
        "validation": validation,
        "test": np.arange(len(test_data), dtype=np.int64),
    }
    if (len(training), len(validation), len(test_data)) != (45000, 5000, 10000):
        raise ValueError("Unexpected CIFAR-10 dataset sizes.")
    np.savez_compressed(args.output / "split_indices.npz", **split_indices)
    model_file = Path(__file__).with_name("model.py")
    manifest = {
        "schema_version": 1,
        "started_at_utc": started,
        "protocol": {
            "dataset": "CIFAR-10",
            "dataset_source": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "dataset_citation": (
                "Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009"
            ),
            "train_examples": 45000,
            "validation_examples": 5000,
            "test_examples": 10000,
            "split_seed": args.split_seed,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0,
            "scheduler": None,
            "image_size": 32,
            "normalization_mean": [0.5] * 3,
            "normalization_std": [0.5] * 3,
            "augmentation": "RandomHorizontalFlip(p=0.5) training only",
            "initialization": "PyTorch default, from scratch; no pretrained weights",
            "checkpoint_selection": "highest validation accuracy; ties use lower validation loss",
            "test_policy": "Evaluate once after all epochs and checkpoint selection for each seed",
            "training_curve_scope": "Augmented pre-update minibatch metrics; changing weights",
            "final_train_scope": "Deterministic unaugmented training split at selected checkpoint",
            "accuracy_interval": "95% Wilson; one fitted model, iid test examples",
            "macro_f1_interval": "95% bootstrap; 1000 multinomial draws from confusion counts",
            "ece_bins": 15,
            "class_names": train_data.classes,
            "random_uniform_accuracy": 0.1,
            "majority_class_accuracy": 0.1,
            "uniform_nll": float(np.log(10)),
            "uniform_brier_score": 0.9,
            "deterministic_algorithms": True,
        },
        "source": {
            "git_revision": git_output("rev-parse", "HEAD"),
            "working_tree_clean": not bool(git_output("status", "--porcelain")),
            "model_source_sha256": sha256(model_file),
            "command": sys.argv,
            "github_run_url": (
                f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
                if os.getenv("GITHUB_RUN_ID")
                else None
            ),
        },
        "data": {
            "split_indices_sha256": sha256(args.output / "split_indices.npz"),
            "training_images_sha256": hashlib.sha256(train_data.data.tobytes()).hexdigest(),
            "training_labels_sha256": hashlib.sha256(targets.astype("<i8").tobytes()).hexdigest(),
            "test_images_sha256": hashlib.sha256(test_data.data.tobytes()).hexdigest(),
            "test_labels_sha256": hashlib.sha256(
                np.asarray(test_data.targets, dtype="<i8").tobytes()
            ).hexdigest(),
            "index_convention": (
                "train/validation indices address official train set; "
                "test indices address official test set"
            ),
        },
        "environment": environment(),
        "dataset_download_and_loading_seconds": download_seconds,
    }
    write_json(args.output / "manifest.json", manifest)
    (args.output / "requirements-frozen.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    (args.output / "torch-build-config.txt").write_text(torch.__config__.show())
    datasets = {
        "augmented_training": Subset(train_data, training.tolist()),
        "train": Subset(evaluation_data, training.tolist()),
        "validation": Subset(evaluation_data, validation.tolist()),
        "test": test_data,
    }
    for seed in args.seeds:
        run_seed(args, seed, datasets, split_indices, train_data.classes)
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--output", type=Path, default=Path("results/cifar10-baseline"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
