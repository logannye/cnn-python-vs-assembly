"""Train the PyTorch CNN using the same fixed binary inputs as the assembly runtime.

This executable deliberately does not use a DataLoader, PIL, an RNG, or a
preprocessing cache during training. See PYTHON_RUNTIME.md and assembly/ABI.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from classifier.model import SimpleCNN

BATCH_SIZE = 128
PARAMETER_COUNT = 5418
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "train_seconds",
    "val_seconds",
    "epoch_seconds",
    "selected_epoch",
)


def read_exact(root: Path, name: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Read every byte into owned memory, rejecting short or trailing data."""
    path = root / name
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected exactly {expected} bytes")
    result = np.fromfile(path, dtype=dtype)
    if result.nbytes != expected:
        raise ValueError(f"{path}: changed size while being read")
    return result.reshape(shape)


def load_inputs(root: Path, epochs: int) -> dict[str, np.ndarray]:
    arrays = {
        "images": read_exact(root, "images.bin", "u1", (60000, 32, 32, 3)),
        "labels": read_exact(root, "labels.bin", "<u4", (60000,)),
        "train": read_exact(root, "train_indices.bin", "<u4", (45000,)),
        "validation": read_exact(root, "val_indices.bin", "<u4", (5000,)),
        "initial": read_exact(root, "init.bin", "<f4", (PARAMETER_COUNT,)),
        "schedule": read_exact(root, "schedule.bin", "<u4", (epochs, 45000, 2)),
    }
    if (arrays["labels"] >= 10).any() or not np.isfinite(arrays["initial"]).all():
        raise ValueError("Invalid labels or nonfinite initial parameters")
    for name in ("train", "validation"):
        ids = arrays[name]
        if (ids >= 50000).any() or (ids[1:] <= ids[:-1]).any():
            raise ValueError(f"{name}: indices must be sorted, unique and in [0, 50000)")
    if not np.array_equal(
        np.sort(np.concatenate((arrays["train"], arrays["validation"]))), np.arange(50000)
    ):
        raise ValueError("Train and validation must partition the original 50000 images")
    for rows in arrays["schedule"]:
        if (rows[:, 1] > 1).any() or not np.array_equal(np.sort(rows[:, 0]), arrays["train"]):
            raise ValueError("Every schedule must permute the train split with binary flip flags")
    arrays["test"] = np.arange(50000, 60000, dtype=np.uint32)
    return arrays


def flatten(model: nn.Module) -> np.ndarray:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).numpy()


def load_parameters(model: nn.Module, flat: np.ndarray) -> None:
    cursor = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(torch.from_numpy(flat[cursor : cursor + count]).view_as(parameter))
            cursor += count
    if cursor != PARAMETER_COUNT:
        raise ValueError("Unexpected model parameter layout")


def normalize(
    images: np.ndarray, indices: np.ndarray, flags: np.ndarray | None = None
) -> torch.Tensor:
    """Compute the exact float32 pixel / 255, subtract .5, divide .5 transform."""
    # Advanced indexing materializes just this minibatch of raw uint8 pixels.
    batch = torch.from_numpy(images[indices]).permute(0, 3, 1, 2).contiguous().float()
    batch.div_(255.0)
    if flags is not None:
        flipped = torch.from_numpy(flags.astype(np.bool_))
        batch[flipped] = batch[flipped].flip(-1)
    batch.sub_(0.5).div_(0.5)
    return batch


def usage() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_SELF)


def rss_bytes(snapshot: resource.struct_rusage) -> int:
    # Darwin reports bytes; Linux reports KiB. Both children run on Darwin in
    # the official experiment; this conversion also permits bounded CI smokes.
    return int(snapshot.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


def cpu_seconds(snapshot: resource.struct_rusage) -> float:
    return snapshot.ru_utime + snapshot.ru_stime


def finite(tensors: list[torch.Tensor], name: str) -> None:
    if not torch.isfinite(torch.cat([tensor.detach().reshape(-1) for tensor in tensors])).all():
        raise FloatingPointError(f"Nonfinite {name}")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> tuple[float, float]:
    loss_sum, correct = 0.0, 0
    for start in range(0, len(indices), BATCH_SIZE):
        selected = indices[start : start + BATCH_SIZE]
        logits = model(normalize(arrays["images"], selected))
        targets = torch.from_numpy(arrays["labels"][selected].astype(np.int64))
        batch_probabilities = logits.softmax(1)
        finite([logits, batch_probabilities], "evaluation logits or probabilities")
        batch_loss = F.cross_entropy(logits, targets, reduction="sum").item()
        if not math.isfinite(batch_loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        loss_sum += batch_loss
        correct += (logits.argmax(1) == targets).sum().item()
        if probabilities is not None:
            probabilities[start : start + len(selected)] = batch_probabilities.numpy()
    return loss_sum / len(indices), correct / len(indices)


@torch.inference_mode()
def forward_timing(model: nn.Module, inputs: torch.Tensor, output: Path) -> None:
    for size in (1, 128):
        batch = inputs[:size]
        for _ in range(20):
            model(batch)
        samples = np.empty(100, dtype="<f8")
        for repeat in range(100):
            started = time.perf_counter()
            model(batch)
            samples[repeat] = (time.perf_counter() - started) * 1000.0
        if not np.isfinite(samples).all() or (samples <= 0).any():
            raise FloatingPointError("Invalid forward timing samples")
        samples.tofile(output / f"timing_batch{size}.bin")


def run(inputs: Path, output: Path, epochs: int, train_limit: int) -> None:
    if not 1 <= epochs <= 25 or not 1 <= train_limit <= 45000:
        raise ValueError("epochs must be 1..25 and train_limit must be 1..45000")
    eval_limit = int(os.environ.get("CNN_ASM_EVAL_LIMIT", "45000"))
    if not 1 <= eval_limit <= 45000:
        raise ValueError("CNN_ASM_EVAL_LIMIT must be 1..45000")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty output directory")
    output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    arrays = load_inputs(inputs, epochs)
    model = SimpleCNN(10).cpu()
    load_parameters(model, arrays["initial"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    indices = {name: arrays[name][:eval_limit] for name in ("train", "validation", "test")}
    probabilities = np.empty((45000, 10), dtype=np.float32)
    best, best_rank, best_epoch = None, (-1.0, -math.inf), 0
    best_checkpoint_writes = 0

    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        loaded_usage = usage()
        training_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_start = epoch_start
            model.train()
            loss_sum, correct = 0.0, 0
            for start in range(0, train_limit, BATCH_SIZE):
                rows = arrays["schedule"][epoch - 1, start : min(start + BATCH_SIZE, train_limit)]
                batch = normalize(arrays["images"], rows[:, 0], rows[:, 1])
                targets = torch.from_numpy(arrays["labels"][rows[:, 0]].astype(np.int64))
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                finite([logits, logits.detach().softmax(1)], "training logits or probabilities")
                loss = F.cross_entropy(logits, targets)
                batch_loss = loss.item()
                if not math.isfinite(batch_loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                finite([parameter.grad for parameter in model.parameters()], "gradients")
                optimizer.step()
                finite(
                    [
                        tensor
                        for parameter in model.parameters()
                        for tensor in (
                            parameter,
                            optimizer.state[parameter]["exp_avg"],
                            optimizer.state[parameter]["exp_avg_sq"],
                        )
                    ],
                    "parameters or Adam moments",
                )
                loss_sum += batch_loss * len(rows)
                correct += (logits.detach().argmax(1) == targets).sum().item()
            train_seconds = time.perf_counter() - train_start
            model.eval()
            val_start = time.perf_counter()
            val_loss, val_accuracy = evaluate(model, arrays, indices["validation"])
            val_seconds = time.perf_counter() - val_start
            rank = (val_accuracy, -val_loss)
            if rank > best_rank:
                best, best_rank, best_epoch = flatten(model), rank, epoch
                best.astype("<f4", copy=False).tofile(output / "best.bin")
                best_checkpoint_writes += 1
            epoch_seconds = time.perf_counter() - epoch_start
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": format(loss_sum / train_limit, ".12g"),
                    "train_accuracy": format(correct / train_limit, ".12g"),
                    "val_loss": format(val_loss, ".12g"),
                    "val_accuracy": format(val_accuracy, ".12g"),
                    "train_seconds": format(train_seconds, ".9f"),
                    "val_seconds": format(val_seconds, ".9f"),
                    "epoch_seconds": format(epoch_seconds, ".9f"),
                    "selected_epoch": best_epoch,
                }
            )
            stream.flush()
            print(
                f"Epoch {epoch}/{epochs}: train loss={loss_sum / train_limit:.6f} "
                f"accuracy={correct / train_limit:.4f}, val loss={val_loss:.6f} "
                f"accuracy={val_accuracy:.4f}, {epoch_seconds:.3f}s",
                flush=True,
            )
        training_seconds = time.perf_counter() - training_start
        training_usage = usage()

    flatten(model).astype("<f4", copy=False).tofile(output / "last.bin")
    if best is None:
        raise RuntimeError("No checkpoint selected")
    load_parameters(model, best)
    model.eval()
    prediction_seconds = {}
    evaluation_start = time.perf_counter()
    for split, selected in indices.items():
        started = time.perf_counter()
        evaluate(model, arrays, selected, probabilities)
        prediction_seconds[split] = time.perf_counter() - started
        probabilities[: len(selected)].astype("<f4", copy=False).tofile(
            output / f"predictions_{split}.bin"
        )
    final_evaluation_seconds = time.perf_counter() - evaluation_start
    timing_inputs = normalize(arrays["images"], arrays["test"][:128])
    forward_timing(model, timing_inputs, output)
    end_usage = usage()
    performance = {
        "training_seconds": training_seconds,
        "training_cpu_seconds": cpu_seconds(training_usage) - cpu_seconds(loaded_usage),
        "peak_rss_loaded_bytes": rss_bytes(loaded_usage),
        "peak_rss_training_bytes": rss_bytes(training_usage),
        "best_epoch": best_epoch,
        "best_checkpoint_writes": best_checkpoint_writes,
        "best_checkpoint_bytes_written": best_checkpoint_writes * PARAMETER_COUNT * 4,
        "parameter_count": PARAMETER_COUNT,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "final_evaluation_seconds": final_evaluation_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_rss_bytes": rss_bytes(end_usage),
        "process_user_cpu_seconds": end_usage.ru_utime,
        "process_system_cpu_seconds": end_usage.ru_stime,
        "epochs": epochs,
        "train_limit": train_limit,
        "evaluation_counts": {split: len(selected) for split, selected in indices.items()},
        "training_timing_scope": (
            "epoch train/validation, best checkpoint writes, CSV flushes and progress; "
            "excludes setup/input loading, final checkpoint, final evaluation and forward timing"
        ),
        "memory_scope": (
            "RUSAGE_SELF lifetime high-water resident bytes, sampled before training, "
            "after training and after final evaluation/timing; includes runtime and loaded inputs"
        ),
        "timing_scope": (
            "forward only; 20 warmups and 100 samples; batch1 and batch128 "
            "PyTorch eager CPU forward with two intraop threads"
        ),
        "floating_point": "float32 model/gradients; float64 reporting sums",
        "framework": {"torch": torch.__version__, "numpy": np.__version__},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "preprocessing": "uint8 NHWC -> contiguous float32 NCHW; /255; recorded flip; -.5; /.5",
        "optimizer": "torch.optim.Adam lr=.001 betas=(.9,.999) eps=1e-8 foreach=False fused=False",
    }
    (output / "performance.json").write_text(json.dumps(performance, indent=2) + "\n")
    print("Training, selected-checkpoint evaluation and timing complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("epochs", type=int, nargs="?", default=25)
    parser.add_argument("train_limit", type=int, nargs="?", default=45000)
    args = parser.parse_args()
    run(args.inputs, args.output, args.epochs, args.train_limit)


if __name__ == "__main__":
    main()
