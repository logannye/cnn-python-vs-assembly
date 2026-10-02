"""Audit a two-epoch executable smoke against the matching PyTorch recipe."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from prepare import digest, flat_parameters, load_parameters, normalize

from classifier.model import SimpleCNN


def verify(inputs, output, binary):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    images = np.memmap(inputs / "images.bin", mode="r", dtype=np.uint8, shape=(60000, 32, 32, 3))
    labels = np.fromfile(inputs / "labels.bin", dtype="<u4")
    schedule = np.fromfile(inputs / "schedule.bin", dtype="<u4").reshape(2, 45000, 2)
    initial = np.fromfile(inputs / "init.bin", dtype="<f4")
    model = load_parameters(SimpleCNN(10), initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    indices = {
        "train": np.fromfile(inputs / "train_indices.bin", dtype="<u4")[:129],
        "validation": np.fromfile(inputs / "val_indices.bin", dtype="<u4")[:129],
        "test": np.arange(50000, 50129, dtype=np.uint32),
    }
    history = list(csv.DictReader((output / "history.csv").open()))
    assert len(history) == 2
    checks = []

    def check(name, actual, expected, atol=1e-5, rtol=1e-4):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
        checks.append(
            {
                "name": name,
                "max_absolute_error": float(np.abs(a - b).max()),
                "atol": atol,
                "rtol": rtol,
            }
        )

    @torch.inference_mode()
    def predict(split):
        predictions = []
        loss, correct = 0.0, 0
        for start in range(0, len(indices[split]), 128):
            selected = indices[split][start : start + 128]
            logits = model(torch.stack([normalize(images[i]) for i in selected]))
            targets = torch.from_numpy(labels[selected].astype(np.int64))
            loss += F.cross_entropy(logits, targets, reduction="sum").item()
            correct += (logits.argmax(1) == targets).sum().item()
            predictions.append(logits.softmax(1).numpy())
        return (
            np.concatenate(predictions),
            loss / len(indices[split]),
            correct / len(indices[split]),
        )

    best, best_rank, best_epoch = None, (-1, float("-inf")), 0
    for epoch in range(2):
        online_loss, online_correct = 0.0, 0
        for start in range(0, 257, 128):
            rows = schedule[epoch, start : min(start + 128, 257)]
            x = torch.stack([normalize(images[i], bool(f)) for i, f in rows])
            targets = torch.from_numpy(labels[rows[:, 0]].astype(np.int64))
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            online_loss += loss.item() * len(rows)
            online_correct += (logits.argmax(1) == targets).sum().item()
        _, validation_loss, validation_accuracy = predict("validation")
        rank = (validation_accuracy, -validation_loss)
        if rank > best_rank:
            best, best_rank, best_epoch = flat_parameters(model).copy(), rank, epoch + 1
        row = history[epoch]
        check(f"epoch{epoch + 1}.train_loss", float(row["train_loss"]), online_loss / 257)
        check(
            f"epoch{epoch + 1}.train_accuracy",
            float(row["train_accuracy"]),
            online_correct / 257,
            1e-10,
            0,
        )
        check(f"epoch{epoch + 1}.val_loss", float(row["val_loss"]), validation_loss)
        check(
            f"epoch{epoch + 1}.val_accuracy",
            float(row["val_accuracy"]),
            validation_accuracy,
            1e-10,
            0,
        )
        assert int(row["selected_epoch"]) == best_epoch
    check(
        "last_parameters",
        np.fromfile(output / "last.bin", dtype="<f4"),
        flat_parameters(model),
        3e-5,
        1e-4,
    )
    check("best_parameters", np.fromfile(output / "best.bin", dtype="<f4"), best, 3e-5, 1e-4)
    load_parameters(model, best)
    for split in indices:
        expected, _, _ = predict(split)
        actual = np.fromfile(output / f"predictions_{split}.bin", dtype="<f4").reshape(129, 10)
        check(f"{split}.probabilities", actual, expected, 3e-6, 3e-5)
    for size in (1, 128):
        times = np.fromfile(output / f"timing_batch{size}.bin", dtype="<f8")
        assert times.shape == (100,) and np.isfinite(times).all() and (times > 0).all()
    performance = json.loads((output / "performance.json").read_text())
    assert performance["best_epoch"] == best_epoch
    assert performance["evaluation_counts"] == dict.fromkeys(indices, 129)
    assert (
        performance["epochs"] == 2
        and performance["train_limit"] == 257
        and performance["threads"] == 2
    )
    result = {
        "status": "passed",
        "binary_sha256": digest(binary),
        "checks": checks,
        "best_epoch": best_epoch,
        "scope": (
            "Two epochs, 257 training records per epoch (128+128+1), "
            "129 evaluation records per split, fixed initialization/schedule. "
            "Verifies exact accuracy and selected checkpoint plus tolerance-based "
            "losses, weights, predictions; 100 finite timing samples each."
        ),
    }
    (output / "runtime-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("results/assembly-smoke-inputs"))
    parser.add_argument("--output", type=Path, default=Path("results/assembly-smoke-v1"))
    parser.add_argument("--binary", type=Path, default=Path("assembly/build/cnn-assembly"))
    args = parser.parse_args()
    verify(args.inputs, args.output, args.binary)
