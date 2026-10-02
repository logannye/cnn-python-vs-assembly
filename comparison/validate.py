"""Run and audit the bounded matched Python smoke before measured experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from python_train import load_inputs, normalize

REPO = Path(__file__).resolve().parents[1]
NUMERICAL_FILES = (
    "best.bin",
    "last.bin",
    "predictions_train.bin",
    "predictions_validation.bin",
    "predictions_test.bin",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(args: argparse.Namespace) -> dict:
    source = REPO / "comparison/python_train.py"
    binary = args.binary.resolve()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(REPO / "src"),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "CNN_ASM_EVAL_LIMIT": "129",
        }
    )
    subprocess.run(
        [sys.executable, str(source), str(args.inputs), str(args.output), "2", "257"],
        env=environment,
        check=True,
    )
    checks = []

    def check(name, actual, expected, atol=0.0, rtol=0.0):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f"{name}: nonfinite numerical output")
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
        checks.append(
            {
                "name": name,
                "max_absolute_error": float(np.max(np.abs(actual - expected))),
                "atol": atol,
                "rtol": rtol,
            }
        )

    # Reuse the established, independently written per-image PyTorch reference.
    # Its audit gives measured differences rather than merely another pass flag.
    subprocess.run(
        [
            sys.executable,
            str(REPO / "assembly/validate_runtime.py"),
            "--inputs",
            str(args.inputs),
            "--output",
            str(args.output),
            "--binary",
            str(binary),
        ],
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    reference = json.loads((args.output / "runtime-audit.json").read_text())
    if reference["status"] != "passed":
        raise AssertionError("Established PyTorch reference audit did not pass")
    exact_names = {"last_parameters", "best_parameters"} | {
        f"{split}.probabilities" for split in ("train", "validation", "test")
    }
    if {item["name"] for item in reference["checks"]} & exact_names != exact_names:
        raise AssertionError("Reference audit omitted required parameter/probability checks")
    for item in reference["checks"]:
        if item["name"] in exact_names:
            check(f"reference_exact.{item['name']}", item["max_absolute_error"], 0)

    # Import the existing transform reference without duplicating its arithmetic.
    sys.path.insert(0, str(REPO / "assembly"))
    from prepare import normalize as reference_normalize

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    arrays = load_inputs(args.inputs, 2)
    rows = arrays["schedule"][0, :257]
    actual = normalize(arrays["images"], rows[:, 0], rows[:, 1])
    expected = torch.stack(
        [reference_normalize(arrays["images"][index], bool(flag)) for index, flag in rows]
    )
    if not torch.equal(actual, expected):
        raise AssertionError("Batched transform differs from the per-image reference")
    checks.append({"name": "transform_257_records_bitwise", "max_absolute_error": 0})
    for name in NUMERICAL_FILES:
        count = 5418 if name in ("best.bin", "last.bin") else 1290
        actual = np.fromfile(args.output / name, dtype="<f4")
        expected = np.fromfile(args.assembly_output / name, dtype="<f4")
        if actual.shape != (count,) or expected.shape != (count,):
            raise AssertionError(f"{name}: invalid binary output length")
        parameters = name in ("best.bin", "last.bin")
        check(
            f"assembly.{name}",
            actual,
            expected,
            atol=3e-5 if parameters else 3e-6,
            rtol=1e-4 if parameters else 3e-5,
        )
        if not parameters:
            if (actual < 0).any() or (actual > 1).any():
                raise AssertionError(f"{name}: probabilities outside [0, 1]")
            check(f"probability_sums.{name}", actual.reshape(129, 10).sum(1), 1, 2e-6)

    with (args.output / "history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    with (args.assembly_output / "history.csv").open() as stream:
        assembly_history = list(csv.DictReader(stream))
    if len(history) != 2 or len(assembly_history) != 2:
        raise AssertionError("Smoke history must contain exactly two epochs")
    for epoch, (row, other) in enumerate(zip(history, assembly_history, strict=True), 1):
        check(f"epoch{epoch}.number", int(row["epoch"]), epoch)
        for field in ("train_loss", "val_loss", "train_accuracy", "val_accuracy", "selected_epoch"):
            check(
                f"epoch{epoch}.assembly.{field}",
                float(row[field]),
                float(other[field]),
                1e-5 if field.endswith("loss") else 0,
                1e-4 if field.endswith("loss") else 0,
            )
        for field in ("train_seconds", "val_seconds", "epoch_seconds"):
            if not math.isfinite(float(row[field])) or float(row[field]) <= 0:
                raise AssertionError(f"Invalid {field} in epoch {epoch}")
    performance = json.loads((args.output / "performance.json").read_text())
    for key, expected in {
        "epochs": 2,
        "train_limit": 257,
        "threads": 2,
        "interop_threads": 1,
        "parameter_count": 5418,
    }.items():
        check(f"protocol.{key}", performance[key], expected)
    check(
        "checkpoint.selected_epoch", performance["best_epoch"], int(history[-1]["selected_epoch"])
    )
    if performance["evaluation_counts"] != dict.fromkeys(("train", "validation", "test"), 129):
        raise AssertionError("Unexpected evaluation counts")
    positive = (
        "training_seconds",
        "training_cpu_seconds",
        "final_evaluation_seconds",
        "peak_rss_loaded_bytes",
        "peak_rss_training_bytes",
        "peak_rss_bytes",
    )
    for key in positive:
        if not math.isfinite(performance[key]) or performance[key] <= 0:
            raise AssertionError(f"Invalid positive performance metric: {key}")
    for value in performance["prediction_seconds"].values():
        if not math.isfinite(value) or value <= 0:
            raise AssertionError("Invalid final prediction timing")
    process_cpu = (
        performance["process_user_cpu_seconds"] + performance["process_system_cpu_seconds"]
    )
    if not math.isfinite(process_cpu) or process_cpu < performance["training_cpu_seconds"]:
        raise AssertionError("Training CPU time exceeds whole-process CPU time")
    if not (
        performance["peak_rss_loaded_bytes"]
        <= performance["peak_rss_training_bytes"]
        <= performance["peak_rss_bytes"]
    ):
        raise AssertionError("Lifetime RSS high-water samples are not monotonic")
    for size in (1, 128):
        values = np.fromfile(args.output / f"timing_batch{size}.bin", dtype="<f8")
        if values.shape != (100,) or not np.isfinite(values).all() or (values <= 0).any():
            raise AssertionError(f"Invalid timing_batch{size}.bin")
    checks.append({"name": "resource_metrics_and_200_timing_samples", "status": "passed"})
    return {
        "status": "passed",
        "source_sha256": digest(source),
        "binary_sha256": digest(binary),
        "reference_source_sha256": digest(REPO / "assembly/validate_runtime.py"),
        "inputs_sha256": {path.name: digest(path) for path in sorted(args.inputs.glob("*.bin"))},
        "python_outputs_sha256": {name: digest(args.output / name) for name in NUMERICAL_FILES},
        "assembly_outputs_sha256": {
            name: digest(args.assembly_output / name) for name in NUMERICAL_FILES
        },
        "checks": checks,
        "reference_checks": reference["checks"],
        "errors": [],
        "scope": "2 epochs; 257 train records (128+128+1); 129 evaluation examples per split",
        "performance": performance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("results/assembly-smoke-inputs"))
    parser.add_argument("--output", type=Path, default=Path("results/matched-python-smoke-gate"))
    parser.add_argument("--assembly-output", type=Path, default=Path("results/matched-asm-smoke"))
    parser.add_argument("--binary", type=Path, default=Path("assembly/build/cnn-assembly"))
    parser.add_argument(
        "--audit", type=Path, default=Path("results/matched-python-smoke-audit.json")
    )
    args = parser.parse_args()
    try:
        result = verify(args)
    except Exception as error:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(
            json.dumps({"status": "failed", "errors": [str(error)]}, indent=2) + "\n"
        )
        raise
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "checks": len(result["checks"]), "audit": str(args.audit)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
