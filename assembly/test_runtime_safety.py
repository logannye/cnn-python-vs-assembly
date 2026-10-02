"""Exercise standalone assembly failure handling and deterministic repeatability.

The executable must already be built. This script only prepares test fixtures,
launches the assembly process and audits its files; it performs no model math.
Input files are hard-linked read-only by convention. A malformed fixture replaces
its target link with a new small file before any write; original data is never
modified. An existing evidence directory is deliberately refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PROBABILITY_FILES = (
    "predictions_train.bin",
    "predictions_validation.bin",
    "predictions_test.bin",
)
DETERMINISTIC_FILES = ("best.bin", "last.bin", *PROBABILITY_FILES)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tree_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("assembly/build/cnn-assembly"))
    parser.add_argument("--inputs", type=Path, default=Path("results/assembly-smoke-inputs"))
    parser.add_argument("--work-dir", type=Path, default=Path("results/assembly-safety-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/assembly-runtime-safety.json"))
    args = parser.parse_args()
    binary, inputs = args.binary.resolve(), args.inputs.resolve()
    work, output = args.work_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing evidence file: {output}")
    work.mkdir(parents=True, exist_ok=False)
    baseline_input_hashes = tree_hashes(inputs)
    evidence: dict = {
        "started_at": datetime.now(UTC).isoformat(),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "inputs": str(inputs),
        "input_sha256": baseline_input_hashes,
        "work_directory": str(work),
        "command_arguments": ["2", "257"],
        "environment": {"CNN_ASM_EVAL_LIMIT": "129"},
        "checks": [],
    }
    env = os.environ.copy()
    env["CNN_ASM_EVAL_LIMIT"] = "129"

    def execute(name: str, input_dir: Path, output_dir: Path) -> dict:
        command = [str(binary), str(input_dir), str(output_dir), "2", "257"]
        result = subprocess.run(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120
        )
        log = work / f"{name}.log"
        log.write_bytes(result.stdout)
        return {
            "name": name,
            "command": command,
            "exit_code": result.returncode,
            "log": str(log),
            "log_sha256": sha256(log),
            "output": str(output_dir),
        }

    def failure_case(name: str, filename: str, contents: bytes) -> None:
        fixture = work / f"{name}-inputs"
        fixture.mkdir()
        for source in inputs.iterdir():
            if source.is_file():
                os.link(source, fixture / source.name)
        destination = fixture / filename
        destination.unlink()
        destination.write_bytes(contents)
        record = execute(name, fixture, work / f"{name}-output")
        record.update(
            {
                "expected": "nonzero exit status",
                "passed": record["exit_code"] != 0,
                "malformed_file": filename,
                "malformed_size_bytes": len(contents),
                "malformed_sha256": sha256(destination),
            }
        )
        evidence["checks"].append(record)

    existing = work / "nonempty-output"
    existing.mkdir()
    (existing / "sentinel.bin").write_bytes(b"Retain this evidence exactly.\x00\xff")
    hashes_before = tree_hashes(existing)
    record = execute("reject-nonempty-output", inputs, existing)
    hashes_after = tree_hashes(existing)
    record.update(
        {
            "expected": "nonzero exit and all existing file bytes retained",
            "before": hashes_before,
            "after": hashes_after,
            "passed": record["exit_code"] != 0 and hashes_before == hashes_after,
        }
    )
    evidence["checks"].append(record)

    initialization = (inputs / "init.bin").read_bytes()
    failure_case("reject-truncated-init", "init.bin", initialization[:-1])
    failure_case("reject-extra-init-byte", "init.bin", initialization + b"\0")
    failure_case("reject-truncated-images", "images.bin", b"\0")
    schedule = (inputs / "schedule.bin").read_bytes()
    failure_case("reject-truncated-schedule", "schedule.bin", schedule[:-1])
    failure_case(
        "reject-nan-init", "init.bin", struct.pack("<f", float("nan")) + initialization[4:]
    )
    failure_case(
        "reject-infinite-init", "init.bin", struct.pack("<f", float("inf")) + initialization[4:]
    )
    train_indices = (inputs / "train_indices.bin").read_bytes()
    failure_case(
        "reject-train-index-out-of-range",
        "train_indices.bin",
        struct.pack("<I", 50000) + train_indices[4:],
    )
    failure_case(
        "reject-unsorted-train-indices",
        "train_indices.bin",
        train_indices[4:8] + train_indices[:4] + train_indices[8:],
    )
    val_indices = (inputs / "val_indices.bin").read_bytes()
    failure_case(
        "reject-validation-index-out-of-range",
        "val_indices.bin",
        struct.pack("<I", 60000) + val_indices[4:],
    )
    failure_case(
        "reject-schedule-index-out-of-range",
        "schedule.bin",
        struct.pack("<I", 60000) + schedule[4:],
    )
    failure_case(
        "reject-test-image-in-training",
        "schedule.bin",
        struct.pack("<I", 50000) + schedule[4:],
    )
    failure_case(
        "reject-validation-image-in-training",
        "schedule.bin",
        val_indices[:4] + schedule[4:],
    )
    failure_case(
        "reject-duplicate-schedule-image",
        "schedule.bin",
        schedule[:8] + schedule[:4] + schedule[12:],
    )
    failure_case(
        "reject-invalid-unsampled-schedule-record",
        "schedule.bin",
        schedule[:-8] + struct.pack("<II", 50000, 0),
    )
    overlapping_validation = sorted(
        [struct.unpack("<I", train_indices[:4])[0]] + list(struct.unpack("<4999I", val_indices[4:]))
    )
    failure_case(
        "reject-overlapping-train-validation-splits",
        "val_indices.bin",
        struct.pack("<5000I", *overlapping_validation),
    )
    failure_case(
        "reject-flip-out-of-range",
        "schedule.bin",
        schedule[:4] + struct.pack("<I", 2) + schedule[8:],
    )
    labels = (inputs / "labels.bin").read_bytes()
    failure_case("reject-label-out-of-range", "labels.bin", struct.pack("<I", 10) + labels[4:])

    repeats = []
    for index in (1, 2):
        destination = work / f"repeat-{index}-output"
        record = execute(f"repeat-{index}", inputs, destination)
        record["expected"] = "successful 2 epoch training and evaluation"
        record["passed"] = record["exit_code"] == 0
        if record["exit_code"] == 0:
            record["deterministic_file_sha256"] = {
                name: sha256(destination / name) for name in DETERMINISTIC_FILES
            }
            with (destination / "history.csv").open(newline="") as handle:
                record["history_without_timings"] = [
                    {key: value for key, value in row.items() if not key.endswith("seconds")}
                    for row in csv.DictReader(handle)
                ]
            record["performance"] = json.loads((destination / "performance.json").read_text())
        evidence["checks"].append(record)
        repeats.append(record)
    repeat_equal = all(record["passed"] for record in repeats) and (
        repeats[0]["deterministic_file_sha256"] == repeats[1]["deterministic_file_sha256"]
        and repeats[0]["history_without_timings"] == repeats[1]["history_without_timings"]
    )
    evidence["checks"].append(
        {
            "name": "bitwise-repeatability",
            "expected": "identical best/last parameters, all predictions, and numerical history",
            "passed": repeat_equal,
            "compared_files": list(DETERMINISTIC_FILES),
            "timings_excluded": True,
        }
    )
    input_hashes_after = tree_hashes(inputs)
    evidence["checks"].append(
        {
            "name": "original-inputs-unchanged",
            "passed": input_hashes_after == baseline_input_hashes,
            "after_sha256": input_hashes_after,
        }
    )
    evidence["completed_at"] = datetime.now(UTC).isoformat()
    evidence["status"] = (
        "passed" if all(item["passed"] for item in evidence["checks"]) else "failed"
    )
    evidence["check_count"] = len(evidence["checks"])
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"{evidence['status']}: {len(evidence['checks'])} checks; evidence {output}")
    for item in evidence["checks"]:
        if not item["passed"]:
            print(f"FAILED: {item['name']}")
    if evidence["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
