"""Run the prespecified, sequential CPU comparison with matched binary inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def snapshot() -> dict:
    """Allowlisted machine conditions; never dump the user's environment."""
    result = {"utc": datetime.now(UTC).isoformat(), "load_average": list(os.getloadavg())}
    for key, args in {
        "power": ("pmset", "-g", "batt"),
        "thermal": ("pmset", "-g", "therm"),
    }.items():
        try:
            result[key] = command(*args)
        except subprocess.CalledProcessError as exc:
            result[key] = {"unavailable": exc.output.strip()}
    return result


def run(args: argparse.Namespace) -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("The matched experiment requires an Apple silicon Mac")
    if command("git", "status", "--porcelain"):
        raise RuntimeError("Commit source changes before running the experiment")
    if "CNN_ASM_EVAL_LIMIT" in os.environ:
        raise RuntimeError("Unset CNN_ASM_EVAL_LIMIT for the full experiment")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite experiment output")
    if args.cooldown < 15:
        raise ValueError("The protocol requires at least 15 seconds between children")

    prepared = json.loads((args.inputs / "manifest.json").read_text())
    if prepared["protocol"]["seeds"] != [42, 43, 44] or prepared["protocol"]["epochs"] != 25:
        raise ValueError("Unexpected prepared protocol")
    binary_hash = digest(args.binary)
    evidence = {
        "numerical-parity.json": Path("results/matched-parity.json"),
        "runtime-parity.json": Path("results/matched-asm-smoke/runtime-audit.json"),
        "runtime-safety.json": Path("results/matched-safety.json"),
        "python-smoke.json": Path("results/matched-python-smoke-audit.json"),
    }
    for name, path in evidence.items():
        record = json.loads(path.read_text())
        if record["status"] != "passed":
            raise RuntimeError(f"Failed gate: {path}")
        if name in ("runtime-parity.json", "runtime-safety.json"):
            if record["binary_sha256"] != binary_hash:
                raise RuntimeError(f"Stale binary audit: {path}")
    parity = json.loads(evidence["numerical-parity.json"].read_text())
    for path, expected in parity["source_hashes"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Stale numerical audit: {path}")
    python_audit = json.loads(evidence["python-smoke.json"].read_text())
    if python_audit["source_sha256"] != digest(Path("comparison/python_train.py")):
        raise RuntimeError("Stale Python smoke audit")

    # Hash all unique input inodes once, before ANY child starts. This also
    # warms the common filesystem cache without privileged cache flushing.
    seen, input_hashes, signatures = {}, {}, {}
    for seed in (42, 43, 44):
        folder = args.inputs / f"seed-{seed}"
        hashes = json.loads((folder / "inputs.json").read_text())["hashes"]
        if hashes != prepared["seeds"][str(seed)]["hashes"] or set(hashes) != {
            "images.bin",
            "labels.bin",
            "train_indices.bin",
            "val_indices.bin",
            "init.bin",
            "schedule.bin",
        }:
            raise RuntimeError(f"Unexpected fixture manifest: {folder}")
        for name, expected in hashes.items():
            path = folder / name
            stat = path.stat()
            signatures[str(path)] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            key = (stat.st_dev, stat.st_ino, stat.st_size)
            if key not in seen:
                seen[key] = digest(path)
            if seen[key] != expected:
                raise RuntimeError(f"Input hash mismatch: {path}")
        input_hashes[str(seed)] = hashes
    for name in ("images.bin", "labels.bin", "train_indices.bin", "val_indices.bin"):
        if len({hashes[name] for hashes in input_hashes.values()}) != 1:
            raise RuntimeError(f"Seeds must share identical {name}")

    def check_inputs() -> None:
        for path, expected in signatures.items():
            stat = Path(path).stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != expected:
                raise RuntimeError(f"Inputs changed during experiment: {path}")

    runs = []
    for seed in (42, 43, 44):
        first = ("python", "assembly") if seed != 43 else ("assembly", "python")
        for repeat, implementations in ((1, first), (2, first[::-1])):
            for implementation in implementations:
                runs.append(
                    {
                        "id": f"seed-{seed}-repeat-{repeat}-{implementation}",
                        "seed": seed,
                        "repeat": repeat,
                        "implementation": implementation,
                        "order": len(runs) + 1,
                    }
                )
    source_revision = command("git", "rev-parse", "HEAD")
    source_files = [
        Path(name)
        for name in command(
            "git", "ls-files", "assembly", "comparison", "src/classifier"
        ).splitlines()
    ]
    source_hashes = {str(path): digest(path) for path in source_files}
    plan = {
        "schema_version": 1,
        "experiment": "cifar10-matched-cpu-25-v1",
        "source_revision": source_revision,
        "source_hashes": source_hashes,
        "binary_sha256": binary_hash,
        "input_hashes": input_hashes,
        "prepared_manifest_sha256": digest(args.inputs / "manifest.json"),
        "protocol": prepared["protocol"],
        "runs": runs,
        "cooldown_seconds": args.cooldown,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "cpu_count": os.cpu_count(),
            "memory_bytes": int(command("sysctl", "-n", "hw.memsize")),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "scikit-learn", "matplotlib")
            },
            "compiler": command("xcrun", "clang", "--version"),
            "linked_libraries": command("otool", "-L", str(args.binary)),
            "undefined_symbols": command("nm", "-u", str(args.binary)),
        },
        "initial_conditions": snapshot(),
    }
    if "AC Power" not in str(plan["initial_conditions"]["power"]):
        raise RuntimeError("Connect AC power before starting the official experiment")
    for forbidden in ("torch", "cblas", "Accelerate", "vDSP", "Py_", "expf", "logf", "pow"):
        if forbidden in plan["environment"]["undefined_symbols"]:
            raise RuntimeError(f"Unexpected numerical dependency: {forbidden}")
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "plan.json", plan)
    shutil.copy2(args.inputs / "manifest.json", args.output / "prepared-manifest.json")
    shutil.copy2(args.binary, args.output / "cnn-assembly")
    for name, path in evidence.items():
        shutil.copy2(path, args.output / name)
    for path in source_files:
        target = args.output / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONHASHSEED": "0",
    }
    for item in runs:
        check_inputs()
        # Only small immutable source files are checked here, never the dataset.
        if command("git", "rev-parse", "HEAD") != source_revision:
            raise RuntimeError("Source revision changed during the experiment")
        if digest(args.binary) != binary_hash or any(
            digest(Path(path)) != expected for path, expected in source_hashes.items()
        ):
            raise RuntimeError("Measured sources changed during the experiment")
        before = snapshot()
        if "AC Power" not in str(before["power"]):
            raise RuntimeError("AC power disconnected; retain completed evidence and stop")
        destination = args.output / "raw" / item["id"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        prefix = (
            [str(args.binary.resolve())]
            if item["implementation"] == "assembly"
            else [sys.executable, "comparison/python_train.py"]
        )
        invocation = prefix + [
            str((args.inputs / f"seed-{item['seed']}").resolve()),
            str(destination.resolve()),
            "25",
            "45000",
        ]
        print(f"[{item['order']}/12] Starting {item['id']}", flush=True)
        with (args.output / f"{item['id']}.log").open("x") as log:
            started = time.perf_counter()
            process = subprocess.run(invocation, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - started
        after = snapshot()
        check_inputs()
        record = {
            **item,
            "returncode": process.returncode,
            "process_wall_seconds": elapsed,
            "source_revision": source_revision,
            "command": invocation,
            "before": before,
            "after": after,
        }
        save(destination / "run.json", record)
        if "AC Power" not in str(after["power"]):
            raise RuntimeError("Run ended off AC power; retain evidence and stop")
        print(f"Completed {item['id']}: exit={process.returncode}, wall={elapsed:.2f}s", flush=True)
        if process.returncode:
            raise RuntimeError(f"Child failed; see {item['id']}.log")
        time.sleep(args.cooldown)
    if command("git", "rev-parse", "HEAD") != source_revision:
        raise RuntimeError("Source revision changed during the last child")
    if digest(args.binary) != binary_hash or any(
        digest(Path(path)) != expected for path, expected in source_hashes.items()
    ):
        raise RuntimeError("Measured source changed during the last child")
    checked = {}
    for seed, hashes in input_hashes.items():
        for name, expected in hashes.items():
            path = args.inputs / f"seed-{seed}" / name
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in checked:
                checked[key] = digest(path)
            if checked[key] != expected:
                raise RuntimeError(f"Final input hash mismatch: {path}")
    save(
        args.output / "completion.json",
        {"status": "completed", "utc": datetime.now(UTC).isoformat()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("results/assembly-inputs-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/matched-cpu-25-v1"))
    parser.add_argument("--binary", type=Path, default=Path("assembly/build/cnn-assembly"))
    parser.add_argument("--cooldown", type=float, default=15)
    run(parser.parse_args())
