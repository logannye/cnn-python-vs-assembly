"""Launch three independent assembly processes and retain their provenance."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from classifier.benchmark import environment, git_output, sha256, write_json


def run(args):
    if git_output("status", "--porcelain"):
        raise RuntimeError("Commit all source changes before the measured experiment")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite experiment output")
    if "CNN_ASM_EVAL_LIMIT" in os.environ:
        raise ValueError("Unset CNN_ASM_EVAL_LIMIT for the full experiment")
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    assert prepared["protocol"]["seeds"] == [42, 43, 44]
    assert prepared["protocol"]["epochs"] == 25
    evidence = {
        "numerical-parity.json": Path("results/assembly-parity-v1.json"),
        "runtime-parity.json": Path("results/assembly-smoke-v1/runtime-audit.json"),
        "runtime-safety.json": Path("results/assembly-runtime-safety.json"),
    }
    for path in evidence.values():
        assert json.loads(path.read_text())["status"] == "passed", path
    for name in ("runtime-parity.json", "runtime-safety.json"):
        assert json.loads(evidence[name].read_text())["binary_sha256"] == sha256(args.binary)
    parity = json.loads(evidence["numerical-parity.json"].read_text())
    for path, expected in parity["source_hashes"].items():
        assert sha256(Path(path)) == expected
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, path in evidence.items():
        shutil.copy2(path, args.output / name)
    source_files = sorted(p for p in Path("assembly").iterdir() if p.is_file())
    source = {
        "git_revision": git_output("rev-parse", "HEAD"),
        "working_tree_clean": True,
        "model_source_sha256": sha256(Path("src/classifier/model.py")),
        "binary_sha256": sha256(args.binary),
        "source_hashes": {str(p): sha256(p) for p in source_files},
        "command": sys.argv,
        "compiler": subprocess.check_output(["xcrun", "clang", "--version"], text=True),
        "linked_libraries": subprocess.check_output(["otool", "-L", str(args.binary)], text=True),
        "undefined_symbols": subprocess.check_output(["nm", "-u", str(args.binary)], text=True),
    }
    for forbidden in ("torch", "cblas", "Accelerate", "vDSP", "Py_", "expf", "logf", "pow"):
        assert forbidden not in source["undefined_symbols"], forbidden
    snapshot = environment()
    snapshot.update(
        {
            "numerical_backend": "handwritten Darwin ARM64 assembly; two pthread workers",
            "python_scope": "Orchestration, fixed input preparation and post-run audit only",
            "assembly_workers": 2,
        }
    )
    manifest = {
        "schema_version": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "protocol": prepared["protocol"],
        "data": prepared["baseline_data"],
        "source": source,
        "environment": snapshot,
        "prepared_manifest_sha256": sha256(args.prepared / "manifest.json"),
        "seed_runs": {},
        "scope_note": (
            "Mathematical model and recipe replication with fixed initial parameters, "
            "data order and flips; floating-point reductions differ from PyTorch. "
            "Python does not compute the assembly training or its predictions."
        ),
    }
    write_json(args.output / "manifest.json", manifest)
    for path in source_files:
        target = args.output / "source" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(args.binary, args.output / "cnn-assembly")
    for seed in (42, 43, 44):
        assert git_output("rev-parse", "HEAD") == source["git_revision"]
        assert not git_output("status", "--porcelain")
        assert sha256(args.binary) == source["binary_sha256"]
        for path in source_files:
            assert sha256(path) == source["source_hashes"][str(path)]
        input_dir = args.prepared / f"seed-{seed}"
        input_record = json.loads((input_dir / "inputs.json").read_text())
        for name, expected in input_record["hashes"].items():
            assert sha256(input_dir / name) == expected, (seed, name)
        destination = args.output / f"seed-{seed}"
        command = [
            str(args.binary.resolve()),
            str(input_dir.resolve()),
            str(destination.resolve()),
            "25",
            "45000",
        ]
        started = datetime.now(UTC).isoformat()
        print(f"Starting assembly CPU seed {seed}", flush=True)
        with (args.output / f"seed-{seed}-console.log").open("x") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Assembly seed {seed} failed with code {return_code}")
        seed_record = {
            "seed": seed,
            "started_at_utc": started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "source": source,
            "environment": snapshot,
            "command": command,
            "input_hashes": input_record["hashes"],
        }
        write_json(destination / "manifest.json", seed_record)
        manifest["seed_runs"][str(seed)] = seed_record
        write_json(args.output / "manifest.json", manifest)
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output / "manifest.json", manifest)
    print("All three assembly CPU runs completed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("results/assembly-inputs-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/assembly-runs-v1"))
    parser.add_argument("--binary", type=Path, default=Path("assembly/build/cnn-assembly"))
    run(parser.parse_args())
