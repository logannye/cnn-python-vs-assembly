"""Prepare a bounded fixture and audit the actual standalone assembly process."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(args):
    for path in (args.inputs, args.output):
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Refusing to overwrite smoke artifacts: {path}")
    args.inputs.mkdir(parents=True, exist_ok=True)
    for name in ("images.bin", "labels.bin", "train_indices.bin", "val_indices.bin", "init.bin"):
        os.link(args.prepared / name, args.inputs / name)
    with (args.prepared / "schedule.bin").open("rb") as stream:
        schedule = stream.read(2 * 45000 * 8)
    assert len(schedule) == 2 * 45000 * 8
    (args.inputs / "schedule.bin").write_bytes(schedule)
    env = {**os.environ, "CNN_ASM_EVAL_LIMIT": "129"}
    subprocess.run(
        [str(args.binary), str(args.inputs), str(args.output), "2", "257"], env=env, check=True
    )
    subprocess.run(
        [
            sys.executable,
            "assembly/validate_runtime.py",
            "--inputs",
            str(args.inputs),
            "--output",
            str(args.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("results/assembly-inputs-v1/seed-42"))
    parser.add_argument("--inputs", type=Path, default=Path("results/assembly-smoke-inputs"))
    parser.add_argument("--output", type=Path, default=Path("results/assembly-smoke-v1"))
    parser.add_argument("--binary", type=Path, default=Path("assembly/build/cnn-assembly"))
    run(parser.parse_args())
