"""Combine compatible per-seed experiment artifacts and generate the report."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from classifier.benchmark import sha256, write_json
from classifier.report import build_report


def collect(inputs: Path, output: Path) -> None:
    manifests = sorted(inputs.glob("*/manifest.json"))
    if not manifests:
        raise ValueError("No experiment manifests found.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing baseline.")
    output.mkdir(parents=True, exist_ok=True)
    combined = json.loads(manifests[0].read_text())
    reference_protocol = {k: v for k, v in combined["protocol"].items() if k != "seeds"}
    combined["seed_runs"] = {}
    seeds = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        protocol = {k: v for k, v in manifest["protocol"].items() if k != "seeds"}
        if (
            protocol != reference_protocol
            or manifest["source"]["git_revision"] != combined["source"]["git_revision"]
        ):
            raise ValueError("Cannot combine different protocols or code revisions.")
        if manifest["data"] != combined["data"]:
            raise ValueError("Dataset or split fingerprints differ.")
        for seed in manifest["protocol"]["seeds"]:
            if seed in seeds:
                raise ValueError("Duplicate seed artifact.")
            seeds.append(seed)
            seed_folder = manifest_path.parent / f"seed-{seed}"
            if not (seed_folder / "performance.json").is_file():
                raise ValueError(f"Incomplete seed {seed}.")
            shutil.copytree(seed_folder, output / f"seed-{seed}")
            shutil.copy2(manifest_path, output / f"seed-{seed}" / "manifest.json")
            shutil.copy2(
                manifest_path.parent / "requirements-frozen.txt",
                output / f"seed-{seed}" / "requirements-frozen.txt",
            )
            shutil.copy2(
                manifest_path.parent / "torch-build-config.txt",
                output / f"seed-{seed}" / "torch-build-config.txt",
            )
            combined["seed_runs"][str(seed)] = manifest
    if sorted(seeds) != [42, 43, 44]:
        raise ValueError(f"Expected the three prespecified seeds; got {seeds}.")
    shutil.copy2(manifests[0].parent / "split_indices.npz", output / "split_indices.npz")
    with np.load(output / "split_indices.npz") as indices:
        assert not np.intersect1d(indices["train"], indices["validation"]).size
    combined["protocol"]["seeds"] = sorted(seeds)
    combined["completed_at_utc"] = max(
        m["completed_at_utc"] for m in combined["seed_runs"].values()
    )
    combined["environment_note"] = "Root environment is seed 42; see seed_runs for each runner."
    write_json(output / "manifest.json", combined)
    build_report(output)
    checksums = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json(output / "checksums.json", checksums)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    collect(arguments.inputs, arguments.output)
