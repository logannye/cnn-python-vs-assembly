"""Export fixed CIFAR-10 inputs; no assembly training is performed by Python."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

from classifier.benchmark import loader
from classifier.data import build_transform
from classifier.model import SimpleCNN


class AugmentationSchedule(Dataset):
    """Draw exactly the RNG used by torchvision RandomHorizontalFlip."""

    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return int(self.indices[index]), int(torch.rand(1).item() < 0.5)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def flat_parameters(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy()


def load_parameters(model, flat):
    cursor = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(
                torch.from_numpy(flat[cursor : cursor + count].copy()).view_as(parameter)
            )
            cursor += count
    assert cursor == 5418
    return model


def normalize(raw, flipped=False):
    value = torch.from_numpy(raw.copy()).permute(2, 0, 1).to(torch.float32).div(255.0)
    if flipped:
        value = value.flip(-1)
    return value.sub(0.5).div(0.5).contiguous()


def prepare(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite prepared inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    shared = args.output / "shared"
    shared.mkdir()
    old = json.loads((args.baseline / "manifest.json").read_text())
    train = CIFAR10(args.data_dir, train=True, download=False)
    test = CIFAR10(args.data_dir, train=False, download=False)
    labels_train = np.asarray(train.targets, dtype="<i8")
    labels_test = np.asarray(test.targets, dtype="<i8")
    for key, array in (
        ("training_images_sha256", train.data),
        ("test_images_sha256", test.data),
        ("training_labels_sha256", labels_train),
        ("test_labels_sha256", labels_test),
    ):
        assert hashlib.sha256(array.tobytes()).hexdigest() == old["data"][key], key
    with np.load(args.baseline / "split_indices.npz") as split:
        indices = {key: split[key].copy() for key in split.files}
    with (shared / "images.bin").open("wb") as stream:
        train.data.tofile(stream)
        test.data.tofile(stream)
    np.concatenate((labels_train, labels_test)).astype("<u4").tofile(shared / "labels.bin")
    indices["train"].astype("<u4").tofile(shared / "train_indices.bin")
    indices["validation"].astype("<u4").tofile(shared / "val_indices.bin")
    records = {}
    for seed in args.seeds:
        root = args.output / f"seed-{seed}"
        root.mkdir()
        for source in shared.iterdir():
            os.link(source, root / source.name)
        torch.manual_seed(seed)
        model = SimpleCNN(num_classes=10)
        flat_parameters(model).astype("<f4").tofile(root / "init.bin")
        after_init_rng = torch.get_rng_state().clone()
        batches = loader(AugmentationSchedule(indices["train"]), 128, seed, shuffle=True)
        schedule = np.empty((args.epochs, 45000, 2), dtype="<u4")
        for epoch in range(args.epochs):
            offset = 0
            for ids, flips in batches:
                size = len(ids)
                schedule[epoch, offset : offset + size, 0] = ids.numpy()
                schedule[epoch, offset : offset + size, 1] = flips.numpy()
                offset += size
            assert offset == 45000
            np.testing.assert_array_equal(np.sort(schedule[epoch, :, 0]), indices["train"])
        schedule.tofile(root / "schedule.bin")
        # Independent check against the real baseline's torchvision pipeline.
        torch.set_rng_state(after_init_rng)
        train.transform = build_transform(32, augment=True)
        actual = loader(Subset(train, indices["train"].tolist()), 128, seed, shuffle=True)
        checked = 0
        for images, labels in actual:
            rows = schedule[0, checked : checked + len(images)]
            expected = torch.stack([normalize(train.data[i], bool(flip)) for i, flip in rows])
            assert torch.equal(images, expected), "Transform/sampler/flip mismatch"
            np.testing.assert_array_equal(labels.numpy(), labels_train[rows[:, 0]])
            checked += len(images)
            if checked >= 256:
                break
        records[str(seed)] = {
            "seed": seed,
            "epochs": args.epochs,
            "batch_size": 128,
            "parameters": 5418,
            "schedule_records": args.epochs * 45000,
            "actual_baseline_loader_images_verified_bitwise": checked,
            "hashes": {path.name: digest(path) for path in sorted(root.iterdir())},
            "rng_note": (
                "Model initialization and actual DataLoader generator order; "
                "one global Torch random draw per flip. "
                "Training has no other stochastic operations."
            ),
        }
        (root / "inputs.json").write_text(json.dumps(records[str(seed)], indent=2) + "\n")
        print(
            json.dumps({"seed": seed, "status": "prepared", "verified_images": checked}), flush=True
        )
    manifest = {
        "schema_version": 1,
        "baseline_source": old["source"],
        "baseline_data": old["data"],
        "protocol": {**old["protocol"], "epochs": args.epochs, "seeds": args.seeds},
        "input_format": (
            "ABI.md; little-endian uint8 NHWC images, "
            "uint32 labels/indices/schedules, float32 weights"
        ),
        "seeds": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--baseline", type=Path, default=Path("../cifar10-mac-cpu-25-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/assembly-inputs-v1"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    prepare(parser.parse_args())
