"""Evaluate saved weights on a labeled, held-out image folder."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from classifier.checkpoint import load_checkpoint
from classifier.config import resolve_device
from classifier.data import load_dataset, make_loader


@torch.inference_mode()
def score(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    """Return sample-weighted loss, accuracy, and counts without changing weights."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    samples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        predictions = logits.argmax(dim=1)
        # Rows are actual classes; columns are predicted classes.
        indices = (labels * num_classes + predictions).cpu()
        confusion += torch.bincount(indices, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
        samples += labels.size(0)
    if samples == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {
        "loss": loss_sum / samples,
        "accuracy": confusion.diag().sum().item() / samples,
        "samples": samples,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(
    checkpoint: str | Path, data_dir: str | Path, batch_size: int = 32, device: str = "cpu"
) -> dict:
    """Evaluate a split whose class folders match the checkpoint's label mapping."""
    model, metadata = load_checkpoint(checkpoint, device=device)
    dataset = load_dataset(
        data_dir, **metadata["preprocessing"], class_to_idx=metadata["class_to_idx"]
    )
    loader = make_loader(dataset, batch_size=batch_size)
    metrics = score(model, loader, resolve_device(device), len(metadata["class_names"]))
    return {"classes": metadata["class_names"], **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data/test", help="Folder containing class folders")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
