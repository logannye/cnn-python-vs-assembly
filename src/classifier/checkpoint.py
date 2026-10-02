"""Store weights together with the labels and preprocessing needed to use them."""

from pathlib import Path

import torch

from classifier.config import resolve_device
from classifier.model import SimpleCNN


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[SimpleCNN, dict]:
    """Reconstruct a model without unpickling arbitrary Python model objects."""
    target = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; train a model with this project first.")
    classes = checkpoint["class_names"]
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(name, str) for name in classes)
        or len(set(classes)) != len(classes)
        or checkpoint["class_to_idx"] != {name: i for i, name in enumerate(classes)}
    ):
        raise ValueError("Checkpoint contains an invalid class mapping.")
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target)
    model.eval()
    return model, checkpoint
