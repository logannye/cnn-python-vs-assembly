"""Predict the class of a single new image using a saved checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from classifier.checkpoint import load_checkpoint
from classifier.config import resolve_device
from classifier.data import build_transform


@torch.inference_mode()
def predict(checkpoint: str | Path, image: str | Path, device: str = "cpu") -> dict:
    model, metadata = load_checkpoint(checkpoint, device=device)
    transform = build_transform(**metadata["preprocessing"])
    with Image.open(image) as source:
        tensor = transform(source.convert("RGB")).unsqueeze(0).to(resolve_device(device))
    probabilities = model(tensor).softmax(dim=1)[0].cpu()
    classes = metadata["class_names"]
    return {
        "label": classes[probabilities.argmax().item()],
        "probabilities": dict(zip(classes, probabilities.tolist(), strict=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(predict(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
