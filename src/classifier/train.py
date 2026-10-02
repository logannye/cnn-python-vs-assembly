"""Train a small CNN and save the checkpoint with the best validation accuracy."""

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from classifier.config import Config, resolve_device
from classifier.data import load_dataset, make_loader
from classifier.evaluate import score
from classifier.model import SimpleCNN


def train(config: Config) -> dict:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    preprocessing = {
        "image_size": config.image_size,
        "mean": list(config.mean),
        "std": list(config.std),
    }
    training = load_dataset(Path(config.data_dir) / "train", **preprocessing, augment=True)
    validation = load_dataset(
        Path(config.data_dir) / "val", **preprocessing, class_to_idx=training.class_to_idx
    )
    train_loader = make_loader(training, config.batch_size, shuffle=True, seed=config.seed)
    val_loader = make_loader(validation, config.batch_size)
    model = SimpleCNN(num_classes=len(training.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    destination = Path(config.checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    best_rank = (-1.0, float("-inf"))
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is not finite; check the data and learning rate.")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        metrics = score(model, val_loader, device, len(training.classes))
        if not math.isfinite(metrics["loss"]):
            raise ValueError("Validation loss is not finite; check the data and learning rate.")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / len(training),
                    "val_loss": metrics["loss"],
                    "val_accuracy": metrics["accuracy"],
                }
            ),
            flush=True,
        )
        # For tied accuracies, prefer the checkpoint with lower validation loss.
        rank = (metrics["accuracy"], -metrics["loss"])
        if rank > best_rank:
            best_rank, best_epoch = rank, epoch
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "class_names": training.classes,
                    "class_to_idx": training.class_to_idx,
                    "preprocessing": preprocessing,
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_metrics": metrics,
                },
                destination,
            )
    return {
        "best_checkpoint": str(destination),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_rank[0],
    }


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=defaults.data_dir)
    parser.add_argument("--checkpoint", default=defaults.checkpoint)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default=defaults.device)
    args = parser.parse_args()
    print(json.dumps(train(Config(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
