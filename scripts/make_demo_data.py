#!/usr/bin/env python3
"""Create a tiny red-versus-blue dataset for checking the full workflow."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image


def generate_dataset(
    output: str | Path,
    *,
    image_size: int = 64,
    train_per_class: int = 64,
    val_per_class: int = 16,
    test_per_class: int = 16,
    seed: int = 42,
) -> dict[str, int]:
    """Write independent, reproducible noisy color images in each split.

    This intentionally easy dataset demonstrates plumbing, not real-world
    classification quality. Existing nonempty directories are never modified.
    """
    counts = {"train": train_per_class, "val": val_per_class, "test": test_per_class}
    if image_size < 4:
        raise ValueError("image_size must be at least 4")
    if any(count <= 0 for count in counts.values()):
        raise ValueError("all split counts must be positive")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Refusing to overwrite nonempty output: {output}")

    rng = random.Random(seed)
    colors = {"blue": (35, 60, 215), "red": (215, 60, 35)}
    for split, count in counts.items():
        for label, color in colors.items():
            destination = output / split / label
            destination.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                # The same generator advances across splits, so images are not
                # copied between training, validation, and test sets.
                brightness = rng.randint(-15, 15)
                pixels = bytes(
                    max(0, min(255, channel + brightness + rng.randint(-20, 20)))
                    for _ in range(image_size * image_size)
                    for channel in color
                )
                image = Image.frombytes("RGB", (image_size, image_size), pixels)
                image.save(destination / f"{index:04d}.png")
    return {split: count * len(colors) for split, count in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--val-per-class", type=int, default=16)
    parser.add_argument("--test-per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        counts = generate_dataset(
            args.output,
            image_size=args.image_size,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            seed=args.seed,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Created synthetic dataset at {args.output}")
    print(f"Images: train={counts['train']}, val={counts['val']}, test={counts['test']}")


if __name__ == "__main__":
    main()
