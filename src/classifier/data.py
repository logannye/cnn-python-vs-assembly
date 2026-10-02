"""Image-folder datasets, shared RGB preprocessing, and reproducible batching."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classifier.config import validate_preprocessing


def build_transform(
    image_size: int = 64,
    mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    std: tuple[float, ...] = (0.5, 0.5, 0.5),
    augment: bool = False,
) -> transforms.Compose:
    """Prepare an RGB image; optional horizontal flips are for training only."""
    validate_preprocessing(image_size, mean, std)
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def load_dataset(
    path: str | Path,
    image_size: int = 64,
    mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    std: tuple[float, ...] = (0.5, 0.5, 0.5),
    class_to_idx: dict[str, int] | None = None,
    augment: bool = False,
) -> datasets.ImageFolder:
    """Load class subfolders and require consistent label names across splits.

    ImageFolder's default PIL loader converts images to RGB, including grayscale
    and RGBA inputs. Every split must contain the same nonempty class folders.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    transform = build_transform(image_size, mean, std, augment)
    try:
        dataset = datasets.ImageFolder(root, transform=transform)
    except FileNotFoundError as error:
        raise ValueError(
            f"Could not load dataset at {root}. Use one nonempty image folder per class. {error}"
        ) from error
    if class_to_idx is not None and dataset.class_to_idx != class_to_idx:
        raise ValueError(
            f"Class mapping mismatch at {root}: expected {class_to_idx}, "
            f"found {dataset.class_to_idx}. Every split must use the same classes."
        )
    return dataset


def make_loader(
    dataset: datasets.ImageFolder,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Batch images with a seeded shuffle and no background worker processes."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
