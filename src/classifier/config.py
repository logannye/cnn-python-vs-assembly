"""Shared settings and validation for training and image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


def validate_preprocessing(
    image_size: int, mean: tuple[float, ...], std: tuple[float, ...]
) -> None:
    """Require an image large enough for two pools and three RGB statistics."""
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 4:
        raise ValueError("image_size must be an integer of at least 4.")
    if len(mean) != 3 or not all(isfinite(value) for value in mean):
        raise ValueError("mean must contain three finite RGB values.")
    if len(std) != 3 or not all(isfinite(value) and value > 0 for value in std):
        raise ValueError("std must contain three finite, positive RGB values.")


@dataclass
class Config:
    """Small, explicit configuration saved together with the trained weights."""

    data_dir: str = "data"
    checkpoint: str = "checkpoints/best.pt"
    image_size: int = 64
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"
    mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    std: tuple[float, ...] = (0.5, 0.5, 0.5)

    def __post_init__(self) -> None:
        validate_preprocessing(self.image_size, self.mean, self.std)
        for name in ("batch_size", "epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, mps, auto.")


def resolve_device(name: str) -> torch.device:
    """Select a device, failing clearly if a requested accelerator is absent."""
    if name not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError("device must be one of: cpu, cuda, mps, auto.")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu or --device auto.")
    return torch.device(name)
