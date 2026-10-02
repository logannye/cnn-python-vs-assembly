"""Two convolutional blocks and a linear image-classification head."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Map RGB image batches to raw class scores (logits)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2.")
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape [batch_size, num_classes]."""
        return self.classifier(self.features(images))
