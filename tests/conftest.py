"""Small deterministic image fixtures; no downloads or GPU required."""

from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    colors = {"blue": (20, 40, 230), "red": (230, 40, 20)}
    for split_index, (split, count) in enumerate((("train", 8), ("val", 3), ("test", 3))):
        for label, color in colors.items():
            directory = root / split / label
            directory.mkdir(parents=True)
            for index in range(count):
                image = Image.new("RGB", (19, 23), color)
                # Every file has distinct content, including across splits.
                image.putpixel((index, split_index), (80, 80 + index, 80 + split_index))
                image.save(directory / f"{index}.png")
    return root
