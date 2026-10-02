from pathlib import Path

import pytest
import torch
from PIL import Image

from classifier.data import build_transform, load_dataset, make_loader


def test_transform_resizes_and_normalizes_rgb():
    image = Image.new("RGB", (12, 20), (255, 0, 0))
    result = build_transform(image_size=16)(image)

    assert result.shape == (3, 16, 16)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0], torch.ones(16, 16))
    torch.testing.assert_close(result[1:], -torch.ones(2, 16, 16))


def test_dataset_converts_grayscale_and_preserves_class_mapping(tmp_path: Path):
    root = tmp_path / "grayscale"
    for label in ("blue", "red"):
        directory = root / label
        directory.mkdir(parents=True)
        Image.new("L", (11, 13), 128).save(directory / "image.png")

    dataset = load_dataset(root, image_size=16, class_to_idx={"blue": 0, "red": 1})
    images, labels = next(iter(make_loader(dataset, batch_size=2)))

    assert dataset.class_to_idx == {"blue": 0, "red": 1}
    assert dataset.classes == ["blue", "red"]
    assert images.shape == (2, 3, 16, 16)
    assert labels.tolist() == [0, 1]
    torch.testing.assert_close(images[:, 0], images[:, 1])
    torch.testing.assert_close(images[:, 1], images[:, 2])


@pytest.mark.parametrize(
    "mapping",
    [
        {"blue": 1, "red": 0},
        {"blue": 0},
        {"blue": 0, "red": 1, "green": 2},
    ],
)
def test_dataset_rejects_incompatible_label_mapping(tiny_dataset: Path, mapping):
    with pytest.raises(ValueError):
        load_dataset(tiny_dataset / "val", class_to_idx=mapping)


def test_shuffled_loaders_are_reproducible(tiny_dataset: Path):
    dataset = load_dataset(tiny_dataset / "train", image_size=16)
    first = list(make_loader(dataset, batch_size=5, shuffle=True, seed=7))
    second = list(make_loader(dataset, batch_size=5, shuffle=True, seed=7))

    assert sum(len(labels) for _, labels in first) == len(dataset)
    for (first_images, first_labels), (second_images, second_labels) in zip(
        first, second, strict=True
    ):
        torch.testing.assert_close(first_images, second_images)
        torch.testing.assert_close(first_labels, second_labels)
