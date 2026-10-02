"""Check the split and measurement path without downloading CIFAR-10."""

from argparse import Namespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from classifier.benchmark import run_seed, stratified_split


def test_stratified_split_is_fixed_disjoint_and_complete():
    labels = np.repeat(np.arange(3), 8)
    train, validation = stratified_split(labels, per_class=2, seed=1729)
    assert len(train) == 18 and len(validation) == 6
    assert not np.intersect1d(train, validation).size
    np.testing.assert_array_equal(np.sort(np.concatenate([train, validation])), np.arange(24))
    np.testing.assert_array_equal(np.bincount(labels[validation]), [2, 2, 2])
    repeat = stratified_split(labels, per_class=2, seed=1729)
    np.testing.assert_array_equal(train, repeat[0])
    np.testing.assert_array_equal(validation, repeat[1])
    with pytest.raises(ValueError):
        stratified_split(labels, per_class=8, seed=1729)


def test_benchmark_retains_reloadable_predictions_and_timings(tmp_path):
    import json

    images = torch.cat([torch.full((4, 3, 8, 8), -0.5), torch.full((4, 3, 8, 8), 0.5)])
    labels = torch.tensor([0] * 4 + [1] * 4)
    dataset = TensorDataset(images, labels)
    datasets = {split: dataset for split in ["augmented_training", "train", "validation", "test"]}
    indices = {split: np.arange(8) for split in ["train", "validation", "test"]}
    args = Namespace(output=tmp_path, batch_size=4, learning_rate=0.001, epochs=1, split_seed=1729)
    run_seed(args, 42, datasets, indices, ["blue", "red"])
    folder = tmp_path / "seed-42"
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["test"]["num_examples"] == 8
    with np.load(folder / "predictions_test.npz") as predictions:
        np.testing.assert_array_equal(predictions["labels"], labels.numpy())
        np.testing.assert_allclose(predictions["probabilities"].sum(1), 1, atol=1e-6)
        np.testing.assert_array_equal(predictions["indices"], np.arange(8))
    performance = json.loads((folder / "performance.json").read_text())
    assert performance["parameter_count"] == 5154
    assert performance["checkpoint_bytes"] > 0
    assert performance["training_seconds"] > 0
    assert performance["inference"]["batch1"]["median_ms"] > 0
    assert (folder / "best.pt").is_file()
