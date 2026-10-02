from pathlib import Path

import pytest
import torch
from PIL import Image

from classifier.checkpoint import load_checkpoint
from classifier.config import Config
from classifier.data import build_transform
from classifier.evaluate import evaluate
from classifier.predict import predict
from classifier.train import train


def test_train_evaluate_predict_round_trip(tiny_dataset: Path, tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    config = Config(
        data_dir=str(tiny_dataset),
        checkpoint=str(checkpoint),
        image_size=16,
        batch_size=4,
        epochs=6,
        learning_rate=0.01,
        seed=42,
        device="cpu",
        mean=(0.4, 0.5, 0.6),
        std=(0.2, 0.3, 0.4),
    )
    training = train(config)

    assert checkpoint.is_file()
    assert Path(training["best_checkpoint"]) == checkpoint
    assert 1 <= training["best_epoch"] <= config.epochs
    assert training["best_val_accuracy"] >= 0.95

    model, metadata = load_checkpoint(checkpoint, device="cpu")
    assert metadata["format_version"] == 1
    assert metadata["class_names"] == ["blue", "red"]
    assert metadata["class_to_idx"] == {"blue": 0, "red": 1}
    assert metadata["preprocessing"]["image_size"] == 16
    assert tuple(metadata["preprocessing"]["mean"]) == config.mean
    assert tuple(metadata["preprocessing"]["std"]) == config.std

    metrics = evaluate(checkpoint, tiny_dataset / "test", batch_size=2)
    assert metrics["samples"] == 6
    assert metrics["classes"] == ["blue", "red"]
    assert metrics["accuracy"] >= 0.95
    assert metrics["loss"] >= 0
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 6

    image_path = tiny_dataset / "test" / "red" / "0.png"
    prediction = predict(checkpoint, image_path)
    assert prediction["label"] == "red"
    assert set(prediction["probabilities"]) == {"blue", "red"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)

    # Inference must use the preprocessing and label order saved at training.
    transform = build_transform(**metadata["preprocessing"])
    with Image.open(image_path) as image:
        inputs = transform(image.convert("RGB")).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        expected = model(inputs).softmax(dim=1)[0].tolist()
    assert prediction["probabilities"] == pytest.approx(
        dict(zip(metadata["class_names"], expected, strict=True))
    )

    # Evaluation must fail instead of silently assigning a new meaning to labels.
    (tiny_dataset / "test" / "red").rename(tiny_dataset / "test" / "green")
    with pytest.raises(ValueError):
        evaluate(checkpoint, tiny_dataset / "test")
