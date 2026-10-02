# Simple CNN

[![Tests](https://github.com/logannye/simple-cnn/actions/workflows/test.yml/badge.svg)](https://github.com/logannye/simple-cnn/actions/workflows/test.yml)

A small, complete image classifier built with Python and PyTorch. Load labeled
images, train a two-block CNN, evaluate it on a held-out test set, and classify
new images. The code is organized so you can read the whole workflow without a
training framework or a notebook.

## Quick start

Use Python 3.11 or newer. CI exercises Python 3.12 on Linux with CPU-only PyTorch.
From your terminal:

```bash
git clone https://github.com/logannye/simple-cnn.git
cd simple-cnn
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
On Linux, you can install smaller CPU-only PyTorch wheels before the final install:

```bash
python -m pip install 'torch>=2.6,<3' 'torchvision>=0.21,<1' --index-url https://download.pytorch.org/whl/cpu
```

Create a tiny synthetic dataset, then run the complete workflow:

```bash
python scripts/make_demo_data.py --output data/demo
cnn-train --data-dir data/demo --epochs 5 --checkpoint checkpoints/demo.pt
cnn-evaluate --checkpoint checkpoints/demo.pt --data-dir data/demo/test
cnn-predict --checkpoint checkpoints/demo.pt --image data/demo/test/red/0000.png
```

The demo contains red and blue images with random pixel variation. It runs
without a dataset download and refuses to overwrite a nonempty destination.
Its accuracy only checks this easy synthetic task; it is not evidence of
performance on real photographs. No pretrained weights are included.

Each command also works as a module: `python -m classifier.train`,
`python -m classifier.evaluate`, or `python -m classifier.predict`, with the
same arguments. Use `--help` to see the options.

## Repository structure

```text
simple-cnn/
├── src/classifier/
│   ├── config.py       # Validated settings and CPU/GPU selection
│   ├── data.py         # Image folders, preprocessing, and batches
│   ├── model.py        # CNN architecture and forward pass
│   ├── train.py        # Loss, gradient updates, validation, saving
│   ├── evaluate.py     # Test loss, accuracy, and confusion matrix
│   ├── predict.py      # Single-image labels and probabilities
│   └── checkpoint.py   # Restore weights, labels, and preprocessing
├── scripts/
│   └── make_demo_data.py
├── tests/
├── .github/workflows/test.yml
├── pyproject.toml
└── README.md
```

`train.py` connects the dataset and model. `evaluate.py` and `predict.py`
reconstruct the same model from a checkpoint. `checkpoint.py` is the one small
shared module for loading saved models consistently.

## Use your own images

Organize images into three splits, with one subfolder per class:

```text
data/my-images/
├── train/
│   ├── cats/
│   └── dogs/
├── val/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

Put JPEG, PNG, or other Pillow-supported ImageFolder images inside the class
folders. Use at least two classes. Every split must contain the same class
names and at least one image in each class folder. Class names and their
numeric indices come from the alphabetically sorted folder names.

Prepare the splits yourself before training. Keep duplicate images, and images
of the same subject when applicable, in the same split to avoid leakage.
Training updates weights; validation selects a checkpoint; test data is used
only when you explicitly run evaluation.

```bash
cnn-train --data-dir data/my-images --epochs 20 --batch-size 32 \
  --image-size 64 --learning-rate 0.001 --checkpoint checkpoints/best.pt
cnn-evaluate --checkpoint checkpoints/best.pt --data-dir data/my-images/test
cnn-predict --checkpoint checkpoints/best.pt --image path/to/new-image.jpg
```

Training resizes RGB images to a square, randomly flips them horizontally, and
normalizes each channel using mean `0.5` and standard deviation `0.5`. Evaluation
and prediction use the same resizing and normalization without random flips.
Grayscale and RGBA images are converted to RGB. If horizontal flips change the
meaning of your labels, set `augment=False` in the training dataset call in
`train.py`.

## Model and training

```text
RGB image [3 × 64 × 64]
  → Conv2d (3 → 16, 3 × 3) → ReLU → MaxPool2d
  → Conv2d (16 → 32, 3 × 3) → ReLU → MaxPool2d
  → Adaptive average pooling → Flatten
  → Linear (32 → number of classes)
  → Raw class scores (logits)
```

The image size is configurable; adaptive pooling keeps the classifier head the
same size. Cross-entropy loss consumes logits during training. Adam updates the
weights. Prediction applies softmax to produce one probability per class; these
probabilities are not calibrated estimates of reliability.

Training prints one JSON record per epoch with training loss, validation loss,
and validation accuracy. It saves the highest validation accuracy, breaking
ties using lower validation loss. Evaluation reports sample-weighted loss,
accuracy between 0 and 1, sample count, class names, and a confusion matrix
whose rows are actual classes and columns are predicted classes.

The checkpoint includes:

- Model weights and a format version.
- Class names and their numeric mapping.
- Image size and normalization statistics.
- Training configuration, selected epoch, and validation metrics.

Weights are saved on CPU and loaded with `weights_only=True`. Use checkpoints
from trusted sources. A checkpoint supports evaluation and prediction; it does
not include optimizer state for resuming training. Reusing a checkpoint path
replaces that file when the new run saves its first model.

All commands default to CPU. Pass `--device cuda` for an available NVIDIA GPU,
`--device mps` for an available Apple GPU, or `--device auto` to select an
available accelerator. Set `--seed` during training for repeatable initialization
and shuffling; exact results may still differ across hardware and PyTorch versions.

## Development

Install experiment/test dependencies with `python -m pip install -e '.[dev,benchmark]'`.

```bash
ruff check .
ruff format --check .
pytest -q
```

Tests create tiny temporary datasets. They check image preparation, consistent
labels, gradients, and the complete train → save → load → evaluate → predict
path. GitHub Actions also runs the documented command-line workflow on each
push and pull request. Datasets, generated model weights, and virtual
environments are excluded from Git.

## Quantitative CIFAR-10 baseline

The optional `cnn-benchmark` command measures the same CNN on the complete
CIFAR-10 dataset with a fixed validation split and three training seeds.
It retains predictions, checkpoints, learning curves, per-class metrics,
calibration, uncertainty intervals, and CPU timing/memory measurements.
See [the prespecified experiment protocol](experiments/PROTOCOL.md) for settings,
reproduction commands, metric definitions, and comparison limits.

The [retained baseline results](experiments/baselines/cifar10-v1/README.md)
achieved **46.36% mean test accuracy** across three seeds after 10 epochs
(0.26 percentage-point sample standard deviation). The
[baseline release](https://github.com/logannye/simple-cnn/releases/tag/baseline-cifar10-v1)
preserves all checkpoints, predictions, split indices, and measurements.

For the underlying library conventions, see PyTorch's
[model saving and loading guide](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)
and torchvision's [ImageFolder documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html).
