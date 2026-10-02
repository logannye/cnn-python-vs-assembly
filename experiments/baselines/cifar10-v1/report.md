# Simple CNN · CIFAR-10 baseline

This is the report mirror. [Download the complete results bundle](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz) for checkpoints, split indices, and per-image predictions. Binary-artifact links below point to that bundle.

This report contains **3 completed training seeds**: 42, 43, 44. Detailed example-level diagnostics use representative seed **42** (seed 42 when available; otherwise the lowest completed seed).

## Evaluation protocol

The CNN is trained from randomly initialized weights, with no pretraining. The official CIFAR-10 training set is split into training and validation subsets; the official test set remains separate. The same saved split is used across training seeds. Within each seed, the checkpoint is selected by highest validation accuracy, breaking ties by lower validation loss. Test results do not select checkpoints or seeds.

Final train, validation, and test metrics below evaluate the selected checkpoint with evaluation preprocessing. Epoch training metrics use augmented images and mix successive model states, so they need not equal final training metrics.

| Split | Images (representative seed) |
|---|---:|
| Train | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |

Uniform random top-1 guessing has expected accuracy **10%** for this 10-class benchmark. This is a reference point, not a trained competing model.

## Test performance across seeds

Every seed uses the same test images. Means and sample standard deviations describe variation across training seeds; repeated test predictions are not independent test sets. Standard deviations of percentage metrics are shown in percentage points. A small number of seeds provides only a preliminary estimate of training variability.

| Metric | Mean | Sample SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Top-1 accuracy | 46.36% | 0.26% | 46.10% | 46.61% |
| Balanced accuracy | 46.36% | 0.26% | 46.10% | 46.61% |
| Macro precision | 46.37% | 0.09% | 46.28% | 46.47% |
| Macro recall | 46.36% | 0.26% | 46.10% | 46.61% |
| Macro F1 | 45.51% | 0.31% | 45.16% | 45.78% |
| Weighted F1 | 45.51% | 0.31% | 45.16% | 45.78% |
| Top-3 accuracy | 79.39% | 0.16% | 79.21% | 79.51% |
| Top-5 accuracy | 91.70% | 0.16% | 91.56% | 91.88% |
| Negative log likelihood | 1.4978 | 0.0032 | 1.4956 | 1.5015 |
| Multiclass Brier score | 0.6776 | 0.0013 | 0.6764 | 0.6790 |
| Expected calibration error | 4.89% | 0.72% | 4.07% | 5.41% |
| Maximum calibration error | 20.88% | 13.81% | 12.64% | 36.82% |
| Mean confidence | 41.48% | 0.49% | 41.20% | 42.05% |
| Confidence when correct | 47.28% | 0.44% | 47.01% | 47.79% |
| Confidence when incorrect | 36.48% | 0.57% | 36.15% | 37.13% |
| Macro one-vs-rest ROC AUC | 0.8694 | 0.0011 | 0.8683 | 0.8705 |
| Macro average precision | 0.4772 | 0.0026 | 0.4746 | 0.4798 |

Lower negative log likelihood, Brier score, ECE, and MCE are better. Brier score sums squared probability error over classes, then averages over images. ECE is the count-weighted absolute accuracy–confidence gap over 15 equal-width bins; MCE is the largest nonempty-bin gap. Both depend on the binning. ROC AUC and average precision use one-vs-rest scores averaged equally over classes. Undefined metrics appear as —.

## Individual runs and uncertainty

| Seed | Selected epoch | Validation accuracy | Test accuracy | Accuracy 95% CI | Macro F1 | Macro F1 95% CI |
|---|---:|---:|---:|---|---:|---|
| 42 | 10 | 46.14% | 46.37% | [45.39%, 47.35%] | 45.78% | [44.81%, 46.69%] |
| 43 | 10 | 45.42% | 46.10% | [45.12%, 47.08%] | 45.16% | [44.21%, 46.02%] |
| 44 | 10 | 44.88% | 46.61% | [45.63%, 47.59%] | 45.59% | [44.51%, 46.62%] |

Accuracy intervals use the 95% Wilson method. Macro F1 intervals use a percentile bootstrap over test examples (1000 resamples; the random seed and exact method are saved in each metrics.json). These intervals describe test-sample uncertainty conditional on a fitted model and an independent, representative sampling assumption; they do not include training-seed variability or distribution shift. They are not intervals for the cross-seed mean.

## Fit and learning curves

| Seed | Final train accuracy | Validation accuracy | Test accuracy | Train − validation gap |
|---|---:|---:|---:|---:|
| 42 | 46.84% | 46.14% | 46.37% | 0.70 pp |
| 43 | 46.79% | 45.42% | 46.10% | 1.37 pp |
| 44 | 46.69% | 44.88% | 46.61% | 1.81 pp |

![Loss and accuracy by epoch](figures/learning_curves.png)

![Validation and test accuracy by seed](figures/accuracy_seed_comparison.png)

## Class-level performance

The table reports seed 42; the figure summarizes all completed seeds.

| Class | Support | Precision | Recall | F1 | ROC AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| airplane | 1000 | 45.21% | 52.80% | 48.71% | 0.8840 | 0.5260 |
| automobile | 1000 | 61.69% | 48.80% | 54.49% | 0.9228 | 0.5777 |
| bird | 1000 | 33.60% | 33.00% | 33.30% | 0.7903 | 0.3080 |
| cat | 1000 | 29.71% | 21.30% | 24.81% | 0.8098 | 0.2791 |
| deer | 1000 | 42.35% | 27.70% | 33.49% | 0.8279 | 0.3687 |
| dog | 1000 | 39.97% | 54.60% | 46.15% | 0.8641 | 0.4246 |
| frog | 1000 | 45.58% | 62.90% | 52.86% | 0.8894 | 0.5520 |
| horse | 1000 | 58.16% | 47.40% | 52.23% | 0.8670 | 0.5474 |
| ship | 1000 | 56.52% | 58.50% | 57.49% | 0.9212 | 0.6155 |
| truck | 1000 | 51.92% | 56.70% | 54.21% | 0.9060 | 0.5472 |

![Per-class F1](figures/per_class_f1.png)

![Normalized confusion matrix](figures/confusion_matrix.png)

Most frequent directed errors for seed 42:

| Actual → predicted | Images | Percentage of actual class |
|---|---:|---:|
| cat → dog | 357 | 35.70% |
| deer → frog | 224 | 22.40% |
| automobile → truck | 213 | 21.30% |
| airplane → ship | 208 | 20.80% |
| ship → airplane | 199 | 19.90% |
| deer → bird | 182 | 18.20% |
| bird → frog | 151 | 15.10% |
| dog → cat | 132 | 13.20% |
| truck → automobile | 129 | 12.90% |
| horse → dog | 129 | 12.90% |

## Confidence and calibration

For seed 42, mean top-class confidence is 41.20%. It is 47.03% on correct predictions and 36.15% on incorrect predictions. Confidence is a model probability, not a guarantee of correctness. The reliability plot shows observed accuracy against mean confidence in each nonempty bin; the lower panel shows how many images fall in each bin. No post-hoc calibration is fitted.

![Reliability plot and confidence distribution](figures/reliability.png)

## Runtime and resource use

| Seed | Train (s) | Final evaluation (s) | Parameters | Checkpoint (MiB) | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| 42 | 136.6963 | 12.0739 | 5418 | 0.0243 | 847.2695 |
| 43 | 200.6421 | 16.9327 | 5418 | 0.0243 | 846.7461 |
| 44 | 151.4993 | 13.0349 | 5418 | 0.0243 | 847.3477 |

| Seed | Batch size | Median batch latency (ms) | P95 batch latency (ms) | Images/s |
|---|---:|---:|---:|---:|
| 42 | 1 | 0.1680 | 0.1874 | 5817.2446 |
| 42 | 128 | 14.5823 | 15.6252 | 8763.5453 |
| 43 | 1 | 0.2630 | 0.2885 | 3715.9504 |
| 43 | 128 | 17.7058 | 21.5946 | 7084.0405 |
| 44 | 1 | 0.2057 | 0.2251 | 4780.0656 |
| 44 | 128 | 14.2430 | 14.7596 | 8878.1879 |

Timing is specific to the recorded host, device, thread count, batch size, warmup, and measurement procedure. Hosted runners can differ between seeds; consult each seed's environment in manifest.json before comparing speed. Latency measurements cover the model forward pass described in performance.json, not an end-to-end image-serving pipeline. Peak RSS is the process high-water mark, including data and evaluation allocations; it is not an isolated model-memory measurement. Per-epoch training and validation timings are retained in history.json.

## Retained evidence and future comparisons

- [Machine-readable summary](summary.json): per-seed scalar values and cross-seed aggregates.
- [Manifest](manifest.json): source revision, protocol, configuration, package versions, and hardware/software environment.
- [Split indices](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz): exact training and validation memberships.
- Seed 42: [metrics](seed-42/metrics.json), [history](seed-42/history.json), [performance](seed-42/performance.json), [checkpoint](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [train predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [validation predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [test predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz).
- Seed 43: [metrics](seed-43/metrics.json), [history](seed-43/history.json), [performance](seed-43/performance.json), [checkpoint](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [train predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [validation predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [test predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz).
- Seed 44: [metrics](seed-44/metrics.json), [history](seed-44/history.json), [performance](seed-44/performance.json), [checkpoint](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [train predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [validation predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz), [test predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-v1/cifar10-baseline-v1.tar.gz).

Prediction archives retain labels, class probabilities, and example indices. A future model can use these same splits, seeds, evaluation code, and indexed test examples for paired comparisons. Fix the next experiment's protocol before examining its test results, select checkpoints using validation only, and report both paired test-example uncertainty and variability across training seeds. Repeated experiment selection based on this test set will turn it into development data.

This is a small CNN baseline at a fixed 10-epoch training budget, not an estimate of the architecture's maximum attainable accuracy. It does not establish optimal hyperparameters, convergence, state-of-the-art performance, robustness to distribution shift, or production suitability.

## Recorded protocol

```json
{
  "accuracy_interval": "95% Wilson; one fitted model, iid test examples",
  "augmentation": "RandomHorizontalFlip(p=0.5) training only",
  "batch_size": 128,
  "betas": [
    0.9,
    0.999
  ],
  "checkpoint_selection": "highest validation accuracy; ties use lower validation loss",
  "class_names": [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
  ],
  "dataset": "CIFAR-10",
  "dataset_citation": "Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009",
  "dataset_source": "https://www.cs.toronto.edu/~kriz/cifar.html",
  "deterministic_algorithms": true,
  "ece_bins": 15,
  "epochs": 10,
  "epsilon": 1e-08,
  "final_train_scope": "Deterministic unaugmented training split at selected checkpoint",
  "image_size": 32,
  "initialization": "PyTorch default, from scratch; no pretrained weights",
  "learning_rate": 0.001,
  "macro_f1_interval": "95% bootstrap; 1000 multinomial draws from confusion counts",
  "majority_class_accuracy": 0.1,
  "normalization_mean": [
    0.5,
    0.5,
    0.5
  ],
  "normalization_std": [
    0.5,
    0.5,
    0.5
  ],
  "optimizer": "Adam",
  "random_uniform_accuracy": 0.1,
  "scheduler": null,
  "seeds": [
    42,
    43,
    44
  ],
  "split_seed": 1729,
  "test_examples": 10000,
  "test_policy": "Evaluate once after all epochs and checkpoint selection for each seed",
  "train_examples": 45000,
  "training_curve_scope": "Augmented pre-update minibatch metrics; changing weights",
  "uniform_brier_score": 0.9,
  "uniform_nll": 2.302585092994046,
  "validation_examples": 5000,
  "weight_decay": 0
}
```

## Source provenance

```json
{
  "command": [
    "/opt/hostedtoolcache/Python/3.12.14/x64/bin/cnn-benchmark",
    "--data-dir",
    "data/cifar10",
    "--output",
    "results/cifar10-baseline",
    "--seeds",
    "42",
    "--split-seed",
    "1729",
    "--epochs",
    "10",
    "--batch-size",
    "128",
    "--learning-rate",
    "0.001",
    "--threads",
    "2"
  ],
  "git_revision": "37dbbbc9a31887ea2fd0781df733c05b0e055fec",
  "github_run_url": "https://github.com/logannye/simple-cnn/actions/runs/37072562495",
  "model_source_sha256": "12f4bb6d92783000e89625408ac2a734892c8eeabe1b4d199aff09c39e77a0b2",
  "working_tree_clean": true
}
```
