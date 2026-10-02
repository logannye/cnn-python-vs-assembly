# Handwritten ARM64 CNN · CIFAR-10

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
| Top-1 accuracy | 51.99% | 0.63% | 51.32% | 52.57% |
| Balanced accuracy | 51.99% | 0.63% | 51.32% | 52.57% |
| Macro precision | 52.02% | 0.74% | 51.21% | 52.66% |
| Macro recall | 51.99% | 0.63% | 51.32% | 52.57% |
| Macro F1 | 51.40% | 0.75% | 50.54% | 51.85% |
| Weighted F1 | 51.40% | 0.75% | 50.54% | 51.85% |
| Top-3 accuracy | 83.42% | 0.83% | 82.47% | 83.92% |
| Top-5 accuracy | 93.87% | 0.46% | 93.34% | 94.14% |
| Negative log likelihood | 1.3443 | 0.0222 | 1.3281 | 1.3696 |
| Multiclass Brier score | 0.6186 | 0.0082 | 0.6123 | 0.6279 |
| Expected calibration error | 3.23% | 0.38% | 2.92% | 3.65% |
| Maximum calibration error | 5.73% | 1.01% | 4.56% | 6.36% |
| Mean confidence | 48.85% | 0.46% | 48.33% | 49.20% |
| Confidence when correct | 55.62% | 0.66% | 54.93% | 56.25% |
| Confidence when incorrect | 41.51% | 0.13% | 41.38% | 41.63% |
| Macro one-vs-rest ROC AUC | 0.8946 | 0.0027 | 0.8916 | 0.8968 |
| Macro average precision | 0.5493 | 0.0104 | 0.5386 | 0.5594 |

Lower negative log likelihood, Brier score, ECE, and MCE are better. Brier score sums squared probability error over classes, then averages over images. ECE is the count-weighted absolute accuracy–confidence gap over 15 equal-width bins; MCE is the largest nonempty-bin gap. Both depend on the binning. ROC AUC and average precision use one-vs-rest scores averaged equally over classes. Undefined metrics appear as —.

## Individual runs and uncertainty

| Seed | Selected epoch | Validation accuracy | Test accuracy | Accuracy 95% CI | Macro F1 | Macro F1 95% CI |
|---|---:|---:|---:|---|---:|---|
| 42 | 25 | 52.78% | 52.08% | [51.10%, 53.06%] | 51.85% | [50.76%, 52.80%] |
| 43 | 25 | 51.18% | 51.32% | [50.34%, 52.30%] | 50.54% | [49.55%, 51.43%] |
| 44 | 25 | 52.10% | 52.57% | [51.59%, 53.55%] | 51.82% | [50.89%, 52.73%] |

Accuracy intervals use the 95% Wilson method. Macro F1 intervals use a percentile bootstrap over test examples (1000 resamples; the random seed and exact method are saved in each metrics.json). These intervals describe test-sample uncertainty conditional on a fitted model and an independent, representative sampling assumption; they do not include training-seed variability or distribution shift. They are not intervals for the cross-seed mean.

## Fit and learning curves

| Seed | Final train accuracy | Validation accuracy | Test accuracy | Train − validation gap |
|---|---:|---:|---:|---:|
| 42 | 53.59% | 52.78% | 52.08% | 0.81 pp |
| 43 | 52.12% | 51.18% | 51.32% | 0.94 pp |
| 44 | 53.29% | 52.10% | 52.57% | 1.19 pp |

![Loss and accuracy by epoch](figures/learning_curves.png)

![Validation and test accuracy by seed](figures/accuracy_seed_comparison.png)

## Class-level performance

The table reports seed 42; the figure summarizes all completed seeds.

| Class | Support | Precision | Recall | F1 | ROC AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| airplane | 1000 | 61.30% | 45.30% | 52.10% | 0.9080 | 0.5973 |
| automobile | 1000 | 66.15% | 59.60% | 62.70% | 0.9467 | 0.6964 |
| bird | 1000 | 35.53% | 43.60% | 39.16% | 0.8198 | 0.3770 |
| cat | 1000 | 36.16% | 37.10% | 36.62% | 0.8446 | 0.3487 |
| deer | 1000 | 51.97% | 34.30% | 41.33% | 0.8669 | 0.4658 |
| dog | 1000 | 49.94% | 44.20% | 46.90% | 0.8904 | 0.5058 |
| frog | 1000 | 52.85% | 66.70% | 58.97% | 0.9197 | 0.6403 |
| horse | 1000 | 56.80% | 57.60% | 57.20% | 0.9035 | 0.6205 |
| ship | 1000 | 57.82% | 68.80% | 62.83% | 0.9354 | 0.6863 |
| truck | 1000 | 58.03% | 63.60% | 60.69% | 0.9329 | 0.6561 |

![Per-class F1](figures/per_class_f1.png)

![Normalized confusion matrix](figures/confusion_matrix.png)

Most frequent directed errors for seed 42:

| Actual → predicted | Images | Percentage of actual class |
|---|---:|---:|
| dog → cat | 245 | 24.50% |
| airplane → ship | 234 | 23.40% |
| cat → dog | 198 | 19.80% |
| automobile → truck | 193 | 19.30% |
| deer → frog | 189 | 18.90% |
| deer → bird | 178 | 17.80% |
| cat → bird | 145 | 14.50% |
| bird → frog | 136 | 13.60% |
| truck → automobile | 124 | 12.40% |
| frog → bird | 120 | 12.00% |

## Confidence and calibration

For seed 42, mean top-class confidence is 49.20%. It is 56.25% on correct predictions and 41.54% on incorrect predictions. Confidence is a model probability, not a guarantee of correctness. The reliability plot shows observed accuracy against mean confidence in each nonempty bin; the lower panel shows how many images fall in each bin. No post-hoc calibration is fitted.

![Reliability plot and confidence distribution](figures/reliability.png)

## Runtime and resource use

Assembly final evaluation excludes Python metrics and audit. Process CPU and peak RSS cover the complete executable; see comparison.json for scope differences from the PyTorch baseline.

| Seed | Train (s) | Final evaluation (s) | Parameters | Checkpoint (MiB) | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| 42 | 98.8176 | 2.0680 | 5418 | 0.0207 | 190.7969 |
| 43 | 99.6008 | 2.1040 | 5418 | 0.0207 | 190.7812 |
| 44 | 99.9329 | 2.0826 | 5418 | 0.0207 | 190.8125 |

| Seed | Batch size | Median batch latency (ms) | P95 batch latency (ms) | Images/s |
|---|---:|---:|---:|---:|
| 42 | 1 | 0.0450 | 0.0468 | 22085.0088 |
| 42 | 128 | 4.2757 | 4.3694 | 29914.9731 |
| 43 | 1 | 0.0448 | 0.0454 | 22168.0342 |
| 43 | 128 | 4.2604 | 4.3247 | 30057.6908 |
| 44 | 1 | 0.0447 | 0.0481 | 22185.0429 |
| 44 | 128 | 4.2360 | 4.3033 | 30243.2276 |

Timing is specific to the recorded host, device, thread count, batch size, warmup, and measurement procedure. Hosted runners can differ between seeds; consult each seed's environment in manifest.json before comparing speed. Latency measurements cover the model forward pass described in performance.json, not an end-to-end image-serving pipeline. Peak RSS is the process high-water mark, including data and evaluation allocations; it is not an isolated model-memory measurement. Per-epoch training and validation timings are retained in history.json.

## Retained evidence and future comparisons

- [Machine-readable summary](summary.json): per-seed scalar values and cross-seed aggregates.
- [Manifest](manifest.json): source revision, protocol, configuration, package versions, and hardware/software environment.
- [Split indices](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz): exact training and validation memberships.
- Seed 42: [metrics](seed-42/metrics.json), [history](seed-42/history.json), [performance](seed-42/performance.json), [checkpoint](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [train predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [validation predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [test predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz).
- Seed 43: [metrics](seed-43/metrics.json), [history](seed-43/history.json), [performance](seed-43/performance.json), [checkpoint](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [train predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [validation predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [test predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz).
- Seed 44: [metrics](seed-44/metrics.json), [history](seed-44/history.json), [performance](seed-44/performance.json), [checkpoint](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [train predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [validation predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz), [test predictions](https://github.com/logannye/simple-cnn/releases/download/baseline-cifar10-assembly-25-v1/cifar10-assembly-25-v1.tar.gz).

Prediction archives retain labels, class probabilities, and example indices. A future model can use these same splits, seeds, evaluation code, and indexed test examples for paired comparisons. Fix the next experiment's protocol before examining its test results, select checkpoints using validation only, and report both paired test-example uncertainty and variability across training seeds. Repeated experiment selection based on this test set will turn it into development data.

This is a small CNN baseline at a fixed 25-epoch training budget, not an estimate of the architecture's maximum attainable accuracy. It does not establish optimal hyperparameters, convergence, state-of-the-art performance, robustness to distribution shift, or production suitability.

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
  "epochs": 25,
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
  "binary_sha256": "fdcb34d39d2f2b1fb84b8c1e56df01afe7148c0a430ae2eddf7e3b3acdfc88e7",
  "command": [
    "assembly/run_experiment.py"
  ],
  "compiler": "Apple clang version 17.0.0 (clang-1700.4.4.1)\nTarget: arm64-apple-darwin25.2.0\nThread model: posix\nInstalledDir: /Library/Developer/CommandLineTools/usr/bin\n",
  "git_revision": "ba36c840759c9bf32db3827bb40180107d97d7db",
  "linked_libraries": "assembly/build/cnn-assembly:\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)\n",
  "model_source_sha256": "12f4bb6d92783000e89625408ac2a734892c8eeabe1b4d199aff09c39e77a0b2",
  "source_hashes": {
    "assembly/ABI.md": "9a437476cb23188e2329c0295a186c7e5049a63857d20dd9795be7b61c8dc34b",
    "assembly/CONVOLUTION.md": "6ffdba0b74f087e7097dd13ef7c62fe38099f860a0b944049ffe2896d6220e2e",
    "assembly/MODEL.md": "d91f9b4fca3c014cd8ee74335672c3e497f3ca605d6bdb240b817e581e40a5cf",
    "assembly/Makefile": "43e881b964ad32031eae57144201822a6e19747cc693649e2275e33a507dd368",
    "assembly/README.md": "3e0a480702fa2726f2694917ef09b98193118f449fd473c8823e2ef913d5d18c",
    "assembly/RUNTIME.md": "1f831064c745f966559e29b6f56dafb422d128db7b320d3dd982255daf6cd0ef",
    "assembly/collect.py": "a8c5bb83d2f8a53baa1ffc3adad3026f38a90ca7bc711a6bf28a36a047424ab4",
    "assembly/convolution.S": "cfbb30c928c7c30814418a4d73e05cf44839083f5767382105e7d07ba0dc8731",
    "assembly/model.S": "308e784016de2e9f45860bcfc4103c49409898b8d240a61cf35cca2b0e761c6d",
    "assembly/prepare.py": "9d152a7bf5a6d833921d79d57757340ce5f5ff08a697196735c3af41d4c3370a",
    "assembly/run_experiment.py": "dbb01042bef9e43ce25a7790591f34c4ca27fafdc4bbd2f6e2c574124a764b93",
    "assembly/runtime.S": "4e1d8a9afd07832f47ba6615b69eeda49af47ad70c7d2b2ea58bd9438d4f776f",
    "assembly/smoke.py": "f4ba5c264927e6a9d7b1d7bf8db2aafc1ee087ed88491edab89a2071343a5eb7",
    "assembly/test_runtime_safety.py": "c81ef5060662ee3a7fbf7e05f282efba3f5446f8a2d2cc7f4193d02d4a1d716a",
    "assembly/validate.py": "b74ad5cca775b32adc5ab687a15c2f51ee9407d86a049c1a2c1b8c0ef282540d",
    "assembly/validate_runtime.py": "188f277df0452bc83a994b6fa07e707270ee7bfad6cf5df4b806b0f0c6ace70c"
  },
  "undefined_symbols": "___error\n_abort\n_atoi\n_calloc\n_closedir\n_exit\n_fclose\n_ferror\n_fflush\n_fgetc\n_fopen\n_fprintf\n_fputs\n_fread\n_free\n_fwrite\n_getenv\n_getrusage\n_mach_absolute_time\n_mach_timebase_info\n_malloc\n_memcpy\n_memset\n_mkdir\n_opendir\n_printf\n_pthread_cond_broadcast\n_pthread_cond_init\n_pthread_cond_signal\n_pthread_cond_wait\n_pthread_create\n_pthread_join\n_pthread_mutex_init\n_pthread_mutex_lock\n_pthread_mutex_unlock\n_puts\n_readdir\n_strcat\n_strcpy\n_strlen\n",
  "working_tree_clean": true
}
```
