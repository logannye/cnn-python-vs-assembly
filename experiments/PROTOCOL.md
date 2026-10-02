# CIFAR-10 baseline v1

This protocol is fixed before inspecting test results. It measures the existing
`SimpleCNN` under a small, fixed training budget. It does not measure the best
accuracy this architecture could reach after tuning or longer training.

## Data and training

- Dataset: complete [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html),
  Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
- Split: reserve 500 examples per class from the official 50,000-image training
  set using NumPy's seeded permutation, seed 1729. This leaves 45,000 training
  and 5,000 validation examples. Use the official 10,000-image test set.
- Preserve original dataset indices and SHA-256 fingerprints. All runs use the
  same split; no test data is used for fitting or checkpoint selection.
- Model: unchanged `classifier.model.SimpleCNN`, 5,418 parameters for 10 classes;
  initialize from scratch. Two convolution/ReLU/pooling blocks, global average
  pooling, and a linear head. No pretraining or ensembling.
- Inputs: native 32 by 32 RGB; mean and standard deviation 0.5 per channel.
  Random horizontal flips with probability 0.5 during training only.
- Train three prespecified seeds: 42, 43, 44. Each receives 10 epochs, batch
  size 128, cross-entropy loss, Adam learning rate 0.001, default betas/epsilon,
  zero weight decay, and no learning-rate schedule or early stopping.
- Select each seed's checkpoint by validation accuracy, breaking ties with
  lower validation loss. Evaluate its test predictions only after training.
- CPU execution, two intra-op threads, one inter-op thread, deterministic Torch
  algorithms. GitHub runner hardware and installed package versions are retained.

The image size and batch size are explicit benchmark settings; the ordinary
`cnn-train` command defaults to 64 by 64 images and batch size 32. The model and
optimizer recipe are the same. The benchmark loop adds measurement and dataset
handling without modifying the existing CNN.

## Retained measurements

- Per-epoch augmented online training loss/accuracy, deterministic validation
  loss/accuracy, training/validation/epoch time, and training throughput.
- At the selected checkpoint, deterministic train, validation, and test
  predictions with original indices, true labels, and all class probabilities.
- Accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, top-3
  and top-5 accuracy, per-class precision/recall/F1 and support, raw and
  row-normalized confusion matrices, one-vs-rest ROC AUC and average precision.
- Negative log likelihood, multiclass Brier score (sum over classes per example),
  mean confidence, confidence for correct/incorrect predictions, and expected
  and maximum calibration error in 15 equal-width confidence bins.
- Per-seed test accuracy 95% Wilson intervals and macro-F1 95% percentile
  intervals from 1,000 multinomial resamples of the confusion matrix.
- Cross-seed means, sample standard deviations, and ranges. Three seeds provide
  a small estimate of observed training variability; per-example intervals do
  not include this variability. Do not treat the three shared test sets as
  30,000 independent test images. Train/validation resampling intervals are
  descriptive, because those examples were used to fit/select the model.
- Parameter count, checkpoint size/hash, process peak resident memory, training
  wall/CPU time, final prediction time, and raw forward-latency samples.
- Batch-1 and batch-128 CPU forward-only inference: 20 warmup iterations and
  100 measured iterations. Report median and p95 batch latency and throughput.
  These exclude image decoding, preprocessing, model loading, and softmax.
  Peak memory includes the dataset and metrics, not just the network.

Timing comparisons require matching CPU/thread settings and scope. Hosted
runner variability means timing differences alone may not reflect code changes.
No energy-use estimate is inferred from these timings.

## Reproduce and compare

Install `python -m pip install -e '.[dev,benchmark]'`, then run:

```bash
cnn-benchmark --output results/cifar10-baseline --seeds 42 43 44
python -c "from pathlib import Path; from classifier.report import build_report; build_report(Path('results/cifar10-baseline'))"
```

The manually dispatched **CIFAR-10 baseline** GitHub Actions workflow runs the
three seeds on separate hosted CPU runners and combines their outputs. Its
artifact contains checkpoints, predictions, split indices, metrics, histories,
environment manifests, frozen dependencies, figures, and checksums. A retained
baseline release and local copy preserve these beyond the 90-day CI artifact
retention period.

For the next experiment, keep split indices, seeds, data preparation, training
budget, and evaluation policy fixed unless a change is the experimental variable.
Use saved test indices/probabilities for paired comparisons. Select any tuning
on validation data; repeated decisions from this test set weaken its role as an
independent estimate. Preserve all three results rather than selecting a seed by
test accuracy. A future architecture or optimization study may need more seeds.
