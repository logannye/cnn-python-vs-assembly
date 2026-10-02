# CIFAR-10 baseline v1 — retained reference

Completed October 2, 2026. [Full report](report.md) · [Complete evidence bundle](https://github.com/logannye/simple-cnn/releases/tag/baseline-cifar10-v1) · [Successful hosted run](https://github.com/logannye/simple-cnn/actions/runs/37072562495).

The unchanged 5,418-parameter CNN was trained from scratch for 10 epochs with each of seeds 42, 43, and 44. The fixed split is 45,000 training / 5,000 validation / 10,000 official test images. Each checkpoint was selected using validation accuracy, never test accuracy. All three selected epoch 10.

| Measurement | Result |
|---|---:|
| Mean test accuracy | 46.36% |
| Test accuracy sample SD across seeds | 0.26 percentage points |
| Individual test accuracies | 46.37%, 46.10%, 46.61% |
| Mean macro F1 | 0.4551 |
| Mean top-3 / top-5 accuracy | 79.39% / 91.70% |
| Mean negative log likelihood | 1.4978 |
| Mean multiclass Brier score | 0.6776 |
| Mean expected calibration error (15 bins) | 4.89 percentage points |
| Training + validation wall time per seed | 136.7–200.6 seconds |
| Warm CPU batch-1 forward median latency | 0.168–0.263 ms |

## Interpretation

This is a reproducible, inexpensive reference with substantial classification error (53.64% on average). It performs well above 10% uniform guessing, but it is a modest classifier at this training budget.

Deterministic training accuracy was 46.69–46.84%, close to test accuracy. Training and validation loss were still falling near epoch 10, and each seed selected the last epoch. That is more consistent with limited fitting under this architecture/training budget than pronounced overfitting. It does not establish whether more epochs, more capacity, or a different optimization recipe will improve performance most.

Classes differ strongly. In representative seed 42, cat recall was 21.3%; 357 of 1,000 cats were predicted as dogs. Frog recall was 62.9%. The report retains all class scores and confusions for every seed. Average confidence (41.48%) was below average accuracy (46.36%); calibration was generally underconfident, with bin-dependent errors retained for inspection.

Three seeds give only a preliminary estimate of training variability. Individual accuracy confidence intervals are approximately ±0.98 percentage points and describe test-example uncertainty conditional on one fitted model; the seed SD is a different quantity. The same 10,000 test images are reused across seeds.

Timing is specific to each recorded hosted CPU and two Torch threads. It excludes dataset download and dependency setup. Inference latency is warmed forward-only timing; it excludes decoding, preprocessing, model loading, and softmax. Peak process memory of about 847 MiB includes the image dataset and measurement code, not only the model.

## Reuse in the next experiment

- Source revision: `37dbbbc9a31887ea2fd0781df733c05b0e055fec`; release tag: `baseline-cifar10-v1`.
- [Prespecified protocol](../../PROTOCOL.md), [manifest](manifest.json), [summary](summary.json), and [independent audit](audit.json).
- The release archive contains all three checkpoints; exact train/validation/test indices; 180,000 indexed probability rows across three runs and three splits; histories; metrics; timing samples; environment and dependency records; figures; and SHA-256 checksums. There are 10,000 unique test examples, not 30,000 independent ones.
- The independent audit checked 39 original artifact file hashes, all prediction rows, split membership, reconstructed label hashes, confusion matrices, accuracy/F1/NLL/Brier values, checkpoint hashes, and aggregate accuracy.
- Keep the splits, seeds, evaluation, and training budget fixed unless one is the intended experimental variable. Use saved test indices for paired comparisons. Tune using validation data and avoid choosing follow-up experiments by repeated inspection of test results.
