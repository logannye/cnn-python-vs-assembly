# Mac CPU baseline: 25 epochs

This rerun establishes a retained local CPU reference for later experiments.
The user specified 25 epochs before the new test results were inspected. The
original 10-epoch CIFAR-10 baseline and its release remain preserved.

## Fixed recipe

Use the unchanged 5,418-parameter `SimpleCNN`, initialized from scratch for
each of the same three seeds, 42, 43, and 44. Run the seeds sequentially in
separate processes on the Mac CPU, with two Torch intra-op threads, one inter-op thread,
zero data-loader workers, and deterministic Torch algorithms. Retaining the
thread count preserves the previous protocol; this is not a benchmark of the
Mac CPU at its maximum parallel throughput.

Use the same full CIFAR-10 data, native 32 by 32 RGB input, normalization mean
and standard deviation 0.5 per channel, and training-only random horizontal
flips with probability 0.5. Split seed 1729 reserves 500 images per class for
validation, leaving 45,000 training, 5,000 validation, and 10,000 official test
examples. Verify dataset fingerprints and actual split indices against the
original retained baseline.

For each seed, train for exactly 25 epochs with batch size 128, cross-entropy
loss, Adam learning rate 0.001, default betas and epsilon, zero weight decay,
and no scheduler or early stopping. Select the checkpoint with the highest
validation accuracy, breaking ties with lower validation loss. Evaluate the
test set only after all epochs and checkpoint selection. Retain all three
seed results; do not select a seed by test performance.

```bash
for seed in 42 43 44; do
  cnn-benchmark --output "results/mac-cpu-25-seed-runs/baseline-seed-$seed" \
    --seeds "$seed" --split-seed 1729 --epochs 25 \
    --batch-size 128 --learning-rate 0.001 --threads 2 || exit 1
done
python scripts/collect_baseline.py --inputs results/mac-cpu-25-seed-runs \
  --output results/cifar10-mac-cpu-25-v1
```

## Measurements and provenance

Retain all measurements defined in [the original protocol](PROTOCOL.md):
per-epoch learning curves and timings; final train, validation, and test
predictions with original indices and probabilities; classification,
per-class, discrimination, calibration, and confidence-interval metrics;
training wall and CPU times; prediction times; forward-only latency samples;
model size; memory; checkpoints; source revision; frozen dependencies; data
hashes; figures; and artifact checksums.

The macOS environment snapshot records the actual chip name, CPU count,
physical memory, GPU model and core count, and available Metal metadata.
It records the observed power-source category at collection time. Hardware
fields use an explicit allowlist that excludes serial numbers, UUIDs,
display details, and battery identifiers. Optional metadata collection failures
are retained without aborting the experiment. GPU metadata describes the
machine; execution remains on the CPU.

Forward timings retain the original scope: preprocessed inputs, 20 warmups,
100 measurements, batch sizes 1 and 128, excluding decoding, transforms,
model loading, and softmax. Process peak RSS covers each seed's whole
process lifetime, including data loading and evaluation; each seed starts in a
fresh process. The macOS RSS value is
already reported in bytes. No energy measurement is inferred from timings.

## Interpretation and later comparison

Relative to the hosted 10-epoch baseline, both the epoch budget and the
hardware/operating-system environment change. Platform-specific package
builds and numerical kernels may also differ; preserve their exact versions.
An accuracy difference is therefore not an isolated causal estimate of
training for 15 additional epochs, and a timing difference is not an isolated
hardware speedup. Preserving seeds does not guarantee identical numerical
trajectories across platforms.

For a later local CPU/GPU comparison, preserve this 25-epoch recipe, splits,
seeds, checkpoint-selection policy, and precision while recording each
backend's timing and memory semantics. CPU results alone do not establish
GPU performance. Local background load, power state changes, and thermal
conditions can affect observed throughput; the initial power snapshot does
not prove that conditions remained constant throughout training.

Report cross-seed means, sample standard deviations, and ranges separately
from each model's test-example confidence intervals. All seeds share the same
10,000 test images, so their test sets are not independent replications.
Repeated reuse of this already inspected test set weakens its role as a fresh
generalization estimate; use validation data for subsequent tuning decisions.
