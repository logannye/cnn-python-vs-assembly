# Matched CPU experiment: CIFAR-10, 25 epochs

This protocol is frozen in Git before the measured runs. It compares the
PyTorch and handwritten ARM64 implementations of the same small CNN on one
Apple M4 Max. The implementation, including its kernels, batching, dispatch
and runtime, is the experimental variable. It is not a comparison of two
different neural-network architectures or a general language-speed benchmark.

## Controls

| Condition | Identical setting |
|---|---|
| Network | Two 3x3 padded convolutions (3→16→32), ReLU/maxpool twice, global average pool, 32→10 linear; 5,418 parameters |
| Data | CIFAR-10: 45,000 train, 5,000 validation, 10,000 test; same original image IDs and labels |
| Initial parameters | Same exported float32 weights for seeds 42, 43, 44 |
| Random choices | Same precomputed, hashed full 25-epoch permutation/flip stream per seed |
| Preprocessing | Raw uint8 pixels gathered, converted to float32, divided by 255, flipped when prescribed, then subtract 0.5 and divide by 0.5 |
| Optimization | Cross-entropy; Adam lr=0.001, betas=(0.9,0.999), epsilon=1e-8; no decay, scheduler, dropout or pretrained weights |
| Batch/epochs | 128, retain final 72 images; exactly 25 epochs |
| Selection | Highest validation accuracy, ties lower validation loss; test only after training and selection |
| Compute | Same Mac CPU, float32, two compute threads; PyTorch one interop thread; no GPU |
| Execution | One measured child at a time, AC power, 15-second fixed pause after each child |
| Safety | Input shape/range/permutation checks before training; finite loss/logits/probabilities/gradients/parameters/Adam moments during training |
| Output | Same flat float32 best/last checkpoints and sorted split probability arrays |

The same raw fixture files are loaded into owned memory before each training
timer. Input hashes are checked once before any child starts, warming a shared
filesystem cache. Neither implementation runs shuffle/flip RNG or PIL inside
the training timer. Both perform gather, normalization and recorded flips there.
The Python training loop is eager PyTorch, whose tensor operations execute
compiled native kernels. The assembly executable uses hand-authored arithmetic
and calls system routines only for memory, files, threads, clocks and printing.

## Prespecified order and repeated measurements

| Seed | Repeat 1 | Repeat 2 |
|---|---|---|
| 42 | Python → assembly | assembly → Python |
| 43 | assembly → Python | Python → assembly |
| 44 | Python → assembly | assembly → Python |

These are 12 complete fits, each initialized from its seed's initial checkpoint.
The two repeats are timing replicates, not six independent training seeds.
Checkpoints, predictions and non-timing learning histories are compared between
repeats. If they match, quality is summarized across three distinct seeds.
Timing reports retain all six observations per implementation, average repeats
within each seed, and report the mean and sample SD of three seed means.
The primary speed ratio is the geometric mean of six paired Python/assembly
training-duration ratios. Seed variability and repeat timing variability remain
separate. No run is discarded for being slower or for its model accuracy.

## Measurement boundaries

- **Training wall time:** all epoch training and validation, best-checkpoint
  writes, CSV records/flushes and progress output. Excludes input/model setup,
  last-checkpoint export, final selected-model evaluation and microbenchmarks.
  Checkpoint-write counts are retained because selection events may differ.
- **Training CPU time:** process user+system CPU delta around the same epoch loop.
  CPU/wall ratio shows utilization; two compute threads do not guarantee equal
  core use or core placement.
- **Process wall time:** parent timer around child launch through successful
  exit, including Python import, setup, evaluation and timing benchmarks.
- **Memory:** macOS `ru_maxrss` bytes, sampled before training, after training
  and before exit. These are lifetime high-water marks through each phase,
  including loaded inputs/framework memory; differences of snapshots are not
  interpreted as isolated training allocations.
- **Forward latency:** pre-normalized first 128 test images, eager forward only,
  no softmax/checks/transforms; batch sizes 1 and 128, 20 warmups and 100 recorded
  samples each. Report p50/p95/p99 and batch throughput. Assembly batch 1 uses
  the calling thread; batch 128 uses its two persistent workers. Both runtimes
  have the same two-thread compute budget, not necessarily equal utilization.
- **Quality:** retained selected-checkpoint probabilities for every train,
  validation and test example. Accuracy, top-k, cross-entropy, macro/weighted
  precision/recall/F1, confusion/per-class metrics, Brier score, ECE, confidence
  and generalization gaps. Pair predictions by original test image ID.

Power, macOS thermal information and load averages are recorded before and
after every child. The fixed pause and reversed order reduce order effects;
they do not fix background processes, clock frequency, OS scheduling or
physical temperature. No claim of perfectly constant physical conditions is
made. macOS may not expose all thermal fields. The machine is not CPU-affinity
pinned. This is a single-machine, small-three-seed experiment.

## Correctness and provenance gates

Before running: numerical forward/backward/Adam parity, partial-batch standalone
runtime parity, corrupt-input/repeatability safety tests and matched Python
smoke/normalization checks must pass. Source must be committed and clean.
Retain the Git revision, source/binary/input SHA256 values, exact package and
compiler versions, commands, run order, raw histories, probabilities, timing
samples, checkpoints and process measurements. Numerical kernels may use
different float32 reduction orders, so tolerances—not bitwise identity across
implementations—define parity. Long-run trajectories can diverge legitimately.

No hyperparameter tuning, seed changes or test-set feedback is permitted during
this run. The earlier Python DataLoader and assembly baselines stay intact as
historical results; their timing ratio is superseded by this matched protocol.
