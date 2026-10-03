# Matched CNN comparison: complete report

Assembly completed the matched training loop **1.69× faster** (geometric mean of six paired ratios). Mean test accuracy was **51.99% for Python** and **51.99% for assembly**; the assembly-minus-Python mean difference was **-0.0033 percentage points**. This small observed difference does not establish statistical equivalence.

| Measurement | Python / PyTorch | ARM64 assembly |
|---|---:|---:|
| Test accuracy (%) | 51.99 ± 0.57 | 51.99 ± 0.63 |
| Test macro F1 | 0.5141 ± 0.0071 | 0.5140 ± 0.0075 |
| Training + validation (s) | 171.68 ± 1.02 | 101.34 ± 0.62 |
| Training CPU time (s) | 257.85 ± 1.65 | 201.18 ± 0.94 |
| Whole process (s) | 177.15 ± 1.01 | 104.01 ± 0.64 |
| Forward batch 1, median (ms) | 0.1190 ± 0.0040 | 0.0449 ± 0.0003 |
| Forward batch 128, median (ms) | 7.061 ± 0.169 | 4.345 ± 0.061 |
| Peak RSS through training (MiB) | 526.3 ± 1.4 | 187.5 ± 0.0 |

Values are mean ± sample SD across three seeds; timing repeats are averaged within seed. At about 52% test accuracy, this deliberately small network provides a modest, inspectable baseline.

## Seed-level quality

| Seed | Python accuracy | Assembly accuracy | Difference (pp) | Prediction disagreements |
|---|---:|---:|---:|---:|
| 42 | 52.09% | 52.08% | -0.010 | 41/10,000 (0.41%) |
| 43 | 51.38% | 51.32% | -0.060 | 33/10,000 (0.33%) |
| 44 | 52.51% | 52.57% | +0.060 | 177/10,000 (1.77%) |

Repeated checkpoints, prediction files and non-timing histories were identical within each implementation/seed. There are three independent training seeds, not six. Every seed uses the same 10,000 test images; paired bootstrap intervals in `summary.json` are conditional on a fitted pair and exclude training-seed uncertainty.

## Test metrics

| Metric | Python / PyTorch | ARM64 assembly |
|---|---:|---:|
| top3_accuracy | 0.83387 ± 0.00797 | 0.83423 ± 0.00826 |
| top5_accuracy | 0.93843 ± 0.00447 | 0.93867 ± 0.00456 |
| negative_log_likelihood | 1.34426 ± 0.02217 | 1.34433 ± 0.02219 |
| balanced_accuracy | 0.51993 ± 0.00571 | 0.51990 ± 0.00630 |
| macro_precision | 0.52008 ± 0.00686 | 0.52015 ± 0.00740 |
| macro_recall | 0.51993 ± 0.00571 | 0.51990 ± 0.00630 |
| macro_f1 | 0.51411 ± 0.00710 | 0.51404 ± 0.00750 |
| weighted_f1 | 0.51411 ± 0.00710 | 0.51404 ± 0.00750 |
| brier_score | 0.61851 ± 0.00820 | 0.61859 ± 0.00820 |
| ece | 0.03254 ± 0.00377 | 0.03232 ± 0.00379 |
| mce | 0.05984 ± 0.00641 | 0.05733 ± 0.01014 |
| mean_confidence | 0.48832 ± 0.00452 | 0.48851 ± 0.00459 |
| multiclass_roc_auc_ovr_macro | 0.89460 ± 0.00273 | 0.89460 ± 0.00273 |

Fractions are in [0,1] where applicable; NLL uses natural logarithms. ECE uses 15 equal-width confidence bins. Detailed class support, precision/recall/F1, ROC/AP, confusion matrices, calibration bins and fixed-model confidence intervals are retained in [per-run metrics](metrics/).

## All measured runs

| Order | Seed | Repeat | Implementation | Selected epoch | Train + val (s) | Process (s) | Training RSS (MiB) |
|---|---:|---:|---|---:|---:|---:|---:|
| 1 | 42 | 1 | python | 25 | 168.930 | 174.487 | 524.7 |
| 2 | 42 | 1 | assembly | 25 | 101.985 | 104.659 | 187.5 |
| 3 | 42 | 2 | assembly | 25 | 101.401 | 104.062 | 187.5 |
| 4 | 42 | 2 | python | 25 | 172.221 | 177.659 | 524.6 |
| 5 | 43 | 1 | assembly | 25 | 101.455 | 104.173 | 187.5 |
| 6 | 43 | 1 | python | 25 | 172.352 | 177.672 | 528.2 |
| 7 | 43 | 2 | python | 25 | 172.833 | 178.466 | 526.8 |
| 8 | 43 | 2 | assembly | 25 | 101.926 | 104.648 | 187.5 |
| 9 | 44 | 1 | python | 25 | 176.444 | 182.016 | 527.0 |
| 10 | 44 | 1 | assembly | 25 | 100.991 | 103.631 | 187.5 |
| 11 | 44 | 2 | assembly | 25 | 100.261 | 102.907 | 187.5 |
| 12 | 44 | 2 | python | 25 | 167.325 | 172.578 | 526.3 |

No measured run was discarded or retried. Run order and the 15-second pauses were specified before measurement. See [runs.csv](runs.csv) for CPU utilization, throughput, p50/p95/p99 latency, checkpoint writes and every resource observation.

## Visual comparisons

![runtime comparison](figures/runtime.png)

![quality comparison](figures/quality.png)

![inference comparison](figures/inference.png)

![memory comparison](figures/memory.png)

![learning comparison](figures/learning.png)

## Experimental controls and interpretation

Both implementations use identical raw pixels, labels, split IDs, initial float32 weights and precomputed epoch/flip schedules; the same 5,418-parameter network, Adam settings, 25 epochs, batch size 128 and selection rule; CPU float32 and two compute threads on the same M4 Max. Gather/normalize/flip, forward/backward/Adam, validation, best-checkpoint writes and logging are inside both training timers. Setup and final evaluation are outside; whole-process duration is separately retained.

PyTorch executes compiled native kernels. Kernel layout, batching, reduction order, scheduling, allocation and runtime overhead remain implementation differences. Assembly calls only system routines outside its handwritten numerical kernels. Cross-implementation float32 trajectories are not bitwise identical.

RSS values are lifetime process high-water marks through each phase and include the framework and inputs. They are not isolated model allocations. Forward timing uses 20 warmups and 100 samples for each batch size. Chart percentile bars average within-run percentiles; their error bars show seed SD, not latency confidence intervals.

AC power was present at all recorded endpoints. The plan and per-run metadata retain load averages and the exact macOS thermal responses. Endpoint checks do not prove uninterrupted power or identical physical temperature; macOS reported no recorded thermal/performance warning status, not a temperature measurement. The observed one-minute load average ranged from 3.55 to 15.74 across endpoints on the 16-core host. No CPU affinity, fixed clock, exclusive machine use or energy measurement is claimed.

## Evidence and reproduction

- Measured revision: [`9978cb0`](https://github.com/logannye/cnn-python-vs-assembly/tree/9978cb0929d64b16a97b0d70dd6f83b93d78e075).
- [Frozen run plan and environment](plan.json), [summary and complete audit](summary.json), [independent NumPy audit](independent-audit.json).
- [All run measurements](runs.csv), [model metrics](models.csv), [paired test comparisons](paired_test.csv), [per-run details](metrics/).
- [Protocol](../../../comparison/PROTOCOL.md) and [reproduction commands](../../../comparison/REPRODUCE.md).
- [Full release](https://github.com/logannye/cnn-python-vs-assembly/releases/tag/comparison-cifar10-matched-cpu-25-v1): exact binary inputs, all best/last checkpoints, all prediction arrays, 2,400 raw latency samples, histories, source snapshot, native executable and checksums.

The earlier baseline reports remain unchanged. Their timing scopes differ, so this experiment supersedes their Python/assembly speed ratio.
