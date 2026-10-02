# Matched CNN implementation experiment: ARM64 assembly

This protocol is fixed before the assembly model's full test results are inspected.
The reference is the retained [25-epoch Mac CPU baseline](MAC_CPU_25_PROTOCOL.md),
source `ec931522316e4c0af783da9425a5797422d4f301`, three seeds 42, 43, 44.

## Experimental variable and meaning of reproduction

Replace the numerical implementation and training runtime with handwritten
Darwin ARM64 assembly. Preserve the mathematical model, its 5,418 parameters,
float32 representation, cross-entropy objective, gradients, Adam hyperparameters,
batch size 128, 25 epochs, two compute workers, split seed 1729, all data and
preprocessing, validation checkpoint selection, and final evaluation protocol.
Train every seed from scratch. No model or hyperparameter changes based on results.

Export exact PyTorch initial weights, the sampler's image order, and horizontal
flip choices into binary input fixtures. The preparation code executes the same
model initialization and DataLoader RNG schedule, verifies each full epoch's
permutation and directly compares the first 256 actual loader images per seed
bitwise. Initialization is not rerun with a different assembly random generator.
Images remain raw uint8; assembly performs normalization and flipping itself.
The native process validates split disjointness and each epoch's complete schedule.

The user-approved boundary permits Python to prepare fixed inputs and independently
audit results, while every model operation, loss, derivative, optimizer update,
minibatch/epoch loop, validation decision, checkpoint and prediction is computed
by assembly. Only libc/OS file, memory, process, thread, clock and formatting
services are linked. No BLAS, Accelerate, Torch, C numerical kernels or libm math
functions implement model computation. Handwritten exp/log and hardware sqrt
support cross-entropy and Adam.

This is an algorithmic reproduction with measured floating-point agreement.
It is not a claim of bitwise identity to PyTorch: vector fused operations,
convolution reductions, per-image computation and two-worker gradient reductions
can round differently. Small numerical differences can change pool winners and
long training trajectories. State this distinction alongside the results.

## Gates before full execution

1. Check exp/log, softmax/cross-entropy, convolution forward/input/weight/bias
   gradients, complete model gradients, 100 Adam updates, and a coupled 10-step
   training trajectory against PyTorch with explicit recorded tolerances.
2. Check the actual executable against PyTorch on two epochs, 257 training
   examples and 129 evaluation examples per split, including batches of
   128, 128 and 1. Compare online metrics, checkpoint selection, weights and
   predictions. These are numerical fixtures, not estimates of model quality.
3. Check repeatability and failures: existing outputs, wrong sizes, nonfinite
   initial state, invalid labels/indices/flips, split overlap, invalid complete
   epoch permutations, and validation/test leakage. Preserve logs and hashes.
4. Inspect imported symbols to verify the assembly computation boundary.
   Compile/link locally for this explicitly requested Apple ARM64 experiment;
   retain exact source revision, compiler, executable and source hashes.
5. Freeze clean source before all three measured processes. Run them sequentially,
   with no concurrent training, numerical audits, builds or metric computation.

## Measurements and comparison

Preserve per-epoch losses, accuracies and wall times; all three selected/final
checkpoints; indexed probabilities for 45,000 train, 5,000 validation and 10,000
test images; the reference's full class/discrimination/calibration metrics and
uncertainty; paired per-seed test comparisons on the same images; initialization,
full schedules, data/source/binary hashes, environment and numerical audit evidence.
Evaluate each official test set once after validation selection. Do not choose
seeds or checkpoints using test performance.

Report mean, sample SD and individual results across the same three seeds.
Per-model confidence intervals and paired test-example intervals do not represent
seed uncertainty; all runs reuse the same 10,000 test images. Tuning based on these
results would reuse inspected test data, so use validation for later decisions.

The reference called from Python executes compiled PyTorch CPU kernels. This
experiment compares that complete implementation with a particular handwritten
assembly implementation; it cannot measure a universal Python-versus-assembly
language effect. No energy estimate is inferred from runtime.

Training wall times include normalization, flips, model/gradient work, updates,
validation and checkpoint writes. Assembly's shuffle/flip random draws are
prepared outside its timed process, whereas the reference draws them inside its
DataLoader. Report this scope difference with any observed runtime ratio.
Input loading and setup are excluded from both training timers. Both use two
compute workers, but batching, dispatch and CPU utilization need not be identical.

Forward-only timing uses the same first 128 normalized test images, 20 warmups,
100 measured passes and batch sizes 1/128, excluding normalization, I/O and
softmax. Assembly batch128 includes worker dispatch; batch1 uses the caller.
These scopes are the most directly comparable performance measurements.

Assembly peak RSS and user/system CPU time cover its whole native process;
Python preparation and post-run metrics are separate. The reference's RSS
includes its dataset and metrics machinery, and its CPU-time field covers only
training. Do not compute a CPU-time speedup between those different scopes.
Likewise the reference's final-evaluation wall time includes classification
metrics and bootstraps; assembly's native evaluation time includes predictions
and writes, with metric/audit time reported separately. Background load and
thermal conditions can affect local timings.
