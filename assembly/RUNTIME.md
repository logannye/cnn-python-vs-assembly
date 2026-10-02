# Standalone assembly training runtime

`runtime.S` provides the process entry point and all experiment control in
handwritten Apple ARM64 assembly. The only imported functions are model kernels
and macOS routines for allocation, byte copying, files, paths, clocks, formatted
output, process exit, environment lookup and pthread synchronization. No Python,
Torch, BLAS, Accelerate or C numerical implementation executes inside this
program.

```
assembly/build/cnn-assembly INPUT_DIRECTORY OUTPUT_DIRECTORY [epochs] [train_limit]
```

The defaults are 25 epochs and 45,000 training examples per epoch. Input formats
are defined in [ABI.md](ABI.md). All files must have the exact documented sizes,
including `schedule.bin`: `epochs * 45000 * 8` bytes even when `train_limit` is
smaller. Epochs must be 1–25 and the training limit must be 1–45,000. Optional
arguments exist for integration checks; published baseline comparisons use the
full defaults.

`CNN_ASM_EVAL_LIMIT=N` caps each evaluation split at the first `N` examples in
its documented sorted order, with a maximum of 45,000. This is exclusively a
smoke-test facility. It also caps per-epoch validation, so it changes checkpoint
selection and must be **unset for full experiments**. The actual training and
evaluation counts are written to `performance.json` to distinguish a smoke
from a full run. Timing measurements always use the first 128 official test
images and always retain 20 warmups plus 100 measured samples.

## Training and reproducibility

Exactly two pthread workers are created once. Each minibatch is split into
contiguous halves, with the first worker receiving `floor(batch_size / 2)`
examples. Each worker normalizes and optionally flips uint8 NHWC images into
float32 CHW images, runs forward/loss/backward, and accumulates its parameter
gradients in float32. The main assembly thread sums worker 0 then worker 1 and
divides by the actual minibatch size. Adam is called once per minibatch. The
final partial minibatch is included. Parameters are read-only while workers
execute and are updated only after both workers signal completion.

Image normalization executes three float32 operations in torchvision order:
`pixel / 255`, subtract `0.5`, then divide by `0.5`. Python preparation supplies
the exact original-image order and horizontal-flip decisions from the reference
loader; random-number generation is not rerun by assembly.

Training loss and accuracy describe predictions before each update. Accuracy
uses logits argmax with first-index tie breaking, matching the Python baseline. Losses
are accumulated as float64 reporting sums. Validation runs after every epoch
without augmentation. The selected checkpoint maximizes validation accuracy,
breaking ties by lower validation loss. Final train, validation and test
probabilities are produced using that checkpoint. `last.bin` separately retains
the final epoch's parameters.

This reproduces the mathematical model and training protocol, not bitwise
PyTorch arithmetic. Per-image convolution and fixed two-worker reductions may
sum floating-point values in a different order from PyTorch's batched CPU
kernels. Kernel, gradient and optimizer parity must therefore be independently
checked with numerical tolerances before a full experiment is interpreted.
The same two-worker count does not imply an identical batching implementation
or identical CPU utilization profile.

## Measurements and retained outputs

`history.csv` is flushed after every epoch. It contains training/validation
loss and accuracy, separate phase durations, total epoch duration including
checkpoint copy/write, and the checkpoint epoch selected so far. A progress
line is printed and flushed once per epoch.

`performance.json` records aggregate training wall time (`training_seconds`),
matching user-plus-system CPU time (`training_cpu_seconds`), selected epoch,
final selected-checkpoint evaluation time, each split's prediction time, and
the actual sample counts. The training interval includes image transforms,
training, per-epoch validation, checkpoint writes and epoch reporting; setup
and final evaluation are excluded. CPU snapshots immediately bracket the wall
interval, so their scope differs only by the resource/clock sampling overhead.
Per-split prediction time excludes writing the binary output, while the
enclosing final evaluation time includes those writes.

macOS `getrusage` reports peak resident size in bytes. `peak_rss_loaded_bytes`
is sampled after inputs, workspaces and worker setup, immediately before
training; `peak_rss_training_bytes` is sampled immediately after the epoch
loop. These are process high-water marks through each boundary, not current
RSS or allocated model memory, so subtracting them does not isolate training
memory. `peak_rss_bytes`, `process_user_cpu_seconds` and
`process_system_cpu_seconds` retain their whole-process scope through setup,
training, evaluation and inference timing. No GPU or machine-wide power
measurement is inferred from these counters.

`timing_batch1.bin` and `timing_batch128.bin` contain 100 little-endian float64
millisecond observations each. Normalization happens before the timed scope;
these measurements contain forward passes only, with no transform, softmax,
loss, gradient or output-file work. Batch 1 executes in the caller using an
idle worker's workspace. Batch 128 dispatches to both persistent workers and
includes dispatch/wakeup overhead. This makes overhead explicit; these timings
are not intended as isolated convolution-kernel microbenchmarks.

All clocks use `mach_absolute_time` converted to seconds with the platform's
reported timebase. `best.bin`, `last.bin` and probability files have the exact
headerless float32 formats documented in the ABI. Source and executable hashes,
compiler/environment provenance and independent numerical/metric audits are
retained by the surrounding preparation and audit workflow.

## Failures and output protection

An existing nonempty output directory is rejected. A new or empty directory
is accepted. Input byte lengths, input labels, sorted split indices, flip flags,
allocation results, file operations and pthread operations are checked. Training
and validation sets must be disjoint. Every epoch of the supplied schedule is
validated as an exact permutation of all 45,000 training indices, preventing
validation/test leakage and missing or duplicated examples. Losses, logits,
probabilities, averaged gradients, initial and updated parameters, and both Adam
moment buffers are checked for NaN or infinity; a detected nonfinite value terminates the
process with a nonzero exit status. There is no fallback execution backend.
An interrupted or failed run may leave partial output and is never silently
resumed or overwritten.
