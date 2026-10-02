# Matched PyTorch runtime

`python_train.py` reuses the unmodified `classifier.model.SimpleCNN` and
`torch.optim.Adam`. It accepts the same byte-for-byte prepared inputs and emits
the same numerical output files as the [assembly runtime](../assembly/ABI.md):

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 PYTHONPATH=src \
  .venv/bin/python comparison/python_train.py INPUT_DIRECTORY OUTPUT_DIRECTORY 25 45000
```

The output directory must be empty. An input `schedule.bin` must contain exactly
`epochs × 45000` index/flip records; selecting fewer training examples does not
change that format. For a bounded check, use the two-epoch smoke inputs, `2 257`
and `CNN_ASM_EVAL_LIMIT=129`, exactly as with the assembly executable. The final
partial training batch then contains one image. The evaluation limit is a smoke
control and must be absent from official runs.

## Shared experimental controls

Both runtimes read all raw images, labels, sorted split indices, initial
parameters, and augmentation schedules into owned memory before training starts.
There is no memory mapping, normalized-image cache, random sampling, or file read
in the training loop. Each batch is converted from uint8 NHWC to contiguous
float32 NCHW, divided by 255, flipped according to the supplied flags, subtracted
by 0.5 and divided by 0.5. The arithmetic has been checked against the original
per-image torchvision/reference transform. Python uses batched tensor operators;
assembly transforms individual images in its two workers.

The model, 5,418-parameter layout, float32 arithmetic, cross-entropy objective,
batch size 128, Adam learning rate .001, betas .9/.999, epsilon 1e-8, zero decay,
25 epochs, and checkpoint selection rule agree. PyTorch sets two intra-op
threads, one inter-op thread, deterministic algorithms, CPU placement, and
unfused/non-foreach Adam. It executes native PyTorch numerical kernels. Assembly
uses two persistent workers and handwritten kernels. Those implementation and
reduction-order differences are the experimental treatment, so bitwise identity
between implementations is not promised.

The two-epoch, 257-example fixed-input smoke reproduces the previous PyTorch
reference's flat parameters and predictions exactly, and is independently
compared with the assembly smoke using the existing numerical tolerances. No
PIL, DataLoader, RNG, dataset download, or preparation runs inside the timed
training region.

## Measurement and artifact contract

`training_seconds` is wall time around the complete epoch loop: transform,
forward/loss/backward, Adam, per-epoch validation, best checkpoint writes, flushed
CSV history, and console progress. Setup and input loading, final `last.bin`
write, selected-checkpoint final evaluation, and forward microbenchmarks are
outside that region. `training_cpu_seconds` is the process user+system CPU delta
over the same scope. CPU seconds sum work across both compute threads and can
exceed wall time.

Both paths reject nonfinite losses, logits, probabilities, gradients, parameters
and Adam moments in the timed training/evaluation paths. Python computes an
explicit softmax for its probability check alongside PyTorch cross-entropy;
assembly's loss routine already returns the softmax. This small implementation
cost is included in the comparison. Checks are excluded from forward-only timing.

`peak_rss_loaded_bytes`, `peak_rss_training_bytes`, and `peak_rss_bytes` are
process lifetime high-water resident size sampled before training, after training,
and after final evaluation/timing. They include framework/process overhead and
loaded input arrays, not only model allocations. They are cumulative high-water
samples, so differences between samples are not independent allocation totals.
The whole-process user/system CPU times include startup and final evaluation.

`history.csv`, `best.bin`, `last.bin`, three prediction arrays and two timing
arrays use the assembly ABI. Final prediction timing includes normalization,
forward, loss/probabilities and checks, but excludes the prediction-file write;
aggregate final evaluation also includes file writes. `best.bin` contains the
highest-validation-accuracy checkpoint, breaking ties by lower validation loss;
final training/validation/test probabilities all use that selected checkpoint.
Test predictions never affect training or checkpoint selection.

Forward timing uses the same normalized first 128 official test images, 20 warmup
calls and 100 measured calls each for batch sizes 1 and 128. Only the model
forward call is timed, with no transforms, loss, probability checks, softmax or
file I/O. Repeated samples within one process are descriptive microbenchmarks,
not independent training runs.
