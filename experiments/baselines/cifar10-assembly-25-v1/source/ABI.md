# ARM64 assembly experiment ABI (implementation contract)

All numerical computation, neural-network forward/backward, Adam, minibatch
updates, epoch control and checkpoint selection are implemented in handwritten
Apple ARM64 assembly. No Torch, BLAS, Accelerate, C or compiler-generated
numerical kernels are linked into the executable. libc is allowed for process,
file, memory, clocks, printing and pthreads. Python prepares fixed reference
inputs and independently audits outputs. All model arithmetic is float32 unless
explicitly identified (Adam bias correction and reporting sums may use float64).
Floating-point agreement is measured with tolerances, not claimed bitwise.

## Parameters
Contiguous float32 values, in PyTorch `state_dict()` order, total 5,418 / 21,672 bytes:

| Array | Count | Byte offset |
|---|---:|---:|
| conv1 weights [16,3,3,3] | 432 | 0 |
| conv1 bias [16] | 16 | 1728 |
| conv2 weights [32,16,3,3] | 4608 | 1792 |
| conv2 bias [32] | 32 | 20224 |
| linear weights [10,32] | 320 | 20352 |
| linear bias [10] | 10 | 21632 |

## Kernel ABI
Darwin AArch64 ABI, exported symbols have leading underscore. Arguments listed
in x0... unless floating arguments explicitly indicated. Callee preserves x19-x28,
x29/x30, d8-d15; x18 reserved. All tensors are contiguous channel-first float32.
All functions reentrant and no mutable global scratch.

- `cnn_conv_forward(input, weights, bias, output, shape)`; shape is three uint32
  values `{input_channels, height_width, output_channels}`. 3x3 stride1 padding1.
- `cnn_conv_backward(input, weights, grad_output, grad_input, grad_weights,
  grad_bias, shape)`; grad_input may be NULL; overwrite grad_input when nonnull;
  accumulate into existing grad_weights/grad_bias. Caller zeros gradients once
  per minibatch/worker. Dimensions as above.

## Whole model ABI
- `cnn_workspace_bytes()` returns size_t, constant runtime workspace size.
- `cnn_forward(parameters, normalized_image, workspace, logits_out)` forwards
  ONE 3x32x32 float image, stores activations in workspace for backward.
- `cnn_backward(parameters, normalized_image, workspace, grad_logits,
  accumulated_parameter_gradients)`; reuses saved activations, adds gradients.
- `cnn_loss(logits, label, probabilities_out, grad_logits_out)` returns float
  loss in s0. grad_logits = softmax - one_hot WITHOUT batch-size division.
  label is uint32 in w1; output pointers may both be required (caller provides).
- `cnn_adam(parameters, mean_gradients, first_moment, second_moment,
  step)`; 5,418 values, lr=.001 beta1=.9 beta2=.999 epsilon=1e-8, zero decay;
  step is uint32 in w4, moments caller-initialized to zero. Must retain PyTorch
  operation ordering as closely as feasible, document differences.
- `cnn_exp(float)` / `cnn_log(float)` float argument s0, return s0; handwritten
  assembly math. Softmax subtracts maximum. Loss uses stable logsumexp so a
  low true-class probability never underflows the cross-entropy loss.

## Standalone executable inputs and outputs
CLI: `cnn-assembly INPUT_DIRECTORY OUTPUT_DIRECTORY [epochs] [train_limit]`.
Epochs defaults25, train_limit defaults45000; optional smaller values for parity
smokes only, never alter the official measurement protocol. Exactly two persistent
pthread workers, each processes one contiguous half of each minibatch. Worker
parameter gradients are summed in worker order then divided by actual batch
size. One optimizer update per batch, final partialbatch handled. Training loss
and accuracy are measured before update. Validation no augmentation every epoch.
Select best validation accuracy, ties lower validation loss. Save all25 epochs.

Input files are little-endian, headerless with exact sizes:
- `images.bin`: 60,000 RGB images as uint8 contiguous NHWC, train50000 then test10000.
- `labels.bin`: 60,000 uint32 class indices.
- `train_indices.bin`: 45,000 uint32 original indices, sorted.
- `val_indices.bin`: 5,000 uint32 original indices, sorted.
- `init.bin`: 5,418 float32 parameters from matching PyTorch initialization.
- `schedule.bin`: `epochs*45000` records of two uint32 `{original_image_index, flip}`,
  preserving EXACT PyTorch DataLoader sampler and RandomHorizontalFlip draws for
  the original seed. Split seed1729 and seeds42,43,44 prepared separately.
  Reference Python validates schedule generation by running the actual loader.
Pixel transform in assembly: float(pixel)/255.0; subtract0.5; divide0.5;
flip width if flag=1, convert NHWC to CHW. No resizing necessary (32x32).

Outputs:
- `history.csv`: epoch,train_loss,train_accuracy,val_loss,val_accuracy,train_seconds,
  val_seconds,epoch_seconds (including checkpoint copy/write), selected_epoch.
- `best.bin`: selected flat parameters, 21,672bytes.
- `last.bin`: final flat parameters, same size.
- `predictions_train.bin`, `predictions_validation.bin`, `predictions_test.bin`:
  float32 probabilities in sorted original-index order; counts45000/5000/10000.
- `performance.json`: training_seconds (train+val+checkpoint, excludes setup and
  final eval), best_epoch, parameter_count5418, threads2, final_evaluation_seconds,
  per-split prediction_seconds, peak_rss_bytes, process user/system CPU seconds.
  Root audit will retain compiler, source, binary, environment and all file hashes.
- `timing_batch1.bin`, `timing_batch128.bin`: float64 milliseconds for100forward-only
  passes after20warmup, using fixed normalized input from first128testimages,
  two-worker dispatch for128, no file/transform/softmax; same scope as baseline.

Executable refuses nonempty output directory. Nonfinite numerical values or
failed file operations are fatal. No silent fallback to Python/native BLAS.
