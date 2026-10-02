# Handwritten model and arithmetic

`model.S` implements the complete two-block CNN head and reverse-mode gradient,
softmax/cross-entropy, Adam, and the elementary exponential/logarithm routines.
Convolutions call the handwritten functions in `convolution.S`. There are no
numerical calls to libc, libm, Torch, Accelerate, BLAS, or compiler-generated C
kernels. Only immutable constants live outside the caller-provided workspace.

The 5,418 parameters and all tensor dimensions follow `ABI.md`. The forward
path is 3x32x32 → Conv(3,16,3,pad1) → ReLU → MaxPool(2) →
Conv(16,32,3,pad1) → ReLU → MaxPool(2) → global average over 8x8 →
Linear(32,10). Parameters, activations, gradients, and optimizer state are float32.
Convolution and reduction arithmetic can differ in final bits from PyTorch:
mathematically identical operations do not imply bitwise-identical kernels.

## Workspace and differentiation

Each image uses a separate 252,160-byte workspace. It holds post-ReLU convolution
activations, pooled activations, uint8 max-pool choices, the average-pooled vector,
and backward intermediates. The allocation is fixed, independent of parameter
values. The functions are reentrant and each pthread supplies its own workspace.

| Byte offset | Contents | Bytes |
|---:|---|---:|
| 0 | First convolution, after ReLU | 65,536 |
| 65,536 | First pooled activation | 16,384 |
| 81,920 | First pool choices | 4,096 |
| 86,016 | Second convolution, after ReLU | 32,768 |
| 118,784 | Second pooled activation | 8,192 |
| 126,976 | Second pool choices | 2,048 |
| 129,024 | Average-pooled activation | 128 |
| 129,152 | Gradient of average-pooled activation | 128 |
| 129,280 | Gradient of second pooled activation | 8,192 |
| 137,472 | Gradient of second convolution | 32,768 |
| 170,240 | Gradient of first pooled activation | 16,384 |
| 186,624 | Gradient of first convolution | 65,536 |

Max pooling uses the first maximum in row-major order, including ties. The
ReLU derivative is zero at zero. Pool gradients scatter to the saved choice only
if that activation was strictly positive. Average-pool gradients divide by 64.
Linear and convolution parameter gradients accumulate into the supplied buffer;
the model does not divide by batch size or reset that buffer. The runtime zeros
one buffer per worker per minibatch and handles batch averaging.

## Loss and elementary functions

Softmax subtracts the largest logit before exponentiation. Cross-entropy uses
`log(sum(exp(logits-max))) - (true_logit-max)`, so loss remains finite even
when the returned true-class probability underflows to zero. `cnn_loss` returns
unaveraged `softmax - one_hot` gradients; the runtime performs batch averaging.

`cnn_exp` reduces its float32 argument in float64 using the nearest integer
multiple of ln(2), evaluates a degree-12 Taylor polynomial on the reduced
interval, and constructs an exact binary scaling factor. It returns float32;
large arguments overflow to infinity and sufficiently negative arguments to zero.
NaN inputs propagate as NaN.
`cnn_log` exactly converts the float32 input to float64, splits exponent and
mantissa, and evaluates the atanh series through the 23rd power on
`(mantissa-1)/(mantissa+1)`. The mantissa lies in [1,2); the series argument is
therefore in [0,1/3). Float32 subnormals normalize automatically during the exact
conversion to double. Positive infinity, zero and negative input are handled as
infinity, negative infinity and NaN respectively. These routines support the
finite domains required by the training algorithm; the executable must reject
nonfinite model values rather than treat them as valid probabilities.

## Adam operation order

Step numbers start at one. The optimizer has learning rate 0.001, beta1 0.9,
beta2 0.999, epsilon 1e-8, and no weight decay. Integer exponentiation by squaring
in float64 computes the two bias corrections, reflecting the double-precision
scalar corrections used by PyTorch. Their derived step size and denominator
scale are rounded to float32 before applying tensor operations.

For each parameter the initial implementation performs:

1. `m = fma(gradient - m, float32(0.1), m)`.
2. `v = v * float32(0.999)` then
   `v = fma(gradient * gradient, float32(0.001), v)`.
3. `denominator = sqrt(v) / float32(sqrt(1 - 0.999**step)) + float32(1e-8)`.
4. `parameter += (m * float32(-0.001 / (1 - 0.9**step))) / denominator`.

The main loop uses four float32 SIMD lanes; the final two parameters use matching
scalar operations. Fused arithmetic and kernel reduction order are explicitly
subject to the independent PyTorch parity gates, rather than assumed to be
bitwise identical. There is no optimizer state hidden in the assembly function.
