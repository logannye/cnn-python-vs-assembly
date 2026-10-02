# Handwritten convolution kernels

`convolution.S` implements float32 3×3, stride-one, zero-padded convolution and
its exact analytic derivatives for the two network shapes in `ABI.md`. Tensor
layout and weight layout match PyTorch: CHW and OIHW, respectively.

The forward kernel tiles four output channels by sixteen adjacent pixels. It
accumulates with NEON `fmla`, then adds the bias. An internal, zero-padded copy
of the input makes edge handling identical to interior handling. The public
forward routine supports the fixed model widths 16 and 32 and output channel
counts divisible by four. The two model layers meet those requirements.

Backward sums each output plane for the bias derivative and uses the same
four-output-channel/sixteen-pixel tile for the input–output-gradient dot
products. The weight and bias derivatives are added to the existing arrays.
The input derivative, when requested, is overwritten with the convolution of
the output derivative and the spatially reversed, channel-transposed weights.
An internal padded output channel allows the optional three-channel RGB input
derivative to use the same NEON forward kernel. A NULL input-gradient pointer
skips that work.

All scratch memory belongs to the current call and is released before return;
there is no mutable global state. The only external calls are `malloc`,
`memset`, `memcpy`, `free`, and fatal allocation-failure handling through
`abort`. Every tensor operation is handwritten assembly. The Darwin ABI is
observed, including preserving x19–x28 and d8–d15 and never using reserved x18.

The real-number convolution and derivatives are identical to the reference.
Float32 fused operations and spatial reduction order differ from the PyTorch
CPU kernels, so agreement is tested numerically rather than claimed bitwise.

## Focused validation

A temporary dylib was built with Apple clang and called through Python ctypes;
PyTorch was used only as the independent reference. Both complete layer shapes
were checked using random, constant, border-only, and single-impulse inputs,
random upstream gradients, and nonzero initial weight/bias gradients. All
forward, input-gradient, weight-gradient, and bias-gradient comparisons passed
`atol=1e-5, rtol=5e-5`. In every case the NULL-input-gradient path produced
bitwise-identical parameter gradients to the full backward path.

| Quantity | Maximum absolute difference across eight cases |
|---|---:|
| Forward output | 9.5367432e-7 |
| Input gradient | 4.1723251e-7 |
| Accumulated weight gradient | 3.9339066e-6 |
| Accumulated bias gradient | 1.6689301e-6 |

These are focused kernel checks. Full-model, optimizer, training, performance,
and reproducibility validation are recorded separately by the experiment.
