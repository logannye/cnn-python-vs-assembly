# Handwritten ARM64 CNN

This directory reproduces the repository's CIFAR-10 `SimpleCNN` in Apple ARM64
assembly: two convolution/ReLU/max-pool blocks, global averaging, a linear head,
stable cross-entropy, backpropagation, Adam, and a standalone two-worker training
runtime. The architecture has the same 5,418 float32 parameters.

| File | Role |
|---|---|
| `convolution.S` | NEON convolution, weight/bias gradients and input gradients |
| `model.S` | Remaining layers, complete forward/backward, loss, exp/log and Adam |
| `runtime.S` | Input validation, transforms, batching, threads, training, checkpoint selection, evaluation and timing |
| `prepare.py` | Export the exact reference initialization, data, order and flips |
| `validate.py`, `validate_runtime.py` | Independent numerical checks against PyTorch |
| `smoke.py`, `test_runtime_safety.py` | Bounded executable parity, repeatability and failure checks |
| `run_experiment.py` | Run three separate native processes with frozen provenance |
| `collect.py` | Audit saved probabilities, compute metrics and compare with the retained baseline |

The native program contains handwritten assembly plus macOS system services for
files, memory, threads and clocks. Python prepares inputs and audits outputs; it
does not execute the native program's CNN or optimizer. No BLAS/Accelerate/Torch
or compiled C numerical implementation is linked into the executable.

Read the [experiment protocol](../experiments/ASSEMBLY_PROTOCOL.md),
[ABI](ABI.md), [convolution notes](CONVOLUTION.md), [model notes](MODEL.md)
and [runtime notes](RUNTIME.md) for the exact boundary and timing scopes.
The replica is mathematically equivalent within tested numerical tolerances;
PyTorch and assembly reductions are not guaranteed bitwise identical.

## Build and verify on an Apple Silicon Mac

Use the repository's existing Python environment with benchmark dependencies
for fixture preparation and independent audits. Install Apple's Command Line
Tools if `xcrun clang` is unavailable. Run these commands from the repository root:

```bash
make -C assembly
PYTHONPATH=src .venv/bin/python assembly/prepare.py
PYTHONPATH=src .venv/bin/python assembly/validate.py
PYTHONPATH=src .venv/bin/python assembly/smoke.py
PYTHONPATH=src .venv/bin/python assembly/test_runtime_safety.py
```

Preparation reuses `data/cifar10` and the retained local baseline at
`../cifar10-mac-cpu-25-v1`; pass `--data-dir`, `--baseline` and `--output` to change
those locations. The baseline archive is available in the private repository's
`baseline-cifar10-mac-cpu-25-v1` release. Use the same pinned dependencies recorded
there to reproduce its initialization and random schedule. Shared binary image
files are hardlinked across fixtures to avoid duplicate dataset storage.
Commands refuse to overwrite existing experiment/evidence directories; choose
fresh output paths for another experiment. The full input fixture is about 202 MiB.

After committing the tested source, unset `CNN_ASM_EVAL_LIMIT` and run:

```bash
PYTHONPATH=src /usr/bin/caffeinate -i .venv/bin/python -u assembly/run_experiment.py
PYTHONPATH=src .venv/bin/python assembly/collect.py
```

The first command runs seeds 42, 43 and 44 sequentially for 25 epochs each. It
requires passing parity and safety evidence and a clean source tree. The second
computes reporting metrics from the assembly probabilities and verifies selected
checkpoint predictions independently; it performs no training.

For a direct native invocation with already prepared inputs:

```bash
assembly/build/cnn-assembly results/assembly-inputs-v1/seed-42 results/my-run 25 45000
```

Output files and optional bounded smoke settings are specified in the ABI and
runtime notes. Existing Python tools still use the original PyTorch model.
