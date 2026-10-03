# Reproduce the matched experiment

Use an Apple silicon Mac with Xcode Command Line Tools for the assembly runtime.
The reference machine was an M4 Max with 16 CPU cores and 128 GiB RAM, running
macOS 26.2; the experiment used CPU float32 and two compute threads. Different
hardware, package versions, compiler versions and background activity may change
both timings and float32 trajectories.

The measured source revision is `9978cb0929d64b16a97b0d70dd6f83b93d78e075`.
Later commits improve report validation, presentation and reproduction guidance;
they do not change the measured training source. `plan.json` retains every
measured source hash and binary hash. The release also contains those source files.

## Environment

After cloning the repository, create a Python 3.12 virtual environment and
install the recorded versions. The reference interpreter was CPython 3.12.13.
These are the versions actually measured, not a claim that newer versions give
identical results.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,benchmark]'
python -m pip install -r comparison/requirements-reference.txt
export PYTHONPATH=src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
unset CNN_ASM_EVAL_LIMIT
```

## Download the retained inputs and results

Using the GitHub CLI, download the two archives and their checksum file:

```bash
mkdir -p results/downloads
gh release download comparison-cifar10-matched-cpu-25-v1 \
  --repo logannye/cnn-python-vs-assembly --dir results/downloads
(cd results/downloads && shasum -a 256 -c SHA256SUMS.txt)
tar -xzf results/downloads/cifar10-matched-inputs-v1.tar.gz -C results
tar -xzf results/downloads/cifar10-matched-cpu-25-v1.tar.gz -C results
```

The input archive expands to `results/assembly-inputs-v1/`. It contains CIFAR-10
raw pixels and labels, exact split IDs, initial float32 weights, and every
image-index/flip pair for all 25 epochs and all three seeds. Common files use
hard links to avoid duplicating raw pixels. Each seed manifest records SHA256
values; the runner verifies them before and after the experiment.

The results archive expands to `results/cifar10-matched-cpu-25-v1/`, including
`raw/`, `source/`, the native executable, `plan.json`, gate evidence, audit and
report. No dataset download or regenerated RNG stream is needed for replay.
To reconstruct inputs from the original CIFAR-10 distribution instead, see
[`assembly/prepare.py`](../assembly/prepare.py) and the historical Mac baseline
release, which retains its `manifest.json` and `split_indices.npz`.

## Build and validate

The compiler recorded for the reference executable was Apple clang 17.0.0
(clang-1700.4.4.1). Rebuilding can change the executable hash, including its build
UUID. Run the gates against your own freshly built executable.

Use fresh output directories: all training tools intentionally reject nonempty
run destinations. The following commands assume the named gate outputs do not
already exist.

```bash
make -C assembly
python assembly/validate.py --output results/matched-parity.json
python assembly/smoke.py \
  --inputs results/assembly-smoke-inputs \
  --output results/matched-asm-smoke
python assembly/test_runtime_safety.py \
  --work-dir results/matched-safety-fixtures \
  --output results/matched-safety.json
python comparison/validate.py
ruff check .
ruff format --check .
pytest -q
```

These check the mathematical kernels, a two-epoch standalone partial-batch run,
corrupt-input handling, repeatability, the matched Python loop, and bitwise
normalization. The Python gate also uses an independently written per-image
PyTorch reference for its checkpoints and predictions. A failure should be
investigated before any performance measurement.

## Run all 12 fits

Connect AC power and keep other substantial local work idle. Source must be
committed and the Git working tree clean; the runner records its current revision.
No test-set result is used to tune the architecture or hyperparameters.

```bash
caffeinate -i python comparison/run.py --output results/my-matched-run
```

This launches three seeds × two repetitions × two implementations sequentially,
reversing implementation order within each seed. It records a 15-second pause
between children, power/thermal/load snapshots, parent process wall time and
source/input fingerprints. Raw training output goes into `raw/<run-id>/` and
individual console logs. If a child or provenance gate fails, it stops and
retains the evidence; it does not silently restart or omit that run.

The runner itself uses the standard library. Do not run audits, chart generation,
compilation or other intensive tasks concurrently with measured training.
The model/optimizer are deterministic within an implementation on the measured
platform; all repeated checkpoints and predictions are checked again afterward.

## Audit and regenerate charts

For your new experiment:

```bash
python comparison/report.py --results results/my-matched-run \
  --inputs results/assembly-inputs-v1 --output results/my-matched-report
```

To regenerate the reference report from downloaded evidence:

```bash
python comparison/report.py --results results/cifar10-matched-cpu-25-v1 \
  --inputs results/assembly-inputs-v1 --output results/regenerated-report
```

For a second calculation of the main results, without importing the classifier's
metric implementation:

```bash
python comparison/audit.py --results results/my-matched-run \
  --inputs results/assembly-inputs-v1 --report results/my-matched-report \
  --output results/my-independent-audit.json
```

The collector validates every run, all expected sizes/counts, selection rules,
finite normalized probability arrays, source/input fingerprints, repeated fits,
and selected-checkpoint inference against PyTorch. It retains per-run metrics,
paired comparisons, raw timing summaries and machine-readable aggregation rules.
It writes five PNG/SVG figures from the measurements, without retraining.
File hashes of newly rendered figures can differ across Matplotlib versions or
SVG metadata even when numerical metrics agree. Compare numerical JSON/CSV
values and retained inputs, not just image bytes.

## What is reproducible?

Exact raw inputs, initial states and random choices are retained. The recorded
source and binary make the original computation inspectable. Same-platform
numerical repeatability is tested; cross-implementation arithmetic uses tolerance
checks. Runtime measurements remain observations under recorded machine
conditions, not universal constants. Three seeds cannot establish statistical
quality equivalence or performance on other architectures.
