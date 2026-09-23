# Neural Network Framework

A small C++ framework for building and experimenting with neural networks.
The repository is mainly intended for learning, experimentation, and documenting
the development process through practical examples.

## Repository Layout

- `include/`: public headers.
- `src/`: framework implementation.
- `tests/`: automated Catch2 tests for tensors, operations, losses, modules, and optimizers.
- `examples/cpp/`: standalone C++ examples such as XOR and MNIST experiments.
- `examples/python/`: Python counterparts used to compare or inspect experiments.
- `examples/data/mnist/`: local directory for the MNIST binary dataset file (ignored by Git).
- `docs/`: additional project documentation.
- `weights/`: generated model weights. This directory is ignored by Git.
- `gradients/`: generated gradient files. This directory is ignored by Git.
- `build/`: local CMake build directory. It is not required in a fresh checkout.

## Requirements

- C++20 compiler
- CMake 3.15 or newer
- Git

The CMake configuration downloads Catch2 and Google Benchmark through CMake's
`FetchContent`, so the first configuration requires an internet connection.

## Examples

The executable examples live under `examples/cpp/` and the comparison scripts
under `examples/python/`:

```bash
./build/xor
./build/mnist
```

On Windows:

```powershell
.\build\Release\xor.exe
.\build\Release\mnist.exe
```

The MNIST example expects the binary dataset to be placed under
`examples/data/mnist/`. The files are intentionally ignored by Git to keep the
repository lightweight. The dataset was sourced from
https://www.kaggle.com/datasets/hojjatk/mnist-dataset.

The folder itself remains tracked via the `.gitkeep` placeholder, so the path is
present in the repo without committing the downloaded binary files.

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build
```

For a Release build on Windows with Visual Studio:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

To build with OpenBLAS using an existing local installation:

```powershell
cmake -B build -DNN_USE_OPENBLAS=ON -DNN_FETCH_OPENBLAS=OFF -DOPENBLAS_ROOT="C:/libs/openblas"
cmake --build build
```

The OpenBLAS installation must contain `include/cblas.h` and the corresponding
library under `lib/`. The library architecture must match the selected Visual
Studio architecture: use an x64 OpenBLAS library for an x64 build.

## Run the Tests

The test programs are built as separate executables. On Linux or macOS:

```bash
./build/TensorTest
./build/OperationTest
./build/LossTest
./build/ModuleTest
./build/OptimizerTest
```

On Windows:

```powershell
.\build\Release\TensorTest.exe
.\build\Release\OperationTest.exe
.\build\Release\LossTest.exe
.\build\Release\ModuleTest.exe
.\build\Release\OptimizerTest.exe
```

These are the tests executed by the GitHub Actions workflow in
`.github/workflows/ci.yml` on Linux and Windows.

## XOR Experiment

`XORTest` is a standalone training experiment rather than part of the CI test
suite. It is useful for manually checking the complete training flow and for
creating the files used by the optional Python script.

After building, run it manually:

```bash
./build/XORTest
```

On Windows:

```powershell
.\build\Release\XORTest.exe
```

The experiment writes the initial weights to `weights/xor_weights.csv` and the
resulting C++ gradients to `gradients/gradients_cpp.json`.

## Profiling

On Linux, profile the complete XOR training flow before optimizing it. The
script builds `XORTest` in Release mode when requested, runs `perf stat`, and
then writes a sampled hotspot report:

```bash
BUILD_DIR=build-linux bash scripts/profile.sh perf --build
```

When using WSL from a Windows checkout, install the Linux toolchain inside WSL
and use a separate build directory from the existing Visual Studio build:

```bash
sudo apt update
sudo apt install cmake g++ linux-tools-generic
BUILD_DIR=build-linux bash scripts/profile.sh perf --build
```

Reports are saved under `profiling/`. Increase the statistical sample size
with `PROFILE_RUNS=10 bash scripts/profile.sh perf`. If `perf` is unavailable,
use Callgrind instead:

```bash
sudo apt install valgrind
BUILD_DIR=build-linux bash scripts/profile.sh callgrind --build
```

`XORTest` is intentionally used because it exercises forward, loss,
backpropagation, and optimizer code. Its workload is small, so for more stable
hotspot percentages repeat the profiling command or replace the executable
with a longer experiment after the first measurement.

## MNIST Benchmark: Naive vs OpenBLAS

`benchmarc_mnist` compares the framework's linear and convolutional MNIST
models. Select a model with `BENCH_MODEL=linear`, `BENCH_MODEL=conv`, or
`BENCH_MODEL=all` (the default). The benchmark prints the model's training and
test metrics and elapsed model time. Batch loading happens before the timer
starts, so these elapsed times measure model execution rather than dataset
loading.

For a fair comparison, build both variants with the same compiler, build type,
model, and batch settings. On Linux or WSL, configure separate Release builds:

```bash
cmake -S . -B build-wsl-naive \
  -DCMAKE_BUILD_TYPE=Release \
  -DNN_USE_OPENBLAS=OFF
cmake --build build-wsl-naive -j"$(nproc)"

cmake -S . -B build-wsl-openblas \
  -DCMAKE_BUILD_TYPE=Release \
  -DNN_USE_OPENBLAS=ON
cmake --build build-wsl-openblas -j"$(nproc)"
```

OpenBLAS is fetched by CMake by default when `NN_USE_OPENBLAS=ON`. The first
configuration therefore needs network access. Run each version several times
and compare the median elapsed time; the first run may include one-time system
effects. For example, to time the convolutional model five times per build:

```bash
for build in build-wsl-naive build-wsl-openblas; do
  echo "=== $build ==="
  for run in 1 2 3 4 5; do
    echo "Run $run"
    BENCH_MODEL=conv OPENBLAS_NUM_THREADS=1 \
      "./$build/benchmarc_mnist"
  done
done
```

Use `BENCH_MODEL=linear` to compare the linear model, or `BENCH_MODEL=all` to
run both. By default, the benchmark uses all batches. To shorten a run, set
`BENCH_MAX_BATCHES`, but use the same value for both builds and record it with
the results. OpenBLAS thread count should also be held constant; use
`OPENBLAS_NUM_THREADS=1` for a single-thread comparison.

### Callgrind instruction percentages (OpenBLAS only)

Run the normal Release timing comparison above first. Those wall-clock times
are the naive-versus-OpenBLAS comparison. Callgrind is a separate profiling
run: it instruments execution and makes it much slower, so do not use its
elapsed time in the comparison. In this workflow, collect Callgrind
percentages only for OpenBLAS.

Build the OpenBLAS variant with debug information, then install Valgrind if
needed:

```bash
cmake -S . -B build-wsl-openblas-profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DNN_USE_OPENBLAS=ON
cmake --build build-wsl-openblas-profile -j"$(nproc)"

sudo apt update
sudo apt install valgrind
```

Profile the same model used for the timing comparison (this example uses the
convolutional model):

```bash
BENCH_MODEL=conv \
BUILD_DIR=build-wsl-openblas-profile \
CALLGRIND_OUT_DIR=callgrind-results-openblas \
bash scripts/profile_mnist_callgrind.sh
```

The script sets `BENCH_MAX_BATCHES=2` and uses one OpenBLAS thread by default
to keep profiling manageable; the batch limit applies only if the benchmark
reads that environment variable. It saves `report-self.txt` and
`report-inclusive.txt` under `callgrind-results-openblas/`. Self percentages
count work attributed directly to a function. Inclusive percentages include
that function and its callees, so nested inclusive percentages must not be
added together. Record the model and profiling batch limit with the top
percentages, separately from the naive and OpenBLAS Release timing results.
These percentages describe the instruction profile, not elapsed-time shares.

### Working from a Windows checkout in WSL

For longer builds and runs, a copy in WSL's Linux filesystem (for example
`~/Neural_Network_framework`) is usually preferable to building under
`/mnt/c`. From the WSL shell, copy the repository while excluding generated
build and profiling output:

```bash
sudo apt update
sudo apt install rsync cmake g++ make git
mkdir -p "$HOME/Neural_Network_framework"
rsync -a \
  --exclude='/build*/' \
  --exclude='/callgrind-results*/' \
  "/mnt/c/Users/Samuele Ragazzo/Desktop/Neural_Network_framework/" \
  "$HOME/Neural_Network_framework/"
cd "$HOME/Neural_Network_framework"
git status --short --branch
```

This copies the current working tree, including its `.git` directory and
uncommitted tracked changes, while leaving the Windows original in place.
Ignored files such as the local MNIST dataset are copied too unless excluded.
Build in this Linux-side copy using the commands above. Since the two copies
are independent, commits made in one are not automatically present in the
other; push/pull or copy changes deliberately when you want to synchronize.

## Optional Python Comparison

The script in `examples/python/xor.py` loads the generated weights into a
small PyTorch model and writes the Python gradients to
`gradients/gradients_python.json`.

Install the optional dependencies in your Python environment:

```bash
python -m pip install torch pandas
```

Then run:

```bash
python examples/python/xor.py
```

Run `XORTest` first so that `weights/xor_weights.csv` exists.

## Continuous Integration

GitHub Actions builds the project and runs the five automated test executables
on Ubuntu and Windows. The XOR experiment is intentionally excluded because it
is a manual experiment that produces local output files rather than a focused,
deterministic unit-test result.

## Articles

The technical background and implementation decisions will be described in
separate Medium articles. This README is intentionally focused on using the
repository rather than explaining the underlying mathematics or architecture.

- Article 1: https://medium.com/@ragazzosamuele7/from-zero-to-backprop-building-a-neural-network-framework-in-c-150677a84112?sharedUserId=ragazzosamuele7
- Article 2: _coming soon_
