# Neural Network Framework

A small C++ framework for building and experimenting with neural networks.
The repository is mainly intended for learning, experimentation, and documenting
the development process through practical examples.

## Repository Layout

- `include/`: public headers.
- `src/`: framework implementation.
- `tests/`: automated Catch2 tests for tensors, operations, losses, modules, and optimizers.
- `examples/`: space for standalone examples.
- `python_scripts/`: optional Python scripts used to compare or inspect experiments.
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

## Optional Python Comparison

The script in `python_scripts/test_xor.py` loads the generated weights into a
small PyTorch model and writes the Python gradients to
`gradients/gradients_python.json`.

Install the optional dependencies in your Python environment:

```bash
python -m pip install torch pandas
```

Then run:

```bash
python python_scripts/test_xor.py
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

- Article 1: _coming soon_
- Article 2: _coming soon_

Links can be added here as the articles are published.
