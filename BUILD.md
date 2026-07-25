# Building GigaLearnRL (Windows)

## NVIDIA CUDA (default high-SPS path)

Prefer a short path without spaces (e.g. `C:\GigaLearnRL`) - `nvcc` can fail on spaces/apostrophes. See [`docs/CUDA_SIM.md`](docs/CUDA_SIM.md).

**CUDA Toolkit:** use **12.8** (tested). Other CUDA 12.x builds usually work; install the toolkit and ensure `nvcc` is on PATH.

```bat
tools\build_cuda.bat
```

Same preset as before:

```bat
cmake --preset windows-cuda-gpu
cmake --build --preset windows-cuda-gpu
```

Output: `build\Release\GigaLearnBot.exe`

## AMD HIP

```bat
tools\build_amd.bat
```

See [`docs/AMD.md`](docs/AMD.md).

## Fix: `cmake not in PATH`

Build helpers look for `cmake` on **PATH**, then common install folders (Kitware, VS-bundled CMake). If nothing is found you get install steps and can still point at a full path:

```bat
set CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe
tools\build_cuda.bat
```

**Install options**

1. **Visual Studio 2022** - workload *Desktop development with C++*, plus individual component **CMake tools for Windows**.
2. **Kitware** - [cmake.org/download](https://cmake.org/download/) Windows x64 installer - enable **Add CMake to the system PATH**.

Then open a **new** terminal and check:

```bat
where cmake
cmake --version
```

Need CMake **3.21+** (`CMakePresets.json`).

Shared resolver: `tools\find_cmake.bat` (used by `build_cuda.bat` / `build_amd.bat`).

## After a successful build

```bat
SETUP_FIRST_RUN.bat
run_fresh_train.bat
```

Or: `cd build\Release` then `GigaLearnBot.exe --from-scratch`.
