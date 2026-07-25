# AMD path (Windows)

GPU sim backends:

| Backend | GPU | Preset | Toolkit |
|---|---|---|---|
| CUDA | NVIDIA | `windows-cuda-gpu` | CUDA 12.x + nvcc |
| HIP | AMD Radeon | `windows-hip` | HIP SDK (Windows) |

Sources under `RocketSimCuda/`. `GIGA_USE_CUDA_SIM` means GPU sim on both backends.

---

## Reference PC (RX 6600 XT)

| Part | Spec |
|---|---|
| OS | Windows 11 |
| Board | A520-M |
| CPU | Ryzen 5 3600 (6c/12t) |
| GPU | RX 6600 XT 8 GB (`gfx1030`) |

| Piece | Device | How |
|---|---|---|
| Env / physics | RX 6600 XT via HIP | `tools\build_amd.bat` |
| PPO (C++ libtorch) | CPU, multi-threaded | `GIGA_TORCH_THREADS` (~10 on 12t) |
| DirectML | Not in C++ Learner | Python only |

HIP runs env on the Radeon; PPO stays on CPU with stock Windows libtorch. Target Overall >= 20000 SPS (`amd_win_20k`): arenas 2048, steps 3, lean512. Without HIP gpuNative, 20k is not guaranteed.

Exe auto-detects NVIDIA vs AMD (`hw_profile.json` -> CUDA POWER or `amd_win_20k`). Opt out: `GIGA_NO_HW_PROFILE=1` / `GIGA_SKIP_AUTO_HW_PROBE=1` / `GIGA_LOCK_HW_PROFILE=1`.

DirectML cannot accelerate `GigaLearnBot.exe` ([DirectML #247](https://github.com/microsoft/DirectML/issues/247)).

### Defaults (`amd_win_20k`)

| Topic | Value |
|---|---|
| Build | `tools\build_amd.bat` (CUDA prebuilts ignore Radeon) |
| Overall target | >= 20000 SPS |
| PPO | CPU libtorch + threads |
| DirectML | Optional Python (`requirements-amd-windows.txt`) |
| Arenas | 2048; 1536/1024 if Overall <20k; avoid 8192 on 8 GB |
| Bank | steps=3 -> ts≈12288 |
| WSL2 | Optional only; see end of doc |

### If Overall < 20000

1. AMD-only PC with CUDA DLLs or no HIP: `tools\build_amd.bat`. Dual-GPU NVIDIA: stay on CUDA.
2. Unset `GIGA_FORCE_CPU=1`.
3. Auto-lean may write `amd_win_20k_next.env` -> restart `run_fresh_train.bat`.
4. Manual: `set GIGA_ENV_ARENAS=1536`.
5. Without HIP, 20k not guaranteed.

### Install

1. Windows 11 + AMD Adrenalin.
2. [HIP SDK for Windows](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html); `hipcc` on PATH.
3. `AMDGPU_TARGETS=gfx1030` for 6600 XT.
4. Prefer a path without spaces.
5. `tools\build_amd.bat` (or cmake preset `windows-hip`).
6. `SETUP_FIRST_RUN.bat`. FAIL if AMD and no NVIDIA and (no hipcc or CUDA-only Release). Dual-GPU AMD+NVIDIA: CUDA OK.
7. Console: `HIP gpuNative=ON`, CPU device, torch threads.
8. Overall >= 20000. If low: steps above / `INIZIO_RAPIDO_IT.md`.

---

## DirectML

| Question | Answer |
|---|---|
| C++ Learner use DirectML? | No |
| `GIGA_TORCH_DIRECTML=1`? | Logged, then CPU |
| torch-directml for? | Optional Python tools |

```bat
python -m venv .venv-dml
.venv-dml\Scripts\activate
pip install -r requirements-amd-windows.txt
python -c "import torch, torch_directml; print(torch_directml.device())"
```

---

## Windows + HIP SDK

1. HIP SDK + Adrenalin.
2. `hipcc` on PATH.
3. CPU libtorch under `GigaLearnCPP\libtorch` (ROCm libtorch on Win is uncommon).
4. Build:

```bat
tools\build_amd.bat
```

5. `python tools\hw_probe.py` -> `amd_win_20k`, `backend=hip`, arenas 2048, steps 3 on 3600+6600 XT.
6. `GIGA_HIP_FORCE=1` if needed.

Target: Overall >= 20000 SPS. Auto-downgrade via `amd_win_20k_next.env` if low.

### CPU PPO knobs

| Env | Meaning |
|---|---|
| `GIGA_TORCH_THREADS` | libtorch intra-op |
| `GIGA_TORCH_INTEROP_THREADS` | inter-op |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | set by probe / bat |

---

## CMake

```text
-DGIGA_USE_CUDA_SIM=ON
-DGIGA_GPU_BACKEND=hip|cuda
-DAMDGPU_TARGETS=gfx1030
```

## Probe

```bat
python tools\hw_probe.py
python tools\hw_probe.py --simulate-amd
```

| Detected | Arenas / profile |
|---|---|
| AMD HIP ~8 GB | 2048 / `amd_win_20k` |
| AMD HIP >=12 GB | 2560-3072 (still CPU PPO) |
| AMD, no HIP | CPU RocketSim; 20k not guaranteed |

## Rebuild?

| Change | Rebuild? |
|---|---|
| First HIP / CUDA<->HIP | Yes |
| Docs / bats / probe | No |
| Rewards / CudaEnvSet | Yes |

---

## Optional: WSL2 / ROCm

Linux ROCm can put env + PPO on AMD GPU. Not required for Win HIP. Details: [`AMD_WSL2.md`](AMD_WSL2.md).

See: [`README.md`](../README.md), `docs/CUDA_SIM.md`, `docs/QUICKSTART.md`, `SHARE.md`.
