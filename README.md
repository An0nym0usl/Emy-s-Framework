# GigaLearnRL

Version **1.0** - [MIT](LICENSE)

C++ framework for training Rocket League bots with **discrete PPO** (RocketSim).  
High-SPS path on **NVIDIA CUDA**; **AMD HIP** on Windows (env on GPU, PPO on CPU).

Default train is a **blank template**: stub rewards, empty opponents, AutoTrainer **off**. You add your own stack.

Italian quick start: [`INIZIO_RAPIDO_IT.md`](INIZIO_RAPIDO_IT.md)

---

## Quick start

1. Unzip (prefer a local folder, not cloud sync).
2. Install **Python 3.10+** (Add to PATH) and GPU drivers.
3. Run `SETUP_FIRST_RUN.bat`.
4. Run `run_fresh_train.bat`.

The exe auto-detects **NVIDIA** (CUDA) or **AMD** (`amd_win_20k`).  
Checkpoints: `build/Release/checkpoints/`.

```bat
SETUP_FIRST_RUN.bat
run_fresh_train.bat
```

| Goal | Command |
|------|---------|
| Train only | `run_fresh_train.bat` |
| Train + AutoTrainer | `run_with_autotrainer.bat` |
| NVIDIA rebuild (CUDA) | `tools\build_cuda.bat` - see [`BUILD.md`](BUILD.md) |
| AMD rebuild (Radeon) | `tools\build_amd.bat` - see [`docs/AMD.md`](docs/AMD.md) |

---

## What you get

| Included | Not included |
|----------|--------------|
| Discrete PPO (`GigaLearnBot.exe`) | Finished reward recipe |
| CUDA / HIP GPU sim | Pre-trained weights |
| Stub rewards (Goal, Touch, VelBallToGoal, VelPlayerToBall) | Continuous / attention (not wired) |
| Optional AutoTrainer | wandb / API secrets |
| RLBot deploy helpers | |

Customize rewards, arenas, opponents: [`docs/CUSTOMIZE.md`](docs/CUSTOMIZE.md).

---

## Requirements

| | Recommended |
|--|-------------|
| OS | Windows 10/11 x64 |
| GPU | NVIDIA + **CUDA Toolkit 12.8** (12.x works; 12.8 tested). AMD: HIP SDK + `tools\build_amd.bat` |
| RAM | 16 GB+ (32 GB better) |
| Disk | SSD; room for checkpoints |
| Runtime | `collision_meshes/soccar/` next to the exe (SETUP links it) |
| Build | CMake 3.21+, MSVC; see [`BUILD.md`](BUILD.md) |

Prebuilt zips are usually **CUDA**. On AMD-only PCs, rebuild HIP once.

---

## Fresh train (from scratch)

Default blank POWER train into `build/Release/checkpoints/`:

```bat
SETUP_FIRST_RUN.bat
run_fresh_train.bat
```

Same path as running the exe directly:

```bat
cd build\Release
GigaLearnBot.exe --from-scratch
```

Plain `GigaLearnBot.exe` (no flags) uses the same from-scratch POWER defaults.  
Rebuild NVIDIA first if needed: `tools\build_cuda.bat` ([`BUILD.md`](BUILD.md)).

CUDA kickoff spawn orientation was aligned with RocketSim (orange mirror); see [`docs/CUDA_SIM.md`](docs/CUDA_SIM.md).

## Docs

| Doc | Topic |
|-----|--------|
| [`INIZIO_RAPIDO_IT.md`](INIZIO_RAPIDO_IT.md) | Italian first run |
| [`docs/CUSTOMIZE.md`](docs/CUSTOMIZE.md) | Rewards, size, opponents |
| [`docs/AMD.md`](docs/AMD.md) | AMD / HIP |
| [`docs/DIRECTML.md`](docs/DIRECTML.md) | DirectML limits (not used by C++ PPO) |
| [`SHARE.md`](SHARE.md) | What to put in a zip |
| [`docs/AUTO_TRAINER.md`](docs/AUTO_TRAINER.md) | AutoTrainer |
| [`BUILD.md`](BUILD.md) | Windows rebuild / `cmake not in PATH` |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Common errors |
| [`docs/CUDA_SIM.md`](docs/CUDA_SIM.md) | RocketSimCuda / spawn notes |
| [`docs/EXPERIMENTAL.md`](docs/EXPERIMENTAL.md) | pure80 / hyperpower / continue-leak |

---

## Layout

```
SETUP_FIRST_RUN.bat
run_fresh_train.bat          # AutoTrainer off (default POWER path)
run_with_autotrainer.bat
src/
  ExampleMain_mod.cpp        # slim product entry (GigaLearnBot)
  TrainProfiles.h            # TRAINING SIZE / CudaPower knobs
  TrainCli.*                 # flag parse + mode resolve
  TrainEnv.*                 # rewards, env create, curriculum
  TrainHw.*                  # hw_probe / amd_win_20k / AutoTrainer spawn
RocketSimCuda/src/
  RocketSimCuda.cu           # thin aggregator
  RocketSimCuda{Physics,Obs,Rewards,Api,…}.cuh
GigaLearnCPP/
autotrainer/                 # optional
collision_meshes/
opponents/                   # empty pool by default
build/Release/               # GigaLearnBot.exe + DLLs
docs/
  EXPERIMENTAL.md            # pure80 / hyperpower / continue-leak
```

**Default path (product):** blank from-scratch POWER train — `GigaLearnBot.exe` / `--from-scratch`.  
**Kept critical:** RocketSimCuda gpuNative, PPO loop, chase learning fixes (maxEp, VelP2B, no Air farming), optional Leak-style console.  
**Secondary / experimental:** `--pure80`, `--hyperpower`, `--continue-leak` — see [`docs/EXPERIMENTAL.md`](docs/EXPERIMENTAL.md).  
**Not in repo:** private reward packs (slater/mkh/…).

Build (NVIDIA):

```bat
tools\build_cuda.bat
```

Or manually: `cmake --preset windows-cuda-gpu` then `cmake --build --preset windows-cuda-gpu`.  
If you see **cmake not in PATH**, see [`BUILD.md`](BUILD.md).

AMD: `tools\build_amd.bat`
