# Troubleshooting

## Setup / Python

| Symptom | Fix |
|---------|-----|
| Python not found | Install 3.10+ with PATH; new terminal; re-run `SETUP_FIRST_RUN.bat` |
| pip fails in SETUP | `python -m pip install -U pip` then `autotrainer\requirements.txt` |
| Traceback in `autotrainer_launch.log` | Read log; use bat launchers; reinstall requirements |

## Build / CMake

| Symptom | Fix |
|---------|-----|
| `cmake not in PATH` | Install VS CMake tools or Kitware CMake + PATH; or `set CMAKE_EXE=...` then `tools\build_cuda.bat` / `tools\build_amd.bat` — [`BUILD.md`](../BUILD.md) |
| nvcc path / spaces | Build from a short path (e.g. `C:\GigaLearnRL`) — [`CUDA_SIM.md`](CUDA_SIM.md) |
| CUDA / nvcc missing | Install **CUDA Toolkit 12.8** (12.x OK), reopen terminal, `where nvcc` - [`BUILD.md`](../BUILD.md) |

## AutoTrainer

| Symptom | Fix |
|---------|-----|
| Blank AT window | Start via `run_with_autotrainer.bat` / `start_autotrainer.bat` |
| `UnicodeEncodeError` | Use bats (UTF-8); avoid raw `orchestrator.py` on cp1252 |
| `GIGA_AUTOTRAINER_EXTERNAL=1` but no brain | Unset it or start AT |

AutoTrainer is **off** by default (`run_fresh_train.bat`). Enable: `run_with_autotrainer.bat` or `GIGA_AUTOTRAINER=1`.

## Hang / freeze

| Symptom | Fix |
|---------|-----|
| Hang at OpponentPool | Cwd = `build\Release`; valid `opponents.json` (`entries: []` OK) |
| Hang after N iters | One `GigaLearnBot.exe`; check VRAM; lower arenas |
| Freeze after reward edits | Rebuild exe + `GigaLearnCPP.dll` |
| Collect/Learn stuck | `set GIGA_ASYNC_OVERLAP=0` |
| Slow first iter | Warmup; later iters show real SPS |

## GPU / AMD / CUDA / HIP

| Path | Env | PPO |
|------|-----|-----|
| NVIDIA CUDA | GPU | GPU |
| Win HIP | AMD GPU | CPU (threaded) |
| No GPU build | CPU | CPU |

| Symptom | Fix |
|---------|-----|
| AMD + CUDA zip | `tools\build_amd.bat` — [`AMD.md`](AMD.md) |
| OOM on 8 GB | `GIGA_ENV_ARENAS=2048` (not 8192) |
| Overall &lt;20k on AMD | Confirm HIP; lower arenas; [`AMD.md`](AMD.md) / [`INIZIO_RAPIDO_IT.md`](../INIZIO_RAPIDO_IT.md) |
| DirectML but CPU Learn | Expected — DirectML is not used by C++ PPO ([`DIRECTML.md`](DIRECTML.md)) |

## Binaries / meshes / checkpoints

| Symptom | Fix |
|---------|-----|
| Exe missing | Copy Release or rebuild; SETUP |
| Crash / tiny DLL | SETUP + `tools\heal_release_dll.bat` |
| Missing meshes | `collision_meshes\soccar`; SETUP links next to exe |
| Write test failed | Move tree off locked sync folders |
| Wrong bot resumed | From-scratch uses `build\Release\checkpoints\` only |
| Want blank POWER train | `run_fresh_train.bat` or `GigaLearnBot.exe --from-scratch` (cwd = Release) |

## Useful env vars

| Variable | Effect |
|----------|--------|
| `GIGA_ENV_ARENAS=N` | Override arenas |
| `GIGA_FORCE_CPU=1` | CPU RocketSim |
| `GIGA_NO_AUTOTRAINER=1` | Force AT off |
| `GIGA_AUTOTRAINER=1` | Opt-in AT |
| `GIGA_TORCH_THREADS=N` | CPU PPO threads (AMD) |
| `GIGA_SKIP_AUTO_HW_PROBE=1` | Skip GPU auto-detect |

Also: [`CUSTOMIZE.md`](CUSTOMIZE.md), [`AMD.md`](AMD.md), [`../SHARE.md`](../SHARE.md), [`AUTO_TRAINER.md`](AUTO_TRAINER.md).
