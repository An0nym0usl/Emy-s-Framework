# SHARE - zip contents

Version **1.0**. Unzip -> `SETUP_FIRST_RUN.bat` -> `run_fresh_train.bat`.  
No machine-specific paths required. Stub rewards/opponents; see `docs/CUSTOMIZE.md`. Primary doc: `README.md`.

## Include (prebuilt)

| Include | Why |
|---|---|
| Source (`src`, `GigaLearnCPP` without libtorch if no rebuild, `autotrainer`, `tools`, `docs`, `opponents`, `rlbot`, `sdk`, `CMake*`, `pyproject.toml`, root `*.bat`) | Setup + optional AT |
| `build\Release\` (exe + Torch/CUDA or HIP DLLs) | Train without compiling |
| `collision_meshes\` (`soccar\`) | RocketSim |
| `opponents\` (empty `opponents.json` + README; no model folders) | Sparring later |
| `README.md`, `INIZIO_RAPIDO_IT.md`, `SHARE.md`, `LICENSE`, `docs\CUSTOMIZE.md`, `docs\AMD.md`, `docs\QUICKSTART.md`, root bats | Entrypoints |

### Exclude from `build\Release`

- `checkpoints\` / `checkpoints_*`
- `wandb\`
- AutoTrainer runtime JSON/logs (SETUP clears these)
- Smoke/logs: `smoke_*`, `*.log`, `diag_*`, `proof_*`

Keep: `GigaLearnBot.exe`, `GigaLearnCPP.dll`, torch/c10/cudnn/cuda/python/vcruntime DLLs, meshes, opponents JSON.

### Optional

- `GigaLearnCPP\libtorch\` only if rebuilding (large)
- Community models under `opponents/` - not shipped

### Without AutoTrainer

Exclude `autotrainer\`, `start_autotrainer.bat`, `run_with_autotrainer.bat`, `tools\launch_autotrainer.bat`. Default train already has AT off:

```bat
run_fresh_train.bat
```

## Always exclude

- `.env` / API keys / wandb tokens
- `.git\`
- `__pycache__\`, `*.pyc`, egg-info
- Personal checkpoint boxes
- Owner smoke/sync scripts
- `RocketSimVis-main\`, `docs/archive\`
- `opponents\nexto\`, `opponents\necto\`
- Dashboard / visualizer companions

## Source-only

1. MSVC + CMake + libtorch
2. SETUP reports missing binaries
3. `tools\build_cuda.bat` (AMD: `tools\build_amd.bat`) — see `BUILD.md` if cmake missing
4. Meshes still required

Reward edits in `ExampleMain.cpp` / `CudaEnvSet.cpp` need a rebuild.

## robocopy example

```powershell
$src = "D:\path\to\GigaLearnRL"
$dst = "D:\Share\GigaLearnRL_ready"
robocopy $src $dst /E /XD `
  .git __pycache__ wandb checkpoints `
  "$src\build\Release\checkpoints" `
  "$src\build\Release\wandb" `
  "$src\GigaLearnCPP\libtorch" `
  /XF *.pdb *.log
```

Unzip to any directory. Zip helper: `make_share_zip.bat` -> `_share_out\`.

## Short message

> 1) Unzip  
> 2) Python 3.10+ (PATH) + NVIDIA or AMD drivers  
> 3) `SETUP_FIRST_RUN.bat`  
> 4) `run_fresh_train.bat` (AT off)  
> 5) Wait for first-iter metrics  
> 6) Optional AT: `run_with_autotrainer.bat`  
> 7) Rewards/opponents: `docs\CUSTOMIZE.md`  
> Docs: `README.md` · IT: `INIZIO_RAPIDO_IT.md`  
> AMD RX 6600 XT: often need `tools\build_amd.bat` / `docs\AMD.md` (HIP env + CPU PPO). Arenas 2048-4096. No WSL required.
