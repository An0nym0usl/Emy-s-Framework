# Inizio rapido - GigaLearnRL (IT)

Versione **1.0**. Windows 11 + AMD (RX 6600 XT) o NVIDIA.  
Path AMD: HIP SDK + Adrenalin + `tools\build_amd.bat` + `run_fresh_train.bat`.  
Discrete stub rewards, AutoTrainer spento di default, GPU auto-detect. WSL non richiesto.  
EN: [`README.md`](README.md) · AMD: [`docs/AMD.md`](docs/AMD.md)

---

1. Scompatta in una cartella locale scrivibile (es. `D:\GigaLearnRL`). Evita sync che bloccano i file.
2. Python 3.10+ (Add to PATH). Riapri il terminale.
3. Driver: AMD Adrenalin oppure NVIDIA.
4. `SETUP_FIRST_RUN.bat` -> `SETUP OK`.
5. Solo AMD (niente NVIDIA): zip spesso CUDA. Installa [HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html), poi `tools\build_amd.bat`.
   - Env: GPU HIP. PPO: CPU. DirectML non accelera il bot C++.
   - SETUP fallisce solo se AMD senza NVIDIA e manca `hipcc` o restano DLL CUDA in Release.
   - Dual-GPU (iGPU AMD + NVIDIA): path CUDA OK.
6. `run_fresh_train.bat` (o `GigaLearnBot.exe --from-scratch`). Auto-detect NVIDIA/AMD. AT spento.
7. Prima iterazione: `HW: NVIDIA path selected` o `HW: AMD path selected` + `HIP gpuNative=ON`. Target Overall >= 20000 con HIP. Checkpoint: `build\Release\checkpoints\`.
8. Rewards: `src\ExampleMain.cpp` + `CudaEnvSet.cpp` (profile 1). Arenas: blocco `TrainingSize` in ExampleMain (2048 / steps=3 su 6600 XT). Rebuild. [`docs/CUSTOMIZE.md`](docs/CUSTOMIZE.md).
9. AT opzionale: `run_with_autotrainer.bat` / `start_autotrainer.bat`.
10. Problemi: [`README.md`](README.md), [`docs/QUICKSTART.md`](docs/QUICKSTART.md).

### Overall < 20000 (HIP)

1. `HIP gpuNative NOT confirmed` o DLL CUDA (PC solo AMD) -> `tools\build_amd.bat`.
2. Togli `GIGA_FORCE_CPU=1` se presente.
3. Auto-lean puo scrivere `amd_win_20k_next.env` -> riavvia `run_fresh_train.bat`.
4. Manuale: `set GIGA_ENV_ARENAS=1536`.
5. Senza HIP i 20k non sono garantiti.

### AMD Win11 + 6600 XT

| Parte | Dove | Note |
|---|---|---|
| Arenas / fisica | GPU (HIP) | vs RocketSim CPU |
| PPO | CPU (~10 thread su 3600) | non CUDA |
| DirectML | Python opzionale | non accelera `GigaLearnBot.exe` |

Target: Overall >= 20000 SPS con HIP (`amd_win_20k`). Senza HIP: non garantito.
