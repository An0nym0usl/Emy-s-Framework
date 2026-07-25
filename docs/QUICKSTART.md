# Quickstart (IT / EN)

IT: [`../INIZIO_RAPIDO_IT.md`](../INIZIO_RAPIDO_IT.md)  
EN troubleshooting: [`../README.md`](../README.md)

C++ trainer for Rocket League. AutoTrainer off by default.

## Steps

1. Unzip (e.g. `D:\GigaLearnRL`); avoid locking cloud sync.
2. Python 3.10+ (PATH) + NVIDIA or AMD drivers (HIP for Radeon: `docs/AMD.md`).
3. `SETUP_FIRST_RUN.bat`
4. `run_fresh_train.bat` (bot only)
5. Optional AT: `run_with_autotrainer.bat` / `start_autotrainer.bat`
6. Wait for first iteration
7. Checkpoints: `build\Release\checkpoints\`
8. Stop: close bot window
9. Problems: `README.md` §5-6

Stub rewards/opponents: fill via `docs/CUSTOMIZE.md`.

## AutoTrainer

```bat
run_fresh_train.bat
run_with_autotrainer.bat
start_autotrainer.bat
set GIGA_AUTOTRAINER=1
run_fresh_train.bat
set GIGA_NO_AUTOTRAINER=1
set GIGA_NO_SSL_AUTONOMY=1
set GIGA_AT_READONLY=1
```

Or: `GigaLearnBot.exe --no-autotrainer` from Release.
