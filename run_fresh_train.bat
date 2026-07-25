@echo off

REM From-scratch train (discrete blank defaults).

REM Checkpoints: build\Release\checkpoints\  (created empty if missing)

REM

REM Canonical:

REM   build\Release\GigaLearnBot.exe --from-scratch

REM

REM Settings (plain exe / --from-scratch / --fresh):

REM   GPU gpuNative (CUDA / HIP), arenas from hw_probe

REM   Blank GPU rewards (Goal 100 | Touch 5 | VelBallToGoal 10 | VelPlayerToBall 2)

REM   skill-eval OFF at boot; AutoTrainer OFF by default

REM   Opt-in AT: run_with_autotrainer.bat  or  set GIGA_AUTOTRAINER=1

REM   Opt-out still works: GIGA_NO_AUTOTRAINER=1 / --no-autotrainer | GIGA_AT_READONLY=1 | GIGA_NO_SSL_AUTONOMY=1

REM   Console Report each iter (GIGA_SKIP_METRICS=1 to silence)

REM

REM MUST run exe from the Release dir (cwd = that folder).

REM Optional: GIGA_RELEASE_DIR  |  GIGA_CUDA_MIRROR (short path if nvcc needs it)

REM Do NOT run a second GigaLearnBot on the same GPU (iters look hung).

REM Skip hw probe: GIGA_NO_HW_PROFILE=1

REM AMD HIP: tools\build_amd.bat / docs\AMD.md

REM First run: SETUP_FIRST_RUN.bat  |  Full guide: README.md



setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"

if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"



echo.

echo ---

echo   GigaLearnRL  -  FROM-SCRATCH TRAIN

echo ---

echo   AutoTrainer :  OFF  ^(default^)

echo   Enable AT   :  run_with_autotrainer.bat

echo                  or set GIGA_AUTOTRAINER=1

echo.

echo   TrainingSize:  edit namespace TrainingSize in

echo                  src\ExampleMain.cpp  then REBUILD

echo                  ^(amd_win_20k: 2048 arenas / steps=3, Overall^>=20k with HIP^)

echo.

echo   Checkpoints :  build\Release\checkpoints\

echo   Italian     :  INIZIO_RAPIDO_IT.md

echo ---

echo.



set "LOCAL_RELEASE=%ROOT%\build\Release"

set "CUDA_RELEASE="

if defined GIGA_CUDA_MIRROR set "CUDA_RELEASE=%GIGA_CUDA_MIRROR%\build\Release"



REM Prefer explicit Release dir, then bat-local Release, then optional CUDA mirror.

set "EXE_DIR="

if defined GIGA_RELEASE_DIR if exist "%GIGA_RELEASE_DIR%\GigaLearnBot.exe" set "EXE_DIR=%GIGA_RELEASE_DIR%"

if not defined EXE_DIR if exist "%LOCAL_RELEASE%\GigaLearnBot.exe" set "EXE_DIR=%LOCAL_RELEASE%"

if not defined EXE_DIR if defined CUDA_RELEASE if exist "%CUDA_RELEASE%\GigaLearnBot.exe" set "EXE_DIR=%CUDA_RELEASE%"



if not defined EXE_DIR (

  echo ERROR: GigaLearnBot.exe not found under:

  echo   %LOCAL_RELEASE%

  if defined GIGA_RELEASE_DIR echo   %GIGA_RELEASE_DIR%

  if defined CUDA_RELEASE echo   %CUDA_RELEASE%

  echo Run SETUP_FIRST_RUN.bat, or copy/build Release next to this repo, then retry.

  echo See README.md

  exit /b 1

)



REM Heal stale/mismatched GigaLearnCPP.dll (exe rebuilt without DLL -> 0xc0000005).

call "%ROOT%\tools\heal_release_dll.bat" "%EXE_DIR%"

if errorlevel 1 (

  REM If chosen dir still looks broken, fall back to optional CUDA mirror Release.

  if defined CUDA_RELEASE if /I not "%EXE_DIR%"=="%CUDA_RELEASE%" if exist "%CUDA_RELEASE%\GigaLearnBot.exe" if exist "%CUDA_RELEASE%\GigaLearnCPP.dll" (

    echo WARNING: %EXE_DIR% still has a bad GigaLearnCPP.dll -- using GIGA_CUDA_MIRROR:

    echo   %CUDA_RELEASE%

    set "EXE_DIR=%CUDA_RELEASE%"

    call "%ROOT%\tools\heal_release_dll.bat" "%EXE_DIR%"

  )

)



REM Single-instance GPU warning (second train makes iters look hung).

tasklist /FI "IMAGENAME eq GigaLearnBot.exe" 2>nul | find /I "GigaLearnBot.exe" >nul

if not errorlevel 1 (

  echo WARNING: GigaLearnBot.exe already running -- same GPU may look hung / thrash VRAM.

  echo   Close the other train window before starting another.

)



set "EXE=%EXE_DIR%\GigaLearnBot.exe"

set "CKPT_DIR=%EXE_DIR%\checkpoints"

set "WATCH=%EXE_DIR%\autotrainer"



cd /d "%EXE_DIR%"

if not exist "%EXE%" (

  echo ERROR: GigaLearnBot.exe not found at:

  echo   %EXE%

  exit /b 1

)

if not exist "%EXE_DIR%\GigaLearnCPP.dll" (

  echo ERROR: GigaLearnCPP.dll missing next to exe -- cannot train.

  echo   %EXE_DIR%

  exit /b 1

)



REM Create empty default box only (does not wipe other checkpoint folders).

if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"



REM Clear continue-leak env if a prior shell left it set.

set GIGA_CONTINUE_LEAK=

set GIGA_MAX_LEARN=

set GIGA_FROM_SCRATCH=1

set GIGA_ENV_LEAN=1



REM Drop stale AutoTrainer state so Stage-1 starter is not poisoned by leftover

REM LKG / aerial SSL pressure (VelBall=3.0, Goal~0.8, weak touch), and so soft

REM distill / HARD_RECOVERY cannot re-arm from a prior run's status/orchestrator.

REM Opt-out: set GIGA_KEEP_AT_OVERRIDES=1

if /I not "%GIGA_KEEP_AT_OVERRIDES%"=="1" (

  if exist "%WATCH%\runtime_overrides.json" (

    echo Clearing stale AutoTrainer overrides: %WATCH%\runtime_overrides.json

    del /f /q "%WATCH%\runtime_overrides.json" >nul 2>&1

  )

  if exist "%WATCH%\snapshots\last_known_good.json" (

    echo Clearing stale LKG snapshot: %WATCH%\snapshots\last_known_good.json

    del /f /q "%WATCH%\snapshots\last_known_good.json" >nul 2>&1

  )

  if exist "%WATCH%\orchestrator_state.json" (

    echo Clearing stale orchestrator state: %WATCH%\orchestrator_state.json

    del /f /q "%WATCH%\orchestrator_state.json" >nul 2>&1

  )

  if exist "%WATCH%\trainer_status.json" (

    echo Clearing stale trainer_status: %WATCH%\trainer_status.json

    del /f /q "%WATCH%\trainer_status.json" >nul 2>&1

  )

  if exist "%WATCH%\dashboard_status.json" (

    del /f /q "%WATCH%\dashboard_status.json" >nul 2>&1

  )

)



REM --- Hardware probe -> adaptive arenas / CPU fallback (AMD etc.) ---

REM Failure is non-fatal: train continues with built-in POWER defaults.

if /I not "%GIGA_NO_HW_PROFILE%"=="1" (

  if exist "%ROOT%\tools\hw_probe.py" (

    echo Running hardware probe...

    set "HW_OK="

    where python >nul 2>&1

    if not errorlevel 1 (

      python "%ROOT%\tools\hw_probe.py" --out "%EXE_DIR%\hw_profile.json" --print-env > "%TEMP%\giga_hw_env.txt" 2> "%TEMP%\giga_hw_err.txt"

      if not errorlevel 1 (

        set "HW_OK=1"

        for /f "usebackq tokens=1,* delims==" %%A in ("%TEMP%\giga_hw_env.txt") do (

          if /I "%%A"=="GIGA_ENV_ARENAS" if not defined GIGA_ENV_ARENAS set "GIGA_ENV_ARENAS=%%B"

          if /I "%%A"=="GIGA_ENV_STEPS" if not defined GIGA_ENV_STEPS set "GIGA_ENV_STEPS=%%B"

          if /I "%%A"=="GIGA_ENV_EPOCHS" if not defined GIGA_ENV_EPOCHS set "GIGA_ENV_EPOCHS=%%B"

          if /I "%%A"=="GIGA_ENV_MAXEP" if not defined GIGA_ENV_MAXEP set "GIGA_ENV_MAXEP=%%B"

          if /I "%%A"=="GIGA_ENV_FP32" if not defined GIGA_ENV_FP32 set "GIGA_ENV_FP32=%%B"

          if /I "%%A"=="GIGA_ENV_LEAN" if not defined GIGA_ENV_LEAN set "GIGA_ENV_LEAN=%%B"

          if /I "%%A"=="GIGA_ASYNC_OVERLAP" if not defined GIGA_ASYNC_OVERLAP set "GIGA_ASYNC_OVERLAP=%%B"

          if /I "%%A"=="GIGA_FORCE_CPU" if not defined GIGA_FORCE_CPU set "GIGA_FORCE_CPU=%%B"

          if /I "%%A"=="GIGA_TORCH_THREADS" if not defined GIGA_TORCH_THREADS set "GIGA_TORCH_THREADS=%%B"

          if /I "%%A"=="OMP_NUM_THREADS" if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=%%B"

          if /I "%%A"=="MKL_NUM_THREADS" if not defined MKL_NUM_THREADS set "MKL_NUM_THREADS=%%B"

          if /I "%%A"=="GIGA_HIP_FORCE" if not defined GIGA_HIP_FORCE set "GIGA_HIP_FORCE=%%B"

          if /I "%%A"=="GIGA_GPU_BACKEND" if not defined GIGA_GPU_BACKEND set "GIGA_GPU_BACKEND=%%B"

          if /I "%%A"=="GIGA_TRAIN_PROFILE" if not defined GIGA_TRAIN_PROFILE set "GIGA_TRAIN_PROFILE=%%B"

        )

        set "GIGA_HW_PROFILE=!EXE_DIR!\hw_profile.json"

        echo HW profile: !GIGA_HW_PROFILE!

        if defined GIGA_ENV_ARENAS echo   arenas=!GIGA_ENV_ARENAS!

        if defined GIGA_TORCH_THREADS echo   torch_threads=!GIGA_TORCH_THREADS! ^(CPU PPO^)

        if /I "!GIGA_FORCE_CPU!"=="1" echo   force_cpu=1 ^(CPU RocketSim - no GPU sim^)

      )

    )

    if not defined HW_OK (

      echo WARNING: hw_probe failed -- using safe POWER defaults ^(arenas/CUDA from exe^).

      if exist "%TEMP%\giga_hw_err.txt" type "%TEMP%\giga_hw_err.txt"

      set "GIGA_HW_PROFILE="

    )

  )

)



REM --- amd_win_20k: force Overall>=20k knobs when AMD HIP is the path ---

REM HIP present (probe backend / GIGA_HIP_FORCE) -> lock lean bank + CPU threads.

REM Without HIP, 20k Overall is NOT guaranteed (CPU RocketSim).

REM Apply auto-downgrade hint from prior run (written by exe when Overall stayed <20k).

if exist "%EXE_DIR%\amd_win_20k_next.env" (

  echo   Applying amd_win_20k_next.env ^(auto-downgrade from prior low Overall^)...

  for /f "usebackq tokens=1,* delims== eol=#" %%A in ("%EXE_DIR%\amd_win_20k_next.env") do (

    if /I "%%A"=="GIGA_ENV_ARENAS" if not defined GIGA_ENV_ARENAS set "GIGA_ENV_ARENAS=%%B"

    if /I "%%A"=="GIGA_ENV_STEPS" if not defined GIGA_ENV_STEPS set "GIGA_ENV_STEPS=%%B"

    if /I "%%A"=="GIGA_TRAIN_PROFILE" if not defined GIGA_TRAIN_PROFILE set "GIGA_TRAIN_PROFILE=%%B"

  )

)

set "GIGA_IS_AMD_HIP="

if /I "%GIGA_GPU_BACKEND%"=="hip" set "GIGA_IS_AMD_HIP=1"

if /I "%GIGA_HIP_FORCE%"=="1" set "GIGA_IS_AMD_HIP=1"

if /I "%GIGA_TRAIN_PROFILE%"=="amd_win_20k" set "GIGA_IS_AMD_HIP=1"

if defined GIGA_IS_AMD_HIP (

  if not defined GIGA_TRAIN_PROFILE set "GIGA_TRAIN_PROFILE=amd_win_20k"

  if not defined GIGA_ENV_ARENAS set "GIGA_ENV_ARENAS=2048"

  if not defined GIGA_ENV_STEPS set "GIGA_ENV_STEPS=3"

  if not defined GIGA_ENV_EPOCHS set "GIGA_ENV_EPOCHS=1"

  if not defined GIGA_ENV_MAXEP set "GIGA_ENV_MAXEP=1.5"

  if not defined GIGA_ENV_FP32 set "GIGA_ENV_FP32=1"

  if not defined GIGA_ENV_LEAN set "GIGA_ENV_LEAN=1"

  if not defined GIGA_ASYNC_OVERLAP set "GIGA_ASYNC_OVERLAP=0"

  if not defined GIGA_TORCH_THREADS set "GIGA_TORCH_THREADS=10"

  if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=!GIGA_TORCH_THREADS!"

  if not defined MKL_NUM_THREADS set "MKL_NUM_THREADS=!GIGA_TORCH_THREADS!"

  echo.

  echo   amd_win_20k: target Overall ^>= 20,000 SPS ^(HIP env + CPU PPO^)

  echo     arenas=!GIGA_ENV_ARENAS! steps=!GIGA_ENV_STEPS! epochs=!GIGA_ENV_EPOCHS! maxEp=!GIGA_ENV_MAXEP!

  echo     torch_threads=!GIGA_TORCH_THREADS! fp32=!GIGA_ENV_FP32! async=!GIGA_ASYNC_OVERLAP!

  if /I "!GIGA_FORCE_CPU!"=="1" (

    echo   WARNING: GIGA_FORCE_CPU=1 - HIP gpuNative OFF; 20k Overall NOT guaranteed.

  )

  if exist "%EXE_DIR%\cudart64_12.dll" if /I not "%GIGA_GPU_BACKEND%"=="hip" (

    echo   WARNING: CUDA DLLs in Release with AMD path - rebuild: tools\build_amd.bat

    echo            Overall^>=20k NOT guaranteed until HIP rebuild.

  )

  echo.

)



REM --- AutoTrainer sidecar (OFF by default) ---

REM Skip AT unless GIGA_AUTOTRAINER=1 (run_with_autotrainer.bat).

REM Failure is non-fatal: train still starts; clear EXTERNAL so exe may self-spawn only if opted in.

if /I "%GIGA_AUTOTRAINER%"=="1" (

  set "GIGA_NO_AUTOTRAINER="

) else (

  set "GIGA_NO_AUTOTRAINER=1"

)

set "GIGA_AUTOTRAINER_EXTERNAL="

if /I "%GIGA_NO_AUTOTRAINER%"=="1" (

  echo AutoTrainer: OFF by default ^(bot-only train^)

  echo   Enable later: run_with_autotrainer.bat  or  start_autotrainer.bat

  REM EXTERNAL=1 blocks exe self-spawn even if an old binary ignores NO_AUTOTRAINER.

  set "GIGA_AUTOTRAINER_EXTERNAL=1"

) else (

  call "%ROOT%\tools\launch_autotrainer.bat" default "%WATCH%"

  if errorlevel 1 (

    echo WARNING: AutoTrainer launch failed -- train continues without sidecar.

    set "GIGA_AUTOTRAINER_EXTERNAL="

  ) else (

    set "GIGA_AUTOTRAINER_EXTERNAL=1"

  )

)



echo Starting FROM-SCRATCH train (POWER) into: %CKPT_DIR%

echo Exe: %EXE%

echo Cwd: %CD%

echo (same as --from-scratch)

"%EXE%" --from-scratch %*

set "RC=%ERRORLEVEL%"

endlocal & exit /b %RC%

