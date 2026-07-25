@echo off
REM =============================================================================
REM GigaLearnRL - first-run checks
REM Run from the repo root (this .bat's folder), then:
REM   run_fresh_train.bat
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "ROOT=%CD%"

echo.
echo === GigaLearnRL SETUP_FIRST_RUN ===
echo Root: %ROOT%
echo.

set "FAIL=0"
set "WARN=0"

REM --- Write test (cloud sync / permissions / read-only media) ---
set "WTEST=%ROOT%\_giga_write_test.tmp"
echo setup_ok>"%WTEST%" 2>nul
if not exist "%WTEST%" (
  echo [FAIL] Cannot write to repo folder.
  echo        Move the tree to a local writable path ^(not a locked sync folder^).
  set "FAIL=1"
) else (
  del /f /q "%WTEST%" >nul 2>&1
  echo [OK]   Write test
)

REM --- Python ---
where python >nul 2>&1
if errorlevel 1 (
  echo [FAIL] Python not found on PATH.
  echo        Install Python 3.10+ from https://www.python.org/downloads/
  echo        and check "Add python.exe to PATH". Open a NEW terminal after install.
  set "FAIL=1"
) else (
  set "PYV="
  for /f "tokens=*" %%V in ('python -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "PYV=%%V"
  if not defined PYV (
    echo [FAIL] python on PATH but failed to print version.
    set "FAIL=1"
  ) else (
    echo [OK]   Python !PYV!
  )
)

REM --- GPU: NVIDIA vs AMD (informational + probe) ---
set "HAS_NVIDIA=0"
set "HAS_AMD=0"
set "HAS_6600=0"
set "HAS_CUDA_DLL=0"
set "EXE_DIR=%ROOT%\build\Release"
if defined GIGA_RELEASE_DIR if exist "%GIGA_RELEASE_DIR%\GigaLearnBot.exe" set "EXE_DIR=%GIGA_RELEASE_DIR%"

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  set "HAS_NVIDIA=1"
  echo [OK]   nvidia-smi present
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
) else (
  echo [INFO] nvidia-smi not found - no NVIDIA GPU tools on PATH.
)

REM Soft AMD detection via adapter name (iGPU + dGPU). hipcc alone does NOT mark AMD.
for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "try { (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; ' } catch { '' }" 2^>nul`) do set "GPU_NAMES=%%G"
echo [INFO] Display adapters: !GPU_NAMES!
echo !GPU_NAMES! | find /I "AMD" >nul && set "HAS_AMD=1"
echo !GPU_NAMES! | find /I "Radeon" >nul && set "HAS_AMD=1"
echo !GPU_NAMES! | find /I "6600" >nul && set "HAS_6600=1"

set "HAS_HIPCC=0"
where hipcc >nul 2>&1
if not errorlevel 1 (
  set "HAS_HIPCC=1"
  echo [OK]   hipcc present ^(AMD HIP SDK^)
)

REM Detect NVIDIA CUDA runtime shipped next to exe (prebuilt SHARE is usually CUDA)
if exist "%EXE_DIR%\cudart64_12.dll" set "HAS_CUDA_DLL=1"
if exist "%EXE_DIR%\torch_cuda.dll" set "HAS_CUDA_DLL=1"
if exist "%EXE_DIR%\c10_cuda.dll" set "HAS_CUDA_DLL=1"

REM Dual-GPU note: AMD iGPU + NVIDIA dGPU is common. FAIL only on AMD-only machines
REM when HIP is missing or Release is still CUDA-only. If NVIDIA is present -> CUDA path OK.
if "!HAS_AMD!"=="1" (
  echo.
  if "!HAS_NVIDIA!"=="1" (
    echo === Dual GPU: AMD adapter + NVIDIA ===
    echo [OK]   NVIDIA present - CUDA train path is OK.
    echo        AMD iGPU alone does not fail SETUP on this machine.
    echo        Optional HIP rebuild only if you want the Radeon as primary GPU sim.
    echo        Guide: docs\AMD.md  ^|  Italian: INIZIO_RAPIDO_IT.md
  ) else (
    echo === AMD / Radeon path ^(no NVIDIA^) ===
    if "!HAS_6600!"=="1" (
      echo [INFO] RX 6600 / 6600 XT class GPU detected ^(8 GB reference^).
    ) else (
      echo [INFO] AMD Radeon detected.
    )
    echo        Reference PC: Win11 + A520 + Ryzen 5 3600 + RX 6600 XT 8GB.
    echo        Path: HIP SDK + tools\build_amd.bat
    echo          - Env/physics on Radeon ^(HIP gpuNative^)
    echo          - PPO learn on CPU ^(threaded; GIGA_TORCH_THREADS^)
    echo        Guide: docs\AMD.md  ^|  Italian: INIZIO_RAPIDO_IT.md
    echo        Arenas: start **2048** / steps=3 on 8GB - do NOT force 8192 ^(hw_probe helps^).
    echo        If Overall ^< 20k: see INIZIO_RAPIDO_IT.md / docs\AMD.md
    echo        DirectML: optional Python only - does NOT speed GigaLearnBot ^(docs\DIRECTML.md^).
    echo        ^(IT^) Percorso: HIP env su GPU + PPO su CPU. Niente WSL. DirectML non accelera il bot.
    REM FAIL only if AMD detected AND NVIDIA NOT detected AND (no hipcc OR CUDA-only Release)
    if "!HAS_CUDA_DLL!"=="1" (
      echo.
      echo [FAIL] NVIDIA CUDA DLLs found in Release, but this PC has AMD only ^(no NVIDIA^).
      echo        CUDA prebuilts will NOT use your Radeon - rebuild HIP before train:
      echo          tools\build_amd.bat
      echo        ^(IT^) Binari CUDA con sola GPU AMD: ricostruisci con HIP ^(obbligatorio^).
      set "FAIL=1"
    )
    if "!HAS_HIPCC!"=="0" (
      echo [FAIL] hipcc not on PATH - install AMD HIP SDK, reopen the shell, then:
      echo          tools\build_amd.bat
      echo        https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
      echo        ^(IT^) Senza hipcc i 20k Overall NON sono garantiti.
      set "FAIL=1"
    )
  )
  echo.
)

if "!HAS_NVIDIA!"=="0" if "!HAS_AMD!"=="0" (
  echo [WARN] No NVIDIA tools and no AMD adapter detected - training may be CPU-only ^(slow^).
  set "WARN=1"
)

REM hw_probe adapts arenas for NVIDIA CUDA / AMD HIP / CPU
if exist "%ROOT%\tools\hw_probe.py" (
  where python >nul 2>&1
  if not errorlevel 1 (
    echo [INFO] Running hw_probe...
    python "%ROOT%\tools\hw_probe.py" 2>nul
    if errorlevel 1 (
      echo [WARN] hw_probe failed - train bats will use built-in defaults.
      set "WARN=1"
    )
  )
)

REM --- collision_meshes ---
if exist "%ROOT%\collision_meshes\soccar" (
  echo [OK]   collision_meshes\soccar
) else (
  echo [FAIL] Missing RocketSim meshes in collision_meshes\soccar
  echo        Copy them into the zip / from whoever shared the framework.
  set "FAIL=1"
)

REM --- libtorch (rebuild only) ---
if exist "%ROOT%\GigaLearnCPP\libtorch" (
  echo [OK]   GigaLearnCPP\libtorch
) else (
  echo [WARN] libtorch missing under GigaLearnCPP\libtorch
  echo        Needed only if you rebuild from source. Prebuilt Release is fine without it.
  set "WARN=1"
)

REM --- Release binaries ---
REM EXE_DIR already set in GPU section (default build\Release)

if exist "%EXE_DIR%\GigaLearnBot.exe" if exist "%EXE_DIR%\GigaLearnCPP.dll" (
  echo [OK]   GigaLearnBot.exe + GigaLearnCPP.dll
  echo        in %EXE_DIR%
  for %%A in ("%EXE_DIR%\GigaLearnCPP.dll") do set "DLLSIZE=%%~zA"
  if defined DLLSIZE (
    if !DLLSIZE! LSS 2000000 (
      echo [WARN] GigaLearnCPP.dll looks tiny ^(!DLLSIZE! bytes^) - possible partial/corrupt copy.
      echo        Re-copy Release from SHARE or rebuild. SETUP will try heal_release_dll.bat.
      set "WARN=1"
    )
  )
) else (
  echo [FAIL] Release binaries not found in %EXE_DIR%
  echo        Options:
  echo          1^) Copy build\Release from the SHARE package ^(recommended^)
  echo          2^) Rebuild NVIDIA: tools\build_cuda.bat  ^(see BUILD.md^)
  echo             ^(or: cmake --preset windows-cuda-gpu ^&^& cmake --build --preset windows-cuda-gpu^)
  echo          3^) Rebuild AMD HIP: tools\build_amd.bat  ^(see docs\AMD.md^)
  echo        If "cmake not in PATH": install CMake / see BUILD.md
  echo        If nvcc fails on spaces in the path, use a short path / GIGA_CUDA_MIRROR.
  set "FAIL=1"
)

REM --- Mesh next to exe (junction or copy) ---
if exist "%EXE_DIR%\GigaLearnBot.exe" (
  if not exist "%EXE_DIR%\collision_meshes\soccar" (
    if exist "%ROOT%\collision_meshes\soccar" (
      echo Linking collision_meshes -^> Release...
      mklink /J "%EXE_DIR%\collision_meshes" "%ROOT%\collision_meshes" >nul 2>&1
      if errorlevel 1 (
        echo [WARN] mklink failed ^(need admin or Developer Mode^) - trying xcopy...
        xcopy /E /I /Y "%ROOT%\collision_meshes" "%EXE_DIR%\collision_meshes\" >nul
      )
    )
  )
  if exist "%EXE_DIR%\collision_meshes\soccar" (
    echo [OK]   meshes next to exe
  ) else (
    echo [FAIL] collision_meshes not available next to GigaLearnBot.exe
    set "FAIL=1"
  )
)

REM --- AutoTrainer Python deps ---
if not "%FAIL%"=="1" (
  if exist "%ROOT%\autotrainer\requirements.txt" (
    where python >nul 2>&1
    if not errorlevel 1 (
      echo.
      echo Installing AutoTrainer deps ^(pip^)...
      set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
      python -m pip install -q -r "%ROOT%\autotrainer\requirements.txt"
      if errorlevel 1 (
        echo [WARN] pip install requirements failed - AutoTrainer may not start.
        echo        Opt-out: set GIGA_NO_AUTOTRAINER=1
        set "WARN=1"
      ) else (
        echo [OK]   pip requirements
      )
      python -m pip install -q -e "%ROOT%" >nul 2>&1
    )
  )
)

REM --- Watch dir clean for first train ---
set "WATCH=%EXE_DIR%\autotrainer"
if exist "%EXE_DIR%\GigaLearnBot.exe" (
  if not exist "%WATCH%" mkdir "%WATCH%"
  if not exist "%WATCH%\profiles" mkdir "%WATCH%\profiles"
  if exist "%ROOT%\autotrainer\profiles" xcopy /Y /E /I "%ROOT%\autotrainer\profiles\*" "%WATCH%\profiles\" >nul 2>&1
  if exist "%ROOT%\autotrainer\config.default.yaml" copy /Y "%ROOT%\autotrainer\config.default.yaml" "%WATCH%\" >nul 2>&1
  if not exist "%EXE_DIR%\checkpoints" mkdir "%EXE_DIR%\checkpoints"
  REM Clear stale AutoTrainer runtime (never ship LKG / overrides / bandit state)
  for %%F in (
    runtime_overrides.json
    orchestrator_state.json
    trainer_status.json
    dashboard_status.json
    coach_summary.json
    meta_bandit_state.json
  ) do if exist "%WATCH%\%%F" del /f /q "%WATCH%\%%F" >nul 2>&1
  if exist "%WATCH%\snapshots\last_known_good.json" del /f /q "%WATCH%\snapshots\last_known_good.json" >nul 2>&1
  if exist "%WATCH%\snapshots" if exist "%WATCH%\snapshots\*" (
    del /f /q "%WATCH%\snapshots\*.json" >nul 2>&1
  )
  if exist "%WATCH%\autotrainer_launch.log" del /f /q "%WATCH%\autotrainer_launch.log" >nul 2>&1
  echo [OK]   watch dir: %WATCH% ^(AutoTrainer runtime cleared^)
)

REM --- Heal DLL if tool present ---
if exist "%EXE_DIR%\GigaLearnBot.exe" if exist "%ROOT%\tools\heal_release_dll.bat" (
  call "%ROOT%\tools\heal_release_dll.bat" "%EXE_DIR%"
)

echo.
if "%FAIL%"=="1" (
  echo === SETUP INCOMPLETE ===
  echo Fix the [FAIL] items above, then re-run this script.
  echo Guide: README.md  ^|  Italian: INIZIO_RAPIDO_IT.md
  echo Zip contents: SHARE.md
  echo AMD: docs\AMD.md
  exit /b 1
)

echo === SETUP OK ===
if "%WARN%"=="1" echo ^(There were WARN items - read them before training.^)
echo.
REM Suggest HIP rebuild only on AMD-only + CUDA Release; NVIDIA (+ optional AMD iGPU) -> train.
if "!HAS_AMD!"=="1" if "!HAS_NVIDIA!"=="0" if "!HAS_CUDA_DLL!"=="1" (
  echo NEXT / PROSSIMO: tools\build_amd.bat  then / poi  run_fresh_train.bat
) else (
  echo NEXT / PROSSIMO: run_fresh_train.bat  ^(AutoTrainer OFF / AT spento^)
)
echo.
echo Next steps:
echo   1. AMD-only: tools\build_amd.bat  ^(docs\AMD.md / INIZIO_RAPIDO_IT.md^)
echo      Native Win: HIP env on GPU + CPU PPO threaded - no WSL required
echo      Dual-GPU ^(AMD iGPU + NVIDIA^): use CUDA Release; HIP optional
echo   2. Run:  run_fresh_train.bat
echo      ^(FROM-SCRATCH train; AutoTrainer OFF by default via GIGA_NO_AUTOTRAINER^)
echo   3. Wait for first-iteration metrics in the bot console
echo   4. Optional AutoTrainer later: run_with_autotrainer.bat
echo.
echo Full guide: README.md  ^|  Italian: INIZIO_RAPIDO_IT.md
echo.
exit /b 0
