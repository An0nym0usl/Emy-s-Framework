@echo off
REM =============================================================================
REM Build GigaLearnRL with AMD HIP backend (Windows HIP SDK)
REM Prefer a clean path (e.g. C:\GigaLearnRL) - same as CUDA mirror advice.
REM Docs: docs\AMD.md
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."
set "ROOT=%CD%"

echo.
echo === GigaLearnRL AMD HIP build ===
echo Root: %ROOT%
echo.

call "%~dp0find_cmake.bat"
if errorlevel 1 exit /b 1

where hipcc >nul 2>&1
if errorlevel 1 (
  echo [FAIL] hipcc not in PATH - install AMD HIP SDK and reopen the shell.
  echo        https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
  echo        Without hipcc this build cannot produce HIP gpuNative ^(20k Overall NOT guaranteed^).
  exit /b 1
)
echo [OK] hipcc found

where ninja >nul 2>&1
if errorlevel 1 (
  echo [WARN] ninja not in PATH - windows-hip preset expects Ninja.
  echo        Install Ninja or use: cmake -G Ninja ...
)

if not exist "%ROOT%\GigaLearnCPP\libtorch" (
  echo [WARN] GigaLearnCPP\libtorch missing - configure may fail.
  echo        For Windows AMD: CPU libtorch is OK for PPO while HIP runs the env.
  echo        DirectML does NOT speed C++ PPO - docs\DIRECTML.md / docs\AMD.md
)

REM Warn if Release still looks like a CUDA-only SHARE drop.
if exist "%ROOT%\build\Release\cudart64_12.dll" (
  echo [INFO] build\Release still has CUDA runtime DLLs.
  echo        After a successful HIP build, deploy HIP outputs into Release ^(or set GIGA_RELEASE_DIR^)
  echo        so run_fresh_train.bat does not keep launching a CUDA binary on AMD.
)

set "GIGA_GPU_BACKEND=hip"
set "GIGA_USE_CUDA_SIM=ON"

echo Configuring preset windows-hip ...
"%CMAKE_EXE%" --preset windows-hip
if errorlevel 1 (
  echo [FAIL] cmake configure failed
  exit /b 1
)

echo Building ...
"%CMAKE_EXE%" --build --preset windows-hip
if errorlevel 1 (
  echo [FAIL] build failed
  exit /b 1
)

echo.
echo [OK] HIP build finished. Probe hardware:
echo   python tools\hw_probe.py
echo Expect: profile=amd_win_20k backend=hip hip_available=1
echo Then copy/use build-hip outputs into build\Release ^(or set GIGA_RELEASE_DIR^).
echo.
echo Win HIP: env on GPU + CPU PPO ^(threaded^) - docs\AMD.md
echo If Overall ^< 20k: INIZIO_RAPIDO_IT.md / amd_win_20k_next.env
echo DirectML pip does NOT speed this C++ learner - docs\DIRECTML.md
echo.
exit /b 0
