@echo off
REM =============================================================================
REM Build GigaLearnRL with NVIDIA CUDA + RocketSimCuda (windows-cuda-gpu preset)
REM Prefer a clean path (e.g. C:\GigaLearnRL) - nvcc fails on spaces/apostrophes.
REM Docs: BUILD.md  |  docs\CUDA_SIM.md
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."
set "ROOT=%CD%"

echo.
echo === GigaLearnRL NVIDIA CUDA build ===
echo Root: %ROOT%
echo.

call "%~dp0find_cmake.bat"
if errorlevel 1 exit /b 1

echo %ROOT% | findstr /R /C:" " /C:"'" >nul
if not errorlevel 1 (
  echo [WARN] Build path contains a space or apostrophe.
  echo        nvcc often fails on such paths. Prefer a clean mirror, e.g.:
  echo          C:\GigaLearnRL
  echo        Then set GIGA_CUDA_MIRROR for the train bats if needed.
  echo        Docs: docs\CUDA_SIM.md
  echo.
)

if not exist "%ROOT%\GigaLearnCPP\libtorch" (
  echo [WARN] GigaLearnCPP\libtorch missing - configure may fail.
  echo        Download libtorch ^(CUDA^) into GigaLearnCPP\libtorch before building.
)

where nvcc >nul 2>&1
if errorlevel 1 (
  echo [WARN] nvcc not in PATH - CUDA toolkit may be missing or not on PATH.
  echo        Install CUDA 12.x and reopen the shell if configure fails.
) else (
  echo [OK]   nvcc found
)

set "GIGA_USE_CUDA_SIM=ON"
set "GIGA_GPU_BACKEND=cuda"

echo Configuring preset windows-cuda-gpu ...
"%CMAKE_EXE%" --preset windows-cuda-gpu
if errorlevel 1 (
  echo [FAIL] cmake configure failed
  exit /b 1
)

echo Building ...
"%CMAKE_EXE%" --build --preset windows-cuda-gpu
if errorlevel 1 (
  echo [FAIL] build failed
  exit /b 1
)

echo.
echo [OK] CUDA build finished.
echo Output: %ROOT%\build\Release\GigaLearnBot.exe
echo Next: SETUP_FIRST_RUN.bat  then  run_fresh_train.bat
echo Docs: docs\CUDA_SIM.md  ^|  BUILD.md
echo.
exit /b 0
