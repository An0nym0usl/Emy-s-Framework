@echo off
REM =============================================================================
REM OPTIONAL: WSL2 + ROCm (not required for Win HIP).
REM Prefer: tools\build_amd.bat + docs\AMD.md + INIZIO_RAPIDO_IT.md
REM This helper only prints WSL2+ROCm steps for people who explicitly want Linux.
REM Full guide: docs\AMD_WSL2.md
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."
set "ROOT=%CD%"
set "DOC=%ROOT%\docs\AMD_WSL2.md"

echo.
echo === GigaLearnRL - WSL2 + ROCm ^(OPTIONAL ADVANCED^) ===
echo Repo: %ROOT%
echo.
echo NOTE: Preferred path is native Windows HIP - not this script.
echo   Use instead: tools\build_amd.bat  ^|  docs\AMD.md  ^|  INIZIO_RAPIDO_IT.md
echo.

echo WHY THIS EXISTS
echo   GigaLearn PPO is C++ LibTorch inside GigaLearnBot - NOT pip torch.
echo   pip install torch-directml does NOT accelerate this trainer.
echo   For Linux/WSL users only.
echo.
echo QUICK MATRIX
echo   Win HIP SDK   env GPU + PPO CPU
echo   WSL2+ROCm     env GPU + PPO GPU   optional advanced only
echo   DirectML pip  Python only         does not speed GigaLearnBot
echo.

if exist "%DOC%" (
  echo Opening: %DOC%
  start "" "%DOC%"
) else (
  echo [WARN] Missing %DOC%
)

echo.
echo --- Step 1: WSL2 Ubuntu ^(admin PowerShell^) ---
echo   wsl --install -d Ubuntu-22.04
echo   wsl --update
echo.

echo --- Step 2: Copy repo onto Linux FS (fast builds) ---
echo   Inside Ubuntu:
echo   mkdir -p ~/src
echo   cp -a "/mnt/c/path/to/GigaLearnRL" ~/src/GigaLearnRL
echo   REM short local mirror example:
echo   cp -a /mnt/c/GigaLearnRL ~/src/GigaLearnRL
echo   cd ~/src/GigaLearnRL
echo.

echo --- Step 3: ROCm + build inside WSL ---
echo   chmod +x tools/setup_wsl2_rocm.sh
echo   ./tools/setup_wsl2_rocm.sh
echo   ^(RX 6600 XT / gfx1030 is often unofficial - see docs\AMD_WSL2.md^)
echo.

where wsl >nul 2>&1
if errorlevel 1 (
  echo [WARN] wsl.exe not on PATH - install WSL2 first, then re-run this bat.
  echo.
  pause
  exit /b 1
)

echo --- Optional: launch Ubuntu setup ---
choice /C YN /M "Open WSL and run setup_wsl2_rocm.sh now"
if errorlevel 2 goto :end
if errorlevel 1 (
  REM Prefer Linux-side copy if already present; else run from /mnt path.
  wsl -e bash -lc "if [ -d ~/src/GigaLearnRL ]; then cd ~/src/GigaLearnRL; elif [ -d /mnt/c/GigaLearnRL ]; then cd /mnt/c/GigaLearnRL; else cd \"$(wslpath '%ROOT%')\"; fi; chmod +x tools/setup_wsl2_rocm.sh 2>/dev/null; GIGA_WSL_SKIP_BUILD=1 ./tools/setup_wsl2_rocm.sh || true; echo; echo 'To build: unset GIGA_WSL_SKIP_BUILD and re-run ./tools/setup_wsl2_rocm.sh'; exec bash -l"
)

:end
echo.
echo Also see: docs\AMD.md  ^|  README.md troubleshooting ^(WSL2 vs Win HIP vs DirectML^)
echo Native Win HIP ^(env only^): tools\build_amd.bat
echo.
pause
endlocal
exit /b 0
