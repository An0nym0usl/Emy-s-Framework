@echo off
REM =============================================================================
REM make_share_zip.bat - build share zip
REM From repo root. Excludes checkpoints/.git/__pycache__/junk; KEEP build\Release
REM binaries + collision_meshes + docs. Prints final zip path.
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "ROOT=%CD%"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "OUT_DIR=%ROOT%\_share_out"
set "STAGE=%OUT_DIR%\GigaLearnRL"
set "ZIP=%OUT_DIR%\GigaLearnRL_share_%STAMP%.zip"

echo.
echo === make_share_zip ===
echo Root: %ROOT%
echo Stage: %STAGE%
echo.

if exist "%STAGE%" rmdir /s /q "%STAGE%" 2>nul
mkdir "%OUT_DIR%" 2>nul
mkdir "%STAGE%" 2>nul

REM --- Core tree (no .git, no libtorch, no personal junk) ---
robocopy "%ROOT%" "%STAGE%" /E /NFL /NDL /NJH /NJS /nc /ns /np ^
  /XD .git __pycache__ .vs .idea wandb checkpoints checkpoints_* ^
     _share_out _bench_session _crash_invest _pure80_runs ^
     gigalearn_autotrainer.egg-info RocketSimVis-main docs\archive ^
     opponents\nexto opponents\necto GigaLearnCPP\libtorch ^
     build\Debug build\x64 build\CMakeFiles build\Release\checkpoints ^
     build\Release\wandb ^
     build\Release\autotrainer_smoke ^
  /XF *.pdb *.pyc *.log diag_* proof_* smoke_*.txt smoke_*_err.txt ^
     _smoke_* hw_profile.json hw_profile_*.json eval_report_periodic.json ^
     product_surface_manifest.json teachers_manifest.json .env *.env ^
  >nul

REM Ensure Release binaries folder exists even if robocopy skipped empty bits
if not exist "%ROOT%\build\Release\GigaLearnBot.exe" (
  echo [WARN] build\Release\GigaLearnBot.exe missing - zip will be source-leaning.
  echo        Rebuild required ^(NVIDIA cmake preset or tools\build_amd.bat^).
) else (
  if not exist "%STAGE%\build\Release" mkdir "%STAGE%\build\Release"
  robocopy "%ROOT%\build\Release" "%STAGE%\build\Release" /E /NFL /NDL /NJH /NJS /nc /ns /np ^
    /XD checkpoints checkpoints_* wandb autotrainer_smoke python_scripts ^
    /XF *.pdb *.log *.txt smoke_* diag_* proof_* hw_profile*.json ^
       eval_report_periodic.json product_surface_manifest.json teachers_manifest.json ^
       runtime_overrides.json orchestrator_state.json trainer_status.json ^
       dashboard_status.json coach_summary.json meta_bandit_state.json ^
       last_known_good.json autotrainer_launch.log .env *.env ^
    >nul
  REM Empty checkpoint box
  if not exist "%STAGE%\build\Release\checkpoints" mkdir "%STAGE%\build\Release\checkpoints"
  echo. > "%STAGE%\build\Release\checkpoints\.gitkeep"
)

REM Meshes are required
if not exist "%STAGE%\collision_meshes\soccar" (
  echo [FAIL] collision_meshes\soccar missing in stage - abort.
  exit /b 1
)

REM Strip AT runtime junk under staged Release\autotrainer if present
if exist "%STAGE%\build\Release\autotrainer" (
  for %%F in (
    runtime_overrides.json orchestrator_state.json trainer_status.json
    dashboard_status.json coach_summary.json meta_bandit_state.json
    autotrainer_launch.log
  ) do if exist "%STAGE%\build\Release\autotrainer\%%F" del /f /q "%STAGE%\build\Release\autotrainer\%%F" >nul 2>&1
  if exist "%STAGE%\build\Release\autotrainer\snapshots" rmdir /s /q "%STAGE%\build\Release\autotrainer\snapshots" 2>nul
)

REM Ensure Italian 1-pager + customize docs are present
if not exist "%STAGE%\INIZIO_RAPIDO_IT.md" if exist "%ROOT%\INIZIO_RAPIDO_IT.md" copy /Y "%ROOT%\INIZIO_RAPIDO_IT.md" "%STAGE%\" >nul
if not exist "%STAGE%\docs\CUSTOMIZE.md" if exist "%ROOT%\docs\CUSTOMIZE.md" (
  if not exist "%STAGE%\docs" mkdir "%STAGE%\docs"
  copy /Y "%ROOT%\docs\CUSTOMIZE.md" "%STAGE%\docs\" >nul
)

echo Zipping ^(may take a few minutes if Release has CUDA DLLs^)...
if exist "%ZIP%" del /f /q "%ZIP%" >nul 2>&1
powershell -NoProfile -Command ^
  "Compress-Archive -Path '%STAGE%' -DestinationPath '%ZIP%' -CompressionLevel Optimal -Force"
if errorlevel 1 (
  echo [FAIL] Compress-Archive failed.
  exit /b 1
)

for %%A in ("%ZIP%") do set "ZIPSIZE=%%~zA"
echo.
echo === SHARE ZIP READY ===
echo Path: %ZIP%
echo Size: !ZIPSIZE! bytes
echo Stage kept at: %STAGE%
echo Next: unzip -^> SETUP_FIRST_RUN.bat -^> run_fresh_train.bat
echo AMD: tools\build_amd.bat / docs\AMD.md / INIZIO_RAPIDO_IT.md
echo.
exit /b 0
