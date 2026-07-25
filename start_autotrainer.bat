@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM Portable: local build\Release, or GIGA_RELEASE_DIR / GIGA_CUDA_MIRROR.
set "EXEDIR=%ROOT%\build\Release"
if defined GIGA_RELEASE_DIR set "EXEDIR=%GIGA_RELEASE_DIR%"
set "WATCH=%EXEDIR%\autotrainer"
set "ALTEXEDIR="
if defined GIGA_CUDA_MIRROR set "ALTEXEDIR=%GIGA_CUDA_MIRROR%\build\Release"
set "ALTEXE="
if defined ALTEXEDIR set "ALTEXE=%ALTEXEDIR%\GigaLearnBot.exe"
set "ALTWATCH="
if defined ALTEXEDIR set "ALTWATCH=%ALTEXEDIR%\autotrainer"

if exist "%EXEDIR%\GigaLearnBot.exe" (
  set "USEDIR=%EXEDIR%"
  set "USEEXE=%EXEDIR%\GigaLearnBot.exe"
  set "USEWATCH=%WATCH%"
) else if defined ALTEXE if exist "%ALTEXE%" (
  set "USEDIR=%ALTEXEDIR%"
  set "USEEXE=%ALTEXE%"
  set "USEWATCH=%ALTWATCH%"
) else (
  echo GigaLearnBot.exe not found under %EXEDIR%
  echo Run SETUP_FIRST_RUN.bat or build/copy Release. See README.md
  exit /b 1
)

if not exist "%USEDIR%\collision_meshes" (
  if exist "%ROOT%\collision_meshes" (
    mklink /J "%USEDIR%\collision_meshes" "%ROOT%\collision_meshes" >nul 2>&1
  ) else if defined GIGA_CUDA_MIRROR if exist "%GIGA_CUDA_MIRROR%\collision_meshes" (
    mklink /J "%USEDIR%\collision_meshes" "%GIGA_CUDA_MIRROR%\collision_meshes" >nul 2>&1
  )
)

if not exist "%USEWATCH%\profiles" mkdir "%USEWATCH%\profiles"
xcopy /Y /E /I "%ROOT%\autotrainer\profiles\*" "%USEWATCH%\profiles\" >nul 2>&1
copy /Y "%ROOT%\autotrainer\config.default.yaml" "%USEWATCH%\" >nul 2>&1

if "%1"=="" (set "PROFILE=default") else (set "PROFILE=%1")

echo Exe: %USEEXE%
echo Watch: %USEWATCH%
echo Profile: %PROFILE%
echo.

REM Heal DLL/exe before launch (stale/partial Release files crash with 0xc0000005).
if exist "%ROOT%\tools\heal_release_dll.bat" call "%ROOT%\tools\heal_release_dll.bat" "%USEDIR%"

tasklist /FI "IMAGENAME eq GigaLearnBot.exe" 2>nul | find /I "GigaLearnBot.exe" >nul
if not errorlevel 1 (
  echo WARNING: stopping existing GigaLearnBot.exe so GPU is free...
  taskkill /F /IM GigaLearnBot.exe >nul 2>&1
  timeout /t 2 /nobreak >nul
)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REM Explicit AutoTrainer opt-in (default train path leaves AT OFF).
set "GIGA_AUTOTRAINER=1"
set "GIGA_NO_AUTOTRAINER="
REM Start AutoTrainer BEFORE the bot so we only set EXTERNAL when the brain is alive.
REM If AT dies on boot, leave EXTERNAL unset so the exe can self-spawn (GIGA_AUTOTRAINER=1).
set "GIGA_AUTOTRAINER_EXTERNAL="
set "PYTHONPATH=%ROOT%"
call "%ROOT%\tools\launch_autotrainer.bat" %PROFILE% "%USEWATCH%"
if errorlevel 1 (
  echo WARNING: AutoTrainer launch failed -- bot will try self-spawn.
  set "GIGA_AUTOTRAINER_EXTERNAL="
) else (
  set "GIGA_AUTOTRAINER_EXTERNAL=1"
)

start "GigaLearnBot" /D "%USEDIR%" "%USEEXE%"
if /I "%2"=="dashboard" (
  if exist "%USEDIR%\AutoTrainerDashboard.exe" (
    start "AutoTrainer-Dashboard" /D "%USEDIR%" "%USEDIR%\AutoTrainerDashboard.exe" --watch-dir "%USEWATCH%"
  ) else (
    start "AutoTrainer-Dashboard" /D "%ROOT%" python "%ROOT%\autotrainer\dashboard\tui.py" --watch-dir "%USEWATCH%"
  )
)

echo Started. First trainer_status.json after ~1 iteration.
echo AutoTrainer disable: GIGA_NO_AUTOTRAINER=1 or --no-autotrainer
echo SSL autonomy disable: GIGA_NO_SSL_AUTONOMY=1
exit /b 0
