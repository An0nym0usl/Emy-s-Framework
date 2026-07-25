@echo off
REM Start AutoTrainer orchestrator in a VISIBLE console (non-blocking).
REM Usage: launch_autotrainer.bat [PROFILE] [WATCH_DIR]
REM Opt-out (caller): set GIGA_NO_AUTOTRAINER=1 before calling.
REM
REM Sets GIGA_AUTOTRAINER_EXTERNAL=1 so ExampleMain does not spawn a second brain.
REM Exit 1 if the child dies on startup (so caller can clear EXTERNAL / let exe self-spawn).
REM
REM Console shows live banner + poll# lines. Same lines also append to
REM   %WATCH%\autotrainer_launch.log
REM (older builds redirected stdout only to the log -> blank AutoTrainer window).

setlocal EnableExtensions
if /I "%GIGA_NO_AUTOTRAINER%"=="1" (
  echo AutoTrainer: skipped ^(GIGA_NO_AUTOTRAINER=1^)
  endlocal & set "GIGA_AUTOTRAINER_EXTERNAL=1" & exit /b 0
)

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
if "%1"=="" (set "PROFILE=default") else (set "PROFILE=%~1")
if "%2"=="" (
  set "WATCH=%ROOT%\build\Release\autotrainer"
) else (
  set "WATCH=%~2"
)

if not exist "%ROOT%\autotrainer\orchestrator.py" (
  echo AutoTrainer: orchestrator.py not found under %ROOT%\autotrainer
  endlocal & exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo AutoTrainer: python not on PATH -- skipped
  endlocal & exit /b 1
)

if not exist "%WATCH%" mkdir "%WATCH%"
if not exist "%WATCH%\profiles" mkdir "%WATCH%\profiles"
xcopy /Y /E /I "%ROOT%\autotrainer\profiles\*" "%WATCH%\profiles\" >nul 2>&1
copy /Y "%ROOT%\autotrainer\config.default.yaml" "%WATCH%\" >nul 2>&1

set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
set "LOG=%WATCH%\autotrainer_launch.log"

REM SSL full autonomy ON by default for from-scratch; opt out with GIGA_NO_SSL_AUTONOMY=1
if /I "%GIGA_NO_SSL_AUTONOMY%"=="1" (
  echo AutoTrainer: SSL autonomy OFF ^(GIGA_NO_SSL_AUTONOMY=1^)
) else (
  if /I "%GIGA_FROM_SCRATCH%"=="1" set "GIGA_SSL_AUTONOMY=1"
  if not defined GIGA_SSL_AUTONOMY set "GIGA_SSL_AUTONOMY=1"
)

REM Avoid stacking many brains titled AutoTrainer.
taskkill /FI "WINDOWTITLE eq AutoTrainer*" /F >nul 2>&1

REM UTF-8 + unbuffered so cp1252 consoles cannot kill the brain on banner/heartbeats.
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

echo ===== AutoTrainer launch %DATE% %TIME% profile=%PROFILE% =====>> "%LOG%"

REM Visible console: tee to log via run_autotrainer_console.py (NOT cmd >> log alone).
REM /K keeps the window open if Python exits so crashes are readable.
start "AutoTrainer" /D "%ROOT%" cmd /k "set PYTHONUNBUFFERED=1&& set PYTHONIOENCODING=utf-8&& set PYTHONPATH=%ROOT%;%PYTHONPATH%&& python -u ""%ROOT%\tools\run_autotrainer_console.py"" ""%ROOT%"" %PROFILE% ""%WATCH%"""

REM Brief settle then verify the child did not die on import/banner (UnicodeEncodeError etc.).
timeout /t 3 /nobreak >nul 2>&1

set "AT_DEAD="
powershell -NoProfile -Command ^
  "$log='%LOG%'; if (-not (Test-Path -LiteralPath $log)) { exit 0 };" ^
  "$t = Get-Content -LiteralPath $log -Raw -ErrorAction SilentlyContinue;" ^
  "if ([string]::IsNullOrEmpty($t)) { exit 0 };" ^
  "$marker = '===== AutoTrainer launch';" ^
  "$i = $t.LastIndexOf($marker);" ^
  "if ($i -lt 0) { exit 0 };" ^
  "$tail = $t.Substring($i);" ^
  "if ($tail -match 'Traceback \(most recent call last\)') { exit 1 };" ^
  "if ($tail -match 'UnicodeEncodeError') { exit 1 };" ^
  "exit 0"
if errorlevel 1 set "AT_DEAD=1"

if defined AT_DEAD (
  echo AutoTrainer: CRASHED on startup -- see %LOG%
  echo   ^(common: UnicodeEncodeError on Windows cp1252; train will try exe self-spawn^)
  endlocal & exit /b 1
)

set "AT_PID=?"
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "try { (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'orchestrator\.py|run_autotrainer_console' } | Sort-Object CreationDate -Descending | Select-Object -First 1).ProcessId } catch { '' }"`) do set "AT_PID=%%P"
if not defined AT_PID set "AT_PID=?"

echo AutoTrainer started ^(visible console + log^)
echo   profile=%PROFILE%
echo   watch=%WATCH%
echo   PID~%AT_PID%
echo   log=%LOG%
echo   Disable: set GIGA_NO_AUTOTRAINER=1  or  GigaLearnBot.exe --no-autotrainer

endlocal & set "GIGA_AUTOTRAINER_EXTERNAL=1" & exit /b 0
