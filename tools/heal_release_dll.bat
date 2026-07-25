@echo off
REM Heal stale/mismatched GigaLearnCPP.dll (+ RocketSimCuda.dll + optional exe) from CUDA mirror.
REM Usage: heal_release_dll.bat <ReleaseDir>
REM Good CUDA builds are ~5MB; broken/partial sync copies were often ~1.4MB.
REM Copies when: missing, size < 3MB, size differs, OR mirror is newer (mtime).
REM Exit 0 = DLL present and >= 3MB; exit 1 = still broken/missing.
REM Never touches libtorch / checkpoints.
REM Mirror is OPTIONAL: set GIGA_CUDA_MIRROR=<repo root> or GIGA_HEAL_MIRROR=<Release dir>.
REM Without a mirror, only validates the local DLL size (local check).

setlocal EnableExtensions EnableDelayedExpansion
set "DIR=%~1"
if "%DIR%"=="" (
  echo heal_release_dll: missing Release dir
  exit /b 1
)
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
set "MIRROR="
if defined GIGA_HEAL_MIRROR set "MIRROR=%GIGA_HEAL_MIRROR%"
if not defined MIRROR if defined GIGA_CUDA_MIRROR set "MIRROR=%GIGA_CUDA_MIRROR%\build\Release"
if not defined MIRROR if exist "C:\GigaLearnRL\build\Release\GigaLearnCPP.dll" set "MIRROR=C:\GigaLearnRL\build\Release"

if not exist "%DIR%\GigaLearnBot.exe" (
  endlocal & exit /b 1
)

REM No mirror configured / present: only validate local DLL (local-only check).
if not defined MIRROR (
  if exist "%DIR%\GigaLearnCPP.dll" (
    for %%A in ("%DIR%\GigaLearnCPP.dll") do (
      if %%~zA LSS 3000000 (endlocal & exit /b 1)
    )
    endlocal & exit /b 0
  )
  endlocal & exit /b 1
)

if not exist "%MIRROR%\GigaLearnCPP.dll" (
  if exist "%DIR%\GigaLearnCPP.dll" (
    for %%A in ("%DIR%\GigaLearnCPP.dll") do (
      if %%~zA LSS 3000000 (endlocal & exit /b 1)
    )
    endlocal & exit /b 0
  )
  endlocal & exit /b 1
)

set "NEED_COPY="
if not exist "%DIR%\GigaLearnCPP.dll" set "NEED_COPY=1"
if not defined NEED_COPY for %%A in ("%DIR%\GigaLearnCPP.dll") do if %%~zA LSS 3000000 set "NEED_COPY=1"
if not defined NEED_COPY (
  for %%A in ("%DIR%\GigaLearnCPP.dll") do for %%B in ("%MIRROR%\GigaLearnCPP.dll") do if %%~zA NEQ %%~zB set "NEED_COPY=1"
)

REM Same size but newer mirror rebuild - still heal (locale-safe mtime via PowerShell).
if not defined NEED_COPY (
  where powershell >nul 2>&1
  if not errorlevel 1 (
    powershell -NoProfile -Command ^
      "if ((Get-Item -LiteralPath '%MIRROR%\GigaLearnCPP.dll').LastWriteTime -gt (Get-Item -LiteralPath '%DIR%\GigaLearnCPP.dll').LastWriteTime) { exit 2 } else { exit 0 }"
    if errorlevel 2 if not errorlevel 3 set "NEED_COPY=1"
  )
)

if defined NEED_COPY (
  echo Healing GigaLearnCPP.dll from CUDA mirror:
  echo   %MIRROR%\GigaLearnCPP.dll  -^>  %DIR%\
  copy /Y "%MIRROR%\GigaLearnCPP.dll" "%DIR%\GigaLearnCPP.dll" >nul
  if errorlevel 1 (
    echo WARNING: could not copy GigaLearnCPP.dll into %DIR%
    endlocal & exit /b 1
  )
  if exist "%MIRROR%\GigaLearnCPP.lib" copy /Y "%MIRROR%\GigaLearnCPP.lib" "%DIR%\GigaLearnCPP.lib" >nul 2>&1
  if exist "%MIRROR%\GigaLearnCPP.exp" copy /Y "%MIRROR%\GigaLearnCPP.exp" "%DIR%\GigaLearnCPP.exp" >nul 2>&1
  REM Also heal exe when sizes diverge (stale exe + fresh DLL).
  if exist "%MIRROR%\GigaLearnBot.exe" if /I not "%DIR%"=="%MIRROR%" (
    for %%A in ("%DIR%\GigaLearnBot.exe") do for %%B in ("%MIRROR%\GigaLearnBot.exe") do if %%~zA NEQ %%~zB (
      echo Healing GigaLearnBot.exe from CUDA mirror...
      copy /Y "%MIRROR%\GigaLearnBot.exe" "%DIR%\GigaLearnBot.exe" >nul 2>&1
    )
    REM Also heal when mirror exe is newer (same size rebuild).
    where powershell >nul 2>&1
    if not errorlevel 1 (
      powershell -NoProfile -Command ^
        "if ((Get-Item -LiteralPath '%MIRROR%\GigaLearnBot.exe').LastWriteTime -gt (Get-Item -LiteralPath '%DIR%\GigaLearnBot.exe').LastWriteTime) { exit 2 } else { exit 0 }"
      if errorlevel 2 if not errorlevel 3 (
        echo Healing GigaLearnBot.exe from CUDA mirror ^(newer mtime^)...
        copy /Y "%MIRROR%\GigaLearnBot.exe" "%DIR%\GigaLearnBot.exe" >nul 2>&1
      )
    )
  )
)

REM Heal RocketSimCuda.dll when present on the mirror (CUDA sim builds).
if exist "%MIRROR%\RocketSimCuda.dll" if /I not "%DIR%"=="%MIRROR%" (
  set "NEED_CUDA="
  if not exist "%DIR%\RocketSimCuda.dll" set "NEED_CUDA=1"
  if not defined NEED_CUDA (
    for %%A in ("%DIR%\RocketSimCuda.dll") do for %%B in ("%MIRROR%\RocketSimCuda.dll") do if %%~zA NEQ %%~zB set "NEED_CUDA=1"
  )
  if not defined NEED_CUDA (
    where powershell >nul 2>&1
    if not errorlevel 1 (
      powershell -NoProfile -Command ^
        "if ((Get-Item -LiteralPath '%MIRROR%\RocketSimCuda.dll').LastWriteTime -gt (Get-Item -LiteralPath '%DIR%\RocketSimCuda.dll').LastWriteTime) { exit 2 } else { exit 0 }"
      if errorlevel 2 if not errorlevel 3 set "NEED_CUDA=1"
    )
  )
  if defined NEED_CUDA (
    echo Healing RocketSimCuda.dll from CUDA mirror...
    copy /Y "%MIRROR%\RocketSimCuda.dll" "%DIR%\RocketSimCuda.dll" >nul 2>&1
  )
)

if not exist "%DIR%\GigaLearnCPP.dll" (endlocal & exit /b 1)
for %%A in ("%DIR%\GigaLearnCPP.dll") do if %%~zA LSS 3000000 (endlocal & exit /b 1)
endlocal & exit /b 0
