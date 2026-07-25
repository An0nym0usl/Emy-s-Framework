@echo off
REM =============================================================================
REM Resolve cmake.exe for Windows builds.
REM Usage (from other bats):
REM   call "%~dp0find_cmake.bat"
REM   if errorlevel 1 exit /b 1
REM   "%CMAKE_EXE%" --version
REM
REM Override: set CMAKE_EXE to a full path before calling.
REM On success: CMAKE_EXE is set and errorlevel is 0.
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion

if defined CMAKE_EXE (
  if exist "!CMAKE_EXE!" (
    set "FOUND=!CMAKE_EXE!"
    goto :have_cmake
  )
  echo [WARN] CMAKE_EXE is set but not found: !CMAKE_EXE!
)

REM 1) PATH (where.exe)
where cmake >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%I in ('where cmake 2^>nul') do (
    if exist "%%~fI" (
      set "FOUND=%%~fI"
      goto :have_cmake
    )
  )
)

REM 2) Common Kitware / Program Files installs
if exist "%ProgramFiles%\CMake\bin\cmake.exe" (
  set "FOUND=%ProgramFiles%\CMake\bin\cmake.exe"
  goto :have_cmake
)
if exist "%ProgramFiles(x86)%\CMake\bin\cmake.exe" (
  set "FOUND=%ProgramFiles(x86)%\CMake\bin\cmake.exe"
  goto :have_cmake
)
if exist "%LocalAppData%\Programs\CMake\bin\cmake.exe" (
  set "FOUND=%LocalAppData%\Programs\CMake\bin\cmake.exe"
  goto :have_cmake
)
if exist "C:\CMake\bin\cmake.exe" (
  set "FOUND=C:\CMake\bin\cmake.exe"
  goto :have_cmake
)

REM 3) Visual Studio bundled CMake (2022 / 2019)
REM PF86 avoids "(x86)" breaking parenthesized for-blocks.
set "PF86=%ProgramFiles(x86)%"
for %%V in (2022 2019) do (
  for %%E in (Community Professional Enterprise BuildTools Preview) do (
    if exist "!ProgramFiles!\Microsoft Visual Studio\%%V\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
      set "FOUND=!ProgramFiles!\Microsoft Visual Studio\%%V\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
      goto :have_cmake
    )
    if exist "!PF86!\Microsoft Visual Studio\%%V\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
      set "FOUND=!PF86!\Microsoft Visual Studio\%%V\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
      goto :have_cmake
    )
  )
)

echo.
echo [FAIL] cmake not in PATH and not found in common install folders.
echo.
echo Install CMake ^(3.21+^), then open a NEW terminal:
echo.
echo   Option A - Visual Studio 2022
echo     1. Install "Desktop development with C++"
echo     2. In Individual components, enable "CMake tools for Windows"
echo     3. Or install Kitware CMake and tick "Add CMake to the system PATH"
echo.
echo   Option B - Kitware installer
echo     https://cmake.org/download/
echo     Windows x64 Installer -^> "Add CMake to the system PATH for all users"
echo     ^(or add ...\CMake\bin to PATH manually^)
echo.
echo   Verify:
echo     where cmake
echo     cmake --version
echo.
echo   If cmake.exe exists but PATH is wrong, set a full path and retry:
echo     set CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe
echo     tools\build_cuda.bat
echo.
echo Docs: BUILD.md  ^(section: cmake not in PATH^)
echo.
endlocal
exit /b 1

:have_cmake
echo [OK]   cmake: !FOUND!
for %%I in ("!FOUND!") do (
  endlocal
  set "CMAKE_EXE=%%~fI"
  exit /b 0
)
