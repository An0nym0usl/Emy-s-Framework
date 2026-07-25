@echo off
REM FROM-SCRATCH train WITH AutoTrainer (explicit opt-in).
REM Default path run_fresh_train.bat leaves AutoTrainer OFF.
REM
REM Same as:
REM   set GIGA_AUTOTRAINER=1
REM   run_fresh_train.bat
REM
REM Resume bot+AT: start_autotrainer.bat

setlocal EnableExtensions
set "GIGA_AUTOTRAINER=1"
set "GIGA_NO_AUTOTRAINER="
call "%~dp0run_fresh_train.bat" %*
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
