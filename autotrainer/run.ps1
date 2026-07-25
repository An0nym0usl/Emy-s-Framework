#Requires -Version 5.1
<#
.SYNOPSIS
  Start GigaLearnBot + AutoTrainer brain (+ optional dashboard).

.EXAMPLE
  cd GigaLearnRL
  .\autotrainer\run.ps1 -Profile default -Dashboard
#>
param(
    [string]$ExePath = "",
    [string]$Profile = "default",
    [switch]$Render,
    [switch]$LLM,
    [switch]$Dashboard,
    [switch]$Tui,
    [switch]$TrainOnly,
    [switch]$BrainOnly
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AutoTrainerSrc = Join-Path $Root "autotrainer"

if (-not $ExePath) {
    $candidates = @(
        (Join-Path $Root "build\Release\GigaLearnBot.exe"),
        (Join-Path $Root "build\GigaLearnBot.exe")
    )
    if ($env:GIGA_RELEASE_DIR) {
        $candidates = @((Join-Path $env:GIGA_RELEASE_DIR "GigaLearnBot.exe")) + $candidates
    }
    if ($env:GIGA_CUDA_MIRROR) {
        $candidates += (Join-Path $env:GIGA_CUDA_MIRROR "build\Release\GigaLearnBot.exe")
    }
    foreach ($c in $candidates) {
        if (Test-Path $c) { $ExePath = (Resolve-Path $c).Path; break }
    }
}
if (-not (Test-Path $ExePath)) {
    Write-Error "GigaLearnBot.exe not found. Build first, run SETUP_FIRST_RUN.bat, or pass -ExePath."
}

$ExeDir = Split-Path -Parent $ExePath
# AutoTrainerBridge writes to {exeDir}/autotrainer (checkpointFolder.parent / autotrainer)
$WatchDir = (Join-Path $ExeDir "autotrainer")

# RocketSim needs collision_meshes next to the exe
$meshNextToExe = Join-Path $ExeDir "collision_meshes"
if (-not (Test-Path $meshNextToExe)) {
    $meshCandidates = @(
        (Join-Path $Root "collision_meshes")
    )
    if ($env:GIGA_CUDA_MIRROR) {
        $meshCandidates += (Join-Path $env:GIGA_CUDA_MIRROR "collision_meshes")
    }
    foreach ($src in $meshCandidates) {
        if (Test-Path $src) {
            cmd /c "mklink /J `"$meshNextToExe`" `"$src`"" | Out-Null
            Write-Host "Linked collision_meshes -> $src"
            break
        }
    }
    if (-not (Test-Path $meshNextToExe)) {
        Write-Error "collision_meshes not found. Place RocketSim meshes in $Root\collision_meshes"
    }
}

# Sync profiles + config next to exe
if (Test-Path $AutoTrainerSrc) {
    New-Item -ItemType Directory -Force -Path $WatchDir | Out-Null
    $profilesDst = Join-Path $WatchDir "profiles"
    New-Item -ItemType Directory -Force -Path $profilesDst | Out-Null
    Copy-Item -Recurse -Force (Join-Path $AutoTrainerSrc "profiles\*") $profilesDst -ErrorAction SilentlyContinue
    Copy-Item -Force (Join-Path $AutoTrainerSrc "config.default.yaml") (Join-Path $WatchDir "config.default.yaml") -ErrorAction SilentlyContinue
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $python) { Write-Error "Python not found on PATH." }

# pip writes notices to stderr — must not use $ErrorActionPreference Stop here
$pipOk = $false
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $python.Source -c "import autotrainer" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $pipOk = $true }
} catch { }
if (-not $pipOk) {
    Write-Host "Installing Python deps (pip install -e .)..."
    & $python.Source -m pip install -q -e $Root 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        & $python.Source -m pip install -q -r (Join-Path $AutoTrainerSrc "requirements.txt") 2>&1 | Out-Null
    }
}
$ErrorActionPreference = $prevEAP

$orchestrator = Join-Path $AutoTrainerSrc "orchestrator.py"
$brainArgs = @($orchestrator, "--profile", $Profile, "--watch-dir", $WatchDir)
if ($LLM) { $brainArgs += "--llm" }

Write-Host ""
Write-Host "AutoTrainer watch dir: $WatchDir"
Write-Host "Exe: $ExePath"
Write-Host ""

if (-not $BrainOnly) {
    if ($Render) {
        $trainJob = Start-Process -FilePath $ExePath -ArgumentList @("--render") -WorkingDirectory $ExeDir -PassThru
    } else {
        $trainJob = Start-Process -FilePath $ExePath -WorkingDirectory $ExeDir -PassThru
    }
    Write-Host "Training started (PID $($trainJob.Id)). First status appears after ~1 iteration."
}

if (-not $TrainOnly) {
    $env:PYTHONPATH = $Root
    Start-Process -FilePath $python.Source -ArgumentList $brainArgs -WorkingDirectory $Root -NoNewWindow
    Write-Host "AutoTrainer brain started (orchestrator.py)."
}

if ($Dashboard) {
    Write-Host "Dashboard companion removed from blank handoff — ignore -Dashboard / use bot+AT console metrics."
}

if (-not $BrainOnly) {
    Write-Host ""
    Write-Host "Press Enter to stop training (brain keeps running in this window)..."
    Read-Host | Out-Null
    if ($trainJob -and -not $trainJob.HasExited) {
        Stop-Process -Id $trainJob.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Training stopped."
    }
}
