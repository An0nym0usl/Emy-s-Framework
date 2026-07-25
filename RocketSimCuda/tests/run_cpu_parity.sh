#!/usr/bin/env bash
# ===========================================================================
# run_cpu_parity.sh — reproducible GPU-less CPU<->CUDA parity report
# ---------------------------------------------------------------------------
# OPT-105 / OPT-106. Configures + builds the RocketSimCudaCpuParity target with
# the host C++ compiler (NO CUDA toolkit, NO GPU, NOT nvcc), runs it, and writes
# a human-readable (.txt) AND machine-readable (.csv/.json) per-scenario
# pass/fail report. The harness exit code is preserved as this script's exit
# code (0 = OVERALL PASS) so it gates CI.
#
# One-line usage (from repo root):
#   bash RocketSimCuda/tests/run_cpu_parity.sh
#
# Reports are written to a committed, byte-stable location by default:
#   RocketSimCuda/tests/reports/cpu_parity_report.{txt,csv,json}
#
# Env overrides:
#   BUILD_DIR  build directory     (default: <RocketSimCuda>/build/cpu_parity)
#   REPORT_DIR report directory    (default: <RocketSimCuda>/tests/reports)
#   CMAKE      cmake binary        (default: autodetected, incl. ~/.local/bin)
# ===========================================================================
set -euo pipefail

RSCUDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$RSCUDA_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$RSCUDA_DIR/build/cpu_parity}"
REPORT_DIR="${REPORT_DIR:-$RSCUDA_DIR/tests/reports}"
REPORT_TXT="$REPORT_DIR/cpu_parity_report.txt"
REPORT_CSV="$REPORT_DIR/cpu_parity_report.csv"
REPORT_JSON="$REPORT_DIR/cpu_parity_report.json"

# --- locate cmake (system PATH, then pip --user install) -------------------
if [[ -n "${CMAKE:-}" ]]; then
    :
elif command -v cmake >/dev/null 2>&1; then
    CMAKE="cmake"
elif [[ -x "$HOME/.local/bin/cmake" ]]; then
    CMAKE="$HOME/.local/bin/cmake"
else
    echo "ERROR: cmake not found. Install it (no GPU/CUDA needed), e.g.:" >&2
    echo "  pip3 install --user --break-system-packages cmake   # or: apt-get install cmake" >&2
    exit 127
fi
echo "Using cmake: $($CMAKE --version | head -1)"

# --- configure + build the host-only parity target (no CUDA, no GPU) -------
"$CMAKE" -S "$RSCUDA_DIR" -B "$BUILD_DIR" -DRSCUDA_BUILD_CPU_PARITY=ON
"$CMAKE" --build "$BUILD_DIR"

# --- run, capture human-readable report, derive machine-readable report ----
mkdir -p "$REPORT_DIR"
set +e
"$BUILD_DIR/RocketSimCudaCpuParity" | tee "$REPORT_TXT"
RC="${PIPESTATUS[0]}"
set -e

python3 "$REPO_ROOT/tools/parity_report.py" \
    --csv "$REPORT_CSV" --json "$REPORT_JSON" < "$REPORT_TXT" || true

echo
echo "Reports written:"
echo "  human-readable : $REPORT_TXT"
echo "  machine (csv)  : $REPORT_CSV"
echo "  machine (json) : $REPORT_JSON"
echo "Harness exit code: $RC"
exit "$RC"
