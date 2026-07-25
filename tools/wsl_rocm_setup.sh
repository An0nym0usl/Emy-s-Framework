#!/usr/bin/env bash
# Backward-compatible alias — prefer tools/setup_wsl2_rocm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/tools/setup_wsl2_rocm.sh" "$@"
