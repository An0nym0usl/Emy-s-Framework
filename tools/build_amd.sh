#!/usr/bin/env bash
# Build GigaLearnRL with AMD ROCm / HIP (Linux). See docs/AMD.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== GigaLearnRL Linux ROCm / HIP build ==="
echo "Root: $ROOT"

if ! command -v cmake >/dev/null; then
  echo "[FAIL] cmake not found"; exit 1
fi
if ! command -v hipcc >/dev/null && [[ ! -d /opt/rocm ]]; then
  echo "[WARN] hipcc /opt/rocm not found — install ROCm first"
fi

export GIGA_GPU_BACKEND=hip
cmake --preset linux-rocm
cmake --build --preset linux-rocm -j"$(nproc 2>/dev/null || echo 4)"

echo "[OK] ROCm build done. Run: python tools/hw_probe.py"
