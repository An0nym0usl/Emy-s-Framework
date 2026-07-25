#!/usr/bin/env bash
# Best-effort WSL2 / Linux ROCm setup + linux-rocm build for GigaLearnRL.
# Safe to re-run. Does not silently wipe data. See docs/AMD_WSL2.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== GigaLearnRL — WSL2 / Linux ROCm setup ==="
echo "Root: $ROOT"
echo

in_wsl=0
if grep -qi microsoft /proc/version 2>/dev/null; then
  in_wsl=1
  echo "[OK] Running under WSL"
else
  echo "[INFO] Not WSL — treating as native Linux ROCm host (same build path)."
fi

# Discourage building from /mnt/c (OneDrive/Desktop) for SPS/build time.
case "$ROOT" in
  /mnt/c/*|/mnt/d/*|/mnt/e/*)
    echo "[WARN] Repo is on a Windows mount ($ROOT)."
    echo "       Copy to ~/src/GigaLearnRL for much faster builds/trains."
    echo "       Example: mkdir -p ~/src && cp -a \"$ROOT\" ~/src/GigaLearnRL"
    ;;
esac

echo
echo "--- Toolchain ---"
need_pkg=0
for c in cmake ninja git python3; do
  if command -v "$c" >/dev/null 2>&1; then
    echo "[OK] $c"
  else
    echo "[MISS] $c"
    need_pkg=1
  fi
done
if [[ "$need_pkg" -eq 1 ]]; then
  echo "Install: sudo apt update && sudo apt install -y build-essential cmake ninja-build git python3 python3-pip"
fi

echo
echo "--- HIP / ROCm ---"
if command -v hipcc >/dev/null 2>&1; then
  echo "[OK] hipcc: $(command -v hipcc)"
elif [[ -x /opt/rocm/bin/hipcc ]]; then
  export PATH="/opt/rocm/bin:${PATH}"
  echo "[OK] hipcc via /opt/rocm/bin"
else
  echo "[MISS] hipcc — install ROCm for this Ubuntu (see https://rocm.docs.amd.com/)."
  echo "       RX 6600 XT (gfx1030) is often unofficial; you may need:"
  echo "         export HSA_OVERRIDE_GFX_VERSION=10.3.0"
  echo "         export AMDGPU_TARGETS=gfx1030"
fi

# RDNA2 consumer defaults (6600 XT class)
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1030}"
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "AMDGPU_TARGETS=$AMDGPU_TARGETS"

echo
echo "--- LibTorch (C++ PPO — required) ---"
if [[ -d "$ROOT/GigaLearnCPP/libtorch" ]]; then
  echo "[OK] GigaLearnCPP/libtorch present"
elif [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  echo "[OK] CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
else
  echo "[WARN] No GigaLearnCPP/libtorch and CMAKE_PREFIX_PATH unset."
  echo "       Download ROCm LibTorch and either extract to GigaLearnCPP/libtorch"
  echo "       or: export CMAKE_PREFIX_PATH=\$HOME/libtorch-rocm"
  echo "       Python torch-directml / pip torch does NOT replace C++ LibTorch."
fi

echo
echo "Docs: docs/AMD_WSL2.md  |  docs/AMD.md"
echo

if [[ "${GIGA_WSL_SKIP_BUILD:-}" == "1" ]]; then
  echo "[SKIP] build (GIGA_WSL_SKIP_BUILD=1)"
  exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[FAIL] cmake missing — cannot build yet."
  exit 1
fi

echo "=== Building linux-rocm ==="
chmod +x "$ROOT/tools/build_amd.sh" 2>/dev/null || true
if [[ -x "$ROOT/tools/build_amd.sh" ]]; then
  "$ROOT/tools/build_amd.sh"
else
  export GIGA_GPU_BACKEND=hip
  cmake --preset linux-rocm
  cmake --build --preset linux-rocm -j"$(nproc 2>/dev/null || echo 4)"
fi

echo
echo "[OK] Build finished (or stopped on error above)."
echo "Next:"
echo "  python3 tools/hw_probe.py"
echo "  cd build-rocm && export HSA_OVERRIDE_GFX_VERSION=10.3.0"
echo "  # ensure collision_meshes next to binary, then:"
echo "  ./GigaLearnBot --from-scratch"
echo
if [[ "$in_wsl" -eq 1 ]]; then
  echo "Reminder: max SPS = HIP env + ROCm LibTorch PPO. CPU LibTorch = Learn bottleneck."
fi
