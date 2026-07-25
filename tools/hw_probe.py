#!/usr/bin/env python3
"""Probe CPU / RAM / GPU and write adaptive train knobs for GigaLearnRL.

Usage:
  python tools/hw_probe.py
  python tools/hw_probe.py --out build/Release/hw_profile.json
  python tools/hw_probe.py --print-env          # bat-friendly KEY=VALUE lines
  python tools/hw_probe.py --simulate-amd       # dry-run AMD recommendations
  python tools/hw_probe.py --force-cpu          # recommend CPU sim even on GPU

Env opt-out / override (read by ExampleMain / bats):
  GIGA_NO_HW_PROFILE=1     ignore profile
  GIGA_FORCE_CPU=1         force CPU RocketSim
  GIGA_ENV_ARENAS=N        override arenas (wins over profile)
  GIGA_HW_PROFILE=path     custom profile path
  GIGA_HIP_FORCE=1         treat AMD as HIP-ready even if SDK probe is soft
  GIGA_TORCH_THREADS=N     LibTorch CPU intra-op threads (Win HIP + CPU PPO)
  GIGA_TORCH_DIRECTML=1    acknowledged; C++ PPO still CPU (no libtorch DirectML)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return ""


def _cpu_info() -> dict[str, Any]:
    brand = platform.processor() or ""
    cores = os.cpu_count() or 1
    threads = cores
    if sys.platform == "win32":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 "
                    "@{N='Name';E={$_.Name}},@{N='Cores';E={$_.NumberOfCores}},"
                    "@{N='Threads';E={$_.NumberOfLogicalProcessors}}) | "
                    "ConvertTo-Json -Compress"
                ),
            ]
        )
        try:
            j = json.loads(out.strip().splitlines()[-1])
            brand = (j.get("Name") or brand).strip()
            cores = int(j.get("Cores") or cores)
            threads = int(j.get("Threads") or threads)
        except Exception:
            pass
    return {"brand": brand, "cores": cores, "threads": threads}


def _ram_gb() -> dict[str, float]:
    total = 0.0
    avail = 0.0
    if sys.platform == "win32":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "$o=Get-CimInstance Win32_OperatingSystem; "
                    "[pscustomobject]@{TotalGB=[math]::Round($o.TotalVisibleMemorySize/1MB,2); "
                    "FreeGB=[math]::Round($o.FreePhysicalMemory/1MB,2)} | ConvertTo-Json -Compress"
                ),
            ]
        )
        try:
            j = json.loads(out.strip().splitlines()[-1])
            total = float(j.get("TotalGB") or 0)
            avail = float(j.get("FreeGB") or 0)
        except Exception:
            pass
    if total <= 0:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                total = round(m.ullTotalPhys / (1024**3), 2)
                avail = round(m.ullAvailPhys / (1024**3), 2)
        except Exception:
            pass
    return {"total_gb": total, "available_gb": avail}


def _nvidia_gpus() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        name, mem = [x.strip() for x in line.split(",", 1)]
        try:
            vram = round(float(mem) / 1024.0, 2)  # MiB -> GiB
        except ValueError:
            vram = 0.0
        gpus.append(
            {
                "vendor": "nvidia",
                "name": name,
                "vram_gb": vram,
                "cuda_capable": True,
                "hip_capable": False,
            }
        )
    return gpus


def _wmi_gpus() -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
            ),
        ]
    )
    text = out.strip()
    if not text:
        return []
    try:
        data = json.loads(text.splitlines()[-1])
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    gpus: list[dict[str, Any]] = []
    for item in data or []:
        name = (item.get("Name") or "").strip()
        if not name:
            continue
        low = name.lower()
        if "microsoft" in low and "basic" in low:
            continue
        vendor = "other"
        if any(x in low for x in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
            vendor = "nvidia"
        elif any(x in low for x in ("amd", "radeon", "rx ", "vega", "instinct")):
            vendor = "amd"
        elif any(x in low for x in ("intel", "arc ", "uhd", "iris")):
            vendor = "intel"
        vram = 0.0
        try:
            ram = int(item.get("AdapterRAM") or 0)
            if ram > 0:
                vram = round(ram / (1024**3), 2)
        except Exception:
            pass
        gpus.append(
            {
                "vendor": vendor,
                "name": name,
                "vram_gb": vram,
                "cuda_capable": vendor == "nvidia",
                "hip_capable": vendor == "amd",
            }
        )
    return gpus


def _hip_runtime_present() -> dict[str, Any]:
    """Detect AMD HIP / ROCm toolchain or runtime (Windows HIP SDK or Linux ROCm)."""
    info: dict[str, Any] = {
        "present": False,
        "hipcc": False,
        "rocminfo": False,
        "paths": [],
        "platform": "none",
    }
    if os.environ.get("GIGA_HIP_FORCE", "").strip() in ("1", "true", "TRUE", "yes"):
        info["present"] = True
        info["platform"] = "forced"
        return info

    if shutil.which("hipcc"):
        info["hipcc"] = True
        info["present"] = True
    if shutil.which("rocminfo") or shutil.which("amd-smi"):
        info["rocminfo"] = True
        info["present"] = True

    candidates = []
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\AMD\ROCm",
            r"C:\Program Files\AMD\HIP",
            os.environ.get("HIP_PATH", ""),
            os.environ.get("ROCM_PATH", ""),
        ]
        # amdhip64.dll on PATH / System32 is enough to *run* a HIP build.
        windir = os.environ.get("WINDIR", r"C:\Windows")
        for dll in (
            os.path.join(windir, "System32", "amdhip64.dll"),
            os.path.join(windir, "System32", "amdhip64_6.dll"),
        ):
            if dll and os.path.isfile(dll):
                info["present"] = True
                info["paths"].append(dll)
                info["platform"] = "windows-hip-runtime"
    else:
        candidates = [
            "/opt/rocm",
            os.environ.get("ROCM_PATH", ""),
            os.environ.get("HIP_PATH", ""),
        ]

    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.is_dir():
            info["paths"].append(str(p))
            info["present"] = True
            if sys.platform == "win32":
                info["platform"] = info.get("platform") or "windows-hip-sdk"
            else:
                info["platform"] = "linux-rocm"

    # Binary already built with HIP? Marker from build scripts / bats.
    if os.environ.get("GIGA_GPU_BACKEND", "").strip().lower() == "hip":
        info["present"] = True
        info["platform"] = info.get("platform") or "env-hip"

    return info


def _torch_gpu_hint() -> str:
    """Best-effort: does this Python see a GPU torch device? (ROCm uses torch.cuda)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"  # CUDA or ROCm HIP via PyTorch cuda API
    except Exception:
        pass
    return "auto"


def _directml_python_available() -> bool:
    """True if torch_directml imports (Python only — not used by C++ Learner)."""
    if sys.platform != "win32":
        return False
    # Explicit off
    if os.environ.get("GIGA_TORCH_DIRECTML", "").strip() in ("0", "false", "FALSE", "no"):
        return False
    try:
        import torch_directml  # type: ignore  # noqa: F401

        return True
    except Exception:
        pass
    # Soft: package name present without import side effects
    try:
        import importlib.util

        return importlib.util.find_spec("torch_directml") is not None
    except Exception:
        return False


def _recommend_torch_threads(cpu: dict[str, Any], hip_gpu_sim: bool) -> int:
    """CPU PPO thread count for C++ libtorch (Win AMD path). Ryzen 5 3600 → ~10.

    Matches Learner::ConfigureCpuTorchThreads: leave ~2 logical cores for OS/HIP host.
    hip_gpu_sim is informational (physics already on GPU); keep same reserve.
    """
    del hip_gpu_sim  # reserved for future tuning
    threads = int(cpu.get("threads") or os.cpu_count() or 4)
    if threads < 1:
        threads = 4
    reserve = 2 if threads >= 8 else 1
    return max(1, threads - reserve)


def detect_gpus(simulate_amd: bool = False) -> list[dict[str, Any]]:
    if simulate_amd:
        # Reference: RX 6600 XT 8 GB (docs/AMD.md).
        return [
            {
                "vendor": "amd",
                "name": "AMD Radeon RX 6600 XT (simulated)",
                "vram_gb": 8.0,
                "cuda_capable": False,
                "hip_capable": True,
            }
        ]
    nvidia = _nvidia_gpus()
    wmi = _wmi_gpus()
    by_name = {g["name"].lower(): g for g in nvidia}
    for g in wmi:
        key = g["name"].lower()
        if key in by_name:
            if by_name[key].get("vram_gb", 0) <= 0 and g.get("vram_gb", 0) > 0:
                by_name[key]["vram_gb"] = g["vram_gb"]
            continue
        if g["vendor"] == "nvidia" and nvidia:
            continue
        by_name[key] = g
    return list(by_name.values())


def _arena_ladder_gpu(vram: float, ram_total: float) -> int:
    """NVIDIA CUDA ladder (VRAM is usually accurate via nvidia-smi)."""
    if vram >= 12 and ram_total >= 24:
        return 8192
    if vram >= 10 and ram_total >= 16:
        return 6144
    if vram >= 8 and ram_total >= 16:
        return 4096
    if vram >= 6:
        return 2048
    return 1024


def _arena_ladder_amd_hip(vram: float, ram_total: float) -> int:
    """AMD HIP ladder — amd_win_20k Overall floor on ~8 GB (RX 6600 XT class).

    Overall = 1/(1/Coll+1/Cons). On Win HIP, Infer+Learn are CPU (3600) while env
    is gpuNative on the Radeon. Blind 8192 OOMs and stalls Cons; 4096 often raises
    Collection alone while Overall drops. Default: 2048 arenas.
    """
    if vram >= 16 and ram_total >= 32:
        return 4096  # still capped — CPU PPO, not CUDA learn
    if vram >= 12 and ram_total >= 24:
        return 3072
    if vram >= 10 and ram_total >= 16:
        return 2560
    if vram >= 8 and ram_total >= 16:
        return 2048  # amd_win_20k sweet spot (6600 XT / 8 GB)
    if vram >= 6:
        return 1536
    return 1024


def _arena_ladder_cpu(cpu: dict[str, Any], ram_total: float) -> int:
    if ram_total >= 32 and (cpu.get("threads") or 1) >= 16:
        return 1024
    if ram_total >= 24:
        return 768
    if ram_total >= 16:
        return 512
    return 256


def recommend(
    cpu: dict[str, Any],
    ram: dict[str, float],
    gpus: list[dict[str, Any]],
    force_cpu: bool = False,
    hip_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hip_info = hip_info or {}
    nvidia = [g for g in gpus if g.get("vendor") == "nvidia" and g.get("cuda_capable")]
    amd = [g for g in gpus if g.get("vendor") == "amd"]
    primary = "none"
    if nvidia:
        primary = "nvidia"
    elif amd:
        primary = "amd"
    elif any(g.get("vendor") == "intel" for g in gpus):
        primary = "intel"
    elif gpus:
        primary = gpus[0].get("vendor", "other")

    ram_total = float(ram.get("total_gb") or 0)
    vram_nv = max((float(g.get("vram_gb") or 0) for g in nvidia), default=0.0)
    if vram_nv <= 0 and nvidia:
        vram_nv = 8.0
    vram_amd = max((float(g.get("vram_gb") or 0) for g in amd), default=0.0)
    if vram_amd <= 0 and amd:
        # WMI AdapterRAM is often wrong/zero on AMD. Assume ~8 GB (6600 XT class)
        # so we do not recommend blind 8192 and OOM on common mid-range cards.
        vram_amd = 8.0

    hip_ready = bool(hip_info.get("present")) or (
        os.environ.get("GIGA_HIP_FORCE", "").strip() in ("1", "true", "TRUE", "yes")
    )
    # Peak AMD path: HIP-built RocketSimCuda (gpuNative). Prefer when AMD + HIP present.
    use_cuda_sim = bool(nvidia) and not force_cpu
    use_hip_sim = bool(amd) and hip_ready and not force_cpu and not use_cuda_sim
    use_gpu_sim = use_cuda_sim or use_hip_sim
    force_cpu_sim = (not use_gpu_sim) or force_cpu

    if use_cuda_sim:
        vram = vram_nv
        arenas = _arena_ladder_gpu(vram, ram_total)
        steps = 8
        backend = "cuda"
        profile_name = "power_cuda" if arenas >= 8192 else "cuda_scaled"
        torch_device = "cuda"
        note = "NVIDIA CUDA path (RocketSimCuda + gpuNative)."
        torch_threads = 0  # GPU learn — thread knobs unused
    elif use_hip_sim:
        vram = vram_amd
        arenas = _arena_ladder_amd_hip(vram, ram_total)
		// amd_win_20k: smaller bank (steps=3 → ts≈arenas*2*3) so CPU PPO clears ≥20k Overall.
        steps = 3
        backend = "hip"
        profile_name = "amd_win_20k"
        # ROCm PyTorch exposes the HIP device as torch.cuda; Windows stock libtorch is CPU.
        torch_device = _torch_gpu_hint()
        if torch_device != "cuda":
            torch_device = "cpu"
        # On native Windows, CUDA torch is almost always NVIDIA — do not claim GPU PPO for HIP Win.
        if sys.platform == "win32" and torch_device == "cuda" and not bool(hip_info.get("rocminfo")):
            torch_device = "cpu"
        torch_threads = _recommend_torch_threads(cpu, hip_gpu_sim=True)
        dml = _directml_python_available()
        note = (
            "amd_win_20k: Overall target >=20,000 SPS on RX 6600 XT + Ryzen 5 3600. "
            "RocketSimCuda gpuNative on the Radeon; C++ PPO on CPU libtorch "
            f"(threads~{torch_threads}). Arenas={arenas} steps={steps} epochs=1 maxEp=1.5 "
            "lean512 FP32 Infer, async off. Auto-downgrade if Overall stays low. "
            "Without HIP, 20k is NOT guaranteed. docs/AMD.md / INIZIO_RAPIDO_IT.md."
        )
        if dml:
            note += " torch-directml seen in this Python (optional tools only; not C++ PPO)."
    else:
        vram = 0.0
        arenas = _arena_ladder_cpu(cpu, ram_total)
        # AMD without HIP yet: still push arenas harder than weak laptops when RAM allows.
        if amd and ram_total >= 32 and (cpu.get("threads") or 1) >= 16:
            arenas = max(arenas, 1536)
        steps = 1
        backend = "cpu"
        profile_name = "cpu_fallback"
        torch_device = "cpu"
        torch_threads = _recommend_torch_threads(cpu, hip_gpu_sim=False)
        if amd and not hip_ready:
            note = (
                "AMD GPU detected but HIP SDK not ready — CPU RocketSim for now. "
                "Install Adrenalin + AMD HIP SDK for Windows, then rebuild: "
                "tools\\build_amd.bat (docs/AMD.md). Native Windows HIP path."
            )
        else:
            note = "No CUDA/HIP GPU path — CPU RocketSim fallback."

    if ram_total > 0 and ram_total < 16:
        arenas = min(arenas, 1024 if use_gpu_sim else 256)
    if ram_total > 0 and ram_total < 8:
        arenas = min(arenas, 512 if use_gpu_sim else 128)

    ts_hint = arenas * 2 * steps
    directml_py = _directml_python_available() if (amd or use_hip_sim) else False

    env = {
        "GIGA_ENV_ARENAS": str(arenas),
        "GIGA_ENV_STEPS": str(steps),
        "GIGA_GPU_BACKEND": backend,
    }
    if force_cpu_sim:
        env["GIGA_FORCE_CPU"] = "1"
    else:
        env["GIGA_FORCE_CPU"] = "0"
    if use_hip_sim:
        env["GIGA_HIP_FORCE"] = "1"
        env["GIGA_TRAIN_PROFILE"] = "amd_win_20k"
        env["GIGA_ENV_EPOCHS"] = "1"
        env["GIGA_ENV_MAXEP"] = "1.5"
        env["GIGA_ENV_FP32"] = "1"
        env["GIGA_ASYNC_OVERLAP"] = "0"
        env["GIGA_ENV_LEAN"] = "1"
    if torch_threads > 0:
        env["GIGA_TORCH_THREADS"] = str(torch_threads)
        env["OMP_NUM_THREADS"] = str(torch_threads)
        env["MKL_NUM_THREADS"] = str(torch_threads)

    return {
        "profile": profile_name,
        "primary_gpu": primary,
        "cuda_available": bool(nvidia),
        "hip_available": bool(amd) and hip_ready,
        "gpu_sim_available": use_gpu_sim,
        "gpu_backend": backend,
        "force_cpu_sim": force_cpu_sim,
        "gpu_native": use_gpu_sim,
        "torch_device": torch_device,
        "torch_threads": torch_threads,
        "directml_python_available": directml_py,
        "directml_in_cpp_learner": False,  # libtorch has no DirectML
        "num_arenas": arenas,
        "steps_per_itr": steps,
        "ts_per_itr_hint": ts_hint,
        "target_min_overall_sps": 20000 if use_hip_sim else None,
        "vram_gb_used": vram,
        "ram_total_gb": ram_total,
        "hip": hip_info,
        "note": note,
        "env": env,
    }


def build_profile(
    out: Path,
    simulate_amd: bool = False,
    force_cpu: bool = False,
) -> dict[str, Any]:
    cpu = _cpu_info()
    ram = _ram_gb()
    gpus = detect_gpus(simulate_amd=simulate_amd)
    hip_info = _hip_runtime_present()
    if simulate_amd:
        # Dry-run assumes HIP SDK available so power_hip path is exercised.
        hip_info = {
            "present": True,
            "hipcc": True,
            "rocminfo": False,
            "paths": ["(simulated)"],
            "platform": "simulated",
        }
    rec = recommend(cpu, ram, gpus, force_cpu=force_cpu, hip_info=hip_info)
    profile = {
        "schema": 2,
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": cpu,
        "ram": ram,
        "gpus": gpus,
        "hip": hip_info,
        "recommendations": rec,
        "simulated_amd": simulate_amd,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile, indent=2)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        try:
            tmp.replace(out)
        except OSError:
            out.write_text(payload, encoding="utf-8")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass
    return profile


def print_human(profile: dict[str, Any]) -> None:
    cpu = profile["cpu"]
    ram = profile["ram"]
    rec = profile["recommendations"]
    print("=== GigaLearnRL hardware probe ===")
    print(f"CPU: {cpu.get('brand')}  cores={cpu.get('cores')} threads={cpu.get('threads')}")
    print(f"RAM: {ram.get('total_gb')} GB total / {ram.get('available_gb')} GB free")
    gpus = profile.get("gpus") or []
    if not gpus:
        print("GPU: (none detected)")
    else:
        for g in gpus:
            print(
                f"GPU: [{g.get('vendor')}] {g.get('name')}  "
                f"VRAM~{g.get('vram_gb')}GB  cuda={g.get('cuda_capable')} "
                f"hip={g.get('hip_capable')}"
            )
    hip = profile.get("hip") or {}
    print(
        f"HIP/ROCm: present={hip.get('present')} platform={hip.get('platform')} "
        f"hipcc={hip.get('hipcc')}"
    )
    print(f"Profile: {rec.get('profile')}  primary={rec.get('primary_gpu')}  backend={rec.get('gpu_backend')}")
    print(
        f"  force_cpu_sim={rec.get('force_cpu_sim')}  gpu_native={rec.get('gpu_native')}  "
        f"gpu_sim={rec.get('gpu_sim_available')}"
    )
    print(
        f"  arenas={rec.get('num_arenas')}  steps={rec.get('steps_per_itr')}  "
        f"ts~{rec.get('ts_per_itr_hint')}"
    )
    if rec.get("torch_threads"):
        print(
            f"  torch_device={rec.get('torch_device')}  "
            f"torch_threads={rec.get('torch_threads')}  "
            f"directml_python={rec.get('directml_python_available')} "
            f"(cpp_dml={rec.get('directml_in_cpp_learner')})"
        )
    print(f"  note: {rec.get('note')}")
    if profile.get("simulated_amd"):
        print("  (dry-run: --simulate-amd)")


def print_env(profile: dict[str, Any]) -> None:
    env = profile["recommendations"].get("env") or {}
    for k, v in env.items():
        print(f"{k}={v}")


def main() -> int:
    ap = argparse.ArgumentParser(description="GigaLearnRL hardware probe")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output hw_profile.json (default: <repo>/build/Release/hw_profile.json)",
    )
    ap.add_argument("--print-env", action="store_true", help="Print KEY=VALUE for bat SET")
    ap.add_argument(
        "--simulate-amd",
        action="store_true",
        help="Dry-run AMD HIP power recommendations",
    )
    ap.add_argument("--force-cpu", action="store_true", help="Recommend CPU sim even on NVIDIA/AMD")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out = args.out or (repo / "build" / "Release" / "hw_profile.json")
    profile = build_profile(out, simulate_amd=args.simulate_amd, force_cpu=args.force_cpu)

    mirror_root = os.environ.get("GIGA_CUDA_MIRROR")
    if mirror_root:
        mirror = Path(mirror_root) / "build" / "Release" / "hw_profile.json"
        if mirror.parent.is_dir() and mirror.resolve() != out.resolve():
            try:
                mirror.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            except OSError:
                pass

    if args.print_env:
        print_env(profile)
    elif not args.quiet:
        print_human(profile)
        print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
