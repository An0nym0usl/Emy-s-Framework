"""RLBot deploy automation — pick best checkpoint and stage for --rlbot.

Supports teachers_manifest, eval_report champion, reconstructed arch folders,
and full product-surface packaging (opponents + reports).

Guide tip (Kue/web): deploy a saved policy version, not a mid-write folder.
Uses the same POLICY+SHARED_HEAD gate as tools/ckpt_hygiene.py prefer.
Prefers from-scratch POWER box (checkpoints) over wiping Leak boxes.

Usage:
  python tools/deploy_rlbot.py
  python tools/deploy_rlbot.py --src C:\\GigaLearnRL\\build\\Release\\checkpoints
  python tools/deploy_rlbot.py --from-eval
  python tools/deploy_rlbot.py --package-surface
  python tools/deploy_rlbot.py --run
  python tools/deploy_rlbot.py --prefer-default
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ROOTS = [
    Path(__file__).resolve().parents[1] / "build" / "Release",
    Path(r"C:\GigaLearnRL\build\Release"),  # optional clean mirror
]


def _is_complete_policy(folder: Path) -> bool:
    """Match ckpt_hygiene: POLICY + SHARED_HEAD required; ARCH optional but preferred."""
    return (
        folder.is_dir()
        and (folder / "POLICY.lt").exists()
        and (folder / "SHARED_HEAD.lt").exists()
    )


def find_latest(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    best_ts = -1
    best: Path | None = None
    try:
        children = list(folder.iterdir())
    except OSError:
        children = []
    for child in children:
        if not child.is_dir() or not child.name.isdigit():
            continue
        if not _is_complete_policy(child):
            continue
        ts = int(child.name)
        if ts > best_ts:
            best_ts = ts
            best = child
    if best is not None:
        return best
    if _is_complete_policy(folder):
        return folder
    return None


def package_surface(exe_dir: Path, champion: Path | None) -> Path:
    """Stage a deployable RLBot folder."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    surface = exe_dir / "product_surface" / stamp
    surface.mkdir(parents=True, exist_ok=True)

    if champion is not None:
        ck = surface / "checkpoint"
        ck.mkdir(exist_ok=True)
        for fname in ("POLICY.lt", "SHARED_HEAD.lt", "RUNNING_STATS.json", "CRITIC.lt", "ARCH.json"):
            srcf = champion / fname
            if srcf.exists():
                shutil.copy2(srcf, ck / fname)

    for name in ("opponents",):
        src = exe_dir / name
        if not src.exists():
            src = Path(r"C:\GigaLearnRL") / name
        if src.exists():
            dst = surface / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

    for report_name in (
        "teachers_manifest.json",
        "eval_report.json",
        "product_surface_manifest.json",
        "port.cfg",
    ):
        src = exe_dir / report_name
        if src.exists():
            shutil.copy2(src, surface / report_name)

    recon = exe_dir / "checkpoints_reconstructed" / "reconstruction_report.json"
    if recon.exists():
        shutil.copy2(recon, surface / "reconstruction_report.json")

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "champion": str(champion) if champion else None,
        "surfaces": [
            "rlbot_checkpoint",
            "opponents_manifest",
            "teachers_manifest",
            "eval_tournament",
            "reconstruction",
            "multi_seed_harness",
        ],
    }
    (surface / "SURFACE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Packaged product surface -> {surface}")
    return surface


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None, help="Checkpoint root or numbered folder")
    ap.add_argument("--dst", type=Path, default=None, help="Deploy target (default: <exe>/checkpoints/<ts>)")
    ap.add_argument("--exe-dir", type=Path, default=Path(r"C:\GigaLearnRL\build\Release"))
    ap.add_argument("--from-eval", action="store_true", help="Use eval_report.json champion")
    ap.add_argument("--package-surface", action="store_true", help="Also write product_surface package")
    ap.add_argument("--run", action="store_true", help="Launch --rlbot after staging")
    ap.add_argument(
        "--prefer-default",
        action="store_true",
        help="Only consider checkpoints (from-scratch POWER box)",
    )
    args = ap.parse_args()

    candidates: list[Path] = []
    if args.from_eval:
        report = args.exe_dir / "eval_report.json"
        if report.exists():
            try:
                data = json.loads(report.read_text(encoding="utf-8-sig"))
                champ = (data.get("champion") or {}).get("path")
                if champ:
                    candidates.append(Path(champ))
            except Exception as exc:
                print(f"WARNING: could not read eval_report.json ({exc})", file=sys.stderr)

    if args.src:
        candidates.append(args.src)
    else:
        names = (
            ("checkpoints",)
            if args.prefer_default
            else (
                "checkpoints",  # from-scratch POWER box (prefer over wiping Leak)
                "checkpoints_default_lean",
                "checkpoints_default",
                "checkpoints_quality",
                "checkpoints_continuous_v2",
                "checkpoints_reconstructed",
                "checkpoints",
            )
        )
        for root in DEFAULT_ROOTS:
            for name in names:
                candidates.append(root / name)

    # Prefer teachers_manifest.json if present
    manifest = args.exe_dir / "teachers_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            for t in data.get("teachers") or []:
                candidates.insert(0, Path(t))
        except Exception as exc:
            print(f"WARNING: could not read teachers_manifest.json ({exc})", file=sys.stderr)

    chosen: Path | None = None
    for c in candidates:
        hit = find_latest(c) if c.is_dir() and not ((c / "POLICY.lt").exists()) else (
            c if (c / "POLICY.lt").exists() else find_latest(c)
        )
        if hit is not None:
            chosen = hit
            break
    if chosen is None:
        print("ERROR: no valid checkpoint found", file=sys.stderr)
        return 1

    dst_root = args.dst or (args.exe_dir / "checkpoints_deployed")
    dst = dst_root / chosen.name
    try:
        if dst.resolve() == chosen.resolve():
            dst_root = args.exe_dir / "checkpoints_deployed"
            dst = dst_root / chosen.name
    except OSError:
        pass
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("POLICY.lt", "SHARED_HEAD.lt", "RUNNING_STATS.json", "CRITIC.lt", "ARCH.json"):
        srcf = chosen / fname
        if not srcf.exists():
            continue
        destf = dst / fname
        try:
            if destf.exists() and destf.resolve() == srcf.resolve():
                continue
        except OSError:
            pass
        shutil.copy2(srcf, destf)
    arch_note = " +ARCH" if (chosen / "ARCH.json").exists() else " (no ARCH.json — ok if lean512/Leak known)"
    print(f"Deployed {chosen} -> {dst}{arch_note}")
    print("Tip: RLBot uses deterministic inference; train stays separate (guide --render / --rlbot split).")

    port_cfg = args.exe_dir / "port.cfg"
    if not port_cfg.exists():
        port_cfg.write_text("42653\n", encoding="utf-8")
        print(f"Wrote default {port_cfg}")

    deploy_meta = {
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(chosen),
        "dest": str(dst),
        "from_eval": bool(args.from_eval),
        "rlbot_port": int(port_cfg.read_text(encoding="utf-8").strip() or "42653"),
    }
    (args.exe_dir / "deploy_manifest.json").write_text(json.dumps(deploy_meta, indent=2), encoding="utf-8")

    if args.package_surface or args.from_eval:
        package_surface(args.exe_dir, chosen)

    if args.run:
        exe = args.exe_dir / "GigaLearnBot.exe"
        if not exe.exists():
            print(f"ERROR: missing {exe}", file=sys.stderr)
            return 1
        print(f"Launching {exe} --rlbot --checkpoint {dst}")
        subprocess.Popen([str(exe), "--rlbot", "--checkpoint", str(dst)], cwd=str(args.exe_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
