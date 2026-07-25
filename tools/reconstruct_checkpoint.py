"""Arch-aware checkpoint reconstruction / validation (ContinuousV2 parity).

Reads opponents.json gigalearn entries + teachers_manifest, verifies POLICY/
SHARED_HEAD presence, optionally reconstructs a standardized deploy folder
with arch sidecar (obs_size, shared_head, policy layers) for InferUnit /
OpponentPool consumers.

Usage:
  python tools/reconstruct_checkpoint.py
  python tools/reconstruct_checkpoint.py --name requiem --dst C:\\GigaLearnRL\\build\\Release\\checkpoints_reconstructed
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_OPP = Path(r"C:\GigaLearnRL\opponents\opponents.json")
DEFAULT_EXE = Path(r"C:\GigaLearnRL\build\Release")


def _is_ckpt(p: Path) -> bool:
    return (p / "POLICY.lt").exists() and (p / "SHARED_HEAD.lt").exists()


def _latest(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    if _is_ckpt(folder):
        return folder
    best_ts, best = -1, None
    for child in folder.iterdir():
        if child.is_dir() and child.name.isdigit() and _is_ckpt(child):
            ts = int(child.name)
            if ts > best_ts:
                best_ts, best = ts, child
    return best


def load_entries(opp_path: Path) -> list[dict]:
    data = json.loads(opp_path.read_text(encoding="utf-8"))
    return [e for e in (data.get("entries") or []) if e.get("type") == "gigalearn"]


def reconstruct_one(entry: dict, dst_root: Path) -> dict:
    name = str(entry.get("name") or "unnamed")
    src = _latest(Path(entry["model"]))
    result = {
        "name": name,
        "src": str(src) if src else None,
        "ok": False,
        "dst": None,
        "arch": {
            "obs_size": entry.get("obs_size"),
            "shared_head": entry.get("shared_head"),
            "policy": entry.get("policy"),
            "layer_norm": entry.get("layer_norm", True),
        },
        "errors": [],
    }
    if src is None:
        result["errors"].append("missing checkpoint at model path")
        return result

    dst = dst_root / name
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("POLICY.lt", "SHARED_HEAD.lt", "CRITIC.lt", "RUNNING_STATS.json"):
        srcf = src / fname
        if srcf.exists():
            shutil.copy2(srcf, dst / fname)
    if not _is_ckpt(dst):
        result["errors"].append("reconstruct failed: POLICY/SHARED_HEAD missing after copy")
        return result

    sidecar = {
        "name": name,
        "source": str(src),
        "obs_size": entry.get("obs_size"),
        "shared_head": entry.get("shared_head"),
        "policy": entry.get("policy"),
        "layer_norm": entry.get("layer_norm", True),
        "beat_bonus_scale": entry.get("beat_bonus_scale"),
        "concede_penalty_scale": entry.get("concede_penalty_scale"),
        "reconstructed": True,
    }
    (dst / "ARCH.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    result["ok"] = True
    result["dst"] = str(dst)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponents", type=Path, default=DEFAULT_OPP)
    ap.add_argument("--exe-dir", type=Path, default=DEFAULT_EXE)
    ap.add_argument("--dst", type=Path, default=None)
    ap.add_argument("--name", type=str, default=None, help="Only reconstruct this opponent name")
    args = ap.parse_args()

    opp = args.opponents
    if not opp.exists():
        alt = args.exe_dir / "opponents" / "opponents.json"
        if alt.exists():
            opp = alt
    if not opp.exists():
        print(f"ERROR: opponents.json not found ({args.opponents})", file=sys.stderr)
        return 1

    entries = load_entries(opp)
    if args.name:
        entries = [e for e in entries if e.get("name") == args.name]
    if not entries:
        print("ERROR: no gigalearn entries to reconstruct", file=sys.stderr)
        return 1

    dst_root = args.dst or (args.exe_dir / "checkpoints_reconstructed")
    dst_root.mkdir(parents=True, exist_ok=True)

    report = {"opponents": str(opp), "dst": str(dst_root), "results": []}
    ok_n = 0
    for e in entries:
        r = reconstruct_one(e, dst_root)
        report["results"].append(r)
        status = "OK" if r["ok"] else "FAIL"
        print(f"[{status}] {r['name']}: {r.get('dst') or r['errors']}")
        if r["ok"]:
            ok_n += 1

    # Also stage teachers_manifest roots into reconstructed/teachers/
    teachers_dst = dst_root / "teachers"
    teachers_dst.mkdir(exist_ok=True)
    manifest = args.exe_dir / "teachers_manifest.json"
    teacher_paths: list[str] = []
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            teacher_paths = list(data.get("teachers") or [])
        except Exception:
            pass
    staged = []
    for i, t in enumerate(teacher_paths):
        src = Path(t)
        if not _is_ckpt(src):
            src2 = _latest(src)
            if src2 is None:
                continue
            src = src2
        slot = teachers_dst / f"teacher_{i}_{src.name}"
        slot.mkdir(parents=True, exist_ok=True)
        for fname in ("POLICY.lt", "SHARED_HEAD.lt", "CRITIC.lt", "RUNNING_STATS.json"):
            if (src / fname).exists():
                shutil.copy2(src / fname, slot / fname)
        staged.append(str(slot))
    report["teachers_staged"] = staged
    report["ok_count"] = ok_n

    out = dst_root / "reconstruction_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({ok_n}/{len(entries)} opponents ok, teachers={len(staged)})")
    return 0 if ok_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
