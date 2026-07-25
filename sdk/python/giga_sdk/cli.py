"""``giga`` command-line interface: manage the checkpoint box and deploy the bot.

Examples::

    giga list
    giga info 47694027776
    giga validate                 # validates the latest checkpoint
    giga export 47694027776 ./deploy_model
    giga prune --keep 8
    giga deploy                   # runs the bot from the latest valid checkpoint
    giga deploy --checkpoint 47694027776
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .checkpoints import CheckpointManager, default_checkpoint_dir


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _flags(info) -> str:
    parts = []
    parts.append("P" if info.has_policy else "-")
    parts.append("C" if info.has_critic else "-")
    parts.append("S" if info.has_shared_head else "-")
    parts.append("O" if info.has_optimizers else "-")
    parts.append("J" if info.has_stats else "-")
    return "".join(parts)


def _find_executable(explicit: str | None, checkpoints: Path) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    # Common locations relative to the checkpoint folder (it usually sits next to the exe).
    candidates = [
        checkpoints.parent / "GigaLearnBot.exe",
        checkpoints.parent / "GigaLearnBot",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = shutil.which("GigaLearnBot")
    return Path(found) if found else None


def cmd_list(mgr: CheckpointManager, args) -> int:
    infos = mgr.list()
    if not infos:
        print(f"No checkpoints found in {mgr.folder}")
        return 0
    print(f"Checkpoints in {mgr.folder}  (flags: P=policy C=critic S=shared O=optim J=stats)")
    for info in infos:
        tag = "  [inference-ready]" if info.is_valid_for_inference else ""
        print(f"  {info.timesteps:>16}  [{_flags(info)}]  {_fmt_size(info.total_bytes):>8}{tag}")
    latest = mgr.latest_valid_for_inference()
    if latest:
        print(f"Latest inference-ready: {latest.timesteps}")
    return 0


def cmd_info(mgr: CheckpointManager, args) -> int:
    info = mgr.get(args.timesteps) if args.timesteps is not None else mgr.latest()
    if info is None:
        print("No matching checkpoint.")
        return 1
    import json
    print(json.dumps(info.as_dict(), indent=2))
    return 0


def cmd_validate(mgr: CheckpointManager, args) -> int:
    if args.timesteps is not None:
        info = mgr.get(args.timesteps)
        if info is None:
            print(f"No checkpoint with timesteps={args.timesteps}")
            return 1
        target = info.path
    else:
        latest = mgr.latest()
        if latest is None:
            print("No checkpoints to validate.")
            return 1
        target = latest.path

    ok, err = mgr.validate(target)
    if ok:
        print(f"OK: {target} is valid.")
        return 0
    print(f"INVALID: {target}: {err}")
    return 1


def cmd_export(mgr: CheckpointManager, args) -> int:
    ok, err = mgr.export(args.timesteps, args.dest)
    if ok:
        print(f"Exported checkpoint {args.timesteps} to {args.dest}")
        return 0
    print(f"Export failed: {err}")
    return 1


def cmd_prune(mgr: CheckpointManager, args) -> int:
    removed = mgr.prune(args.keep)
    print(f"Pruned {removed} checkpoint(s), keeping the {args.keep} newest.")
    return 0


def cmd_versions(mgr: CheckpointManager, args) -> int:
    infos = mgr.list_policy_versions()
    if not infos:
        print("No policy versions found.")
        return 0
    for info in infos:
        print(f"  {info.timesteps}")
    return 0


def cmd_deploy(mgr: CheckpointManager, args) -> int:
    if args.checkpoint is not None:
        info = mgr.get(args.checkpoint)
        if info is None:
            print(f"No checkpoint with timesteps={args.checkpoint}")
            return 1
    else:
        info = mgr.latest_valid_for_inference()
        if info is None:
            print("No inference-ready checkpoint found.")
            return 1

    exe = _find_executable(args.exe, mgr.folder)
    if exe is None:
        print(
            "Could not locate the GigaLearnBot executable.\n"
            "Build the framework and pass --exe <path>, or run from next to the exe."
        )
        return 1

    cmd = [str(exe), "--rlbot", "--checkpoint", str(info.path)]
    print(f"Launching: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except OSError as e:
        print(f"Failed to launch bot: {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="giga", description="GigaLearnRL checkpoint + deployment SDK")
    parser.add_argument(
        "--checkpoints", default=str(default_checkpoint_dir()),
        help="Path to the checkpoints/ folder (default: $GIGA_CHECKPOINT_DIR or ./checkpoints)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all checkpoints").set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="Show checkpoint metadata (latest if no timesteps given)")
    p_info.add_argument("timesteps", nargs="?", type=int, default=None)
    p_info.set_defaults(func=cmd_info)

    p_val = sub.add_parser("validate", help="Validate a checkpoint (latest if no timesteps given)")
    p_val.add_argument("timesteps", nargs="?", type=int, default=None)
    p_val.set_defaults(func=cmd_validate)

    p_exp = sub.add_parser("export", help="Copy a checkpoint's inference files to a folder")
    p_exp.add_argument("timesteps", type=int)
    p_exp.add_argument("dest")
    p_exp.set_defaults(func=cmd_export)

    p_prune = sub.add_parser("prune", help="Delete old checkpoints, keeping the newest N")
    p_prune.add_argument("--keep", type=int, default=8)
    p_prune.set_defaults(func=cmd_prune)

    sub.add_parser("versions", help="List policy-version snapshots").set_defaults(func=cmd_versions)

    p_dep = sub.add_parser("deploy", help="Run the RLBot bot from a checkpoint")
    p_dep.add_argument("--checkpoint", type=int, default=None, help="Timesteps of the checkpoint (default: latest valid)")
    p_dep.add_argument("--exe", default=None, help="Path to the GigaLearnBot executable")
    p_dep.set_defaults(func=cmd_deploy)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mgr = CheckpointManager(args.checkpoints)
    return args.func(mgr, args)


if __name__ == "__main__":
    sys.exit(main())
