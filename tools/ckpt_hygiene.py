"""Checkpoint hygiene helpers (guide: keep versions for old-opponent sparring; don't wipe Leak).

- list: show timestamped policy folders under a ckpt root
- prune: delete oldest numeric folders, keep newest N (never touches checkpoints_default*)
- prefer: print best candidate for RLBot deploy

Usage:
  python tools/ckpt_hygiene.py list --root C:\\GigaLearnRL\\build\\Release\\checkpoints
  python tools/ckpt_hygiene.py prune --root ... --keep 8 --dry-run
  python tools/ckpt_hygiene.py prefer --root ...
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROTECTED_NAME_FRAGMENTS = (
    "checkpoints_default",
    "checkpoints_default_lean",
)


def _is_policy_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit() and (p / "POLICY.lt").exists() and (p / "SHARED_HEAD.lt").exists()


def list_ckpts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    kids = [c for c in root.iterdir() if _is_policy_dir(c)]
    return sorted(kids, key=lambda c: int(c.name))


def protected_root(root: Path) -> bool:
    """Refuse prune on Leak boxes (name or any path segment)."""
    parts = [p.lower() for p in root.parts]
    name = root.name.lower()
    if any(frag in name for frag in PROTECTED_NAME_FRAGMENTS):
        return True
    return any(any(frag in part for frag in PROTECTED_NAME_FRAGMENTS) for part in parts)


def cmd_list(root: Path) -> int:
    rows = list_ckpts(root)
    if not rows:
        print(f"No policy checkpoints under {root}")
        return 1
    for p in rows:
        extras = [f.name for f in ("CRITIC.lt", "RUNNING_STATS.json", "ARCH.json") if (p / f).exists()]
        print(f"{p.name}\t{p}\t{','.join(extras) or 'policy+shared'}")
    print(f"total={len(rows)}  latest={rows[-1].name}")
    return 0


def cmd_prune(root: Path, keep: int, dry_run: bool) -> int:
    if protected_root(root):
        print(f"REFUSE: will not prune protected Leak box {root}")
        return 2
    rows = list_ckpts(root)
    if len(rows) <= keep:
        print(f"Nothing to prune ({len(rows)} <= keep={keep})")
        return 0
    drop = rows[: max(0, len(rows) - keep)]
    for p in drop:
        print(f"{'DRY ' if dry_run else ''}DELETE {p}")
        if not dry_run:
            shutil.rmtree(p)
    print(f"kept newest {keep}; removed {len(drop)}")
    return 0


def cmd_prefer(root: Path) -> int:
    rows = list_ckpts(root)
    if not rows:
        print(f"No checkpoints in {root}", file=sys.stderr)
        return 1
    # Prefer newest that has ARCH.json (deploy hygiene); else newest complete policy
    with_arch = [p for p in rows if (p / "ARCH.json").exists()]
    best = with_arch[-1] if with_arch else rows[-1]
    print(best)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Checkpoint hygiene (GigaLearnRL)")
    ap.add_argument("action", choices=("list", "prune", "prefer"))
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "build"
        / "Release"
        / "checkpoints",
    )
    ap.add_argument("--keep", type=int, default=8, help="prune: keep newest N")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.action == "list":
        return cmd_list(args.root)
    if args.action == "prune":
        return cmd_prune(args.root, args.keep, args.dry_run)
    return cmd_prefer(args.root)


if __name__ == "__main__":
    sys.exit(main())
