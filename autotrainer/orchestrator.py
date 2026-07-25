#!/usr/bin/env python3
"""
AutoTrainer entry point — run from GigaLearnRL root:

  python autotrainer/orchestrator.py --watch-dir build/Release/autotrainer

Or after `pip install -e .`:

  python -m autotrainer --watch-dir build/Release/autotrainer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `python autotrainer/orchestrator.py` without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from autotrainer.brain.engine import AutoTrainerEngine, print_banner, print_decision
    from autotrainer.brain.io_utils import configure_stdio, load_yaml, read_json
    from autotrainer.brain.profile_builder import load_or_create_profile
except ImportError:
    from brain.engine import AutoTrainerEngine, print_banner, print_decision
    from brain.io_utils import configure_stdio, load_yaml, read_json
    from brain.profile_builder import load_or_create_profile


def main() -> None:
    # MUST run before print_banner — cp1252 chokes on Unicode in banner/heartbeats.
    configure_stdio()

    ap = argparse.ArgumentParser(description="GigaLearnRL AutoTrainer orchestrator")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--watch-dir", type=Path, default=None)
    ap.add_argument("--poll-sec", type=float, default=15.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Dump full JSON patches each cycle (default: one rich status line)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = args.config or (root / "config.default.yaml")
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if args.llm and cfg.get("llm"):
        cfg["llm"]["enabled"] = True
    if args.verbose:
        cfg.setdefault("logging", {})["verbose"] = True

    profile = load_or_create_profile(root / "profiles", args.profile)
    watch_dir = args.watch_dir or (Path.cwd() / "autotrainer")
    watch_dir.mkdir(parents=True, exist_ok=True)

    print_banner(profile, watch_dir, cfg)
    engine = AutoTrainerEngine(watch_dir, root / "profiles", profile, cfg)
    verbose = bool((cfg.get("logging") or {}).get("verbose"))

    consecutive_errors = 0
    while True:
        try:
            decision = engine.tick()
            consecutive_errors = 0
            if decision and verbose:
                status = read_json(watch_dir / "trainer_status.json")
                print_decision(decision, status or {}, verbose=True)
        except KeyboardInterrupt:
            print("[AutoTrainer] stopped (KeyboardInterrupt)", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 — never one-shot exit after first apply
            consecutive_errors += 1
            print(
                f"[AutoTrainer] tick error ({consecutive_errors}): {type(exc).__name__}: {exc}",
                flush=True,
            )
            # Brief backoff; keep polling forever unless --once
            if not args.once:
                time.sleep(min(60.0, args.poll_sec * max(1, consecutive_errors)))

        if args.once:
            break
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
