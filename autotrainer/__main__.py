#!/usr/bin/env python3
"""GigaLearnRL AutoTrainer — `python -m autotrainer`"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autotrainer.brain.engine import AutoTrainerEngine, print_banner, print_decision
from autotrainer.brain.io_utils import configure_stdio, load_yaml
from autotrainer.brain.profile_builder import load_or_create_profile, profile_from_text


def main() -> None:
    # MUST run before print_banner — Windows cp1252 kills AT on UnicodeEncodeError.
    configure_stdio()

    ap = argparse.ArgumentParser(description="GigaLearnRL AutoTrainer")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--watch-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--poll-sec", type=float, default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--llm", action="store_true", help="Enable LLM advisor (needs OPENAI_API_KEY)")
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Dump full JSON patches each cycle (default: one rich status line)",
    )
    ap.add_argument(
        "--profile-text",
        type=Path,
        default=None,
        help="Natural-language bot description → generate profile YAML",
    )
    ap.add_argument("--import-profile", action="store_true", help="With --profile-text, write YAML")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = args.config or (root / "config.default.yaml")
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if args.llm and cfg.get("llm"):
        cfg["llm"]["enabled"] = True
    if getattr(args, "verbose", False):
        cfg.setdefault("logging", {})["verbose"] = True

    profiles_dir = root / "profiles"
    if args.profile_text:
        text = args.profile_text.read_text(encoding="utf-8")
        prof = profile_from_text(
            text,
            name=args.profile,
            profiles_dir=profiles_dir if args.import_profile else None,
            use_llm=args.llm,
        )
        if args.import_profile:
            print(f"Wrote profile: {profiles_dir / (prof['name'] + '.yaml')}")
            if args.once and not args.watch_dir:
                return
    else:
        prof = load_or_create_profile(profiles_dir, args.profile, args.profile_text)

    watch_dir = args.watch_dir or (Path.cwd() / "autotrainer")
    watch_dir.mkdir(parents=True, exist_ok=True)

    poll = args.poll_sec if args.poll_sec is not None else float(cfg.get("poll_interval_sec", 15))

    if cfg.get("logging", {}).get("print_banner", True):
        print_banner(prof, watch_dir)

    engine = AutoTrainerEngine(watch_dir, profiles_dir, prof, cfg)
    verbose = bool((cfg.get("logging") or {}).get("verbose"))

    consecutive_errors = 0
    while True:
        status_path = watch_dir / "trainer_status.json"
        try:
            if not status_path.exists():
                print(
                    "Waiting for trainer_status.json from GigaLearnBot...",
                    flush=True,
                )
            else:
                from autotrainer.brain.io_utils import read_json

                decision = engine.tick()
                consecutive_errors = 0
                if decision and verbose:
                    status = read_json(status_path)
                    print_decision(decision, status, verbose=True)
        except KeyboardInterrupt:
            print("[AutoTrainer] stopped (KeyboardInterrupt)", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 — never one-shot exit after first apply
            consecutive_errors += 1
            print(
                f"[AutoTrainer] tick error ({consecutive_errors}): {type(exc).__name__}: {exc}",
                flush=True,
            )
            if not args.once:
                time.sleep(min(60.0, poll * max(1, consecutive_errors)))

        if args.once:
            break
        time.sleep(poll)


if __name__ == "__main__":
    main()
