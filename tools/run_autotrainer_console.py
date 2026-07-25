#!/usr/bin/env python3
"""Visible AutoTrainer console with tee to autotrainer_launch.log.

Used by tools/launch_autotrainer.bat so the AutoTrainer window shows live
poll# lines while still appending to the launch log (no blank redirected cmd).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: run_autotrainer_console.py ROOT PROFILE WATCH [extra orchestrator args...]",
            file=sys.stderr,
        )
        return 2

    root = Path(sys.argv[1]).resolve()
    profile = sys.argv[2]
    watch = Path(sys.argv[3]).resolve()
    extra = sys.argv[4:]

    watch.mkdir(parents=True, exist_ok=True)
    log_path = watch / "autotrainer_launch.log"
    orch = root / "autotrainer" / "orchestrator.py"
    if not orch.is_file():
        print(f"ERROR: missing {orch}", file=sys.stderr)
        return 1

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(root) + (os.pathsep + pp if pp else "")

    cmd = [
        sys.executable,
        "-u",
        str(orch),
        "--profile",
        profile,
        "--watch-dir",
        str(watch),
        *extra,
    ]

    with log_path.open("a", encoding="utf-8", errors="replace") as logf:
        logf.write(
            f"\n===== AutoTrainer console {profile} watch={watch} =====\n"
        )
        logf.flush()
        print(f"[AutoTrainer] logging also to {log_path}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        tee = _Tee(sys.stdout, logf)
        for line in proc.stdout:
            tee.write(line)
        return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
