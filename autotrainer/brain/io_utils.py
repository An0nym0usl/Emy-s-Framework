"""Shared JSON / YAML I/O for AutoTrainer."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def configure_stdio() -> None:
    """Make Windows cp1252 consoles / cmd redirects UTF-8-safe before any print.

    Without this, a single Unicode arrow/dash in print_banner kills the process
    with UnicodeEncodeError and AutoTrainer never polls.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(  # type: ignore[attr-defined]
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        except Exception:
            try:
                stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
            except Exception:
                pass


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("Install PyYAML: pip install pyyaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    if yaml is None:
        raise SystemExit("Install PyYAML: pip install pyyaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        # utf-8-sig tolerates Windows BOM from PowerShell Set-Content / editors
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def write_json_atomic(path: Path, data: dict) -> None:
    """Atomic JSON write (tmp + replace). Robust on Windows when dest already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    tmp.write_text(payload, encoding="utf-8")
    try:
        tmp.replace(path)
    except OSError:
        # Windows: replace can race with a reader holding the dest briefly.
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
        try:
            tmp.replace(path)
        except OSError:
            # Last resort: direct write so AutoTrainer never silently drops overrides.
            path.write_text(payload, encoding="utf-8")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def write_overrides(watch_dir: Path, patch: dict, note: str) -> None:
    out = dict(patch)
    out["note"] = note
    out["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(watch_dir / "runtime_overrides.json", out)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def merge_overrides(current: dict, patch: dict) -> dict:
    """Merge patch into active overrides (reward_weights merge by key).

    If patch sets ``reward_weights_replace: true``, replace the whole
    ``reward_weights`` map (drops stale LKG keys like TouchReward / VelBall=3.0).
    The replace flag itself is not persisted.
    """
    out = dict(current)
    replace_rw = bool(patch.get("reward_weights_replace"))
    for k, v in patch.items():
        if k in ("note", "updated_at", "reward_weights_replace"):
            continue
        if k == "reward_weights" and isinstance(v, dict):
            if replace_rw:
                out["reward_weights"] = dict(v)
            else:
                rw = dict(out.get("reward_weights") or {})
                rw.update(v)
                out["reward_weights"] = rw
        else:
            out[k] = v
    return out
