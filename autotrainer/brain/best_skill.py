"""Best-by-skill checkpoint — keep a copy of the strongest bot by Elo/skill.

Writes under ``<ckpt_box>/best_skill/`` (default checkpoints/best_skill).
Uses periodic-eval Elo + live Rating/* when available. Does not touch Leak ckpts.

Opt-out: best_skill.enabled: false  or  GIGA_NO_BEST_SKILL=1
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from .io_utils import write_json_atomic
from .commands import request_checkpoint


STATE_KEY = "best_skill"


def best_skill_enabled(cfg: dict[str, Any] | None = None) -> bool:
    if os.environ.get("GIGA_NO_BEST_SKILL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    bs = (cfg or {}).get("best_skill") or {}
    return bool(bs.get("enabled", True))


def _is_ckpt(p: Path) -> bool:
    return (p / "POLICY.lt").exists() and (p / "SHARED_HEAD.lt").exists()


def _latest_numbered(folder: Path) -> Path | None:
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


def resolve_ckpt_box(watch_dir: Path, cfg: dict[str, Any] | None) -> Path:
    bs = (cfg or {}).get("best_skill") or {}
    if bs.get("checkpoint_box"):
        return Path(str(bs["checkpoint_box"]))
    parent = Path(watch_dir).resolve().parent
    return parent / "checkpoints"


def _copy_ckpt(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "POLICY.lt",
        "SHARED_HEAD.lt",
        "CRITIC.lt",
        "RUNNING_STATS.json",
        "OPTIMIZER.lt",
        "AGENT_META.json",
    ):
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)


def maybe_save_best_skill(
    watch_dir: Path,
    status: dict[str, Any],
    state: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
    eval_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    If current skill/Elo beats previous best, copy latest ckpt → best_skill/.

    Also requests C++ ``save_best_skill`` so live weights can be dumped without
    waiting for tsPerSave (bridge handles if present; Python copy is primary).
    """
    if not best_skill_enabled(cfg):
        return None

    bs = (cfg or {}).get("best_skill") or {}
    min_improve = float(bs.get("min_improve", 1.0))
    pe = dict(state.get("periodic_eval") or {})
    summary = eval_summary or pe.get("last_summary") or {}

    skill = None
    if summary.get("elo") is not None:
        try:
            skill = float(summary["elo"])
        except (TypeError, ValueError):
            pass
    if skill is None and pe.get("last_elo") is not None:
        try:
            skill = float(pe["last_elo"])
        except (TypeError, ValueError):
            pass
    if skill is None and summary.get("live_skill") is not None:
        try:
            skill = float(summary["live_skill"])
        except (TypeError, ValueError):
            pass
    if skill is None:
        return None

    st = dict(state.get(STATE_KEY) or {})
    best = st.get("best_skill")
    if best is not None and skill < float(best) + min_improve:
        return None

    box = resolve_ckpt_box(watch_dir, cfg)
    latest = _latest_numbered(box)
    dest = box / str(bs.get("subdir") or "best_skill")

    # Prefer requesting live save so best_skill matches in-memory policy
    try:
        from .commands import request_save_best_skill

        request_save_best_skill(Path(watch_dir))
    except Exception:
        pass

    copied_from = None
    if latest is not None and _is_ckpt(latest):
        try:
            _copy_ckpt(latest, dest)
            copied_from = str(latest)
        except OSError as exc:
            print(f"[AutoTrainer] best_skill copy failed: {exc}")
            return None
    else:
        # No numbered ckpt yet — ask for a normal save; next cycle may copy
        request_checkpoint(Path(watch_dir))
        st["pending_copy"] = True
        st["pending_skill"] = skill
        state[STATE_KEY] = st
        return {"pending": True, "skill": skill}

    meta = {
        "best_skill": skill,
        "timesteps": int(status.get("total_timesteps") or 0),
        "copied_from": copied_from,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elo": summary.get("elo"),
        "live_skill": summary.get("live_skill"),
        "fitness": summary.get("fitness"),
        "vs_experts": summary.get("vs_experts") or {},
    }
    write_json_atomic(dest / "BEST_SKILL.json", meta)
    write_json_atomic(Path(watch_dir) / "best_skill_status.json", meta)

    prev = best
    st["best_skill"] = skill
    st["best_path"] = str(dest)
    st["last_saved_at"] = meta["saved_at"]
    st["pending_copy"] = False
    state[STATE_KEY] = st

    print(
        f"[AutoTrainer] BEST SKILL {prev if prev is not None else 'n/a'} -> {skill:.1f} "
        f"-> {dest}"
    )
    return meta


def flush_pending_best_skill(
    watch_dir: Path,
    status: dict[str, Any],
    state: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Retry copy after a save_checkpoint request filled a numbered folder."""
    st = dict(state.get(STATE_KEY) or {})
    if not st.get("pending_copy"):
        return None
    # Force re-eval of threshold by temporarily lowering best
    pending = st.get("pending_skill")
    if pending is None:
        return None
    # Clear best so maybe_save runs
    st.pop("best_skill", None)
    state[STATE_KEY] = st
    return maybe_save_best_skill(
        watch_dir,
        status,
        state,
        cfg=cfg,
        eval_summary={"elo": pending},
    )
