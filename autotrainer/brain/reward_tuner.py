"""Adjust reward multipliers toward profile skill priorities + default GPU stack."""

from __future__ import annotations

from typing import Any

from .full_control import BLANK_REWARD_KEYS, ensure_blank_reward_keys
from .ssl_guide import reward_shaping_decay


def tune_rewards(
    profile: dict,
    phase: int,
    gate_failures: list[str],
    manifest: dict,
    current_mult: dict[str, float],
    step: str,
    *,
    timesteps: int = 0,
    ssl_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return delta multipliers to merge into reward_weights patch."""
    priorities = profile.get("priorities") or []
    out: dict[str, float] = dict(current_mult)

    # SSL guide §4 — timestep reward shaping decay (early touch → late goal)
    if ssl_cfg is not None:
        out = reward_shaping_decay(timesteps, phase, out, manifest, ssl_cfg)

    # Phase-based focus groups
    focus_ids: set[str] = set()
    if phase == 0:
        focus_ids = {"chase_basics"}
    elif phase == 1:
        focus_ids = {"speedflip_kickoff", "fundamentals"}
    else:
        focus_ids = {p["id"] for p in priorities}

    for p in priorities:
        pid = p.get("id", "")
        if phase >= 2 or pid in focus_ids or pid == "speedflip_kickoff":
            boost = 1.08 if step == "boost" else 1.04
            for rname in p.get("rewards") or []:
                base = out.get(rname, 1.0)
                out[rname] = min(3.0, base * boost)

    # If chase gates fail, boost chase-related rewards (CPU + GPU default names)
    fail_text = " ".join(gate_failures).lower()
    if "touch" in fail_text or phase == 0:
        for name in (
            "TouchBallReward",
            "VelocityPlayerToBallReward",
            "FaceBallReward",
            "StrongTouchReward",
            "TouchAccelReward",
        ):
            out[name] = min(3.0, out.get(name, 1.0) * 1.12)

    if "kickoff" in fail_text or "veltowardball" in fail_text.replace(" ", ""):
        for name in ("KickoffProximity", "VelocityPlayerToBallReward"):
            out[name] = min(3.0, out.get(name, 1.0) * 1.15)

    if phase >= 2 and "air" in fail_text:
        for name in (
            "AirReward",
            "AirDribbleReward",
        ):
            out[name] = min(3.0, out.get(name, 1.0) * 1.1)

    if "goal" in fail_text or "finish" in fail_text:
        for name in ("GoalReward", "VelocityBallToGoalReward", "TouchBallReward", "VelocityPlayerToBallReward"):
            out[name] = min(3.0, out.get(name, 1.0) * 1.1)

    out = ensure_blank_reward_keys(out, manifest)
    valid = {e["name"] for e in (manifest.get("rewards") or []) if "name" in e}
    if valid:
        keep = set(BLANK_REWARD_KEYS) | valid
        out = {k: v for k, v in out.items() if k in keep}

    return out
