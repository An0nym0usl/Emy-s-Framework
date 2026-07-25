"""
AutoTrainer full autonomous control helpers.

When full_control / ssl_autonomy is ON (default for from-scratch), the brain freely
tunes reward weights + PPO/league/GPU knobs and persists them so C++ ReapplyPostApex
wins over static Apex every iteration.

Opt-out:
  - GIGA_NO_SSL_AUTONOMY=1
  - GIGA_AT_READONLY=1
  - ssl_autonomy.enabled / full_control: false in config.default.yaml
"""

from __future__ import annotations

import os
from typing import Any


# Blank GPU stack keys (default from-scratch template).
BLANK_REWARD_KEYS: tuple[str, ...] = (
    "GoalReward",
    "TouchBallReward",
    "VelocityBallToGoalReward",
    "VelocityPlayerToBallReward",
)

def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def env_readonly() -> bool:
    return _env_truthy("GIGA_AT_READONLY")


def full_control_enabled(cfg: dict[str, Any] | None) -> bool:
    """
    Default ON (especially from-scratch).
    Opt-out: ssl_autonomy.full_control false | GIGA_NO_SSL_AUTONOMY=1 | GIGA_AT_READONLY=1
    """
    if env_readonly() or _env_truthy("GIGA_NO_SSL_AUTONOMY"):
        return False
    root = cfg or {}

    auto = root.get("ssl_autonomy")
    if isinstance(auto, dict) and auto:
        if "enabled" in auto and not bool(auto.get("enabled")):
            return False
        if "full_control" in auto and not bool(auto.get("full_control")):
            return False
        return bool(auto.get("enabled", True)) and bool(auto.get("full_control", True))

    # Legacy flat keys
    if "full_control" in root:
        return bool(root.get("full_control"))
    at = root.get("autotrainer") or {}
    if "full_control" in at:
        return bool(at.get("full_control"))

    # From-scratch / explicit autonomy env → ON
    if _env_truthy("GIGA_FROM_SCRATCH") or _env_truthy("GIGA_SSL_AUTONOMY"):
        return True
    return True  # config default: autonomous when AutoTrainer runs


def warmup_steps(cfg: dict[str, Any] | None) -> int:
    root = cfg or {}
    auto = root.get("ssl_autonomy") if isinstance(root.get("ssl_autonomy"), dict) else {}
    if auto.get("sparring_warmup_steps") is not None:
        return int(auto["sparring_warmup_steps"])
    at = root.get("autotrainer") or {}
    fc = at.get("full_control_cfg") or root.get("full_control_cfg") or {}
    return int(fc.get("warmup_steps", root.get("warmup_steps", 30_000_000)))


def autonomy_flags(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Flags C++ reads so ReapplyPostApex always reasserts AutoTrainer knobs."""
    if not full_control_enabled(cfg):
        return {"ssl_guide_post_apex": True}
    return {
        "autotrainer_full_control": True,
        "full_control": True,
        "ssl_autonomy": True,
        "ssl_guide_post_apex": True,
    }


def post_warmup_sparring(
    timesteps: int,
    cfg: dict[str, Any] | None,
    *,
    phase: int = 0,
) -> dict[str, Any]:
    """
    After warmup M steps: re-enable skill-eval + light old-version sparring
    (from-scratch boots with both OFF for fast iters).

    Prefer ssl_guide.warmup_arms + league_sampling when full SSL patch runs;
    this helper is the fallback merge for non-SSL engine paths.
    """
    if not full_control_enabled(cfg):
        return {}
    root = cfg or {}
    auto = root.get("ssl_autonomy") if isinstance(root.get("ssl_autonomy"), dict) else {}
    # Keep defaults aligned with config.default.yaml (~30M / ~40M).
    sparring_w = int(auto.get("sparring_warmup_steps", warmup_steps(cfg)))
    skill_w = int(auto.get("skill_eval_warmup_steps", max(sparring_w, 40_000_000)))
    versions_w = int(auto.get("policy_versions_warmup_steps", 25_000_000))

    if timesteps < skill_w and timesteps < sparring_w:
        out = {
            "skill_tracker_enabled": False,
            "skill_tracker_interval": 256,
            "train_against_old_chance": 0.0,
            "opponent_pool_chance": 0.0,
        }
        if timesteps < versions_w:
            out["save_policy_versions"] = False
        return out

    out: dict[str, Any] = {}
    if timesteps >= skill_w:
        out["skill_tracker_enabled"] = True
        out["skill_tracker_interval"] = 128 if phase <= 0 else 64
    else:
        out["skill_tracker_enabled"] = False
        out["skill_tracker_interval"] = 256

    if timesteps >= versions_w:
        out["save_policy_versions"] = True
    else:
        out["save_policy_versions"] = False

    if timesteps >= sparring_w:
        if phase <= 0:
            out["train_against_old_chance"] = 0.08
            out["opponent_pool_chance"] = 0.02
        elif phase == 1:
            out["train_against_old_chance"] = 0.18
            out["opponent_pool_chance"] = 0.08
        else:
            out["train_against_old_chance"] = 0.30
            out["opponent_pool_chance"] = 0.10
    else:
        out["train_against_old_chance"] = 0.0
        out["opponent_pool_chance"] = 0.0
    return out


def ensure_blank_reward_keys(
    weights: dict[str, float],
    manifest: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Keep blank GPU keys present so C++ remap always has something to apply."""
    out = dict(weights)
    valid = {e["name"] for e in (manifest or {}).get("rewards") or [] if "name" in e}
    for k in BLANK_REWARD_KEYS:
        if k not in out:
            if not valid or k in valid:
                out[k] = 1.0
    if valid:
        out = {k: v for k, v in out.items() if k in valid or k in BLANK_REWARD_KEYS}
    return out


def format_reward_log(
    weights: dict[str, float] | None,
    *,
    limit: int | None = None,
) -> str:
    """Compact one-line summary. ``limit=None`` prints every key (no ``...(+N)``)."""
    if not weights:
        return "(none)"
    items = sorted(weights.items(), key=lambda kv: (kv[0].lower(), kv[0]))
    if limit is not None:
        # Legacy: show largest deviations first, then truncate
        items = sorted(weights.items(), key=lambda kv: abs(float(kv[1]) - 1.0), reverse=True)
        shown = items[:limit]
        parts = [f"{k}={float(v):.3f}" for k, v in shown]
        if len(items) > limit:
            parts.append(f"...(+{len(items) - limit})")
        return ", ".join(parts)
    return ", ".join(f"{k}={float(v):.3f}" for k, v in items)


def format_rewards_now_block(weights: dict[str, float] | None) -> str:
    """Multi-line dump of ALL reward weights (sorted by name). Never truncated."""
    if not weights:
        return "[AutoTrainer] rewards_now:\n  (none)"
    lines = ["[AutoTrainer] rewards_now:"]
    for k in sorted(weights.keys(), key=lambda s: (str(s).lower(), str(s))):
        try:
            v = float(weights[k])
            lines.append(f"  {k}={v:.2f}")
        except (TypeError, ValueError):
            lines.append(f"  {k}={weights[k]}")
    return "\n".join(lines)


def reward_weights_changed(
    prev: dict[str, Any] | None,
    new: dict[str, Any] | None,
    *,
    eps: float = 1e-4,
) -> bool:
    """True if keys differ or any weight moved by more than ``eps``."""
    a = {str(k): float(v) for k, v in (prev or {}).items() if v is not None}
    b = {str(k): float(v) for k, v in (new or {}).items() if v is not None}
    if set(a) != set(b):
        return True
    for k in a:
        if abs(a[k] - b[k]) > eps:
            return True
    return False


def should_log_rewards_now(
    *,
    prev: dict[str, Any] | None,
    new: dict[str, Any] | None,
    patch_had_rewards: bool = False,
    note: str = "",
    force: bool = False,
    eps: float = 1e-4,
) -> bool:
    """When to print the full rewards_now block (avoid spam on float jitter)."""
    if force:
        return True
    if not new:
        return False
    note_l = (note or "").lower()
    if "adjusted reward" in note_l or "reward adjust" in note_l:
        return True
    if "restored reward" in note_l:
        return True
    if patch_had_rewards and reward_weights_changed(prev, new, eps=eps):
        return True
    if reward_weights_changed(prev, new, eps=eps):
        return True
    return False


def log_rewards_now(
    weights: dict[str, Any] | None,
    *,
    prev: dict[str, Any] | None = None,
    patch_had_rewards: bool = False,
    note: str = "",
    force: bool = False,
    eps: float = 1e-4,
) -> bool:
    """Print full ``rewards_now`` block when rewards changed / forced. Returns True if printed."""
    if not should_log_rewards_now(
        prev=prev,
        new=weights,
        patch_had_rewards=patch_had_rewards,
        note=note,
        force=force,
        eps=eps,
    ):
        return False
    print(format_rewards_now_block(weights))
    return True


def stamp_full_control(patch: dict[str, Any], cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Merge autonomy flags into any outbound patch."""
    out = dict(patch)
    out.update(autonomy_flags(cfg))
    return out
