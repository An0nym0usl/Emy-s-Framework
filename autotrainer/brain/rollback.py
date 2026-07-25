"""Snapshot / rollback when metrics degrade — including same-cycle LKG restore.

CRITICAL: Last-known-good must never poison HARD RECOVERY.
During entropy death / red zone, reward-drop rollback may restore reward_weights
from LKG but entropy_scale + OP mutes always come from the recovery template.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Floor below which an LKG entropy_scale must not be restored during death
_RECOVERY_ENTROPY_FLOOR = 0.08
_HARD_RECOVERY_TARGET = 0.35

# Keys that HARD RECOVERY owns — never take these from a poisoned / healthy-era LKG
# while entropy is dead (reward_weights are intentionally NOT in this set).
_RECOVERY_OWNED_KEYS = frozenset(
    {
        "entropy_scale",
        "entropy_death_recovery",
        "entropy_death_tier",
        "hard_recovery",
        "freeze_op_chaos",
        "safety_zone",
        "es_noise_scale",
        "es_enabled",
        "pbt_paused",
        "op_destructive_paused",
        "truncation_pbt_paused",
        "meta_gradients_paused",
        "env_architect_paused",
        "compete_rotate_paused",
        "priority_sampling",
        "skill_tracker_enabled",
        "epochs",
        "event_advantage_boost",
        "policy_lr",
        "critic_lr",
        "var_max",
        "var_min",
        "clip_range",
        "mask_entropy",
        "opponent_pool_chance",
        "train_against_old_chance",
        "opponent_weight_nexto",
        "opponent_weight_necto",
        "opponent_weight_nexto_tled",
        "opponent_weight_requiem",
        "gpu_reset_kickoff",
        "gpu_reset_fuzzed",
        "gpu_reset_aerial",
        "kickoff_weight",
        "fuzzed_weight",
        "aerial_weight",
        "w_icm",
        "w_rnd",
        "entropy_recovery_rollback",
        "note",
    }
)


def save_snapshot(watch_dir: Path, overrides: dict, status: dict, tag: str) -> Path:
    snap_dir = watch_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = snap_dir / f"{tag}_{ts}.json"
    payload = {
        "tag": tag,
        "time": datetime.now(timezone.utc).isoformat(),
        "overrides": overrides,
        "status_metrics": status.get("last_metrics", {}),
        "timesteps": status.get("total_timesteps"),
        "phase": status.get("curriculum_phase"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def lkg_path(watch_dir: Path) -> Path:
    return watch_dir / "snapshots" / "last_known_good.json"


def save_last_known_good(
    watch_dir: Path,
    overrides: dict[str, Any],
    status: dict[str, Any],
) -> Path:
    """Persist last-known-good overrides + key metrics (healthy cycle only)."""
    snap_dir = watch_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = lkg_path(watch_dir)
    metrics = status.get("last_metrics") or {}
    payload = {
        "tag": "last_known_good",
        "time": datetime.now(timezone.utc).isoformat(),
        "overrides": dict(overrides),
        "status_metrics": {
            k: metrics.get(k)
            for k in (
                "Average Step Reward",
                "Policy Entropy",
                "KL Div Loss",
                "Mean KL Divergence",
                "Touch Ratio",
                "Overall Steps/Second",
            )
            if k in metrics
        },
        "timesteps": status.get("total_timesteps"),
        "phase": status.get("curriculum_phase"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_last_known_good(watch_dir: Path) -> dict[str, Any] | None:
    path = lkg_path(watch_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metric_entropy(metrics: dict[str, Any] | None) -> float:
    m = metrics or {}
    try:
        return float(m.get("Policy Entropy", m.get("policy_entropy", 0.5)))
    except (TypeError, ValueError):
        return float("nan")


def lkg_is_poisoned(lkg: dict[str, Any] | None) -> bool:
    """True if LKG was snapshotted during death / carries recovery flags / dead H."""
    if not lkg:
        return False
    ov = lkg.get("overrides") or {}
    if ov.get("entropy_death_recovery") or ov.get("hard_recovery") or ov.get("freeze_op_chaos"):
        return True
    h = _metric_entropy(lkg.get("status_metrics"))
    if math.isnan(h) or math.isinf(h):
        return True
    from .event_driven import ENTROPY_RECOVER_HOLD, is_entropy_dead

    if is_entropy_dead(h) or h < ENTROPY_RECOVER_HOLD:
        return True
    try:
        es = float(ov.get("entropy_scale") or 0.0)
    except (TypeError, ValueError):
        es = 0.0
    # Absurdly low coef is not a usable "known good"
    if es > 0.0 and es < 0.005:
        return True
    return False


def discard_poisoned_lkg(watch_dir: Path) -> bool:
    """Delete poisoned LKG file. Returns True if removed."""
    lkg = load_last_known_good(watch_dir)
    if not lkg_is_poisoned(lkg):
        return False
    path = lkg_path(watch_dir)
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def entropy_scale_below_recovery_floor(overrides: dict[str, Any] | None) -> bool:
    """True if overrides would undo HARD RECOVERY (coef below recovery floor)."""
    ov = overrides or {}
    try:
        es = float(ov.get("entropy_scale") or 0.0)
    except (TypeError, ValueError):
        return True
    return es < _RECOVERY_ENTROPY_FLOOR


def stage_safe_recovery_template(
    policy_entropy: float,
    *,
    baseline_entropy_scale: float = 0.022,
    cycle: int = 1,
) -> dict[str, Any]:
    """Stage-safe HARD RECOVERY overrides (never trust LKG entropy/OP knobs)."""
    from .event_driven import entropy_death_recovery_patch, is_entropy_dead

    if not is_entropy_dead(policy_entropy):
        return {}
    patch, log = entropy_death_recovery_patch(
        policy_entropy=policy_entropy,
        baseline_entropy_scale=baseline_entropy_scale,
        prev_coef=baseline_entropy_scale,
        cycle=max(1, int(cycle)),
    )
    if not patch:
        return {}
    try:
        es = float(patch.get("entropy_scale") or 0.0)
    except (TypeError, ValueError):
        es = 0.0
    # Critical death: pin to recovery target; otherwise enforce floor
    if policy_entropy < 0.001:
        es = max(es, _HARD_RECOVERY_TARGET)
    else:
        es = max(es, _RECOVERY_ENTROPY_FLOOR)
    patch["entropy_scale"] = min(_HARD_RECOVERY_TARGET, max(es, _RECOVERY_ENTROPY_FLOOR))
    if policy_entropy < 0.001:
        patch["entropy_scale"] = _HARD_RECOVERY_TARGET
    patch["note"] = log
    return patch


def is_healthy_for_lkg(
    metrics: dict[str, Any] | None,
    cfg: dict | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    in_recovery: bool = False,
    safety_zone: str | None = None,
) -> bool:
    """Healthy enough to refresh LKG snapshot.

    Never save LKG while entropy_death / red zone / H < soft threshold / recovery active.
    """
    from .event_driven import ENTROPY_DEATH_SOFT, ENTROPY_RECOVER_HOLD, is_entropy_dead

    if in_recovery:
        return False
    if safety_zone and str(safety_zone).lower() in ("red", "yellow"):
        return False

    ov = overrides or {}
    if ov.get("entropy_death_recovery") or ov.get("hard_recovery") or ov.get("freeze_op_chaos"):
        return False
    # LKG entropy_scale must be in the sane post-recovery band (not 0.35 spike)
    try:
        es = float(ov.get("entropy_scale")) if "entropy_scale" in ov else float("nan")
    except (TypeError, ValueError):
        es = float("nan")
    if not math.isnan(es) and not (0.01 <= es <= 0.08):
        return False

    m = metrics or {}
    entropy = _metric_entropy(m)
    if math.isnan(entropy) or math.isinf(entropy):
        return False
    if is_entropy_dead(entropy) or entropy < ENTROPY_RECOVER_HOLD or entropy < ENTROPY_DEATH_SOFT:
        return False
    try:
        reward = float(m.get("Average Step Reward", m.get("avg_reward", 0.0)))
    except (TypeError, ValueError):
        reward = 0.0
    if math.isnan(reward) or math.isinf(reward):
        return False
    try:
        kl = float(m.get("KL Div Loss", m.get("Mean KL Divergence", 0.0)))
    except (TypeError, ValueError):
        kl = 0.0
    if math.isnan(kl) or math.isinf(kl):
        return False
    kl_lim = float(((cfg or {}).get("safety") or {}).get("kl_explosion_threshold", 0.12))
    if kl > kl_lim:
        return False
    return True


def load_last_good(watch_dir: Path) -> dict | None:
    # Prefer dedicated LKG file (skip if poisoned)
    lkg = load_last_known_good(watch_dir)
    if lkg and not lkg_is_poisoned(lkg):
        return lkg
    if lkg and lkg_is_poisoned(lkg):
        discard_poisoned_lkg(watch_dir)
    state_path = watch_dir / "orchestrator_state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    snap = state.get("last_good_snapshot")
    if not snap:
        return None
    p = Path(snap)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def should_rollback(
    cfg: dict,
    before_metrics: dict,
    after_metrics: dict,
) -> tuple[bool, str]:
    rb = cfg.get("rollback") or {}
    if not rb.get("enabled", True):
        return False, ""

    def f(m: dict, k: str) -> float | None:
        v = m.get(k)
        return float(v) if v is not None else None

    before_r = f(before_metrics, "Average Step Reward")
    after_r = f(after_metrics, "Average Step Reward")
    if before_r and after_r and before_r > 0:
        drop = (before_r - after_r) / before_r
        if drop > float(rb.get("reward_drop_threshold", 0.15)):
            return True, f"Average Step Reward dropped {drop:.1%}"

    before_t = f(before_metrics, "Player/Ball Touch Ratio")
    after_t = f(after_metrics, "Player/Ball Touch Ratio")
    if before_t and after_t:
        if before_t - after_t > float(rb.get("touch_ratio_drop_threshold", 0.03)):
            return True, f"Touch ratio dropped {before_t - after_t:.3f}"

    # Entropy death / NaN / KL — same-cycle style triggers for pending rollback checks
    after_e = f(after_metrics, "Policy Entropy")
    if after_e is not None:
        if math.isnan(after_e) or after_e < float(rb.get("entropy_death_threshold", 0.05)):
            return True, f"Policy Entropy collapsed ({after_e})"
    after_kl = f(after_metrics, "KL Div Loss") or f(after_metrics, "Mean KL Divergence")
    if after_kl is not None and (
        math.isnan(after_kl)
        or after_kl > float(((cfg.get("safety") or {}).get("kl_explosion_threshold", 0.12)))
    ):
        return True, f"KL toxic ({after_kl})"

    return False, ""


def restore_snapshot(watch_dir: Path, snapshot: dict) -> dict:
    """Raw restore — prefer safe_rollback_patch during entropy death."""
    ov = dict(snapshot.get("overrides") or {})
    ov["note"] = "ROLLBACK: restored last known good overrides"
    return ov


def merge_lkg_keep_recovery(
    snapshot_overrides: dict[str, Any],
    metrics: dict[str, Any] | None,
    *,
    reason: str = "",
    baseline_entropy_scale: float = 0.022,
    current_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore reward_weights (and non-OP knobs) from LKG; HARD RECOVERY wins on entropy/OP."""
    from .event_driven import is_entropy_dead

    h = _metric_entropy(metrics)
    if math.isnan(h) or math.isinf(h):
        h = 0.0

    cur = current_overrides or {}
    try:
        held = float(cur.get("entropy_scale") or 0.0)
    except (TypeError, ValueError):
        held = 0.0

    rec = stage_safe_recovery_template(
        h,
        baseline_entropy_scale=baseline_entropy_scale,
        cycle=max(1, int(cur.get("entropy_recovery_cycles") or 4)),
    )
    # Never drop below an already-active recovery spike or hard target while dead
    floor = _HARD_RECOVERY_TARGET if (is_entropy_dead(h) or h < 0.05) else _RECOVERY_ENTROPY_FLOOR
    rec["entropy_scale"] = max(
        float(rec["entropy_scale"]),
        held if held >= _RECOVERY_ENTROPY_FLOOR else 0.0,
        floor,
    )

    # Start from recovery template, then pull non-owned keys (esp. reward_weights) from LKG
    out = dict(rec)
    for k, v in (snapshot_overrides or {}).items():
        if k in _RECOVERY_OWNED_KEYS:
            continue
        if k in ("updated_at",):
            continue
        out[k] = v

    # Explicitly keep reward_weights from LKG when present
    rw = (snapshot_overrides or {}).get("reward_weights")
    if isinstance(rw, dict) and rw:
        out["reward_weights"] = dict(rw)

    out["entropy_death_recovery"] = True
    out["hard_recovery"] = True
    out["freeze_op_chaos"] = True
    out["es_noise_scale"] = 0.0
    out["priority_sampling"] = False
    out["skill_tracker_enabled"] = False
    out["epochs"] = 1
    detail = _sanitize_recovery_detail(reason)
    out["note"] = (
        f"[AutoTrainer] ROLLBACK+HARD_RECOVERY ({detail}) "
        f"restored rewards only, entropy_scale={float(out['entropy_scale']):.3f} es=0 OP=MUTED"
    )
    return out


def _sanitize_recovery_detail(reason: str | None) -> str:
    """Avoid nested ``ROLLBACK+HARD_RECOVERY ([AutoTrainer] ROLLBACK+...)`` notes."""
    import re

    detail = (reason or "").strip() or "metric drop during entropy death"
    if "ROLLBACK" in detail or detail.startswith("[AutoTrainer]"):
        # Prefer deepest short parenthetical that is not itself a banner
        for cand in reversed(re.findall(r"\(([^()]{2,100})\)", detail)):
            c = cand.strip()
            if c and "ROLLBACK" not in c and not c.startswith("[AutoTrainer]"):
                return c[:160]
        return "recovery"
    return detail[:160]


def safe_rollback_patch(
    snapshot: dict[str, Any],
    metrics: dict[str, Any] | None,
    *,
    reason: str = "",
    in_hard_recovery: bool = False,
    current_overrides: dict[str, Any] | None = None,
    baseline_entropy_scale: float = 0.022,
) -> tuple[dict[str, Any], str]:
    """Build rollback overrides. Recovery always wins over LKG entropy/OP mutes.

    - Red zone / HARD_RECOVERY / dead H: merge rewards from snapshot, keep recovery.
    - LKG entropy_scale below recovery floor: discard entropy knobs, use template.
    - Green / healthy: restore snapshot as-is.
    """
    from .event_driven import is_entropy_dead

    ov = dict(snapshot.get("overrides") or {})
    h = _metric_entropy(metrics)
    if math.isnan(h) or math.isinf(h):
        h = 0.0
    cur = current_overrides or {}
    recovering = bool(
        in_hard_recovery
        or is_entropy_dead(h)
        or cur.get("hard_recovery")
        or cur.get("entropy_death_recovery")
        or cur.get("freeze_op_chaos")
    )
    poisoned_coef = entropy_scale_below_recovery_floor(ov)

    if recovering or poisoned_coef:
        merged = merge_lkg_keep_recovery(
            ov,
            metrics,
            reason=reason,
            baseline_entropy_scale=baseline_entropy_scale,
            current_overrides=cur,
        )
        return merged, str(merged.get("note") or "")

    ov["note"] = f"[AutoTrainer] ROLLBACK ({reason})" if reason else (
        "[AutoTrainer] ROLLBACK: restored last known good overrides"
    )
    return ov, str(ov["note"])


def instant_lkg_recovery_patch(
    watch_dir: Path,
    metrics: dict[str, Any] | None,
    cfg: dict | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """If metrics are toxic, return LKG rewards + HARD RECOVERY (same cycle)."""
    from .safety import metrics_are_toxic

    toxic, why = metrics_are_toxic(metrics, (cfg or {}).get("safety"))
    if not toxic:
        return None, ""

    if discard_poisoned_lkg(watch_dir):
        # Poisoned LKG removed — use stage-safe template only
        h = _metric_entropy(metrics)
        if math.isnan(h):
            h = 0.0
        ov = stage_safe_recovery_template(h)
        ov["entropy_recovery_rollback"] = True
        ov["note"] = (
            f"[AutoTrainer] CRITICAL RECOVERY (poisoned LKG discarded): {why} "
            f"entropy_scale={float(ov['entropy_scale']):.3f}"
        )
        return ov, why

    lkg = load_last_known_good(watch_dir) or load_last_good(watch_dir)
    if not lkg:
        h = _metric_entropy(metrics)
        if math.isnan(h):
            h = 0.0
        ov = stage_safe_recovery_template(h)
        ov["entropy_recovery_rollback"] = True
        ov["note"] = (
            f"[AutoTrainer] CRITICAL RECOVERY (no LKG): {why} "
            f"entropy_scale={float(ov['entropy_scale']):.3f}"
        )
        return ov, why

    ov, note = safe_rollback_patch(
        lkg,
        metrics,
        reason=why,
        in_hard_recovery=True,
    )
    ov["entropy_recovery_rollback"] = True
    ov["note"] = note
    return ov, why
