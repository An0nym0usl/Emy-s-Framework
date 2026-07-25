"""
SSL-enterprise guide + full-autonomy autopilot for GigaLearnRL AutoTrainer.

Taken from the pasted AutoTrainer guide (feasible NOW on RocketSim/CUDA):
  §2 lightweight metrics heuristics (Phase 2 full savestate deferred)
  §3 procedural fuzzy scenario mix -> GPU reset curriculum weights
  §4 reward shaping decay by timesteps + stagnation escape
  §5 league sampling 60% current / 30% old / 10% hardened experts

Also adapted from https://gigalearn-guide.netlify.app/ (Kue-inspired community guide):
  Stage 1 contact -> suppress goals / amplify touch+air
  Stage 2 purpose -> introduce VelocityBallToGoal + Goal + StrongTouch
  Stage 3 strategy -> reduce concede, gamma bumps, old-version sparring
  LR 2e-4 -> 1e-4 -> <0.8e-4; epochs 1->2; AirReward critical
  Gamma increments every ~1–1.5B (conservative +0.0005) on POWER baseline
  See docs/GIGALEARN_GUIDE_ADAPTATION.md for full mapping + what we skipped (tsPerItr 50k, etc.)

Full autonomy (ssl_autonomy / full_control):
  Long-horizon SSL-track phases (mechanical -> aerial -> mid -> late -> SSL pressure).
  Writes the full override surface every cycle; C++ re-applies after Apex.
  Logs: "SSL-track phase X" until metrics warrant — never fake SSL.

Opt-out: GIGA_NO_SSL_AUTONOMY=1  or  ssl_autonomy.enabled / full_control: false
"""

from __future__ import annotations

import os
from typing import Any

from .event_driven import (
    ENTROPY_CLEAR_EXIT,
    ENTROPY_FULLY_RECOVERED,
    ENTROPY_HEALTHY,
    entropy_death_recovery_patch,
    taper_entropy_scale,
)


def play_hours(timesteps: int | float, tick_skip: int = 6) -> float:
    """Kue / web guide: Hours = (timesteps * tick_skip) / (120 * 3600)."""
    return (float(timesteps) * float(tick_skip)) / (120.0 * 3600.0)


# Early pressure (touch / chase) -> late possession / shot / goal focus
# Includes CPU class names + GPU-native registry aliases (CudaEnvSet GpuRewardRegistryName).
_EARLY_TOUCH = (
    "TouchBallReward",
    "VelocityPlayerToBallReward",
    "FaceBallReward",
    "TouchAccelReward",
    "StrongTouchReward",
)
_LATE_GOAL = (
    "GoalReward",
    "VelocityBallToGoalReward",
    "ShotReward",
    "SaveReward",
    "PossessionReward",
    "BallPossessionReward",
    "EnergyReward",
    "BoostPickupReward",
    "PickupBoostReward",
    "SaveBoostReward",
)

# Canonical Stage-1 / from-scratch blank multipliers (× GPU base).
# NOT absolute reward weights — C++ does base × RuntimeRewardRegistry.
# Blank template: Goal + Touch + VelBallToGoal + VelPlayerToBall.
_STARTER_STAGE1: dict[str, float] = {
    "GoalReward": 1.0,
    "TouchBallReward": 1.0,
    "VelocityBallToGoalReward": 1.0,
    "VelocityPlayerToBallReward": 1.0,
}
# Guide: AirReward is critical early or bots "forget" how to jump.
_AERIAL = (
    "AirReward",
    "AirDribbleReward",
    "AerialTouchHeightReward",
)
_STRONG_TOUCH = (
    "StrongTouchReward",
    "StrongTouch",
    "TouchAccelReward",
    "TouchAccel",
)
_KICKOFF = (
    "KickoffProximity",
)
_BOOST = (
    "PickupBoostReward",
    "BoostPickupReward",
    "SaveBoostReward",
)

# Long-horizon SSL-track program (phase labels — not rank claims).
# Step gates aligned with gigalearn-guide.netlify.app roadmap (0–100M / 100–500M / 500M+),
# plus two finer late phases for POWER from-scratch autonomy.
_SSL_TRACK = (
    {
        "id": "mechanical",
        "label": "SSL-track phase 1/5 (mechanical / contact)",
        "min_steps": 0,
        "touch_min": 0.08,
        "air_min": 0.0,
        "reward_delta_min": -1e9,
        "guide_stage": 1,  # Make Contact — no goal focus yet
    },
    {
        "id": "aerial",
        "label": "SSL-track phase 2/5 (aerial / purpose)",
        "min_steps": 100_000_000,  # guide Stage 2 start
        "touch_min": 0.12,
        "air_min": 0.04,
        "reward_delta_min": 0.0,
        "guide_stage": 2,  # Hit With Purpose
    },
    {
        "id": "mid",
        "label": "SSL-track phase 3/5 (midfield / possession)",
        "min_steps": 250_000_000,
        "touch_min": 0.16,
        "air_min": 0.06,
        "reward_delta_min": 0.0,
        "guide_stage": 2,
    },
    {
        "id": "late",
        "label": "SSL-track phase 4/5 (late pressure)",
        "min_steps": 500_000_000,  # guide Stage 3 start
        "touch_min": 0.18,
        "air_min": 0.08,
        "reward_delta_min": 0.0,
        "guide_stage": 3,  # Strategy & Refinement
    },
    {
        "id": "ssl_pressure",
        "label": "SSL-track phase 5/5 (SSL pressure — still training)",
        "min_steps": 1_200_000_000,
        "touch_min": 0.20,
        "air_min": 0.10,
        "reward_delta_min": 0.0,
        "guide_stage": 3,
    },
)


def _cfg(ssl: dict[str, Any] | None) -> dict[str, Any]:
    return ssl or {}


def _autonomy_block(ssl: dict[str, Any] | None, root_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge ssl_autonomy from root config or nested under ssl_guide."""
    root = root_cfg or {}
    nested = _cfg(ssl).get("autonomy") or _cfg(ssl).get("ssl_autonomy") or {}
    top = root.get("ssl_autonomy") if isinstance(root.get("ssl_autonomy"), dict) else {}
    # Nested/ssl keys win over empty; top-level ssl_autonomy is the master block.
    out = dict(nested)
    out.update(top or {})
    return out


def enabled(ssl: dict[str, Any] | None) -> bool:
    return bool(_cfg(ssl).get("enabled", True))


def autonomy_opted_out() -> bool:
    """Env kill-switch (checked first)."""
    v = os.environ.get("GIGA_NO_SSL_AUTONOMY", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def from_scratch_context() -> bool:
    for key in ("GIGA_FROM_SCRATCH", "GIGA_SSL_AUTONOMY"):
        v = os.environ.get(key, "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
    return False


def full_control_enabled(
    ssl: dict[str, Any] | None,
    root_cfg: dict[str, Any] | None = None,
) -> bool:
    """
    Master switch for full-autonomy autopilot.
    Default ON when AutoTrainer runs with from-scratch / GIGA_SSL_AUTONOMY,
    or when config ssl_autonomy.enabled / full_control is true (config default).
    Opt-out: GIGA_NO_SSL_AUTONOMY=1 or config enabled/full_control false.
    Also: GIGA_AT_READONLY=1 -> observer (no writes / C++ ignores).
    """
    if autonomy_opted_out():
        return False
    if os.environ.get("GIGA_AT_READONLY", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    root = root_cfg or {}
    # Flat keys (config.default.yaml top-level full_control: true)
    if "full_control" in root and not bool(root.get("full_control")):
        return False
    if bool(root.get("full_control")):
        return True
    auto = _autonomy_block(ssl, root_cfg)
    if "enabled" in auto and not bool(auto.get("enabled")):
        return False
    if "full_control" in auto and not bool(auto.get("full_control")):
        return False
    # Explicit enable
    if bool(auto.get("enabled", False)) or bool(auto.get("full_control", False)):
        return True
    # Default: on for from-scratch / fresh train when auto_enable_from_scratch
    if bool(auto.get("auto_enable_from_scratch", True)) and from_scratch_context():
        return True
    # Config default in config.default.yaml sets enabled+full_control true —
    # if block present with neither key, treat as on when guide enabled.
    if auto and enabled(ssl):
        return bool(auto.get("enabled", True)) and bool(auto.get("full_control", True))
    return False


def progress01(timesteps: int, horizon: int = 2_000_000_000) -> float:
    """0 early -> 1 late (clamped)."""
    if horizon <= 0:
        return 0.0
    return max(0.0, min(1.0, float(timesteps) / float(horizon)))


def starter_reward_multipliers(ssl: dict[str, Any] | None = None) -> dict[str, float]:
    """
    Canonical Stage-1 / from-scratch blank multipliers (× GPU base).

    Default blank: Goal + Touch + VelocityPlayerToBall @ 1.0
      (bases: Goal 100, Touch 5, VelP2B 2 → effective same).
    Config: ssl_guide.web_guide.starter_rewards (partial override) or reward_shaping.starter.
    """
    out = dict(_STARTER_STAGE1)
    wg = _cfg(ssl).get("web_guide") or {}
    rs = _cfg(ssl).get("reward_shaping") or {}
    for src in (rs.get("starter"), wg.get("starter_rewards")):
        if isinstance(src, dict):
            for k, v in src.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
    # Keep Goal / touch peaks aligned with stage1_* knobs when set (blank stack only)
    if "stage1_goal_mult" in wg or "stage1_goal_mult" in rs:
        g = float(wg.get("stage1_goal_mult", rs.get("stage1_goal_mult", out.get("GoalReward", 1.0))))
        out["GoalReward"] = g
    if "stage1_touch_mult" in wg:
        t = float(wg["stage1_touch_mult"])
        out["TouchBallReward"] = max(float(out.get("TouchBallReward", 1.0)), t)
        out["VelocityPlayerToBallReward"] = max(
            float(out.get("VelocityPlayerToBallReward", 1.0)), t
        )
    return out


def format_starter_rewards_log(weights: dict[str, float] | None) -> str:
    """One-line full starter dump for boot."""
    if not weights:
        return "[AutoTrainer] starter rewards (blank): (none)"
    parts = ", ".join(
        f"{k}={float(v):.2f}"
        for k, v in sorted(weights.items(), key=lambda kv: (kv[0].lower(), kv[0]))
    )
    return f"[AutoTrainer] starter rewards (blank): {parts}"


def _looks_like_late_reward_pollution(mult: dict[str, float] | None) -> bool:
    """True if leftover LKG / aerial SSL pressure would poison early boot."""
    m = mult or {}
    if not m:
        return False
    try:
        vb = float(m.get("VelocityBallToGoalReward") or m.get("VelocityBallToGoal") or 0)
        touch = float(m.get("TouchBallReward") or m.get("TouchReward") or 0)
        goal = float(m.get("GoalReward") or 0)
    except (TypeError, ValueError):
        return False
    # VelBall maxed while touch not high → classic bad early-mid leftover
    if vb >= 2.4 and touch > 0 and touch < 1.55:
        return True
    # Goal near 1.0+ with weak touch (late pressure leftover)
    if goal >= 0.90 and touch > 0 and touch < 1.50:
        return True
    return False


def should_seed_starter_rewards(
    timesteps: int,
    track_id: str | None,
    current_mult: dict[str, float] | None,
    ssl: dict[str, Any] | None,
    *,
    state: dict[str, Any] | None = None,
    force: bool = False,
) -> bool:
    """
    Seed the canonical starter set (replace, don't blend LKG leftovers) when:
      - force / autonomy bootstrap
      - mechanical track
      - from-scratch (always until 100M — ignore stale ts / on-disk mult)
      - early timesteps with late-stage reward pollution in current_mult
    """
    if force:
        return True
    wg = _cfg(ssl).get("web_guide") or {}
    if not bool(wg.get("enabled", True)):
        return False
    if not bool(wg.get("seed_starter_on_boot", True)):
        return False
    early_steps = int(wg.get("starter_seed_max_steps", 100_000_000))
    st = (state or {}).get("ssl_autonomy") or {}
    # From-scratch: force starter until 100M (phase schedule). Ignore poisoned
    # on-disk multipliers / stale orchestrator flags from a prior run.
    if from_scratch_context() and timesteps < early_steps:
        return True
    if track_id == "mechanical":
        return True
    if timesteps < early_steps and _looks_like_late_reward_pollution(current_mult):
        return True
    # First OP autonomy tick (no prior ssl starter applied)
    if timesteps < early_steps and not st.get("starter_rewards_applied"):
        return True
    return False


def reward_shaping_decay(
    timesteps: int,
    phase: int,
    current_mult: dict[str, float],
    manifest: dict[str, Any] | None,
    ssl: dict[str, Any] | None,
    *,
    track_id: str | None = None,
    state: dict[str, Any] | None = None,
    force_starter: bool = False,
) -> dict[str, float]:
    """
    §4 — Early: amplify touch / vel-to-ball / air. Later: decay those; raise goal-oriented.

    Guide-backed (gigalearn-guide.netlify.app) on POWER/default:
      Stage 1 (mechanical): touch/chase/air HIGH; Goal present (~0.45×) not mute-to-~0
      Stage 2 (aerial/mid): introduce goals + StrongTouch (no VelBall×3 clamp yet)
      Stage 3 (late+): goal-dominant shaping, touch floored; VelBall/VelPlayer ratio

    Returns full multiplier map (merged with current, or starter-seeded).
    Live-tunable via RuntimeRewardRegistry (reward_weights overrides).
    Values are MULTIPLIERS on GPU default bases (Goal 400, VelB2G 20, Touch 3, …).
    """
    if not enabled(ssl):
        return dict(current_mult)

    rs = _cfg(ssl).get("reward_shaping") or {}
    if not rs.get("enabled", True):
        return dict(current_mult)

    wg = _cfg(ssl).get("web_guide") or {}
    guide_on = bool(wg.get("enabled", True))

    seed_starter = should_seed_starter_rewards(
        timesteps, track_id, current_mult, ssl, state=state, force=force_starter
    )
    starter = starter_reward_multipliers(ssl) if seed_starter else None

    horizon = int(rs.get("horizon_steps", 2_000_000_000))
    p = progress01(timesteps, horizon)
    # Phase also advances decay (chase holds early longer)
    if phase <= 0:
        p = min(p, 0.25)
    elif phase == 1:
        p = min(max(p, 0.35), 0.65)
    else:
        p = max(p, 0.55)

    # SSL-track overrides curriculum phase for shaping bias
    if track_id == "mechanical":
        p = min(p, 0.20)
    elif track_id == "aerial":
        p = min(max(p, 0.30), 0.45)
    elif track_id == "mid":
        p = min(max(p, 0.45), 0.65)
    elif track_id == "late":
        p = min(max(p, 0.60), 0.80)
    elif track_id == "ssl_pressure":
        p = max(p, 0.75)

    early_peak = float(rs.get("early_touch_mult", 1.55))
    late_floor = float(rs.get("late_touch_mult", 0.85))
    late_goal_peak = float(rs.get("late_goal_mult", 1.45))
    early_goal_floor = float(rs.get("early_goal_mult", 0.95))

    # Stage 1: Goal present (~0.45× → ~180 effective) — never crush to ~0.02 (was mute).
    if guide_on and track_id == "mechanical":
        early_goal_floor = float(wg.get("stage1_goal_mult", rs.get("stage1_goal_mult", 0.45)))
        early_peak = max(early_peak, float(wg.get("stage1_touch_mult", 1.85)))
    elif guide_on and track_id == "aerial":
        # Stage 2 intro — goals climb in but don't dominate yet
        early_goal_floor = min(early_goal_floor, float(wg.get("stage2_goal_mult", 0.65)))
        late_goal_peak = max(late_goal_peak, float(wg.get("stage2_goal_peak", 1.15)))

    touch_m = early_peak * (1.0 - p) + late_floor * p
    goal_m = early_goal_floor * (1.0 - p) + late_goal_peak * p
    # Guide: AirReward critical early — keep aerial above 1.0 through contact/purpose.
    # With DefaultAction + maskEntropy, bots already go aerial ~10x more (Kue) — don't over-amp.
    if guide_on and track_id in ("mechanical", "aerial"):
        air_base = float(wg.get("early_air_mult", 1.35))
        air_base *= float(wg.get("action_masked_air_scale", 0.96))
        aerial_m = air_base * (1.0 - 0.25 * p) + (0.95 + 0.45 * p) * 0.35
        aerial_m = max(1.05, min(1.65, aerial_m))
    else:
        aerial_m = 0.95 + 0.45 * min(1.0, max(0.0, (p - 0.25) / 0.55))
    kickoff_m = 1.35 * (1.0 - 0.55 * p) + 0.90 * p
    boost_m = 1.05 + 0.25 * p
    # StrongTouch: moderate Stage 1; raise Stage 2+
    if guide_on and track_id == "mechanical":
        strong_m = float(wg.get("stage1_strong_touch_mult", 1.00))
    elif guide_on and track_id in ("aerial", "mid"):
        strong_m = float(wg.get("stage2_strong_touch_mult", 1.35))
    else:
        strong_m = 1.0 + 0.15 * p

    # Mechanical / boot: REPLACE with starter (do not blend poisoned LKG / aerial leftovers).
    if seed_starter and starter is not None and (
        track_id == "mechanical" or timesteps < int(wg.get("starter_seed_max_steps", 100_000_000))
    ):
        out = dict(starter)
        # Light schedule nudge on top of starter (stay near recipe).
        # Exclude StrongTouch* from early-touch blend so moderate stage1_strong stays ~1.0.
        _strong_set = set(_STRONG_TOUCH)
        for name in _EARLY_TOUCH:
            if name in out and name not in _strong_set:
                out[name] = max(0.01, min(3.0, 0.70 * float(out[name]) + 0.30 * touch_m))
        for name in _LATE_GOAL:
            if name in out:
                # Keep Goal/VelBall near starter floor — schedule may not yank Goal to ~0.02
                floor = float(starter.get(name, 0.40))
                out[name] = max(0.05, min(3.0, 0.75 * float(out[name]) + 0.25 * max(goal_m, floor)))
        for name in _AERIAL:
            if name in out:
                out[name] = max(0.05, min(3.0, 0.70 * float(out[name]) + 0.30 * aerial_m))
        for name in _KICKOFF:
            if name in out:
                out[name] = max(0.05, min(3.0, 0.70 * float(out[name]) + 0.30 * kickoff_m))
        for name in _BOOST:
            if name in out:
                out[name] = max(0.05, min(3.0, 0.70 * float(out[name]) + 0.30 * boost_m))
        for name in _STRONG_TOUCH:
            if name in out:
                # Pin near starter strong (moderate); tiny schedule drift only
                base = float(starter.get(name, strong_m))
                out[name] = max(0.05, min(3.0, 0.85 * base + 0.15 * strong_m))
    else:
        # Blend scheduled multipliers with any existing AutoTrainer / env-architect values
        # so OP ticks do not thrash reward_weights every cycle.
        out = dict(current_mult) if current_mult else {}
        if starter and not out:
            out = dict(starter)
        for name in _EARLY_TOUCH:
            prev = float(out.get(name, 1.0))
            out[name] = max(0.01, min(3.0, 0.55 * prev + 0.45 * touch_m))

        for name in _LATE_GOAL:
            prev = float(out.get(name, 1.0))
            floor = 0.05
            if guide_on and track_id == "mechanical":
                # Hard schedule toward stage1 goal floor (present, not mute)
                out[name] = max(floor, min(3.0, goal_m))
            else:
                out[name] = max(floor, min(3.0, 0.55 * prev + 0.45 * goal_m))

        for name in _AERIAL:
            prev = float(out.get(name, 1.0))
            out[name] = max(0.05, min(3.0, 0.55 * prev + 0.45 * aerial_m))

        for name in _KICKOFF:
            prev = float(out.get(name, 1.0))
            out[name] = max(0.05, min(3.0, 0.55 * prev + 0.45 * kickoff_m))

        for name in _BOOST:
            prev = float(out.get(name, 1.0))
            out[name] = max(0.05, min(3.0, 0.55 * prev + 0.45 * boost_m))

        for name in _STRONG_TOUCH:
            prev = float(out.get(name, 1.0))
            out[name] = max(0.05, min(3.0, 0.55 * prev + 0.45 * strong_m))

    # Guide weight-balance (relative multipliers): VelBallToGoal ~2–3× VelPlayerToBall
    # only from mid+ (NOT aerial) — early aerial with ratio→3.0 fights contact learning.
    if guide_on and track_id in ("mid", "late", "ssl_pressure"):
        out = _enforce_vel_goal_ratio(out, wg)

    valid = {e["name"] for e in (manifest or {}).get("rewards") or [] if "name" in e}
    # GPU-native path may have empty / CPU-only manifest — keep GPU aliases anyway
    if valid:
        # Keep keys that are either in manifest OR known GPU aliases
        gpu_alias = set(_EARLY_TOUCH + _LATE_GOAL + _AERIAL + _KICKOFF + _BOOST + _STRONG_TOUCH)
        out = {k: v for k, v in out.items() if k in valid or k in gpu_alias}
    return out


def _enforce_vel_goal_ratio(
    mult: dict[str, float],
    wg: dict[str, Any],
) -> dict[str, float]:
    """Keep VelocityBallToGoal multipliers ≥ ratio × VelocityPlayerToBall (guide 2–3×)."""
    out = dict(mult)
    ratio = float(wg.get("vel_ball_to_goal_vs_player_ratio", 2.5))
    ratio = max(1.5, min(3.5, ratio))
    vp_keys = ("VelocityPlayerToBallReward", "VelocityPlayerToBall")
    vb_keys = ("VelocityBallToGoalReward", "VelocityBallToGoal")
    vp = None
    for k in vp_keys:
        if k in out:
            vp = float(out[k])
            break
    if vp is None or vp <= 0:
        return out
    target = vp * ratio
    for k in vb_keys:
        if k in out:
            out[k] = max(float(out[k]), min(3.0, target))
        else:
            # GPU alias may be absent from blended map — seed it for RuntimeRewardRegistry
            out[k] = min(3.0, target)
    return out


def league_sampling_overrides(
    timesteps: int,
    phase: int,
    ssl: dict[str, Any] | None,
    *,
    hardened_available: bool = True,
    sps_safe: bool = False,
    track_id: str | None = None,
    sparring_armed: bool = True,
) -> dict[str, Any]:
    """
    §5 — Target mix among sparring draws:
      current (self-play) / old exploiters / hardened experts = 60 / 30 / 10

    C++ order: OpponentPool first, then old versions, else current.
      P(expert) = opp_chance
      P(old) = (1 - opp_chance) * old_chance
      P(current) = 1 - P(expert) - P(old)

    Solving for 0.10 / 0.30 / 0.60:
      opp_chance = 0.10
      old_chance = 0.30 / 0.90 ≈ 0.333

    Evolves with SSL-track skill phase when autonomy is on.
    """
    if not enabled(ssl):
        return {}

    lg = _cfg(ssl).get("league") or {}
    if not lg.get("enabled", True):
        return {}

    if not sparring_armed:
        return {
            "opponent_pool_chance": 0.0,
            "train_against_old_chance": 0.0,
            "ssl_league_current": 1.0,
            "ssl_league_old": 0.0,
            "ssl_league_expert": 0.0,
            "ssl_guide_post_apex": True,
            "ssl_sparring_warmup": True,
        }

    current = float(lg.get("current", 0.60))
    old_frac = float(lg.get("old_exploiters", 0.30))
    expert_frac = float(lg.get("hardened_experts", 0.10))

    # Track-aware league mix (overrides chase/foundation soft ramp)
    if track_id == "mechanical":
        current, old_frac, expert_frac = 0.92, 0.08, 0.0
    elif track_id == "aerial":
        current, old_frac, expert_frac = 0.78, 0.18, 0.04
    elif track_id == "mid":
        current, old_frac, expert_frac = 0.65, 0.25, 0.10
    elif track_id == "late":
        current, old_frac, expert_frac = 0.55, 0.32, 0.13
    elif track_id == "ssl_pressure":
        current, old_frac, expert_frac = 0.50, 0.35, 0.15
    else:
        total = max(1e-6, current + old_frac + expert_frac)
        current, old_frac, expert_frac = current / total, old_frac / total, expert_frac / total
        # Soft ramp: chase phase stays mostly self-play
        if phase <= 0:
            expert_frac *= float(lg.get("chase_expert_scale", 0.0))
            old_frac *= float(lg.get("chase_old_scale", 0.15))
            rem = max(0.0, 1.0 - expert_frac - old_frac)
            current = rem
        elif phase == 1:
            expert_frac *= float(lg.get("foundation_expert_scale", 0.6))
            old_frac = min(old_frac, 0.22)
            rem = max(0.0, 1.0 - expert_frac - old_frac)
            current = rem

    total = max(1e-6, current + old_frac + expert_frac)
    current, old_frac, expert_frac = current / total, old_frac / total, expert_frac / total

    if not hardened_available:
        # Fold experts into old exploiters, keep current share
        old_frac += expert_frac
        expert_frac = 0.0

    if sps_safe:
        scale = float(lg.get("sps_safe_scale", 0.55))
        expert_frac *= scale
        old_frac *= scale
        rem = max(0.0, 1.0 - expert_frac - old_frac)
        current = rem

    opp_chance = max(0.0, min(0.5, expert_frac))
    denom = max(1e-6, 1.0 - opp_chance)
    old_chance = max(0.0, min(0.5, old_frac / denom))

    out: dict[str, Any] = {
        "opponent_pool_chance": round(opp_chance, 4),
        "train_against_old_chance": round(old_chance, 4),
        "ssl_league_current": round(current, 4),
        "ssl_league_old": round(old_frac, 4),
        "ssl_league_expert": round(expert_frac, 4),
        "ssl_guide_post_apex": True,
    }

    # Weight hardened experts when pool has them (C++ SetOpponentWeight)
    weights = lg.get("expert_weights") or {
        "nexto": 1.0,
        "nexto_tled": 1.05,
        "necto": 0.85,
        "requiem": 1.15,
    }
    if hardened_available and expert_frac > 0:
        # Slight requiem bias in late / ssl_pressure
        scale_req = 1.0
        if track_id in ("late", "ssl_pressure"):
            scale_req = 1.15
        for name, w in weights.items():
            ww = float(w)
            if name == "requiem":
                ww *= scale_req
            out[f"opponent_weight_{name}"] = ww
        out["opponent_beat_bonus"] = float(lg.get("beat_bonus", 70.0 if sps_safe else 100.0))
    else:
        out["ssl_hardened_skipped"] = True

    # Guide Stage 3: reduce concede penalty to encourage aggression (even without experts)
    wg = _cfg(ssl).get("web_guide") or {}
    base_concede = float(lg.get("concede_penalty", -28.0 if sps_safe else -40.0))
    if bool(wg.get("enabled", True)) and track_id in ("late", "ssl_pressure"):
        scale = float(wg.get("late_concede_scale", 0.65))
        base_concede = max(-100.0, min(0.0, base_concede * scale))
        out["opponent_concede_penalty"] = base_concede
    elif hardened_available and expert_frac > 0:
        out["opponent_concede_penalty"] = base_concede

    # Mild skill tracker once foundation+ / aerial+
    if phase >= 1 or (track_id and track_id != "mechanical"):
        out["skill_tracker_enabled"] = bool(lg.get("skill_tracker", True))

    _ = timesteps  # reserved for future timestep-scaled mix
    return out


def scenario_mix_overrides(
    timesteps: int,
    phase: int,
    ssl: dict[str, Any] | None,
    *,
    stagnating: bool = False,
    skill_hint: float = 0.0,
    track_id: str | None = None,
) -> dict[str, Any]:
    """
    §3 — Procedural fuzzy scenario mix -> kickoff / fuzzed / aerial GPU reset weights.
    No fake wall-shot setter: aerial+fuzzed approximate high-ball / awkward states.
    """
    if not enabled(ssl):
        return {}

    sc = _cfg(ssl).get("scenarios") or {}
    if not sc.get("enabled", True):
        return {}

    horizon = int(sc.get("horizon_steps", 2_000_000_000))
    p = progress01(timesteps, horizon)

    if track_id == "mechanical":
        kickoff, fuzzed, aerial = 0.60, 0.35, 0.05
    elif track_id == "aerial":
        kickoff, fuzzed, aerial = 0.35, 0.30, 0.35
    elif track_id == "mid":
        kickoff, fuzzed, aerial = 0.30, 0.35, 0.35
    elif track_id == "late":
        kickoff, fuzzed, aerial = 0.25, 0.30, 0.45
    elif track_id == "ssl_pressure":
        kickoff, fuzzed, aerial = 0.22, 0.28, 0.50
    elif phase <= 0:
        kickoff, fuzzed, aerial = 0.55, 0.40, 0.05
    elif phase == 1:
        kickoff, fuzzed, aerial = 0.40, 0.35, 0.25
    else:
        kickoff, fuzzed, aerial = 0.28, 0.32, 0.40

    # Skill / timestep push aerial later
    aerial += 0.08 * p
    if skill_hint > 0:
        aerial += min(0.10, skill_hint * 0.001)
    if stagnating:
        # Force diversity: more fuzzed + aerial, less pure kickoff
        kickoff *= 0.75
        fuzzed += 0.10
        aerial += 0.08

    s = max(1e-6, kickoff + fuzzed + aerial)
    kickoff, fuzzed, aerial = kickoff / s, fuzzed / s, aerial / s

    # Also map into env-architect CombinedState keys (CPU hybrid path)
    # ball_chase ≈ fuzzed midfield mess; random ≈ residual; kickoff = kickoff
    return {
        "kickoff_weight": round(kickoff, 4),
        "fuzzed_weight": round(fuzzed, 4),
        "aerial_weight": round(aerial, 4),
        "gpu_reset_kickoff": round(kickoff, 4),
        "gpu_reset_fuzzed": round(fuzzed, 4),
        "gpu_reset_aerial": round(aerial, 4),
        "ball_chase_weight": round(fuzzed * 0.7 + aerial * 0.15, 4),
        "random_state_weight": round(fuzzed * 0.3 + aerial * 0.25, 4),
        "ssl_guide_post_apex": True,
    }


def detect_training_stagnation(
    history: list[float],
    entropy_history: list[float] | None = None,
    *,
    window: int = 6,
    reward_eps: float = 1e-3,
    entropy_eps: float = 5e-3,
) -> bool:
    """§4 end — require BOTH reward and entropy flat (avoids false positives on noisy early reward)."""
    if len(history) < window:
        return False
    # Without entropy history, do not fire — early iters often have flat avg reward noise.
    if not entropy_history or len(entropy_history) < window:
        return False
    recent = history[-window:]
    # Ignore all-zero reward windows (metrics not wired yet / skipped).
    if max(abs(x) for x in recent) < 1e-9:
        return False
    reward_flat = (max(recent) - min(recent)) < reward_eps
    er = entropy_history[-window:]
    if max(abs(x) for x in er) < 1e-9:
        return False
    entropy_flat = (max(er) - min(er)) < entropy_eps
    return reward_flat and entropy_flat


def stagnation_escape_overrides(
    ssl: dict[str, Any] | None,
    *,
    current_entropy: float = 0.015,
) -> dict[str, Any]:
    """Bump entropy, force scenario diversity, light event advantage."""
    if not enabled(ssl):
        return {}
    st = _cfg(ssl).get("stagnation") or {}
    if not st.get("enabled", True):
        return {}

    bump = float(st.get("entropy_bump", 0.008))
    return {
        "entropy_scale": min(0.04, max(0.01, current_entropy + bump)),
        "event_advantage_boost": float(st.get("event_boost", 1.65)),
        "es_noise_scale": float(st.get("es_noise", 0.035)),
        "var_max": float(st.get("var_max", 0.75)),
        "priority_sampling": True,
        "ssl_stagnation_escape": True,
        "ssl_guide_post_apex": True,
    }


def cheap_geometry_heuristics(metrics: dict[str, Any] | None) -> dict[str, float]:
    """
    §2 simplified — no savestate rewind. Log-friendly heuristics from existing metrics.
    Phase 2: true geometric error / intercept predictor.
    """
    m = metrics or {}
    out: dict[str, float] = {}

    def _f(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in m and m[k] is not None:
                try:
                    return float(m[k])
                except (TypeError, ValueError):
                    pass
        return default

    touch = _f("Player/Ball Touch Ratio", "Touch Ratio")
    vel = _f("Kickoff/VelTowardBall")
    air = _f("Player/In Air Ratio", "In Air Ratio", "Player Air Ratio")
    goal = _f("Game/Goal Speed", "Goal Rate", "Goals Per Episode")
    # Proxy "positioning error": low touch + low vel-toward-ball while mid training
    out["ssl_proxy_chase_error"] = max(0.0, 1.0 - touch) * max(0.0, 1.0 - min(1.0, vel / 800.0))
    out["ssl_proxy_aerial_gap"] = max(0.0, 0.15 - air)
    out["ssl_proxy_finish_gap"] = max(0.0, 1.0 - min(1.0, goal / 1000.0 if goal > 1 else goal * 50))
    # Guide: action masking -> ~10x aerials; spam-flip if air ratio high while touch still weak
    if air > 0.22 and touch < 0.14:
        out["ssl_proxy_spam_flip"] = min(1.0, (air - 0.22) / 0.20 + (0.14 - touch) / 0.14)
    # Guide graph tips (only meaningful when skipPPOMetrics=false / --full-metrics)
    clip = _f("SB3 Clip Fraction", "Clip Fraction", "PPO Clip Fraction")
    kl = _f("Mean KL Divergence", "KL Divergence", "Approx KL")
    if clip > 0:
        out["ssl_proxy_clip_frac"] = clip
        # Healthy ~0.08 for multi-epoch; near-zero is normal for 1-epoch (ignore)
        out["ssl_proxy_lr_hot"] = 1.0 if clip > 0.15 else 0.0
    if kl > 0:
        out["ssl_proxy_mean_kl"] = kl
    # ZeroSum farming illusion: large raw continuous rewards vs tiny step reward
    raw_vp = _f("Rewards/VelocityPlayerToBallReward", "VelocityPlayerToBallReward")
    if raw_vp > 0.8:
        out["ssl_proxy_zerosum_farm"] = min(1.0, raw_vp / 2.0)
    return out


def hardened_experts_available(status: dict[str, Any] | None, ssl: dict[str, Any] | None = None) -> bool:
    """
    Graceful skip when Nexto/Requiem/Necto weights are absent.
    Prefer status flags from C++; else look for known metric / override hints.
    """
    st = status or {}
    if "opponent_pool_loaded" in st:
        try:
            return int(st["opponent_pool_loaded"]) > 0
        except (TypeError, ValueError):
            pass
    metrics = st.get("last_metrics") or {}
    for k in ("Curriculum/RequiemWeight", "Curriculum/OpponentChance", "Training Vs Fixed Opponent"):
        if k in metrics:
            try:
                if float(metrics[k]) > 0:
                    return True
            except (TypeError, ValueError):
                pass
    # Explicit force in config
    force = (_cfg(ssl).get("league") or {}).get("assume_hardened", None)
    if force is not None:
        return bool(force)
    # Default: allow chance; C++ OpponentPool skips empty pool without crashing
    return True


def _metric_f(metrics: dict[str, Any] | None, *keys: str, default: float = 0.0) -> float:
    m = metrics or {}
    for k in keys:
        if k in m and m[k] is not None:
            try:
                return float(m[k])
            except (TypeError, ValueError):
                pass
    return default


def resolve_ssl_track(
    timesteps: int,
    metrics: dict[str, Any] | None,
    ssl: dict[str, Any] | None,
    root_cfg: dict[str, Any] | None = None,
    *,
    reward_history: list[float] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Pick SSL-track phase from timesteps + graduation gates.
    Returns (track_dict, graduation_log). Never claims SSL rank.
    """
    auto = _autonomy_block(ssl, root_cfg)
    track_cfg = auto.get("ssl_track") or {}
    phases = list(_SSL_TRACK)
    # Allow config override of min_steps per phase
    overrides = track_cfg.get("phase_min_steps") or {}
    touch = _metric_f(metrics, "Player/Ball Touch Ratio", "Touch Ratio")
    air = _metric_f(metrics, "Player/In Air Ratio")
    rating = _metric_f(metrics, "Rating/1v1", "Rating/2v2")
    avg_r = _metric_f(metrics, "Average Step Reward")
    hist = list(reward_history or [])
    reward_trend = 0.0
    if len(hist) >= 4:
        reward_trend = sum(hist[-2:]) / 2.0 - sum(hist[-4:-2]) / 2.0

    # Highest phase whose min_steps + soft gates are met
    idx = 0
    gates_log: list[dict[str, Any]] = []
    for i, ph in enumerate(phases):
        min_steps = int(overrides.get(ph["id"], ph["min_steps"]))
        ok_steps = timesteps >= min_steps
        ok_touch = touch >= float(ph["touch_min"]) or timesteps < min_steps + 50_000_000
        ok_air = air >= float(ph["air_min"]) or ph["air_min"] <= 0 or timesteps < min_steps + 80_000_000
        # Early phases: don't block forever on missing metrics (metrics may be sparse)
        if i <= 1 and touch <= 0 and air <= 0:
            ok_touch = ok_steps
            ok_air = ok_steps
        graduated = ok_steps and ok_touch and ok_air
        gates_log.append(
            {
                "id": ph["id"],
                "min_steps": min_steps,
                "ok_steps": ok_steps,
                "ok_touch": ok_touch,
                "ok_air": ok_air,
                "graduated": graduated,
            }
        )
        if graduated:
            idx = i
        else:
            # Stick at last graduated; don't skip ahead
            break

    # Stagnation: hold phase but flag diversify (caller uses stagnating)
    track = dict(phases[idx])
    grad = {
        "ssl_track_id": track["id"],
        "ssl_track_label": track["label"],
        "ssl_track_index": idx,
        "ssl_track_gates": gates_log,
        "ssl_touch": round(touch, 4),
        "ssl_air": round(air, 4),
        "ssl_rating": round(rating, 2),
        "ssl_avg_reward": round(avg_r, 6),
        "ssl_reward_trend": round(reward_trend, 6),
        # Only "metrics_pressure" — never "is_ssl"
        "ssl_claim": "none",
        "ssl_note": (
            "True SSL needs months of compute; autonomy is an autopilot schedule, not instant SSL."
        ),
    }
    # Optional soft "approaching" signal — still not a rank claim
    if idx >= 4 and rating >= float(track_cfg.get("approaching_rating", 1200)):
        grad["ssl_claim"] = "approaching_metrics"
        grad["ssl_track_label"] = "SSL-track phase 5/5 (pressure — metrics warming, not SSL yet)"

    if state is not None:
        prev = (state.get("ssl_autonomy") or {}).get("track_id")
        if prev and prev != track["id"]:
            grad["ssl_track_graduated_from"] = prev
    return track, grad


def guide_gamma_bump(
    timesteps: int,
    base_gamma: float,
    ssl: dict[str, Any] | None,
) -> float:
    """
    gigalearn-guide.netlify.app: +0.0005 every ~1–1.5B steps (conservative).
    Applied on POWER baseline (~0.9973 @ tickSkip=6), NOT guide's 0.99 @ tickSkip=8.
    """
    wg = _cfg(ssl).get("web_guide") or {}
    if not bool(wg.get("enabled", True)):
        return base_gamma
    every = int(wg.get("gamma_bump_every_steps", 1_250_000_000))
    bump = float(wg.get("gamma_bump", 0.0005))
    max_extra = float(wg.get("gamma_bump_max_extra", 0.0025))
    if every <= 0 or bump <= 0:
        return base_gamma
    n = max(0, timesteps // every)
    extra = min(max_extra, n * bump)
    # Cap below safety ceiling (0.999)
    return min(0.9985, base_gamma + extra)


def hyperparam_schedule(
    track_id: str,
    phase: int,
    ssl: dict[str, Any] | None,
    root_cfg: dict[str, Any] | None = None,
    *,
    sps_safe: bool = True,
    stagnating: bool = False,
    timesteps: int = 0,
) -> dict[str, Any]:
    """Entropy / LR / GAE / ES / episode duration / checkpoint cadence by track.

    Guide-backed LR (adapted for POWER from-scratch):
      early/mechanical: 2e-4 -> mid: 1e-4 -> late: ≤0.8e-4
    Gamma bumps every ~1.25B on POWER baseline (tickSkip=6).
    Epochs: 1 early (high SPS), 2 from aerial+ (guide middle ground).
    """
    auto = _autonomy_block(ssl, root_cfg)
    hp = auto.get("hyperparams") or {}
    wg = _cfg(ssl).get("web_guide") or {}
    guide_on = bool(wg.get("enabled", True))

    # Defaults tuned for POWER ≥400k path (don't tank Collect with crazy maxEp)
    # LR numbers follow the web guide when web_guide.enabled (else prior schedule).
    table = {
        "mechanical": {
            "entropy_scale": 0.024,
            "policy_lr": 2e-4 if guide_on else 1e-4,
            "critic_lr": 2e-4 if guide_on else 1e-4,
            "gae_gamma": 0.9973,
            "gae_lambda": 0.95,
            "event_advantage_boost": 1.25,
            "priority_sampling": False,
            "es_noise_scale": 0.02,
            "var_max": 0.65,
            "max_episode_duration": 2.0,
            "ts_per_save": 2_000_000,
            "ts_per_version": 8_000_000,
            "save_policy_versions": False,
            "epochs": 1,
            "mask_entropy": True,
        },
        "aerial": {
            "entropy_scale": 0.022,
            "policy_lr": 1e-4,
            "critic_lr": 1e-4,
            "gae_gamma": 0.9973,
            "gae_lambda": 0.95,
            "event_advantage_boost": 1.40,
            "priority_sampling": True,
            "es_noise_scale": 0.025,
            "var_max": 0.70,
            "max_episode_duration": 2.0,
            "ts_per_save": 2_000_000,
            "ts_per_version": 6_000_000,
            "save_policy_versions": True,
            "epochs": 2,
            "mask_entropy": True,
        },
        "mid": {
            "entropy_scale": 0.019,
            "policy_lr": 8e-5 if guide_on else 8e-5,
            "critic_lr": 8e-5,
            "gae_gamma": 0.9973,
            "gae_lambda": 0.95,
            "event_advantage_boost": 1.55,
            "priority_sampling": True,
            "es_noise_scale": 0.03,
            "var_max": 0.60,
            "max_episode_duration": 2.5,
            "ts_per_save": 1_500_000,
            "ts_per_version": 5_000_000,
            "save_policy_versions": True,
            "epochs": 2,
            "mask_entropy": True,
        },
        "late": {
            "entropy_scale": 0.016,
            "policy_lr": 7e-5 if guide_on else 7e-5,  # guide: <0.8e-4
            "critic_lr": 7e-5,
            "gae_gamma": 0.9975,
            "gae_lambda": 0.96,
            "event_advantage_boost": 1.70,
            "priority_sampling": True,
            "es_noise_scale": 0.028,
            "var_max": 0.55,
            "max_episode_duration": 3.0,
            "ts_per_save": 1_000_000,
            "ts_per_version": 4_000_000,
            "save_policy_versions": True,
            "epochs": 2,
            "mask_entropy": True,
        },
        "ssl_pressure": {
            "entropy_scale": 0.014,
            "policy_lr": 5e-5,
            "critic_lr": 5e-5,
            "gae_gamma": 0.9975,
            "gae_lambda": 0.96,
            "event_advantage_boost": 1.85,
            "priority_sampling": True,
            "es_noise_scale": 0.025,
            "var_max": 0.50,
            "max_episode_duration": 3.5,
            "ts_per_save": 1_000_000,
            "ts_per_version": 4_000_000,
            "save_policy_versions": True,
            "epochs": 2,
            "mask_entropy": True,
        },
    }
    # Optional config overrides for guide LR peaks
    if guide_on:
        early_lr = float(wg.get("early_lr", 2e-4))
        mid_lr = float(wg.get("mid_lr", 1e-4))
        late_lr = float(wg.get("late_lr", 0.7e-4))
        table["mechanical"]["policy_lr"] = early_lr
        table["mechanical"]["critic_lr"] = early_lr
        table["aerial"]["policy_lr"] = mid_lr
        table["aerial"]["critic_lr"] = mid_lr
        table["mid"]["policy_lr"] = min(mid_lr, 9e-5)
        table["mid"]["critic_lr"] = min(mid_lr, 9e-5)
        table["late"]["policy_lr"] = late_lr
        table["late"]["critic_lr"] = late_lr

    base = dict(table.get(track_id) or table["mechanical"])
    # Config overrides per-track
    for k, v in (hp.get(track_id) or {}).items():
        base[k] = v

    # Guide gamma progression on POWER baseline
    base["gae_gamma"] = guide_gamma_bump(timesteps, float(base["gae_gamma"]), ssl)

    # Always keep entropy masking on POWER/SSL path (guide + DefaultAction)
    if guide_on or bool(wg.get("force_mask_entropy", True)):
        base["mask_entropy"] = True

    if stagnating:
        base["entropy_scale"] = min(0.04, float(base["entropy_scale"]) + 0.006)
        base["priority_sampling"] = True
        base["event_advantage_boost"] = max(float(base["event_advantage_boost"]), 1.65)
        # Guide: if unstable, lower LR first
        if guide_on:
            base["policy_lr"] = max(1e-5, float(base["policy_lr"]) * 0.85)
            base["critic_lr"] = max(1e-5, float(base["critic_lr"]) * 0.85)
    if not sps_safe:
        # Quality path can take slightly more episode length
        base["max_episode_duration"] = min(5.0, float(base["max_episode_duration"]) + 0.5)
    # Cap maxEp so ≥400k SPS path stays sane on POWER
    max_ep_cap = float(auto.get("max_episode_duration_cap", 4.0 if sps_safe else 6.0))
    base["max_episode_duration"] = min(max_ep_cap, float(base["max_episode_duration"]))
    _ = phase
    return base


def warmup_arms(
    timesteps: int,
    ssl: dict[str, Any] | None,
    root_cfg: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Skill-eval / sparring / policy_versions stay off at boot; arm after warmup."""
    auto = _autonomy_block(ssl, root_cfg)
    sparring_w = int(auto.get("sparring_warmup_steps", 30_000_000))
    skill_w = int(auto.get("skill_eval_warmup_steps", 40_000_000))
    versions_w = int(auto.get("policy_versions_warmup_steps", 25_000_000))
    return {
        "sparring": timesteps >= sparring_w,
        "skill_eval": timesteps >= skill_w,
        "policy_versions": timesteps >= versions_w,
    }


def full_control_surface(
    *,
    timesteps: int,
    phase: int,
    metrics: dict[str, Any] | None,
    current_mult: dict[str, float],
    manifest: dict[str, Any] | None,
    ssl: dict[str, Any] | None,
    root_cfg: dict[str, Any] | None = None,
    reward_history: list[float] | None = None,
    entropy_history: list[float] | None = None,
    hardened_available: bool = True,
    sps_safe: bool = False,
    state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """
    Compose the full autonomy override set every cycle.
    Returns (patch, reason, autonomy_state_delta).
    """
    track, grad = resolve_ssl_track(
        timesteps, metrics, ssl, root_cfg, reward_history=reward_history, state=state
    )
    track_id = str(track["id"])
    arms = warmup_arms(timesteps, ssl, root_cfg)

    hist = list(reward_history or [])
    try:
        hist.append(float((metrics or {}).get("Average Step Reward") or 0))
    except (TypeError, ValueError):
        pass
    ent_hist = list(entropy_history or [])
    try:
        ent_hist.append(float((metrics or {}).get("Policy Entropy") or 0))
    except (TypeError, ValueError):
        pass
    stagnating = detect_training_stagnation(hist, ent_hist)

    reasons: list[str] = [f"full_control:{track_id}"]
    patch: dict[str, Any] = {
        "full_control": True,
        "autotrainer_full_control": True,
        "ssl_autonomy": True,
        "ssl_guide_post_apex": True,
        "ssl_track_id": track_id,
        "ssl_track_label": track["label"],
        "note": track["label"],
    }

    # Rewards — seed starter on mechanical / from-scratch boot (ignore LKG pollution)
    seed_starter = should_seed_starter_rewards(
        timesteps, track_id, current_mult, ssl, state=state
    )
    base_mult = {} if seed_starter else dict(current_mult or {})
    rs = reward_shaping_decay(
        timesteps,
        phase,
        base_mult,
        manifest,
        ssl,
        track_id=track_id,
        state=state,
        force_starter=seed_starter,
    )
    if rs:
        patch["reward_weights"] = rs
        if seed_starter:
            # Replace whole map so stale keys (TouchReward, VelBall=3.0 LKG) cannot linger
            patch["reward_weights_replace"] = True
            reasons.append("reward_starter")
        else:
            reasons.append("reward_full")

    # Policy entropy (used to gate league / skill-eval / hardened experts)
    # Missing / boot-zero is NOT death — require a real Policy Entropy metric.
    from .event_driven import (
        entropy_metric_unreliable,
        is_entropy_dead,
        read_policy_entropy,
    )

    _pe = read_policy_entropy(metrics)
    entropy_unreliable = entropy_metric_unreliable(_pe, timesteps=timesteps)
    try:
        policy_entropy = float(_pe) if _pe is not None else 0.5
    except (TypeError, ValueError):
        policy_entropy = 0.5
    # Sticky recovery active must NOT mark H≥0.18 as unhealthy — that re-armed
    # HARD RECOVERY (coef→0.35) forever after a successful rebound.
    ent_rec_state = (state or {}).get("entropy_recovery") or {}
    recovery_active = bool(ent_rec_state.get("active")) and not entropy_unreliable
    entropy_dead = (not entropy_unreliable) and is_entropy_dead(policy_entropy)
    entropy_unhealthy = entropy_dead or (
        recovery_active and policy_entropy < ENTROPY_HEALTHY
    )
    recovery_exiting = recovery_active and policy_entropy >= ENTROPY_HEALTHY

    # League / sparring (armed after warmup). Delay hardened experts while entropy is dead —
    # aerial gate (~100–180M) + experts was crushing exploration at ~168M.
    # Yellow/green exit taper: allow league gradually when H is healthy again.
    use_hardened = hardened_available and not entropy_unhealthy and not recovery_exiting
    league = league_sampling_overrides(
        timesteps,
        phase,
        ssl,
        hardened_available=use_hardened,
        sps_safe=sps_safe,
        track_id=track_id if not entropy_unhealthy else "mechanical",
        sparring_armed=arms["sparring"] and not entropy_unhealthy,
    )
    patch.update(league)
    if entropy_unhealthy:
        reasons.append("league_entropy_deferred")
    elif arms["sparring"]:
        reasons.append("league_armed")
    else:
        reasons.append("league_warmup")

    # Skill-eval ramp — off while entropy dead/holding; re-enable on yellow/green exit
    if arms["skill_eval"] and (
        not entropy_unhealthy
        or (recovery_exiting and policy_entropy >= ENTROPY_CLEAR_EXIT)
        or policy_entropy >= ENTROPY_FULLY_RECOVERED
    ):
        patch["skill_tracker_enabled"] = True
        # Light -> denser as track advances
        if track_id in ("late", "ssl_pressure"):
            patch["skill_tracker_interval"] = 64
        else:
            patch["skill_tracker_interval"] = 128
        reasons.append("skill_eval_on" if not recovery_exiting else "skill_eval_exit_taper")
    else:
        patch["skill_tracker_enabled"] = False
        reasons.append(
            "skill_eval_entropy_hold" if entropy_unhealthy else "skill_eval_warmup"
        )

    # Hypers
    hp = hyperparam_schedule(
        track_id,
        phase,
        ssl,
        root_cfg,
        sps_safe=sps_safe,
        stagnating=stagnating,
        timesteps=timesteps,
    )
    if not arms["policy_versions"]:
        hp["save_policy_versions"] = False
    patch.update(hp)
    reasons.append("hyper_schedule")

    # Entropy death recovery MUST win over aerial/mid schedule (~0.022->tiny).
    # Only when H is actually dead (soft/hard/critical) — never tier=ok / boot-zero.
    if entropy_dead and not recovery_exiting and not entropy_unreliable:
        baseline = float(hp.get("entropy_scale") or 0.022)
        prev = float(ent_rec_state.get("prev_coef") or baseline)
        rec, rec_log = entropy_death_recovery_patch(
            policy_entropy=policy_entropy,
            baseline_entropy_scale=baseline,
            prev_coef=prev,
            multiplier=16.0 if policy_entropy < 0.05 else 12.0,
            cycle=int(ent_rec_state.get("cycles") or 0),
        )
        if rec:
            if ent_rec_state.get("coef"):
                rec["entropy_scale"] = max(
                    float(rec["entropy_scale"]), float(ent_rec_state["coef"])
                )
                rec_log = (
                    f"[AutoTrainer] CRITICAL RECOVERY entropy_death -> coef {prev:.4f} -> "
                    f"{rec['entropy_scale']:.4f}"
                )
                rec["note"] = rec_log
            patch.update(rec)
            reasons.append("entropy_death_recovery")
            if rec_log:
                reasons.append(rec_log)
    elif recovery_exiting and ent_rec_state.get("coef"):
        # Keep tapering from sticky coef — do not let schedule overwrite to 0.017 cliff
        baseline = float(hp.get("entropy_scale") or 0.022)
        prev_c = float(ent_rec_state.get("coef") or 0.35)
        patch["entropy_scale"] = taper_entropy_scale(
            prev_c,
            baseline,
            ok_streak=int(ent_rec_state.get("ok_streak") or 1),
            taper_cycles=4,
        )
        patch["recovery_exiting"] = True
        reasons.append("entropy_exit_taper")

    if stagnating and not entropy_unhealthy:
        try:
            cur_e = float((metrics or {}).get("Policy Entropy") or 0.015)
        except (TypeError, ValueError):
            cur_e = 0.015
        # Use absolute entropy floor, not the broken cur_e*0.05 shrink
        patch.update(
            stagnation_escape_overrides(
                ssl, current_entropy=max(0.018, min(0.03, cur_e if cur_e > 0.01 else 0.02))
            )
        )
        # Diversify league slightly on stagnation (not while death-recovering)
        if arms["sparring"]:
            patch["train_against_old_chance"] = min(
                0.45, float(patch.get("train_against_old_chance", 0.2)) + 0.05
            )
            patch["opponent_pool_chance"] = min(
                0.20, float(patch.get("opponent_pool_chance", 0.05)) + 0.03
            )
        reasons.append("stagnation_escape")

    skill = _metric_f(metrics, "Rating/1v1", "Rating/2v2")
    scen = scenario_mix_overrides(
        timesteps,
        phase,
        ssl,
        stagnating=stagnating or entropy_unhealthy,
        skill_hint=skill,
        track_id=track_id if not entropy_unhealthy else "mechanical",
    )
    if entropy_unhealthy:
        # Force diversity resets; never re-enable skill-eval via scenario helpers
        scen = dict(scen or {})
        scen.update(
            {
                "gpu_reset_kickoff": 0.30,
                "gpu_reset_fuzzed": 0.40,
                "gpu_reset_aerial": 0.30,
                "kickoff_weight": 0.30,
                "fuzzed_weight": 0.40,
                "aerial_weight": 0.30,
            }
        )
        scen.pop("skill_tracker_enabled", None)
    patch.update(scen)
    if scen:
        reasons.append("scenario_mix")
    # Re-assert recovery freeze after scenario_mix (must win)
    if entropy_unhealthy and patch.get("entropy_death_recovery"):
        patch["skill_tracker_enabled"] = False
        patch["priority_sampling"] = False
        patch["es_noise_scale"] = 0.0
        patch["opponent_pool_chance"] = min(float(patch.get("opponent_pool_chance", 0)), 0.01)
        for _ok in list(patch.keys()):
            if str(_ok).startswith("opponent_weight_"):
                patch[_ok] = 0.0

    geo = cheap_geometry_heuristics(metrics)
    if geo and (_cfg(ssl).get("geometry") or {}).get("log_heuristics", True):
        # Guide: with 1 epoch, clip fraction is near-zero — mark ignore so dashboards don't panic
        if int(hp.get("epochs", 1)) <= 1:
            geo["ssl_proxy_clip_ignore_1epoch"] = 1.0
            geo["ssl_proxy_lr_hot"] = 0.0
        patch["ssl_geometry"] = geo
        reasons.append("geo_heuristics")
        # Guide: hot clip fraction -> lower LR first (only when PPO metrics enabled AND epochs≥2)
        if geo.get("ssl_proxy_lr_hot", 0) >= 1.0 and int(hp.get("epochs", 1)) >= 2:
            patch["policy_lr"] = max(1e-5, float(patch.get("policy_lr", hp["policy_lr"])) * 0.9)
            patch["critic_lr"] = max(1e-5, float(patch.get("critic_lr", hp["critic_lr"])) * 0.9)
            reasons.append("clip_lr_cut")
        # ZeroSum farming illusion -> soft-cut chase multipliers
        if geo.get("ssl_proxy_zerosum_farm", 0) > 0.5 and "reward_weights" in patch:
            rw = dict(patch["reward_weights"])
            for name in ("VelocityPlayerToBallReward", "FaceBallReward", "VelocityPlayerToBallReward"):
                if name in rw:
                    rw[name] = max(0.05, float(rw[name]) * 0.92)
            patch["reward_weights"] = rw
            reasons.append("zerosum_farm_softcut")
        # Guide (action masking): spam-flip -> soft-cut air rewards (don't disable masking)
        spam = float(geo.get("ssl_proxy_spam_flip", 0) or 0)
        wg = _cfg(ssl).get("web_guide") or {}
        spam_thr = float(wg.get("spam_flip_threshold", 0.45))
        if spam >= spam_thr and "reward_weights" in patch and bool(wg.get("enabled", True)):
            cut = float(wg.get("spam_flip_air_cut", 0.88))
            rw = dict(patch["reward_weights"])
            for name in _AERIAL:
                if name in rw:
                    rw[name] = max(0.05, float(rw[name]) * cut)
            patch["reward_weights"] = rw
            reasons.append("spam_flip_air_softcut")

    # Guide playtime diagnostic (status-only; stripped before C++)
    tick_skip = 6
    try:
        tick_skip = int(os.environ.get("GIGA_TICK_SKIP", "6") or "6")
    except ValueError:
        tick_skip = 6
    patch["ssl_play_hours"] = round(play_hours(timesteps, tick_skip), 3)
    patch["ssl_tick_skip"] = tick_skip

    # Graduation diagnostics (stripped by safety before C++; kept in state)
    patch["ssl_graduation"] = grad

    auto_state = {
        "track_id": track_id,
        "track_label": track["label"],
        "graduation": grad,
        "arms": arms,
        "stagnating": stagnating,
        "full_control": True,
        "play_hours": patch.get("ssl_play_hours"),
        "entropy_unhealthy": entropy_unhealthy,
        "policy_entropy": policy_entropy,
        "starter_rewards_applied": bool(
            seed_starter
            or ((state or {}).get("ssl_autonomy") or {}).get("starter_rewards_applied")
        ),
        "starter_rewards_pending_log": bool(
            seed_starter
            and not ((state or {}).get("ssl_autonomy") or {}).get("starter_rewards_logged")
        ),
    }
    return patch, "ssl_autonomy:" + "+".join(reasons), auto_state


def build_ssl_patch(
    *,
    timesteps: int,
    phase: int,
    metrics: dict[str, Any] | None,
    current_mult: dict[str, float],
    manifest: dict[str, Any] | None,
    ssl: dict[str, Any] | None,
    reward_history: list[float] | None = None,
    entropy_history: list[float] | None = None,
    hardened_available: bool = True,
    sps_safe: bool = False,
    root_cfg: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Compose a single OP/curriculum patch from all SSL-guide features."""
    if not enabled(ssl) and not full_control_enabled(ssl, root_cfg):
        return {}, "ssl_off"

    # Full autonomy path — drives bot largely alone toward SSL-track pressure
    if full_control_enabled(ssl, root_cfg):
        patch, reason, auto_state = full_control_surface(
            timesteps=timesteps,
            phase=phase,
            metrics=metrics,
            current_mult=current_mult,
            manifest=manifest,
            ssl=ssl,
            root_cfg=root_cfg,
            reward_history=reward_history,
            entropy_history=entropy_history,
            hardened_available=hardened_available,
            sps_safe=sps_safe,
            state=state,
        )
        if state is not None:
            state["ssl_autonomy"] = auto_state
        return patch, reason

    reasons: list[str] = []
    patch: dict[str, Any] = {}

    # Reward decay
    rs = reward_shaping_decay(timesteps, phase, current_mult, manifest, ssl)
    if rs:
        patch["reward_weights"] = rs
        reasons.append("reward_decay")

    # League mix
    league = league_sampling_overrides(
        timesteps, phase, ssl, hardened_available=hardened_available, sps_safe=sps_safe
    )
    patch.update(league)
    if league:
        reasons.append("league_60_30_10" if hardened_available else "league_no_experts")

    # Stagnation
    hist = list(reward_history or [])
    try:
        hist.append(float((metrics or {}).get("Average Step Reward") or 0))
    except (TypeError, ValueError):
        pass
    ent_hist = list(entropy_history or [])
    try:
        ent_hist.append(float((metrics or {}).get("Policy Entropy") or 0))
    except (TypeError, ValueError):
        pass

    stagnating = detect_training_stagnation(hist, ent_hist)
    if stagnating:
        try:
            cur_e = float((metrics or {}).get("Policy Entropy") or 0.015)
        except (TypeError, ValueError):
            cur_e = 0.015
        patch.update(stagnation_escape_overrides(ssl, current_entropy=max(0.01, cur_e * 0.05 + 0.012)))
        reasons.append("stagnation_escape")

    # Scenarios
    skill = 0.0
    try:
        skill = float((metrics or {}).get("Rating/1v1") or (metrics or {}).get("Rating/2v2") or 0)
    except (TypeError, ValueError):
        pass
    scen = scenario_mix_overrides(
        timesteps, phase, ssl, stagnating=stagnating, skill_hint=skill
    )
    patch.update(scen)
    if scen:
        reasons.append("scenario_mix")

    # Cheap heuristics (status-only; written as diagnostic floats, not PPO knobs)
    geo = cheap_geometry_heuristics(metrics)
    if geo and (_cfg(ssl).get("geometry") or {}).get("log_heuristics", True):
        patch["ssl_geometry"] = geo
        reasons.append("geo_heuristics")

    return patch, "ssl:" + "+".join(reasons) if reasons else "ssl_noop"


def build_autonomy_patch(
    *,
    timesteps: int,
    phase: int,
    metrics: dict[str, Any] | None,
    current_mult: dict[str, float],
    manifest: dict[str, Any] | None,
    ssl: dict[str, Any] | None,
    reward_history: list[float] | None = None,
    entropy_history: list[float] | None = None,
    hardened_available: bool = True,
    sps_safe: bool = False,
    root_cfg: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Entry used by OP stack / four-level orchestrator.
    When ssl_autonomy.full_control is on, emits the full control surface every cycle.
    """
    # Prefer root_cfg; allow ssl["_root_autonomy"] injection from OPStack
    root = dict(root_cfg or {})
    if not root.get("ssl_autonomy") and isinstance((ssl or {}).get("_root_autonomy"), dict):
        root["ssl_autonomy"] = ssl["_root_autonomy"]  # type: ignore[index]
    patch, reason = build_ssl_patch(
        timesteps=timesteps,
        phase=phase,
        metrics=metrics,
        current_mult=current_mult,
        manifest=manifest,
        ssl=ssl,
        reward_history=reward_history,
        entropy_history=entropy_history,
        hardened_available=hardened_available,
        sps_safe=sps_safe,
        root_cfg=root,
        state=state,
    )
    # Ensure default/GPU keys stay present for gpuNative remap + C++ full_control flags
    from .full_control import ensure_blank_reward_keys, stamp_full_control

    if patch.get("reward_weights"):
        patch["reward_weights"] = ensure_blank_reward_keys(
            dict(patch["reward_weights"]), manifest
        )
        # One-shot boot log (full list) when starter recipe was seeded
        if state is not None and patch.get("reward_weights_replace"):
            auto = state.setdefault("ssl_autonomy", {})
            if not auto.get("starter_rewards_logged"):
                print(format_starter_rewards_log(patch["reward_weights"]))
                auto["starter_rewards_logged"] = True
                auto["starter_rewards_applied"] = True
    if full_control_enabled(ssl, root) or bool(root.get("full_control")):
        patch = stamp_full_control(patch, root if root.get("full_control") is not None else {
            **root,
            "full_control": True,
            "ssl_autonomy": root.get("ssl_autonomy") or {"enabled": True, "full_control": True},
        })
    else:
        patch.setdefault("ssl_guide_post_apex", True)
    return patch, reason
