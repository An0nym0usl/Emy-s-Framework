"""Error-pressure + real savestate replay (−1.5s ring when CUDA available).

When metrics detect bad events (reward crash, no-touch / low-touch, concede /
goal-gap patterns):
  1) Sticky harder fuzzed/aerial GPU reset mix (curriculum pressure)
  2) Enable CudaEnvSet state-ring + rate-limited restore-with-fuzz via
     runtime_overrides → AutoTrainerBridge → TriggerErrorStateReplay

Notes:
  - Ring restores **arena physics** (cars/ball/pads) ~1.5s lookback — NOT full
    RL trajectory / GAE rewind. Closest real thing on RocketSimCuda Get/Set APIs.
  - Phase curriculum (gpu_reset_*) still runs as diversity when ring is empty.

Opt-out: error_pressure.enabled: false  or  GIGA_NO_ERROR_PRESSURE=1
         state_replay: false to keep curriculum-only (no ring restore)
"""

from __future__ import annotations

import math
import os
from typing import Any


STATE_KEY = "error_pressure"


def error_pressure_enabled(cfg: dict[str, Any] | None = None) -> bool:
    if os.environ.get("GIGA_NO_ERROR_PRESSURE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    ep = (cfg or {}).get("error_pressure") or {}
    return bool(ep.get("enabled", True))


def state_replay_enabled(cfg: dict[str, Any] | None = None) -> bool:
    """Real ring-buffer restore (not just curriculum). Default ON with error_pressure."""
    if os.environ.get("GIGA_NO_STATE_REPLAY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    ep = (cfg or {}).get("error_pressure") or {}
    return bool(ep.get("state_replay", True)) and error_pressure_enabled(cfg)


def _f(metrics: dict[str, Any] | None, *keys: str, default: float = float("nan")) -> float:
    m = metrics or {}
    for k in keys:
        if k in m and m[k] is not None:
            try:
                v = float(m[k])
                if math.isnan(v) or math.isinf(v):
                    return float("nan")
                return v
            except (TypeError, ValueError):
                pass
    return default


def detect_bad_events(
    metrics: dict[str, Any] | None,
    state: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Return list of triggered event tags."""
    ep = (cfg or {}).get("error_pressure") or {}
    events: list[str] = []
    mon = state.get("reward_monitor") or {}
    if mon.get("in_crash") or mon.get("in_drop"):
        events.append("reward_crash")

    touch = _f(metrics, "Player/Ball Touch Ratio", "Touch Ratio")
    touch_floor = float(ep.get("touch_floor", 0.06))
    if not math.isnan(touch) and touch < touch_floor:
        events.append("no_touch")

    concede = _f(
        metrics,
        "Curriculum/ConcedePenalty",
        "Game/Concede Rate",
        "Concede Rate",
    )
    goals_against = _f(metrics, "Game/Goals Against", "Goals Against")
    goal_speed = _f(metrics, "Game/Goal Speed", "Goal Rate")
    if not math.isnan(goals_against) and goals_against > float(ep.get("goals_against_fire", 0.15)):
        events.append("concede_pattern")
    if (
        not math.isnan(goal_speed)
        and goal_speed < float(ep.get("goal_speed_floor", 200.0))
        and not math.isnan(touch)
        and touch > 0.12
    ):
        events.append("finish_gap")

    geo = (state.get("ssl_guide") or {}).get("last_geometry") or {}
    if float(geo.get("ssl_proxy_chase_error") or 0) > float(ep.get("chase_error_fire", 0.55)):
        events.append("chase_error")
    if float(geo.get("ssl_proxy_aerial_gap") or 0) > float(ep.get("aerial_gap_fire", 0.08)):
        events.append("aerial_gap")

    _ = concede
    return events


def error_pressure_patch(
    *,
    metrics: dict[str, Any] | None,
    state: dict[str, Any],
    zone: str,
    recovering: bool,
    timesteps: int,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Sticky GPU-reset bias + optional real state-ring replay flags.

    During entropy_death recovery, do NOT fight recovery recipe — return {}.
    """
    if not error_pressure_enabled(cfg):
        return {}
    if recovering or str(zone).lower() == "red":
        st = dict(state.get(STATE_KEY) or {})
        if st.get("active"):
            st["active"] = False
            st["cleared_by_recovery"] = True
            state[STATE_KEY] = st
        return {}

    ep = (cfg or {}).get("error_pressure") or {}
    window_steps = int(ep.get("window_steps", 8_000_000))
    events = detect_bad_events(metrics, state, cfg=cfg)

    st = dict(state.get(STATE_KEY) or {})
    until = int(st.get("until_timesteps") or 0)
    newly_fired = False

    if events:
        was_inactive = not st.get("active")
        until = max(until, timesteps + window_steps)
        # Only treat as "newly fired" when entering a sticky window (not event flicker)
        newly_fired = was_inactive
        st["active"] = True
        st["until_timesteps"] = until
        st["last_events"] = events
        st["fired_at"] = timesteps
        if was_inactive:
            st["replay_armed"] = False
            st["replay_triggers"] = 0
        state[STATE_KEY] = st
        if was_inactive:
            replay_note = " +state-ring-replay" if state_replay_enabled(cfg) else " (curriculum-only)"
            print(
                f"[AutoTrainer] ERROR-PRESSURE{replay_note}: {','.join(events)} "
                f"until={until}"
            )
    elif timesteps >= until:
        if st.get("active"):
            st["active"] = False
            st["replay_armed"] = False
            st["replay_triggers"] = 0
            state[STATE_KEY] = st
            print("[AutoTrainer] ERROR-PRESSURE window cleared")
        return {}

    if not st.get("active") and timesteps < until:
        st["active"] = True
        state[STATE_KEY] = st

    if not st.get("active"):
        return {}

    kickoff = float(ep.get("kickoff", 0.22))
    fuzzed = float(ep.get("fuzzed", 0.45))
    aerial = float(ep.get("aerial", 0.33))
    last = list(st.get("last_events") or events)
    if "aerial_gap" in last or "finish_gap" in last:
        aerial = min(0.50, aerial + 0.08)
        fuzzed = max(0.30, fuzzed - 0.04)
        kickoff = max(0.15, 1.0 - fuzzed - aerial)
    if "no_touch" in last or "chase_error" in last:
        fuzzed = min(0.55, fuzzed + 0.08)
        kickoff = max(0.12, kickoff - 0.04)
        aerial = max(0.20, 1.0 - kickoff - fuzzed)

    phase = 2 if state_replay_enabled(cfg) else 1
    out: dict[str, Any] = {
        "error_pressure_active": True,
        "error_pressure_phase": phase,
        "error_pressure_events": last,
        "gpu_reset_kickoff": kickoff,
        "gpu_reset_fuzzed": fuzzed,
        "gpu_reset_aerial": aerial,
        "kickoff_weight": kickoff,
        "fuzzed_weight": fuzzed,
        "aerial_weight": aerial,
        "ssl_guide_post_apex": True,
        "train_against_old_chance": max(
            float(ep.get("old_chance", 0.12)),
            float((state.get("active_overrides") or {}).get("train_against_old_chance") or 0),
        ),
        "note": f"error_pressure:phase{phase}:{','.join(last) or 'sticky'}",
    }
    if "no_touch" in last:
        out["no_touch_seconds"] = float(ep.get("no_touch_seconds", 4.0))

    if state_replay_enabled(cfg):
        # Heavy caps — never rewind most of the batch (was restoring ~all 983 arenas).
        frac = min(0.05, float(ep.get("replay_frac", 0.05)))
        max_arenas = max(1, int(ep.get("replay_max_arenas", 64)))
        min_steps = max(5_000_000, int(ep.get("replay_min_steps", 15_000_000)))
        max_per_window = max(1, int(ep.get("replay_max_per_window", 3)))
        out["error_pressure_replay"] = True
        out["state_ring_enable"] = True
        out["state_ring_depth"] = int(ep.get("ring_depth", 10))
        out["state_ring_capture_every"] = int(ep.get("ring_capture_every", 3))
        out["state_ring_restore_frac"] = frac
        out["error_replay_frac"] = frac
        out["error_replay_max_arenas"] = max_arenas
        out["error_replay_min_steps"] = min_steps
        out["error_replay_max_per_window"] = max_per_window
        out["error_replay_lookback"] = int(ep.get("replay_lookback", 7))
        out["error_replay_fuzz"] = float(ep.get("replay_fuzz", 1.0))
        # One-shot only on first fire of a sticky window — not every event flicker
        # (C++ also enforces min_steps cooldown; explicitTrig no longer bypasses it).
        if newly_fired and not st.get("replay_armed"):
            out["error_pressure_replay_now"] = True
            st["replay_armed"] = True
            state[STATE_KEY] = st

    return out


def merge_error_pressure(
    base: dict[str, Any],
    pressure: dict[str, Any] | None,
    *,
    recovering: bool,
) -> dict[str, Any]:
    if not pressure or recovering:
        return dict(base)
    out = dict(base)
    if out.get("entropy_death_recovery") or out.get("hard_recovery"):
        return out
    for k, v in pressure.items():
        if k == "note":
            prev = str(out.get("note") or "")
            out["note"] = f"{prev}; {v}" if prev else str(v)
        else:
            out[k] = v
    return out
