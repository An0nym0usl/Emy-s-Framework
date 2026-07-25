"""
Diagnose → Plan → Act intelligence layer for AutoTrainer.

Replaces twitchy per-cycle meta nudges with one coherent intervention:
  1. Observe H / reward trend / Elo / SPS / zone / viz / crashes
  2. Diagnose a single primary issue (or ``healthy``)
  3. Plan one action with cooldown
  4. Act + log DIAGNOSE / PLAN / ACT

Priority (hard): Safety > Diagnose/Plan > SSL schedule > Meta explore.

Opt-out: ``GIGA_AT_LEGACY=1`` or ``intelligence.enabled: false``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json_atomic
from .meta_brain import ACTION_NAMES, build_action_patch


STATE_FILENAME = "intelligence_state.json"
STATE_KEY = "intelligence"

# Opposite pairs — ban thrashing (league_ease then league_pressure, etc.)
_OPPOSITES: dict[str, str] = {
    "league_ease": "league_pressure",
    "league_pressure": "league_ease",
    "entropy_nudge_up": "entropy_nudge_down",
    "entropy_nudge_down": "entropy_nudge_up",
    "reward_touch_mix": "reward_goal_mix",
    "reward_goal_mix": "reward_touch_mix",
    "lr_nudge_up": "lr_nudge_down",
    "lr_nudge_down": "lr_nudge_up",
}

_DIAGNOSIS_PRIORITY: tuple[str, ...] = (
    "entropy_risk",
    "reward_volatile",
    "reward_hack_suspect",
    "vs_nexto_weak",
    "elo_stalled",
    "sps_ok_learning",
    "healthy",
)


def intelligence_enabled(cfg: dict[str, Any] | None = None) -> bool:
    """Master switch. Legacy path when GIGA_AT_LEGACY=1 or enabled/diagnose_plan_act false."""
    if os.environ.get("GIGA_AT_LEGACY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    intel = (cfg or {}).get("intelligence") or {}
    if intel.get("enabled") is False:
        return False
    return bool(intel.get("diagnose_plan_act", True))


def _intel_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return dict((cfg or {}).get("intelligence") or {})


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


@dataclass
class Observation:
    entropy: float = float("nan")
    reward: float = float("nan")
    reward_delta_pct: float | None = None
    reward_crashed: bool = False
    sps: float = float("nan")
    zone: str = "unk"
    elo: float = float("nan")
    elo_delta: float = float("nan")
    elo_best: float = float("nan")
    elo_flat_evals: int = 0
    vs_expert_weak: bool = False
    viz_no_touch: bool = False
    viz_hack_risk: bool = False
    timesteps: int = 0
    green_h_streak: int = 0


@dataclass
class IntelligencePlan:
    diagnosis: str
    plan_text: str
    action: str
    hold_steps: int = 50_000_000
    reason_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "plan_text": self.plan_text,
            "action": self.action,
            "hold_steps": self.hold_steps,
            "reason_detail": self.reason_detail,
        }


@dataclass
class IntelligenceResult:
    observation: Observation
    plan: IntelligencePlan
    patch: dict[str, Any] = field(default_factory=dict)
    acted: bool = False
    suppress_meta: bool = True
    log_lines: list[str] = field(default_factory=list)
    diagnosis_changed: bool = False


@dataclass
class IntelligenceState:
    cycles: int = 0
    last_diagnosis: str = ""
    last_action: str = "hold"
    last_plan_text: str = ""
    last_major_cycle: int = -999
    last_major_timesteps: int = 0
    banned_until_cycle: dict[str, int] = field(default_factory=dict)
    elo_history: list[float] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    green_h_streak: int = 0
    unmute_level: float = 0.0  # 0=fully muted toys, 1=full OP
    episodes: int = 0
    active_plan_until_ts: int = 0
    active_plan_until_cycle: int = 0
    ab_enabled_mirror: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "kind": "diagnose_plan_act",
            "updated": datetime.now(timezone.utc).isoformat(),
            "cycles": self.cycles,
            "last_diagnosis": self.last_diagnosis,
            "last_action": self.last_action,
            "last_plan_text": self.last_plan_text,
            "last_major_cycle": self.last_major_cycle,
            "last_major_timesteps": self.last_major_timesteps,
            "banned_until_cycle": {k: int(v) for k, v in self.banned_until_cycle.items()},
            "elo_history": [float(x) for x in self.elo_history[-16:]],
            "reward_history": [float(x) for x in self.reward_history[-24:]],
            "green_h_streak": self.green_h_streak,
            "unmute_level": self.unmute_level,
            "episodes": self.episodes,
            "active_plan_until_ts": self.active_plan_until_ts,
            "active_plan_until_cycle": self.active_plan_until_cycle,
            "ab_enabled_mirror": self.ab_enabled_mirror,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> IntelligenceState:
        if not d:
            return cls()
        return cls(
            cycles=int(d.get("cycles") or 0),
            last_diagnosis=str(d.get("last_diagnosis") or ""),
            last_action=str(d.get("last_action") or "hold"),
            last_plan_text=str(d.get("last_plan_text") or ""),
            last_major_cycle=int(d.get("last_major_cycle") if d.get("last_major_cycle") is not None else -999),
            last_major_timesteps=int(d.get("last_major_timesteps") or 0),
            banned_until_cycle={
                str(k): int(v) for k, v in (d.get("banned_until_cycle") or {}).items()
            },
            elo_history=[float(x) for x in (d.get("elo_history") or [])][-16:],
            reward_history=[float(x) for x in (d.get("reward_history") or [])][-24:],
            green_h_streak=int(d.get("green_h_streak") or 0),
            unmute_level=float(d.get("unmute_level") or 0.0),
            episodes=int(d.get("episodes") or 0),
            active_plan_until_ts=int(d.get("active_plan_until_ts") or 0),
            active_plan_until_cycle=int(d.get("active_plan_until_cycle") or 0),
            ab_enabled_mirror=bool(d.get("ab_enabled_mirror", True)),
        )


def _reward_variance(history: list[float]) -> float:
    if len(history) < 3:
        return 0.0
    xs = history[-12:]
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)


def _fmt_num(v: float, digits: int = 0) -> str:
    if math.isnan(v):
        return "?"
    if digits == 0:
        return f"{v:.0f}"
    return f"{v:.{digits}f}"


def observe(
    *,
    metrics: dict[str, Any] | None,
    zone: str,
    reward_delta_pct: float | None,
    reward_crashed: bool,
    elo_signal: dict[str, float] | None,
    viz_state: dict[str, Any] | None,
    timesteps: int,
    intel_state: IntelligenceState,
    cfg: dict[str, Any] | None = None,
) -> Observation:
    """Build a compact observation snapshot for diagnosis."""
    ic = _intel_cfg(cfg)
    h = _f(metrics, "Policy Entropy", "policy_entropy")
    reward = _f(metrics, "Average Step Reward", "avg_reward")
    sps = _f(metrics, "Overall Steps/Second", "SPS")
    sig = elo_signal or {}
    elo = float(sig.get("elo", float("nan")))
    elo_delta = float(sig.get("elo_delta", float("nan")))

    # Update rolling histories on the state (caller persists)
    if not math.isnan(reward):
        intel_state.reward_history.append(float(reward))
        intel_state.reward_history = intel_state.reward_history[-24:]
    if not math.isnan(elo):
        prev = intel_state.elo_history[-1] if intel_state.elo_history else None
        # Append only when Elo value moved (new eval), not every cycle with a stale delta
        if prev is None or abs(elo - prev) > 0.5:
            intel_state.elo_history.append(float(elo))
            intel_state.elo_history = intel_state.elo_history[-16:]

    elo_best = max(intel_state.elo_history) if intel_state.elo_history else float("nan")
    stall_delta = float(ic.get("elo_stall_delta", 5.0))
    stall_n = max(2, int(ic.get("elo_stall_evals", 2)))
    flat = 0
    if len(intel_state.elo_history) >= stall_n:
        recent = intel_state.elo_history[-stall_n:]
        if max(recent) - min(recent) <= stall_delta:
            flat = stall_n
        elif not math.isnan(elo_best) and elo < elo_best - stall_delta:
            flat = stall_n

    viz = viz_state or {}
    soft = list(viz.get("soft_features") or [])
    viz_no_touch = "viz_no_touch" in soft or str(viz.get("tag") or "") in (
        "no_contact",
        "viz_idle",
    )
    viz_hack = "viz_reward_hack_risk" in soft or str(viz.get("viz_hint") or "") == "reward_hack_watch"

    # Expert weakness: require a clear negative Elo delta + low Elo.
    # Flat/low Elo alone is noise — do not thrash league pressure.
    vs_weak = False
    stall_d = float(ic.get("elo_stall_delta", 5.0))
    if not math.isnan(elo) and not math.isnan(elo_delta):
        if elo_delta < -stall_d and elo < float(ic.get("vs_nexto_elo_floor", 1100.0)):
            vs_weak = True
    summary = (viz.get("last_summary") if isinstance(viz.get("last_summary"), dict) else None)
    _ = summary

    unmute_h = float(ic.get("unmute_h_threshold", 0.5))
    if str(zone).lower() == "green" and not math.isnan(h) and h >= unmute_h:
        intel_state.green_h_streak += 1
    else:
        intel_state.green_h_streak = 0

    return Observation(
        entropy=h,
        reward=reward,
        reward_delta_pct=reward_delta_pct,
        reward_crashed=reward_crashed,
        sps=sps,
        zone=str(zone or "unk"),
        elo=elo,
        elo_delta=elo_delta,
        elo_best=elo_best,
        elo_flat_evals=flat,
        vs_expert_weak=vs_weak,
        viz_no_touch=viz_no_touch,
        viz_hack_risk=viz_hack,
        timesteps=int(timesteps),
        green_h_streak=intel_state.green_h_streak,
    )


def diagnose(obs: Observation, intel_state: IntelligenceState, cfg: dict[str, Any] | None = None) -> str:
    """Pick one primary diagnosis (never multiple fighting plans)."""
    ic = _intel_cfg(cfg)
    h = obs.entropy
    zone = obs.zone.lower()

    # Safety owns red — intelligence reports but does not act (caller suppresses)
    if zone == "red":
        return "entropy_risk"

    if not math.isnan(h) and h < float(ic.get("entropy_risk_threshold", 0.18)):
        return "entropy_risk"

    if obs.reward_crashed:
        return "reward_volatile"

    var = _reward_variance(intel_state.reward_history)
    var_thr = float(ic.get("reward_var_threshold", 0.04))
    if var >= var_thr and len(intel_state.reward_history) >= 4:
        delta = obs.reward_delta_pct
        if delta is not None and abs(float(delta)) >= 20.0:
            return "reward_volatile"

    if obs.viz_hack_risk or (obs.viz_no_touch and not math.isnan(obs.reward) and obs.reward > 0.15):
        # High train reward + no touch on viz → hack suspicion
        if obs.viz_hack_risk or (
            obs.viz_no_touch
            and obs.reward_delta_pct is not None
            and float(obs.reward_delta_pct) > 5.0
        ):
            return "reward_hack_suspect"

    if obs.vs_expert_weak:
        return "vs_nexto_weak"

    if obs.elo_flat_evals >= max(2, int(ic.get("elo_stall_evals", 2))):
        return "elo_stalled"

    # Healthy learning signals
    h_ok = (not math.isnan(h)) and h >= 0.25
    r_up = obs.reward_delta_pct is not None and float(obs.reward_delta_pct) > 2.0
    elo_up = (not math.isnan(obs.elo_delta)) and obs.elo_delta > 1.0
    sps_ok = (not math.isnan(obs.sps)) and obs.sps >= 50_000

    if h_ok and sps_ok and (r_up or elo_up):
        return "sps_ok_learning"
    if h_ok and zone == "green" and not obs.reward_crashed:
        return "healthy"
    if h_ok:
        return "sps_ok_learning"
    return "healthy"


def plan_for(
    diagnosis: str,
    obs: Observation,
    cfg: dict[str, Any] | None = None,
) -> IntelligencePlan:
    """One coherent plan per diagnosis."""
    ic = _intel_cfg(cfg)
    hold = int(ic.get("plan_hold_steps", 50_000_000))

    if diagnosis == "entropy_risk":
        return IntelligencePlan(
            diagnosis=diagnosis,
            plan_text="raise entropy + pause OP toys until H clears",
            action="entropy_nudge_up",
            hold_steps=hold // 2,
            reason_detail=f"H={_fmt_num(obs.entropy, 3)} zone={obs.zone}",
        )
    if diagnosis == "reward_volatile":
        return IntelligencePlan(
            diagnosis=diagnosis,
            plan_text="stabilize: ease league + mild entropy hold",
            action="league_ease",
            hold_steps=hold,
            reason_detail=f"reward_delta={obs.reward_delta_pct}",
        )
    if diagnosis == "reward_hack_suspect":
        return IntelligencePlan(
            diagnosis=diagnosis,
            plan_text="rebalance rewards toward real touch (hack suspicion)",
            action="reward_touch_mix",
            hold_steps=hold,
            reason_detail="viz no-touch / train reward ahead",
        )
    # Green + healthy H + Elo flat within noise → HOLD (no expert thrash)
    h_healthy = (not math.isnan(obs.entropy)) and obs.entropy >= float(
        ic.get("unmute_h_threshold", 0.5)
    )
    elo_flat_noise = obs.elo_flat_evals >= max(2, int(ic.get("elo_stall_evals", 2)))
    elo_clearly_down = (not math.isnan(obs.elo_delta)) and obs.elo_delta < -float(
        ic.get("elo_stall_delta", 5.0)
    )
    prefer_hold = (
        str(obs.zone).lower() == "green"
        and h_healthy
        and elo_flat_noise
        and not elo_clearly_down
        and not obs.reward_crashed
    )

    if diagnosis == "vs_nexto_weak":
        if prefer_hold or not elo_clearly_down:
            return IntelligencePlan(
                diagnosis=diagnosis,
                plan_text="HOLD — vs-expert Elo soft/flat; avoid league thrash",
                action="hold",
                hold_steps=hold,
                reason_detail=(
                    f"elo={_fmt_num(obs.elo)} delta={_fmt_num(obs.elo_delta, 1)} "
                    f"(need clear drop before pressure)"
                ),
            )
        return IntelligencePlan(
            diagnosis=diagnosis,
            plan_text="increase league/teacher pressure vs experts",
            action="league_pressure",
            hold_steps=hold,
            reason_detail=f"elo={_fmt_num(obs.elo)} delta={_fmt_num(obs.elo_delta, 1)}",
        )
    if diagnosis == "elo_stalled":
        if prefer_hold:
            return IntelligencePlan(
                diagnosis=diagnosis,
                plan_text="HOLD — Elo flat within noise; green + H healthy",
                action="hold",
                hold_steps=hold,
                reason_detail=(
                    f"elo={_fmt_num(obs.elo)} flat {obs.elo_flat_evals} evals "
                    f"(best={_fmt_num(obs.elo_best)})"
                ),
            )
        return IntelligencePlan(
            diagnosis=diagnosis,
            plan_text=f"touch-mix / mild diversify for {hold // 1_000_000}M steps (Elo soft)",
            action="reward_touch_mix",
            hold_steps=hold,
            reason_detail=(
                f"elo={_fmt_num(obs.elo)} flat {obs.elo_flat_evals} evals "
                f"(best={_fmt_num(obs.elo_best)})"
            ),
        )
    if diagnosis == "sps_ok_learning":
        return IntelligencePlan(
            diagnosis=diagnosis,
            plan_text="HOLD — learning healthy; gentle LR decay only",
            action="lr_nudge_down",
            hold_steps=hold,
            reason_detail=f"H={_fmt_num(obs.entropy, 2)} reward_delta={obs.reward_delta_pct}",
        )
    # healthy
    return IntelligencePlan(
        diagnosis="healthy",
        plan_text="HOLD — no major change",
        action="hold",
        hold_steps=hold,
        reason_detail=f"H={_fmt_num(obs.entropy, 2)} zone={obs.zone}",
    )


def _cooldown_ready(state: IntelligenceState, obs: Observation, cfg: dict[str, Any] | None) -> bool:
    ic = _intel_cfg(cfg)
    min_c = max(1, int(ic.get("cooldown_cycles", 8)))
    min_s = max(0, int(ic.get("cooldown_steps", 15_000_000)))
    if state.cycles - state.last_major_cycle < min_c:
        return False
    if obs.timesteps - state.last_major_timesteps < min_s:
        return False
    # Active plan window still open → hold the line
    if obs.timesteps < state.active_plan_until_ts and state.cycles < state.active_plan_until_cycle:
        return False
    return True


def _action_banned(state: IntelligenceState, action: str) -> bool:
    until = int(state.banned_until_cycle.get(action) or 0)
    return state.cycles < until


def _ban_opposite(state: IntelligenceState, action: str, cfg: dict[str, Any] | None) -> None:
    ic = _intel_cfg(cfg)
    if not ic.get("ban_oscillation", True):
        return
    opp = _OPPOSITES.get(action)
    if not opp:
        return
    ban_for = max(4, int(ic.get("oscillation_ban_cycles", 12)))
    state.banned_until_cycle[opp] = state.cycles + ban_for


def _describe_act(
    action: str,
    patch: dict[str, Any],
    current: dict[str, Any] | None,
) -> str:
    cur = current or {}
    bits: list[str] = []
    if "opponent_pool_chance" in patch:
        before = float(cur.get("opponent_pool_chance", 0.1) or 0.1)
        after = float(patch["opponent_pool_chance"])
        bits.append(f"opponent_pool {before:.2f}→{after:.2f}")
    if "train_against_old_chance" in patch:
        before = float(cur.get("train_against_old_chance", 0.15) or 0.15)
        after = float(patch["train_against_old_chance"])
        bits.append(f"old {before:.2f}→{after:.2f}")
    if "entropy_scale" in patch:
        before = float(cur.get("entropy_scale", 0.022) or 0.022)
        after = float(patch["entropy_scale"])
        bits.append(f"entropy_scale {before:.3f}→{after:.3f}")
    if "policy_lr" in patch:
        before = float(cur.get("policy_lr", 1e-4) or 1e-4)
        after = float(patch["policy_lr"])
        bits.append(f"policy_lr {before:.2e}→{after:.2e}")
    rw = patch.get("reward_weights")
    if isinstance(rw, dict):
        if "TouchReward" in rw or "TouchBall" in str(rw):
            bits.append("reward TouchBall↑" if action == "reward_touch_mix" else "reward Goal↑")
        else:
            bits.append("reward_weights updated")
    if action == "hold":
        return "hold"
    if not bits:
        return action
    return ", ".join(bits)


def gradual_op_unmute_patch(
    *,
    obs: Observation,
    state: IntelligenceState,
    current: dict[str, Any] | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unmute OP toys gradually when green + H healthy for a streak.

    Avoids forever ``muted=OP`` in green. Never fights red recovery.
    """
    ic = _intel_cfg(cfg)
    if obs.zone.lower() != "green":
        state.unmute_level = 0.0
        return {}
    h_thr = float(ic.get("unmute_h_threshold", 0.5))
    streak_need = max(1, int(ic.get("unmute_streak", 3)))
    if math.isnan(obs.entropy) or obs.entropy < h_thr or obs.green_h_streak < streak_need:
        return {}

    # Already fully live — no-op (avoids redundant patches + ACT spam every green tick)
    if state.unmute_level >= 1.0 - 1e-12:
        return {}

    # Step unmute 0 → 1 over several healthy cycles
    step = float(ic.get("unmute_step", 0.25))
    state.unmute_level = min(1.0, state.unmute_level + step)
    lvl = state.unmute_level
    cur = current or {}
    out: dict[str, Any] = {
        "intelligence_unmute": True,
        "unmute_level": lvl,
    }
    # Clear hard mutes once we start unmuting
    out["op_destructive_paused"] = False
    out["freeze_op_chaos"] = False
    out["pbt_paused"] = lvl < 0.5
    out["es_enabled"] = lvl >= 0.5
    out["truncation_pbt_paused"] = lvl < 0.75
    out["compete_rotate_paused"] = lvl < 0.75
    out["meta_gradients_paused"] = lvl < 1.0
    out["env_architect_paused"] = lvl < 1.0
    # Soft ES noise ramp
    base_noise = float(cur.get("es_noise_scale", 0.05) or 0.05)
    if base_noise <= 0:
        base_noise = 0.05
    out["es_noise_scale"] = max(0.0, min(0.12, base_noise * lvl))
    out["priority_sampling"] = lvl >= 0.75
    out["skill_tracker_enabled"] = True
    out["note"] = f"intelligence:unmute_OP level={lvl:.2f}"
    return out


def _boost_league_pressure(patch: dict[str, Any], current: dict[str, Any] | None) -> dict[str, Any]:
    """Stronger coherent league bump than a tiny meta nudge."""
    cur = current or {}
    pool = float(cur.get("opponent_pool_chance", 0.1) or 0.1)
    old = float(cur.get("train_against_old_chance", 0.15) or 0.15)
    patch["opponent_pool_chance"] = min(0.25, max(pool, pool + 0.08, 0.15))
    patch["train_against_old_chance"] = min(0.40, max(old, old + 0.08, 0.20))
    return patch


def _boost_touch_mix(patch: dict[str, Any], current: dict[str, Any] | None, cfg: dict[str, Any] | None) -> dict[str, Any]:
    base = build_action_patch("reward_touch_mix", current, cfg=cfg)
    patch.update({k: v for k, v in base.items() if k != "note"})
    return patch


class IntelligenceController:
    """Diagnose → Plan → Act controller with persisted cooldowns + optional A/B."""

    def __init__(self, watch_dir: Path, cfg: dict[str, Any] | None = None) -> None:
        self.watch_dir = Path(watch_dir)
        self.cfg = cfg or {}
        self.state_path = self.watch_dir / STATE_FILENAME
        self.state = IntelligenceState.from_dict(read_json(self.state_path))
        self.ab = None
        try:
            from .ab_plans import ABPlanController, ab_plans_enabled

            if ab_plans_enabled(self.cfg):
                self.ab = ABPlanController(self.watch_dir, self.cfg)
                self.state.ab_enabled_mirror = True
            else:
                self.state.ab_enabled_mirror = False
        except Exception:
            self.ab = None
            self.state.ab_enabled_mirror = False

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state.to_dict())
        if self.ab is not None:
            try:
                self.ab.save()
            except Exception:
                pass

    def note_elo_eval(self) -> None:
        if self.ab is not None:
            self.ab.note_elo_eval()

    def step(
        self,
        *,
        metrics: dict[str, Any] | None,
        status: dict[str, Any] | None,
        zone: str,
        current_overrides: dict[str, Any] | None,
        gov_recovery: bool = False,
        reward_delta_pct: float | None = None,
        reward_crashed: bool = False,
        elo_signal: dict[str, float] | None = None,
        viz_state: dict[str, Any] | None = None,
        safety_owns: bool = False,
    ) -> IntelligenceResult:
        """
        One intelligence cycle.

        When ``safety_owns`` / recovery / red: diagnose for logs but do not act.
        """
        if not intelligence_enabled(self.cfg):
            return IntelligenceResult(
                observation=Observation(zone=zone),
                plan=IntelligencePlan("healthy", "legacy", "hold"),
                suppress_meta=False,
            )

        self.state.cycles += 1
        st = status or {}
        ts = int(st.get("total_timesteps") or 0)
        obs = observe(
            metrics=metrics,
            zone=zone,
            reward_delta_pct=reward_delta_pct,
            reward_crashed=reward_crashed,
            elo_signal=elo_signal,
            viz_state=viz_state,
            timesteps=ts,
            intel_state=self.state,
            cfg=self.cfg,
        )
        diagnosis = diagnose(obs, self.state, self.cfg)
        planned = plan_for(diagnosis, obs, self.cfg)
        diagnosis_changed = diagnosis != self.state.last_diagnosis

        log_lines: list[str] = []
        patch: dict[str, Any] = {}
        acted = False
        suppress_meta = True
        var = _reward_variance(self.state.reward_history)

        # Long-credit A/B scoring every cycle (even when holding / mid-cooldown)
        if self.ab is not None and not (safety_owns or gov_recovery or zone.lower() == "red"):
            ab_line = self.ab.maybe_score_and_rotate(
                cycle=self.state.cycles,
                timesteps=ts,
                elo=obs.elo,
                elo_delta=obs.elo_delta,
                reward_var=var,
                zone=zone,
                crashed=reward_crashed,
            )
            if ab_line:
                log_lines.append(ab_line)

        # Safety always wins — no act, still may unmute only when green
        if safety_owns or gov_recovery or zone.lower() == "red":
            detail = planned.reason_detail or diagnosis
            if diagnosis_changed or self.state.last_diagnosis != diagnosis:
                log_lines.append(
                    f"[AutoTrainer] DIAGNOSE: {diagnosis} ({detail}) — safety owns"
                )
            self.state.last_diagnosis = diagnosis
            # Still allow gradual unmute only when already green (not red)
            if zone.lower() == "green" and not gov_recovery:
                unmute = gradual_op_unmute_patch(
                    obs=obs, state=self.state, current=current_overrides, cfg=self.cfg
                )
                if unmute:
                    patch.update(unmute)
            self.save()
            return IntelligenceResult(
                observation=obs,
                plan=planned,
                patch=patch,
                acted=False,
                suppress_meta=True,
                log_lines=log_lines,
                diagnosis_changed=diagnosis_changed,
            )

        # Unmute path always available in green healthy streak
        prev_unmute = float(self.state.unmute_level or 0.0)
        unmute = gradual_op_unmute_patch(
            obs=obs, state=self.state, current=current_overrides, cfg=self.cfg
        )

        # HOLD diagnoses: log only when diagnosis changes; allow soft meta explore
        if planned.action == "hold" or diagnosis in ("healthy",):
            suppress_meta = False  # meta may explore mildly under intelligence ε
            if diagnosis_changed:
                log_lines.append(
                    f"[AutoTrainer] DIAGNOSE: {diagnosis} ({planned.reason_detail})"
                )
                log_lines.append(f"[AutoTrainer] PLAN: {planned.plan_text}")
                log_lines.append("[AutoTrainer] ACT: hold")
            if unmute:
                patch.update(unmute)
                new_lvl = float(unmute.get("unmute_level", self.state.unmute_level) or 0.0)
                # Only announce when level actually advanced (not every green tick at 1.0)
                if new_lvl > prev_unmute + 1e-9:
                    log_lines.append(
                        f"[AutoTrainer] ACT: unmute OP toys level={self.state.unmute_level:.2f}"
                    )
            self.state.last_diagnosis = diagnosis
            self.state.last_action = "hold"
            self.state.last_plan_text = planned.plan_text
            self.save()
            return IntelligenceResult(
                observation=obs,
                plan=planned,
                patch=patch,
                acted=False,
                suppress_meta=suppress_meta,
                log_lines=log_lines,
                diagnosis_changed=diagnosis_changed,
            )

        # sps_ok_learning → gentle LR only (still a major-ish act, use cooldown)
        ready = _cooldown_ready(self.state, obs, self.cfg)
        action = planned.action
        if action not in ACTION_NAMES:
            action = "hold"

        if _action_banned(self.state, action):
            # Try hold instead of thrashing
            if diagnosis_changed:
                log_lines.append(
                    f"[AutoTrainer] DIAGNOSE: {diagnosis} ({planned.reason_detail})"
                )
                log_lines.append(
                    f"[AutoTrainer] PLAN: {planned.plan_text} (deferred — oscillation ban)"
                )
                log_lines.append("[AutoTrainer] ACT: hold")
            if unmute:
                patch.update(unmute)
            self.state.last_diagnosis = diagnosis
            self.save()
            return IntelligenceResult(
                observation=obs,
                plan=planned,
                patch=patch,
                acted=False,
                suppress_meta=True,
                log_lines=log_lines,
                diagnosis_changed=diagnosis_changed,
            )

        if not ready and not diagnosis_changed:
            # Same diagnosis mid-cooldown — quiet hold
            if unmute:
                patch.update(unmute)
            self.state.last_diagnosis = diagnosis
            self.save()
            return IntelligenceResult(
                observation=obs,
                plan=planned,
                patch=patch,
                acted=False,
                suppress_meta=True,
                log_lines=log_lines,
                diagnosis_changed=False,
            )

        # A/B race: seed pair; prefer active arm action when racing
        # Skip while paused / after flat ties (HOLD wins)
        if (
            self.ab is not None
            and action != "hold"
            and not self.ab.paused(self.state.cycles)
        ):
            self.ab.ensure_pair(
                diagnosis=diagnosis,
                primary_action=action,
                primary_plan_text=planned.plan_text,
                cycle=self.state.cycles,
                timesteps=ts,
                elo=obs.elo,
                reward_var=var,
            )
            arm = self.ab.active_arm()
            if arm.active and arm.action in ACTION_NAMES:
                action = arm.action
                planned = IntelligencePlan(
                    diagnosis=diagnosis,
                    plan_text=f"{planned.plan_text} | AB[{arm.slot}]={arm.action}",
                    action=action,
                    hold_steps=planned.hold_steps,
                    reason_detail=planned.reason_detail,
                )
        elif self.ab is not None and self.ab.paused(self.state.cycles):
            action = "hold"
            planned = IntelligencePlan(
                diagnosis=diagnosis,
                plan_text="HOLD — A/B paused after flat ties",
                action="hold",
                hold_steps=planned.hold_steps,
                reason_detail=planned.reason_detail,
            )

        # Act (new diagnosis or cooldown ready)
        log_lines.append(
            f"[AutoTrainer] DIAGNOSE: {diagnosis} ({planned.reason_detail})"
        )
        log_lines.append(f"[AutoTrainer] PLAN: {planned.plan_text}")

        if action == "hold":
            act_desc = "hold"
        else:
            if self.ab is not None and self.ab.active_arm().active:
                patch = self.ab.patch_for_active(current_overrides, self.cfg)
                # Still apply coherent amplifications for curriculum diagnoses on primary
                if diagnosis in ("elo_stalled", "vs_nexto_weak") and action == "league_pressure":
                    patch = _boost_league_pressure(patch, current_overrides)
                    if diagnosis == "elo_stalled":
                        patch = _boost_touch_mix(patch, current_overrides, self.cfg)
                if diagnosis == "reward_hack_suspect" and action == "reward_touch_mix":
                    patch = _boost_touch_mix(patch, current_overrides, self.cfg)
            else:
                patch = build_action_patch(action, current_overrides, cfg=self.cfg)
                # Coherent amplifications for curriculum diagnoses
                if diagnosis in ("elo_stalled", "vs_nexto_weak"):
                    patch = _boost_league_pressure(patch, current_overrides)
                    if diagnosis == "elo_stalled":
                        patch = _boost_touch_mix(patch, current_overrides, self.cfg)
                        patch["meta_action"] = "league_pressure+reward_touch_mix"
                if diagnosis == "reward_hack_suspect":
                    patch = _boost_touch_mix(patch, current_overrides, self.cfg)
                    # Soften goals a bit more
                    rw = dict(patch.get("reward_weights") or {})
                    if "GoalReward" in rw:
                        rw["GoalReward"] = max(0.05, float(rw["GoalReward"]) * 0.85)
                        patch["reward_weights"] = rw
            patch["intelligence"] = True
            patch["intelligence_diagnosis"] = diagnosis
            note = str(patch.get("note") or "")
            tag = f"intelligence:{diagnosis}:{action}"
            patch["note"] = f"{note}; {tag}" if note else tag
            act_desc = _describe_act(action, patch, current_overrides)
            if patch.get("ab_slot"):
                act_desc = f"AB[{patch['ab_slot']}] {act_desc}"
            acted = True
            self.state.last_major_cycle = self.state.cycles
            self.state.last_major_timesteps = ts
            self.state.active_plan_until_ts = ts + int(planned.hold_steps)
            self.state.active_plan_until_cycle = self.state.cycles + max(
                4, int(_intel_cfg(self.cfg).get("cooldown_cycles", 8))
            )
            self.state.episodes += 1
            _ban_opposite(self.state, action, self.cfg)

        if unmute and not (patch.get("freeze_op_chaos") or patch.get("op_destructive_paused")):
            # Merge unmute under intelligence act (unmute never fights entropy_risk act)
            if diagnosis != "entropy_risk":
                for k, v in unmute.items():
                    if k == "note":
                        continue
                    patch.setdefault(k, v)

        log_lines.append(f"[AutoTrainer] ACT: {act_desc}")
        self.state.last_diagnosis = diagnosis
        self.state.last_action = action
        self.state.last_plan_text = planned.plan_text
        self.save()
        return IntelligenceResult(
            observation=obs,
            plan=planned,
            patch=patch,
            acted=acted,
            suppress_meta=True,
            log_lines=log_lines,
            diagnosis_changed=True,
        )


def merge_intelligence_patch(
    base: dict[str, Any],
    intel: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge intelligence patch; never clear recovery locks."""
    if not intel:
        return dict(base)
    out = dict(base)
    if (
        out.get("entropy_death_recovery")
        or out.get("hard_recovery")
        or out.get("freeze_op_chaos")
        or str(out.get("safety_zone") or "").lower() == "red"
    ):
        # Only allow unmute-clearing when base already left red AND intel asks unmute
        if not intel.get("intelligence_unmute"):
            return out
        # If still red in base, refuse unmute
        if str(out.get("safety_zone") or "").lower() == "red" or out.get("hard_recovery"):
            return out

    locked = {
        "entropy_death_recovery",
        "hard_recovery",
        "safety_zone",
    }
    for k, v in intel.items():
        if k in locked:
            continue
        if k == "reward_weights" and isinstance(v, dict):
            rw = dict(out.get("reward_weights") or {})
            rw.update(v)
            out["reward_weights"] = rw
        elif k == "note":
            prev = str(out.get("note") or "")
            out["note"] = f"{prev}; {v}" if prev else str(v)
        else:
            out[k] = v
    return out
