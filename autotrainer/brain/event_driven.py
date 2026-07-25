"""
Event-driven training brain - react to metric *events*, not fixed tick intervals.

Instead of waiting N steps blindly, fire decisions when the learning signal changes:
reward collapse, entropy death, goal spike, touch drought, KL explosion, SPS stall.

Entropy death recovery is sticky and multi-lever: once armed, HARD RECOVERY overrides
re-apply every OP tick until Policy Entropy clears the safe floor. OP/PBT/ES chaos is
frozen while recovering - safety veto wins over clever experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# Soft / hard / critical floors (normalized masked entropy ~0..1)
# Soft 0.10 catches collapse earlier than 0.05; hard/critical escalate response.
ENTROPY_DEATH_SOFT = 0.10
ENTROPY_DEATH_HARD = 0.05
ENTROPY_DEATH_CRITICAL = 0.001
# Exit hysteresis: soft hold ≥0.18, clear exit ≥0.25 (fully recovered ≫ both)
ENTROPY_HEALTHY = 0.18
ENTROPY_RECOVER_HOLD = 0.15
ENTROPY_CLEAR_EXIT = 0.25
ENTROPY_FULLY_RECOVERED = 0.50

# Baseline schedule coefs used for ×N spikes when live coef unknown
_DEFAULT_BASELINE_ENTROPY_SCALE = 0.022
_RECOVERY_MAX_ENTROPY_SCALE = 0.35
_RECOVERY_MIN_ENTROPY_SCALE = 0.08
# Post-exit soft floor — never cliff to schedule poison (~0.014–0.017)
_POST_RECOVERY_SOFT_FLOOR = 0.02
# Taper steps from recovery spike toward track baseline (no cliff)
_RECOVERY_TAPER_CYCLES = 4
# Still-dead after this many sticky OP cycles -> rollback LKG / freeze OP knobs
_ROLLBACK_AFTER_CYCLES = 8


def recovery_ok_cycles_needed(policy_entropy: float, default: int = 3) -> int:
    """Fewer consecutive healthy ticks when H is clearly recovered."""
    h = float(policy_entropy)
    if h >= ENTROPY_FULLY_RECOVERED:
        return 1  # force-exit: leave red immediately
    if h >= ENTROPY_CLEAR_EXIT:
        return 2
    return max(1, int(default))


def track_baseline_entropy(track_id: str | None = None, fallback: float = 0.022) -> float:
    """Aerial/mid/late schedule baselines — taper target, not recovery spike."""
    return {
        "mechanical": 0.024,
        "aerial": 0.022,
        "mid": 0.019,
        "late": 0.016,
        "ssl_pressure": 0.014,
    }.get(str(track_id or "aerial"), float(fallback))


def sane_post_recovery_coef(baseline: float) -> float:
    """Target after taper: mild explore band (0.02–0.05), never ~0.017 poison."""
    base = max(_POST_RECOVERY_SOFT_FLOOR, float(baseline))
    return max(0.02, min(0.05, base if base >= 0.02 else 0.03))


def taper_entropy_scale(
    current: float,
    baseline: float,
    *,
    ok_streak: int = 0,
    taper_cycles: int = _RECOVERY_TAPER_CYCLES,
) -> float:
    """Blend recovery spike down toward a sane track baseline over a few ok cycles.

    Never cliffs to schedule micro-coefs (~0.014–0.017). Soft floor ~0.02.
    """
    cur = max(float(current), 0.0)
    target = sane_post_recovery_coef(baseline)
    if cur <= target * 1.08:
        return max(target, min(cur, _RECOVERY_MAX_ENTROPY_SCALE))
    n = max(1, int(taper_cycles))
    progress = min(1.0, max(0.0, float(max(0, int(ok_streak))) / float(n)))
    t = progress * progress * (3.0 - 2.0 * progress)
    blended = cur + (target - cur) * t
    # Early taper keeps exploration up (first healthy tick ≠ dump to baseline)
    soft_floor = target + (cur - target) * max(0.0, 1.0 - progress) * 0.40
    if ok_streak <= 1:
        soft_floor = max(soft_floor, cur * 0.55 + target * 0.45)
    return max(target, min(_RECOVERY_MAX_ENTROPY_SCALE, max(blended, soft_floor)))

@dataclass
class MetricEvent:
    name: str
    severity: float  # 0..1+
    detail: str
    suggest: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventState:
    last_metrics: dict[str, float] = field(default_factory=dict)
    last_event_timesteps: int = 0
    event_count: int = 0
    cooldown_steps: int = 5_000_000
    history: list[dict[str, Any]] = field(default_factory=list)
    # Sticky entropy-death recovery (survives OP ticks that would otherwise overwrite)
    entropy_recovery_active: bool = False
    entropy_recovery_coef: float = 0.0
    entropy_recovery_prev_coef: float = 0.0
    entropy_recovery_started_ts: int = 0
    entropy_recovery_log: str = ""
    entropy_recovery_cycles: int = 0
    entropy_recovery_tier: str = ""
    entropy_recovery_need_rollback: bool = False
    entropy_ok_streak: int = 0
    # After leaving red: keep tapering coef for N cycles (zone may already be green)
    entropy_taper_remaining: int = 0
    entropy_taper_from: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_metrics": dict(self.last_metrics),
            "last_event_timesteps": self.last_event_timesteps,
            "event_count": self.event_count,
            "cooldown_steps": self.cooldown_steps,
            "history": list(self.history[-32:]),
            "entropy_recovery_active": self.entropy_recovery_active,
            "entropy_recovery_coef": self.entropy_recovery_coef,
            "entropy_recovery_prev_coef": self.entropy_recovery_prev_coef,
            "entropy_recovery_started_ts": self.entropy_recovery_started_ts,
            "entropy_recovery_log": self.entropy_recovery_log,
            "entropy_recovery_cycles": self.entropy_recovery_cycles,
            "entropy_recovery_tier": self.entropy_recovery_tier,
            "entropy_recovery_need_rollback": self.entropy_recovery_need_rollback,
            "entropy_ok_streak": self.entropy_ok_streak,
            "entropy_taper_remaining": self.entropy_taper_remaining,
            "entropy_taper_from": self.entropy_taper_from,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> EventState:
        if not d:
            return cls()
        return cls(
            last_metrics={k: float(v) for k, v in (d.get("last_metrics") or {}).items()},
            last_event_timesteps=int(d.get("last_event_timesteps") or 0),
            event_count=int(d.get("event_count") or 0),
            cooldown_steps=int(d.get("cooldown_steps") or 5_000_000),
            history=list(d.get("history") or []),
            entropy_recovery_active=bool(d.get("entropy_recovery_active")),
            entropy_recovery_coef=float(d.get("entropy_recovery_coef") or 0),
            entropy_recovery_prev_coef=float(d.get("entropy_recovery_prev_coef") or 0),
            entropy_recovery_started_ts=int(d.get("entropy_recovery_started_ts") or 0),
            entropy_recovery_log=str(d.get("entropy_recovery_log") or ""),
            entropy_recovery_cycles=int(d.get("entropy_recovery_cycles") or 0),
            entropy_recovery_tier=str(d.get("entropy_recovery_tier") or ""),
            entropy_recovery_need_rollback=bool(d.get("entropy_recovery_need_rollback")),
            entropy_ok_streak=int(d.get("entropy_ok_streak") or 0),
            entropy_taper_remaining=int(d.get("entropy_taper_remaining") or 0),
            entropy_taper_from=float(d.get("entropy_taper_from") or 0),
        )


def _f(metrics: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                pass
    return default


# Boot: C++ often publishes Policy Entropy=0 before the first learn — not death.
_BOOT_ENTROPY_GUARD_STEPS = 2_000_000
_MIN_REAL_ENTROPY = 1e-8


def read_policy_entropy(metrics: dict[str, Any] | None) -> float | None:
    """Return Policy Entropy only when the metric is present and finite."""
    m = metrics or {}
    for k in ("Policy Entropy", "policy_entropy"):
        if k not in m or m[k] is None:
            continue
        try:
            v = float(m[k])
        except (TypeError, ValueError):
            continue
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return None


def entropy_metric_unreliable(
    policy_entropy: float | None,
    *,
    timesteps: int = 0,
) -> bool:
    """True when H is missing or a boot-time zero placeholder (not real death)."""
    if policy_entropy is None:
        return True
    try:
        h = float(policy_entropy)
    except (TypeError, ValueError):
        return True
    if math.isnan(h) or math.isinf(h):
        return True
    if int(timesteps) < _BOOT_ENTROPY_GUARD_STEPS and h < _MIN_REAL_ENTROPY:
        return True
    return False


def entropy_tier(policy_entropy: float) -> str:
    """Return critical | hard | soft | ok for Policy Entropy."""
    h = float(policy_entropy)
    if h < ENTROPY_DEATH_CRITICAL:
        return "critical"
    if h < ENTROPY_DEATH_HARD:
        return "hard"
    if h < ENTROPY_DEATH_SOFT:
        return "soft"
    return "ok"


def is_entropy_dead(policy_entropy: float | None) -> bool:
    """True only for a real soft/hard/critical death — never missing/ok/boot-zero."""
    if policy_entropy is None:
        return False
    try:
        h = float(policy_entropy)
    except (TypeError, ValueError):
        return False
    if math.isnan(h) or math.isinf(h):
        return False
    return entropy_tier(h) != "ok"


def entropy_death_recovery_patch(
    *,
    policy_entropy: float,
    baseline_entropy_scale: float = _DEFAULT_BASELINE_ENTROPY_SCALE,
    prev_coef: float | None = None,
    multiplier: float | None = None,
    max_coef: float = _RECOVERY_MAX_ENTROPY_SCALE,
    cycle: int = 0,
) -> tuple[dict[str, Any], str]:
    """Aggressive sticky HARD RECOVERY for near-zero Policy Entropy.

    Multi-lever: spike entropy_scale, raise policy LR / cut critic if stuck,
    freeze priority sampling + ES noise + PBT chaos, ease opponent pressure,
    force GPU scenario diversity, skill-eval OFF. Re-applied every cycle until
    entropy clears ENTROPY_RECOVER_HOLD / ENTROPY_HEALTHY.

    Refuses to arm when tier=ok (no tiny coef churn / nonsense logs).
    """
    tier = entropy_tier(policy_entropy)
    if tier == "ok":
        return {}, ""
    base = max(1e-4, float(baseline_entropy_scale))
    prev = float(prev_coef) if prev_coef is not None and prev_coef > 0 else base

    if multiplier is None:
        if tier == "critical":
            multiplier = 20.0
        elif tier == "hard":
            multiplier = 16.0
        else:
            multiplier = 10.0

    # Escalate further if sticky recovery is not working
    if cycle >= 4:
        multiplier = max(multiplier, 20.0)
    if cycle >= _ROLLBACK_AFTER_CYCLES // 2:
        multiplier = max(multiplier, 22.0)

    new_coef = max(_RECOVERY_MIN_ENTROPY_SCALE, min(max_coef, prev * multiplier))
    new_coef = max(new_coef, min(max_coef, base * multiplier))
    if tier == "critical":
        new_coef = max(new_coef, min(max_coef, 0.22))
    elif tier == "hard":
        new_coef = max(new_coef, min(max_coef, 0.16))

    # Policy needs room to move logits; critic can overfit the dead policy
    if tier == "critical":
        policy_lr, critic_lr = 2.2e-4, 8.0e-5
    elif tier == "hard":
        policy_lr, critic_lr = 1.8e-4, 1.0e-4
    else:
        policy_lr, critic_lr = 1.5e-4, 1.2e-4

    detail = (
        f"tier={tier} H={policy_entropy:.4e} coef {prev:.4f}->{new_coef:.4f} "
        f"cycle={cycle} ES/PBT/meta/env MUTED epochs=1"
    )
    log = f"[AutoTrainer] HARD RECOVERY entropy_death -> {detail}"
    patch: dict[str, Any] = {
        "entropy_scale": new_coef,
        "entropy_death_recovery": True,
        "entropy_death_tier": tier,
        "hard_recovery": True,
        "freeze_op_chaos": True,
        "safety_zone": "red",
        "epochs": 1,
        # Freeze chaos amplifiers - do NOT inject ES noise while dead
        "es_noise_scale": 0.0,
        "es_enabled": False,
        "pbt_paused": True,
        "op_destructive_paused": True,
        "truncation_pbt_paused": True,
        "meta_gradients_paused": True,
        "env_architect_paused": True,
        "compete_rotate_paused": True,
        "priority_sampling": False,
        "var_max": 1.0,
        "var_min": 0.25,
        "clip_range": 0.25,
        # eventAdvantage DOWN during collapse (concentration worsens death)
        "event_advantage_boost": 1.05,
        # Ease deterministic / expert pressure - diversify via old versions
        "opponent_pool_chance": 0.0,
        "train_against_old_chance": 0.12,
        "opponent_weight_nexto": 0.0,
        "opponent_weight_necto": 0.0,
        "opponent_weight_nexto_tled": 0.0,
        "opponent_weight_requiem": 0.0,
        "skill_tracker_enabled": False,
        "skill_tracker_interval": 256,
        "mask_entropy": True,
        "ssl_guide_post_apex": True,
        "policy_lr": policy_lr,
        "critic_lr": critic_lr,
        # Force scenario diversity (GPU resets)
        "gpu_reset_kickoff": 0.30,
        "gpu_reset_fuzzed": 0.40,
        "gpu_reset_aerial": 0.30,
        "kickoff_weight": 0.30,
        "fuzzed_weight": 0.40,
        "aerial_weight": 0.30,
        "w_icm": 0.0,
        "w_rnd": 0.0,
        "note": log,
    }
    return patch, log


def update_entropy_recovery_state(
    state: EventState,
    metrics: dict[str, Any],
    *,
    timesteps: int = 0,
    baseline_entropy_scale: float = _DEFAULT_BASELINE_ENTROPY_SCALE,
    force_arm: bool = False,
    rollback_after_cycles: int = _ROLLBACK_AFTER_CYCLES,
    prolonged_cycles: int | None = None,
    tick_cycle: bool = True,
) -> dict[str, Any]:
    """Arm / hold / clear sticky recovery. Returns patch if recovery should apply.

    tick_cycle: increment cycle counter once per OP tick (False on re-apply same tick).
    prolonged_cycles: alias for rollback_after_cycles (SafetyGovernor / OP stack).

    Exit hysteresis (do NOT re-arm just because recovery was active):
      - soft exit when H >= ENTROPY_HEALTHY (0.18) for ok_streak >= N
      - force exit red immediately when H >= ENTROPY_FULLY_RECOVERED (0.5)
      - N=1 if H>=0.5, N=2 if H>=0.25, else N=3
      - while exiting / after clear, taper entropy_scale toward 0.02-0.05 (no cliff)
      - force_arm is IGNORED when H is already healthy (prevents exit deadlock)
      - missing / boot-zero Policy Entropy never arms HARD RECOVERY
    """
    if prolonged_cycles is not None:
        rollback_after_cycles = int(prolonged_cycles)
    entropy_raw = read_policy_entropy(metrics)
    if entropy_metric_unreliable(entropy_raw, timesteps=timesteps):
        # Boot / missing metric: clear stale sticky arm from a prior run, no HARD RECOVERY
        if state.entropy_recovery_active and int(timesteps) < _BOOT_ENTROPY_GUARD_STEPS:
            state.entropy_recovery_active = False
            state.entropy_recovery_cycles = 0
            state.entropy_recovery_need_rollback = False
            state.entropy_recovery_log = ""
            state.entropy_recovery_tier = ""
            state.entropy_ok_streak = 0
        return {}
    entropy = float(entropy_raw)  # type: ignore[arg-type]
    tier = entropy_tier(entropy)
    # Never treat "still recovering" as a fresh death when H is healthy — that
    # pinned zone=red + coef=0.35 forever (force_arm=ev.entropy_recovery_active).
    if force_arm and (entropy >= ENTROPY_HEALTHY or not is_entropy_dead(entropy)):
        force_arm = False
    dead = is_entropy_dead(entropy) or force_arm
    recover_ok_cycles = recovery_ok_cycles_needed(entropy, default=3)
    base = max(1e-4, float(baseline_entropy_scale))

    def _exit_patch(*, cleared: bool, prev_coef: float, tapered: float) -> dict[str, Any]:
        """Tapered exit overrides — yellow/green friendly."""
        if cleared:
            note = (
                f"[AutoTrainer] RECOVERY COMPLETE H={entropy:.4f} -> zone=green "
                f"entropy_scale {prev_coef:.4f}->{tapered:.4f}"
            )
            zone = "green"
            skill = True
        else:
            note = (
                f"[AutoTrainer] RECOVERY EXIT H={entropy:.4f} "
                f"ok_streak={state.entropy_ok_streak}/{recover_ok_cycles} "
                f"coef {prev_coef:.4f}->{tapered:.4f} (taper)"
            )
            zone = "yellow"
            skill = entropy >= ENTROPY_CLEAR_EXIT and state.entropy_ok_streak >= 2
        return {
            "entropy_scale": tapered,
            "entropy_death_recovery": False if cleared else True,
            "hard_recovery": False,
            "freeze_op_chaos": not cleared,
            "safety_zone": zone,
            "recovery_exiting": True,
            "es_noise_scale": 0.0 if not cleared else 0.01,
            "es_enabled": cleared,
            "pbt_paused": not cleared,
            "op_destructive_paused": not cleared,
            "truncation_pbt_paused": not cleared,
            "meta_gradients_paused": not cleared,
            "env_architect_paused": not cleared,
            "compete_rotate_paused": not cleared,
            "skill_tracker_enabled": skill,
            "priority_sampling": False,
            "epochs": 1 if not cleared else 2,
            "note": note,
        }

    def _continue_post_clear_taper() -> dict[str, Any]:
        """After leaving sticky red, keep stepping coef down for a few cycles."""
        if state.entropy_taper_remaining <= 0:
            return {}
        prev = (
            state.entropy_recovery_coef
            if state.entropy_recovery_coef > 0
            else (state.entropy_taper_from or _RECOVERY_MAX_ENTROPY_SCALE)
        )
        done = _RECOVERY_TAPER_CYCLES - state.entropy_taper_remaining + 1
        tapered = taper_entropy_scale(
            prev, base, ok_streak=max(1, done), taper_cycles=_RECOVERY_TAPER_CYCLES
        )
        if tick_cycle:
            state.entropy_taper_remaining = max(0, state.entropy_taper_remaining - 1)
        state.entropy_recovery_coef = tapered
        note = (
            f"[AutoTrainer] RECOVERY TAPER H={entropy:.4f} "
            f"entropy_scale {prev:.4f}->{tapered:.4f} "
            f"(remaining={state.entropy_taper_remaining})"
        )
        state.entropy_recovery_log = note
        return {
            "entropy_scale": tapered,
            "entropy_death_recovery": False,
            "hard_recovery": False,
            "freeze_op_chaos": False,
            "safety_zone": "green",
            "recovery_exiting": state.entropy_taper_remaining > 0,
            "note": note,
        }

    # Post-clear taper (zone already green; sticky recovery off)
    if not state.entropy_recovery_active and state.entropy_taper_remaining > 0:
        if entropy >= ENTROPY_HEALTHY:
            return _continue_post_clear_taper()
        state.entropy_taper_remaining = 0

    # Hysteresis: stay in recovery until entropy clears exit threshold for N ticks
    if state.entropy_recovery_active and entropy >= ENTROPY_HEALTHY:
        if tick_cycle:
            state.entropy_ok_streak += 1
        needed = recovery_ok_cycles_needed(entropy, default=recover_ok_cycles)
        clear_ready = state.entropy_ok_streak >= needed and (
            entropy >= ENTROPY_CLEAR_EXIT or entropy >= ENTROPY_FULLY_RECOVERED
        )
        prev_coef = (
            state.entropy_recovery_coef
            if state.entropy_recovery_coef > 0
            else _RECOVERY_MAX_ENTROPY_SCALE
        )
        step = max(1, state.entropy_ok_streak)
        if clear_ready:
            tapered = taper_entropy_scale(
                prev_coef, base, ok_streak=1, taper_cycles=_RECOVERY_TAPER_CYCLES
            )
            exit_patch = _exit_patch(cleared=True, prev_coef=prev_coef, tapered=tapered)
            state.entropy_recovery_log = str(exit_patch["note"])
            state.entropy_recovery_active = False
            state.entropy_recovery_coef = tapered
            state.entropy_taper_from = prev_coef
            state.entropy_taper_remaining = max(0, _RECOVERY_TAPER_CYCLES - 1)
            state.entropy_recovery_cycles = 0
            state.entropy_recovery_tier = ""
            state.entropy_recovery_need_rollback = False
            state.entropy_ok_streak = 0
            return exit_patch
        tapered = taper_entropy_scale(
            prev_coef, base, ok_streak=step, taper_cycles=_RECOVERY_TAPER_CYCLES
        )
        exit_hold = _exit_patch(cleared=False, prev_coef=prev_coef, tapered=tapered)
        state.entropy_recovery_coef = tapered
        state.entropy_recovery_log = str(exit_hold["note"])
        return exit_hold

    if state.entropy_recovery_active and entropy >= ENTROPY_RECOVER_HOLD and tier == "ok":
        if tick_cycle:
            state.entropy_ok_streak += 1
        held = taper_entropy_scale(
            max(state.entropy_recovery_coef, _RECOVERY_MIN_ENTROPY_SCALE),
            base,
            ok_streak=state.entropy_ok_streak,
            taper_cycles=_RECOVERY_TAPER_CYCLES + 2,
        )
        state.entropy_recovery_coef = held
        log = (
            f"[AutoTrainer] RECOVERY HOLD H={entropy:.4f} "
            f"ok_streak={state.entropy_ok_streak}/{recover_ok_cycles} "
            f"coef->{held:.4f} (soft)"
        )
        state.entropy_recovery_log = log
        if tick_cycle:
            state.entropy_recovery_cycles += 1
        return {
            "entropy_scale": held,
            "entropy_death_recovery": True,
            "hard_recovery": True,
            "freeze_op_chaos": True,
            "safety_zone": "yellow",
            "es_noise_scale": 0.0,
            "es_enabled": False,
            "pbt_paused": True,
            "op_destructive_paused": True,
            "skill_tracker_enabled": False,
            "priority_sampling": False,
            "epochs": 1,
            "note": log,
        }

    if state.entropy_recovery_active:
        if tick_cycle:
            state.entropy_ok_streak = 0
    elif not (dead or state.entropy_recovery_active):
        if tick_cycle:
            state.entropy_ok_streak = 0
        return {}

    if not (dead or state.entropy_recovery_active):
        return {}

    # Still actually dead (or freshly armed) — full HARD RECOVERY
    # tier=ok must never emit HARD RECOVERY (tiny coef churn / nonsense logs)
    if not is_entropy_dead(entropy):
        return {}
    prev = (
        state.entropy_recovery_prev_coef
        if state.entropy_recovery_prev_coef > 0
        else baseline_entropy_scale
    )
    if state.entropy_recovery_active:
        cycles = state.entropy_recovery_cycles + (1 if tick_cycle else 0)
    else:
        cycles = 1 if tick_cycle else max(1, state.entropy_recovery_cycles)
    patch, log = entropy_death_recovery_patch(
        policy_entropy=entropy,
        baseline_entropy_scale=baseline_entropy_scale,
        prev_coef=prev,
        cycle=cycles,
    )
    if not patch:
        return {}

    if state.entropy_recovery_active:
        held = max(state.entropy_recovery_coef, _RECOVERY_MIN_ENTROPY_SCALE)
        if tier in ("critical", "hard"):
            held = max(held, float(patch["entropy_scale"]))
        if entropy < ENTROPY_HEALTHY:
            held = max(held, float(patch["entropy_scale"]))
        patch["entropy_scale"] = held
        log = (
            f"[AutoTrainer] HARD RECOVERY entropy_death -> tier={tier} H={entropy:.4e} "
            f"coef {state.entropy_recovery_prev_coef:.4f}->{held:.4f} "
            f"(hold, cycle={cycles} ok_streak={state.entropy_ok_streak}/{recover_ok_cycles}) "
            f"ES/PBT/meta/env MUTED"
        )
        patch["note"] = log
        state.entropy_recovery_cycles = cycles
    else:
        state.entropy_recovery_prev_coef = prev
        state.entropy_recovery_started_ts = timesteps
        state.entropy_recovery_cycles = cycles
        log = (
            f"[AutoTrainer] HARD RECOVERY entropy_death -> tier={tier} H={entropy:.4e} "
            f"coef {prev:.4f}->{patch['entropy_scale']:.4f} cycle={cycles}"
        )
        patch["note"] = log

    state.entropy_recovery_active = True
    state.entropy_recovery_coef = float(patch["entropy_scale"])
    state.entropy_taper_remaining = 0
    state.entropy_recovery_tier = tier if tier != "ok" else state.entropy_recovery_tier or "soft"
    state.entropy_recovery_log = log
    state.entropy_recovery_need_rollback = (
        state.entropy_recovery_cycles >= int(rollback_after_cycles)
        and entropy < ENTROPY_DEATH_HARD
    )
    if state.entropy_recovery_need_rollback:
        patch["entropy_recovery_rollback"] = True
        patch["note"] = (
            f"{log} | ROLLBACK+HARD_RECOVERY after {state.entropy_recovery_cycles} "
            f"dead cycles (keep recovery entropy, restore rewards only)"
        )
        state.entropy_recovery_log = str(patch["note"])
    return patch



def detect_events(
    metrics: dict[str, Any],
    prev: dict[str, float],
    *,
    phase: int = 0,
    timesteps: int = 0,
) -> list[MetricEvent]:
    """Compare current vs previous metrics and emit actionable events."""
    events: list[MetricEvent] = []
    if not metrics:
        return events

    reward = _f(metrics, "Average Step Reward", "avg_reward")
    prev_reward = prev.get("Average Step Reward", prev.get("avg_reward", reward))
    entropy_opt = read_policy_entropy(metrics)
    entropy_unreliable = entropy_metric_unreliable(entropy_opt, timesteps=timesteps)
    entropy = float(entropy_opt) if entropy_opt is not None else float("nan")
    prev_entropy = prev.get("Policy Entropy", prev.get("policy_entropy", entropy))
    kl = _f(metrics, "KL Div Loss", "kl", "Mean KL Divergence")
    touch = _f(metrics, "Touch Ratio", "touch_ratio")
    prev_touch = prev.get("Touch Ratio", prev.get("touch_ratio", touch))
    goal = _f(metrics, "Goal Rate", "goal_rate", "Goals Per Episode")
    prev_goal = prev.get("Goal Rate", prev.get("goal_rate", prev.get("Goals Per Episode", goal)))
    sps = _f(metrics, "Overall Steps/Second")
    prev_sps = prev.get("Overall Steps/Second", sps)

    # Reward collapse -> boost exploration (ES noise + entropy) - skipped if entropy dead
    if (
        prev
        and reward < prev_reward * 0.85
        and prev_reward != 0
        and not entropy_unreliable
        and not is_entropy_dead(entropy)
    ):
        drop = (prev_reward - reward) / max(abs(prev_reward), 1e-6)
        events.append(
            MetricEvent(
                "reward_collapse",
                severity=min(2.0, drop * 4),
                detail=f"reward {prev_reward:.4f}->{reward:.4f}",
                suggest={
                    "entropy_scale": min(0.08, max(0.03, 0.04)),
                    "es_noise_scale": 0.02,
                    "event_advantage_boost": 1.6,
                    "var_max": 0.7,
                },
            )
        )

    # Entropy death -> policy collapsed; HARD RECOVERY (sticky path also arms).
    # Require a real Policy Entropy metric — never arm on missing / boot-zero / tier=ok.
    if not entropy_unreliable and is_entropy_dead(entropy):
        rec, log = entropy_death_recovery_patch(
            policy_entropy=entropy,
            baseline_entropy_scale=_DEFAULT_BASELINE_ENTROPY_SCALE,
            prev_coef=_DEFAULT_BASELINE_ENTROPY_SCALE,
        )
        if rec and log:
            tier = entropy_tier(entropy)
            events.append(
                MetricEvent(
                    "entropy_death",
                    severity=2.5 if tier == "critical" else (2.0 if tier == "hard" else 1.6),
                    detail=f"entropy={entropy:.4e} | {log}",
                    suggest=rec,
                )
            )
    elif (
        not entropy_unreliable
        and prev
        and not math.isnan(entropy)
        and not math.isnan(float(prev_entropy))
        and entropy < float(prev_entropy) * 0.7
        and entropy < ENTROPY_DEATH_SOFT
    ):
        rec, log = entropy_death_recovery_patch(
            policy_entropy=entropy,
            baseline_entropy_scale=_DEFAULT_BASELINE_ENTROPY_SCALE,
            prev_coef=_DEFAULT_BASELINE_ENTROPY_SCALE,
        )
        if rec and log:
            tier = entropy_tier(entropy)
            events.append(
                MetricEvent(
                    "entropy_death",
                    severity=2.5 if tier == "critical" else (2.0 if tier == "hard" else 1.6),
                    detail=f"entropy={entropy:.4e} | {log}",
                    suggest=rec,
                )
            )

    # KL explosion -> too aggressive updates
    if kl > 0.08:
        events.append(
            MetricEvent(
                "kl_explosion",
                severity=min(2.0, kl / 0.04),
                detail=f"kl={kl:.4f}",
                suggest={
                    "policy_lr": 5e-5,
                    "critic_lr": 8e-5,
                    "clip_range": 0.12,
                    "epochs": 1,
                    "es_noise_scale": 0.0,
                },
            )
        )

    # Touch drought (ball contact skill stall)
    if prev and touch < prev_touch * 0.9 and touch < 0.25 and phase >= 0:
        if not is_entropy_dead(entropy):
            events.append(
                MetricEvent(
                    "touch_drought",
                    severity=0.9,
                    detail=f"touch {prev_touch:.3f}->{touch:.3f}",
                    suggest={
                        "event_advantage_boost": 1.8,
                        "gae_lambda": 0.96,
                        "entropy_scale": 0.03,
                    },
                )
            )

    # Goal spike -> exploit (reduce noise, sharpen policy)
    # Skip while entropy is dead - exploit would deepen collapse
    if prev and goal > prev_goal * 1.25 and goal > 0.01 and not is_entropy_dead(entropy):
        events.append(
            MetricEvent(
                "goal_spike",
                severity=1.0,
                detail=f"goal {prev_goal:.4f}->{goal:.4f}",
                suggest={
                    "es_noise_scale": 0.005,
                    "entropy_scale": 0.012,
                    "event_advantage_boost": 1.35,
                    "epochs": 2 if phase >= 1 else 1,
                    "clip_range": 0.18,
                },
            )
        )

    # Throughput stall (possible GPU/CPU imbalance) - nudge batching-related knobs lightly
    if prev_sps > 1000 and sps < prev_sps * 0.7:
        events.append(
            MetricEvent(
                "sps_stall",
                severity=0.5,
                detail=f"sps {prev_sps:.0f}->{sps:.0f}",
                suggest={"epochs": 1, "es_noise_scale": 0.0},
            )
        )

    # Healthy plateau with good reward -> mild ES explore to escape local optima
    if prev and abs(reward - prev_reward) / max(abs(prev_reward), 1e-6) < 0.02 and reward > 0:
        if entropy > 0.2 and kl < 0.04:
            events.append(
                MetricEvent(
                    "plateau",
                    severity=0.4,
                    detail=f"reward flat at {reward:.4f}",
                    suggest={
                        "es_noise_scale": 0.03,
                        "event_advantage_boost": 1.25,
                        "var_max": min(1.0, _f(metrics, "var_max", default=0.5) + 0.1),
                    },
                )
            )

    # SSL §4 stagnation: reward + entropy both stuck -> diversity shock
    if prev and abs(reward - prev_reward) / max(abs(prev_reward), 1e-6) < 0.01:
        if abs(entropy - prev_entropy) < 0.01 and entropy < 0.45 and not is_entropy_dead(entropy):
            events.append(
                MetricEvent(
                    "ssl_stagnation",
                    severity=1.1,
                    detail=f"reward+entropy flat r={reward:.4f} e={entropy:.3f}",
                    suggest={
                        "entropy_scale": min(0.08, max(0.022, 0.018 + entropy * 0.05)),
                        "event_advantage_boost": 1.65,
                        "es_noise_scale": 0.02,
                        "var_max": 0.75,
                        "gpu_reset_kickoff": 0.25,
                        "gpu_reset_fuzzed": 0.40,
                        "gpu_reset_aerial": 0.35,
                        "fuzzed_weight": 0.40,
                        "aerial_weight": 0.35,
                        "ssl_guide_post_apex": True,
                    },
                )
            )

    events.sort(key=lambda e: e.severity, reverse=True)
    return events


def should_fire(state: EventState, timesteps: int, events: list[MetricEvent]) -> bool:
    if state.entropy_recovery_active:
        return True
    if not events:
        return False
    if any(e.name == "entropy_death" for e in events):
        # Entropy death bypasses cooldown entirely - recovery must land every tick
        return True
    if timesteps - state.last_event_timesteps < state.cooldown_steps:
        # Critical events bypass short cooldown
        if events[0].severity >= 1.4 and timesteps - state.last_event_timesteps >= state.cooldown_steps // 3:
            return True
        return False
    return True


def merge_event_suggestions(events: list[MetricEvent], max_events: int = 3) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    # Apply lower-severity first so entropy_death (high severity) wins last
    ordered = sorted(events[:max_events], key=lambda e: e.severity)
    for ev in ordered:
        patch.update(ev.suggest)
    return patch


def snapshot_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    keys = (
        "Average Step Reward",
        "avg_reward",
        "Policy Entropy",
        "policy_entropy",
        "KL Div Loss",
        "kl",
        "Mean KL Divergence",
        "Touch Ratio",
        "touch_ratio",
        "Goal Rate",
        "goal_rate",
        "Goals Per Episode",
        "Overall Steps/Second",
    )
    out: dict[str, float] = {}
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                out[k] = float(metrics[k])
            except (TypeError, ValueError):
                pass
    return out
