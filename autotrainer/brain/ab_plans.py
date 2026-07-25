"""
A/B automatic plan racing for AutoTrainer intelligence.

Maintains two candidate plans/windows. After a scoring horizon (Elo evals
and/or AT cycles), keep the winner by Elo delta + reward stability and
regenerate the loser. Safety veto still always wins — AB never acts in red.

Equal / within-epsilon scores are ties: do not crown A forever; diversify
the challenger and pause A/B after repeated ties.

Opt-out: ``ab_plans.enabled: false`` or ``GIGA_NO_AB_PLANS=1``.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json_atomic
from .meta_brain import ACTION_NAMES, build_action_patch

STATE_FILENAME = "ab_plans_state.json"

# Mild alternative actions per diagnosis for the challenger arm
_CHALLENGER: dict[str, tuple[str, ...]] = {
    "entropy_risk": ("entropy_nudge_up", "pause_op_toys", "hold"),
    "reward_volatile": ("league_ease", "lr_nudge_down", "entropy_nudge_up"),
    "reward_hack_suspect": ("reward_touch_mix", "league_ease", "gpu_reset_diversify"),
    "vs_nexto_weak": ("reward_touch_mix", "lr_nudge_down", "hold", "gpu_reset_diversify"),
    "elo_stalled": ("reward_touch_mix", "hold", "lr_nudge_down", "gpu_reset_diversify"),
    "sps_ok_learning": ("hold", "lr_nudge_down", "entropy_nudge_down"),
    "healthy": ("hold", "lr_nudge_down", "entropy_nudge_down"),
}

# Prefer these when diversifying after flat ties (avoid expert thrash)
_DIVERSIFY_POOL: tuple[str, ...] = (
    "hold",
    "reward_touch_mix",
    "lr_nudge_down",
    "gpu_reset_diversify",
    "entropy_nudge_up",
)


def ab_plans_enabled(cfg: dict[str, Any] | None = None) -> bool:
    if os.environ.get("GIGA_NO_AB_PLANS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    if os.environ.get("GIGA_AT_LEGACY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    ab = (cfg or {}).get("ab_plans") or {}
    if ab.get("enabled") is False:
        return False
    intel = (cfg or {}).get("intelligence") or {}
    if intel.get("enabled", True) and ab.get("enabled") is None:
        return True
    return bool(ab.get("enabled", True))


def _f(v: Any, default: float = float("nan")) -> float:
    if v is None:
        return default
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


@dataclass
class PlanArm:
    slot: str  # "A" or "B"
    diagnosis: str = "healthy"
    action: str = "hold"
    plan_text: str = ""
    cycle_started: int = 0
    timesteps_started: int = 0
    elo_started: float = float("nan")
    elo_evals_started: int = 0
    reward_var_started: float = 0.0
    score: float | None = None
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "diagnosis": self.diagnosis,
            "action": self.action,
            "plan_text": self.plan_text,
            "cycle_started": self.cycle_started,
            "timesteps_started": self.timesteps_started,
            "elo_started": self.elo_started,
            "elo_evals_started": self.elo_evals_started,
            "reward_var_started": self.reward_var_started,
            "score": self.score,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, default_slot: str = "A") -> PlanArm:
        if not d:
            return cls(slot=default_slot)
        return cls(
            slot=str(d.get("slot") or default_slot),
            diagnosis=str(d.get("diagnosis") or "healthy"),
            action=str(d.get("action") or "hold"),
            plan_text=str(d.get("plan_text") or ""),
            cycle_started=int(d.get("cycle_started") or 0),
            timesteps_started=int(d.get("timesteps_started") or 0),
            elo_started=_f(d.get("elo_started")),
            elo_evals_started=int(d.get("elo_evals_started") or 0),
            reward_var_started=float(d.get("reward_var_started") or 0.0),
            score=None if d.get("score") is None else float(d["score"]),
            active=bool(d.get("active")),
        )


@dataclass
class ABPlanState:
    arms: dict[str, PlanArm] = field(default_factory=dict)
    active_slot: str = "A"
    cycles: int = 0
    trials: int = 0
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    consecutive_ties: int = 0
    pause_until_cycle: int = 0
    last_winner: str = ""
    last_score_a: float = 0.0
    last_score_b: float = 0.0
    last_result: str = ""  # win_a | win_b | tie | pause
    elo_eval_count: int = 0
    seed: int = 42
    recent_challengers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "kind": "ab_plans",
            "updated": datetime.now(timezone.utc).isoformat(),
            "arms": {k: v.to_dict() for k, v in self.arms.items()},
            "active_slot": self.active_slot,
            "cycles": self.cycles,
            "trials": self.trials,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "consecutive_ties": self.consecutive_ties,
            "pause_until_cycle": self.pause_until_cycle,
            "last_winner": self.last_winner,
            "last_score_a": self.last_score_a,
            "last_score_b": self.last_score_b,
            "last_result": self.last_result,
            "elo_eval_count": self.elo_eval_count,
            "seed": self.seed,
            "recent_challengers": list(self.recent_challengers[-8:]),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ABPlanState:
        if not d:
            st = cls()
            st.arms = {"A": PlanArm(slot="A"), "B": PlanArm(slot="B")}
            return st
        arms_raw = d.get("arms") or {}
        arms = {
            "A": PlanArm.from_dict(arms_raw.get("A"), "A"),
            "B": PlanArm.from_dict(arms_raw.get("B"), "B"),
        }
        recent = d.get("recent_challengers") or []
        return cls(
            arms=arms,
            active_slot=str(d.get("active_slot") or "A"),
            cycles=int(d.get("cycles") or 0),
            trials=int(d.get("trials") or 0),
            wins_a=int(d.get("wins_a") or 0),
            wins_b=int(d.get("wins_b") or 0),
            ties=int(d.get("ties") or 0),
            consecutive_ties=int(d.get("consecutive_ties") or 0),
            pause_until_cycle=int(d.get("pause_until_cycle") or 0),
            last_winner=str(d.get("last_winner") or ""),
            last_score_a=float(d.get("last_score_a") or 0.0),
            last_score_b=float(d.get("last_score_b") or 0.0),
            last_result=str(d.get("last_result") or ""),
            elo_eval_count=int(d.get("elo_eval_count") or 0),
            seed=int(d.get("seed") or 42),
            recent_challengers=[str(x) for x in recent][-8:],
        )


def score_arm(
    *,
    elo_start: float,
    elo_now: float,
    elo_delta: float,
    reward_var: float,
    zone: str,
    crashed: bool,
) -> float:
    """Scalar score from *window* Elo progress + stability. Higher is better.

    ``elo_delta`` is accepted for API compat but ignored when ``elo_start`` /
    ``elo_now`` are valid — global/stale Elo delta must not make A and B identical.
    """
    s = 0.0
    window_delta = float("nan")
    if not math.isnan(elo_start) and not math.isnan(elo_now):
        window_delta = elo_now - elo_start
        s += max(-1.0, min(1.0, window_delta / 20.0))
    elif not math.isnan(elo_delta):
        # Fallback only when window Elo unavailable
        s += max(-0.5, min(0.5, elo_delta / 20.0))
    # Stability bonus
    if reward_var < 0.02:
        s += 0.25
    elif reward_var < 0.05:
        s += 0.10
    elif reward_var > 0.12:
        s -= 0.35
    if crashed or str(zone).lower() == "red":
        s -= 1.0
    if str(zone).lower() == "green":
        s += 0.1
    return max(-2.0, min(2.0, s))


class ABPlanController:
    """Race two plan arms; promote winner by Elo/stability."""

    def __init__(self, watch_dir: Path, cfg: dict[str, Any] | None = None) -> None:
        self.watch_dir = Path(watch_dir)
        self.cfg = cfg or {}
        self.state_path = self.watch_dir / STATE_FILENAME
        self.state = ABPlanState.from_dict(read_json(self.state_path))
        if "A" not in self.state.arms:
            self.state.arms["A"] = PlanArm(slot="A")
        if "B" not in self.state.arms:
            self.state.arms["B"] = PlanArm(slot="B")
        ab = self.cfg.get("ab_plans") or {}
        if "seed" in ab:
            self.state.seed = int(ab["seed"])
        self._rng = random.Random(self.state.seed + self.state.trials + self.state.ties)

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state.to_dict())

    def _ab(self) -> dict[str, Any]:
        return self.cfg.get("ab_plans") or {}

    def _horizon_cycles(self) -> int:
        ab = self._ab()
        intel = self.cfg.get("intelligence") or {}
        default = int(intel.get("score_horizon_cycles", 16))
        return max(4, int(ab.get("horizon_cycles", default)))

    def _min_elo_evals(self) -> int:
        # Default 2 so window Elo can differentiate; smoke may set 1
        return max(1, int(self._ab().get("min_elo_evals", 2)))

    def _horizon_steps(self) -> int:
        return max(5_000_000, int(self._ab().get("horizon_steps", 25_000_000)))

    def _score_epsilon(self) -> float:
        return max(0.0, float(self._ab().get("score_epsilon", 0.05)))

    def _tie_pause_after(self) -> int:
        return max(1, int(self._ab().get("tie_pause_after", 2)))

    def _tie_pause_cycles(self) -> int:
        return max(4, int(self._ab().get("tie_pause_cycles", 48)))

    def paused(self, cycle: int | None = None) -> bool:
        c = int(cycle if cycle is not None else self.state.cycles)
        return c < int(self.state.pause_until_cycle)

    def note_elo_eval(self) -> None:
        self.state.elo_eval_count += 1
        self.save()

    def _challenger_action(
        self,
        diagnosis: str,
        primary: str,
        *,
        diversify: bool = False,
    ) -> str:
        if diversify:
            opts = [a for a in _DIVERSIFY_POOL if a in ACTION_NAMES and a != primary]
        else:
            opts = list(_CHALLENGER.get(diagnosis, ("hold", "lr_nudge_down")))
            opts = [a for a in opts if a in ACTION_NAMES and a != primary]
        # Avoid repeating the same challenger endlessly
        recent = set(self.state.recent_challengers[-4:])
        fresh = [a for a in opts if a not in recent]
        if fresh:
            opts = fresh
        if not opts:
            opts = [a for a in ACTION_NAMES if a != primary][:4] or ["hold"]
        # Deterministic rotation with entropy from trials/ties (not sticky first pick)
        self._rng = random.Random(
            self.state.seed + 17 * self.state.trials + 31 * self.state.ties + len(opts)
        )
        pick = self._rng.choice(opts)
        self.state.recent_challengers = (self.state.recent_challengers + [pick])[-8:]
        return pick

    def ensure_pair(
        self,
        *,
        diagnosis: str,
        primary_action: str,
        primary_plan_text: str,
        cycle: int,
        timesteps: int,
        elo: float,
        reward_var: float,
    ) -> None:
        """Seed A=primary, B=challenger when empty or diagnosis changed."""
        self.state.cycles = max(self.state.cycles, cycle)
        if self.paused(cycle):
            return
        a = self.state.arms["A"]
        b = self.state.arms["B"]
        need_new = (
            not a.active
            and not b.active
        ) or (
            a.diagnosis != diagnosis and b.diagnosis != diagnosis and primary_action != "hold"
        )
        if not need_new and a.active:
            return
        # Don't re-seed expert-thrash pairs when primary is hold
        if primary_action == "hold":
            return
        chal = self._challenger_action(diagnosis, primary_action)
        a.diagnosis = diagnosis
        a.action = primary_action if primary_action in ACTION_NAMES else "hold"
        a.plan_text = primary_plan_text or f"primary:{a.action}"
        a.cycle_started = cycle
        a.timesteps_started = timesteps
        a.elo_started = elo
        a.elo_evals_started = self.state.elo_eval_count
        a.reward_var_started = reward_var
        a.score = None
        a.active = True

        b.diagnosis = diagnosis
        b.action = chal
        b.plan_text = f"AB challenger: {chal} (vs {a.action})"
        b.cycle_started = cycle
        b.timesteps_started = timesteps
        b.elo_started = elo
        b.elo_evals_started = self.state.elo_eval_count
        b.reward_var_started = reward_var
        b.score = None
        b.active = True

        self.state.active_slot = "A"
        self.save()

    def active_arm(self) -> PlanArm:
        return self.state.arms.get(self.state.active_slot) or self.state.arms["A"]

    def _seed_hold_pair(
        self,
        *,
        diagnosis: str,
        cycle: int,
        timesteps: int,
        elo: float,
        reward_var: float,
        reason: str,
    ) -> None:
        """Park A/B on HOLD after ties / pause — no expert thrash."""
        hold = PlanArm(
            slot="A",
            diagnosis=diagnosis,
            action="hold",
            plan_text=f"AB pause HOLD ({reason})",
            cycle_started=cycle,
            timesteps_started=timesteps,
            elo_started=elo,
            elo_evals_started=self.state.elo_eval_count,
            reward_var_started=reward_var,
            score=None,
            active=True,
        )
        chal_action = self._challenger_action(diagnosis, "hold", diversify=True)
        challenger = PlanArm(
            slot="B",
            diagnosis=diagnosis,
            action=chal_action,
            plan_text=f"AB challenger: {chal_action} (paused)",
            cycle_started=cycle,
            timesteps_started=timesteps,
            elo_started=elo,
            elo_evals_started=self.state.elo_eval_count,
            reward_var_started=reward_var,
            score=None,
            active=False,  # do not race while paused
        )
        self.state.arms = {"A": hold, "B": challenger}
        self.state.active_slot = "A"

    def maybe_score_and_rotate(
        self,
        *,
        cycle: int,
        timesteps: int,
        elo: float,
        elo_delta: float,
        reward_var: float,
        zone: str,
        crashed: bool = False,
    ) -> str | None:
        """If active arm window done, score it; when both scored, pick winner."""
        self.state.cycles = max(self.state.cycles, cycle)
        if self.paused(cycle):
            return None

        arm = self.active_arm()
        if not arm.active:
            return None
        elapsed_c = cycle - arm.cycle_started
        elapsed_ts = timesteps - arm.timesteps_started
        elo_evals = self.state.elo_eval_count - arm.elo_evals_started
        due = (
            elapsed_c >= self._horizon_cycles()
            or elo_evals >= self._min_elo_evals()
            or elapsed_ts >= self._horizon_steps()
            or crashed
            or str(zone).lower() == "red"
        )
        if not due:
            return None

        sc = score_arm(
            elo_start=arm.elo_started,
            elo_now=elo,
            elo_delta=elo_delta,
            reward_var=reward_var,
            zone=zone,
            crashed=crashed,
        )
        arm.score = sc
        if arm.slot == "A":
            self.state.last_score_a = sc
        else:
            self.state.last_score_b = sc

        # Switch to other unscored arm if present
        other_slot = "B" if arm.slot == "A" else "A"
        other = self.state.arms[other_slot]
        if other.active and other.score is None:
            # Start other arm window now
            other.cycle_started = cycle
            other.timesteps_started = timesteps
            other.elo_started = elo
            other.elo_evals_started = self.state.elo_eval_count
            other.reward_var_started = reward_var
            self.state.active_slot = other_slot
            self.save()
            return (
                f"[AutoTrainer] AB: scored {arm.slot}={sc:+.2f} → testing {other_slot} "
                f"({other.action})"
            )

        # Both scored (or other inactive) → decide winner
        sa = self.state.arms["A"].score
        sb = self.state.arms["B"].score
        if sa is None:
            sa = sc if arm.slot == "A" else -999.0
        if sb is None:
            sb = sc if arm.slot == "B" else -999.0

        eps = self._score_epsilon()
        tied = abs(float(sa) - float(sb)) <= eps
        self.state.trials += 1
        self.state.last_score_a = float(sa)
        self.state.last_score_b = float(sb)

        if tied:
            self.state.ties += 1
            self.state.consecutive_ties += 1
            self.state.last_winner = "tie"
            self.state.last_result = "tie"
            diagnosis = self.state.arms["A"].diagnosis or self.state.arms["B"].diagnosis
            # Prefer the less-aggressive of the two (hold > touch_mix > pressure)
            _prefer = ("hold", "lr_nudge_down", "reward_touch_mix", "entropy_nudge_up")
            cand = [self.state.arms["A"].action, self.state.arms["B"].action]
            kept_action = "hold"
            for pref in _prefer:
                if pref in cand:
                    kept_action = pref
                    break
            else:
                kept_action = cand[0] if cand[0] != "league_pressure" else (cand[1] if len(cand) > 1 else "hold")

            pause_now = self.state.consecutive_ties >= self._tie_pause_after()
            if pause_now:
                self.state.pause_until_cycle = cycle + self._tie_pause_cycles()
                self._seed_hold_pair(
                    diagnosis=diagnosis,
                    cycle=cycle,
                    timesteps=timesteps,
                    elo=elo,
                    reward_var=reward_var,
                    reason=f"ties={self.state.consecutive_ties}",
                )
                self.save()
                return (
                    f"[AutoTrainer] AB: tie A={sa:+.2f} B={sb:+.2f} (ε={eps:.2f}) "
                    f"— pause A/B for {self._tie_pause_cycles()} cycles "
                    f"keep=hold (trials={self.state.trials} ties={self.state.ties})"
                )

            # Soft tie: keep milder action as A, diversify challenger (not same league_pressure)
            kept = PlanArm(
                slot="A",
                diagnosis=diagnosis,
                action=kept_action if kept_action in ACTION_NAMES else "hold",
                plan_text=f"AB tie keep:{kept_action}",
                cycle_started=cycle,
                timesteps_started=timesteps,
                elo_started=elo,
                elo_evals_started=self.state.elo_eval_count,
                reward_var_started=reward_var,
                score=None,
                active=True,
            )
            chal_action = self._challenger_action(diagnosis, kept.action, diversify=True)
            challenger = PlanArm(
                slot="B",
                diagnosis=diagnosis,
                action=chal_action,
                plan_text=f"AB challenger: {chal_action} (post-tie)",
                cycle_started=cycle,
                timesteps_started=timesteps,
                elo_started=elo,
                elo_evals_started=self.state.elo_eval_count,
                reward_var_started=reward_var,
                score=None,
                active=True,
            )
            self.state.arms = {"A": kept, "B": challenger}
            self.state.active_slot = "A"
            self.save()
            return (
                f"[AutoTrainer] AB: tie A={sa:+.2f} B={sb:+.2f} (ε={eps:.2f}) "
                f"keep={kept.action} next_challenger={chal_action} "
                f"(trials={self.state.trials} ties={self.state.ties} "
                f"consec={self.state.consecutive_ties})"
            )

        # Clear win — require real score delta
        winner = "A" if sa > sb else "B"
        self.state.last_winner = winner
        self.state.last_result = f"win_{winner.lower()}"
        self.state.consecutive_ties = 0
        if winner == "A":
            self.state.wins_a += 1
        else:
            self.state.wins_b += 1

        # Keep winner as A; regenerate B as new challenger
        win_arm = self.state.arms[winner]
        kept = PlanArm(
            slot="A",
            diagnosis=win_arm.diagnosis,
            action=win_arm.action,
            plan_text=win_arm.plan_text + " [AB winner]",
            cycle_started=cycle,
            timesteps_started=timesteps,
            elo_started=elo,
            elo_evals_started=self.state.elo_eval_count,
            reward_var_started=reward_var,
            score=None,
            active=True,
        )
        chal_action = self._challenger_action(kept.diagnosis, kept.action)
        challenger = PlanArm(
            slot="B",
            diagnosis=kept.diagnosis,
            action=chal_action,
            plan_text=f"AB challenger: {chal_action}",
            cycle_started=cycle,
            timesteps_started=timesteps,
            elo_started=elo,
            elo_evals_started=self.state.elo_eval_count,
            reward_var_started=reward_var,
            score=None,
            active=True,
        )
        self.state.arms = {"A": kept, "B": challenger}
        self.state.active_slot = "A"
        self.save()
        return (
            f"[AutoTrainer] AB: winner={winner} A={sa:+.2f} B={sb:+.2f} "
            f"keep={kept.action} next_challenger={chal_action} "
            f"(trials={self.state.trials} W_A={self.state.wins_a} W_B={self.state.wins_b})"
        )

    def patch_for_active(
        self,
        current: dict[str, Any] | None,
        cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.paused():
            return {
                "ab_slot": "A",
                "ab_action": "hold",
                "ab_plans": True,
                "ab_paused": True,
            }
        arm = self.active_arm()
        if not arm.active or arm.action == "hold":
            return {
                "ab_slot": arm.slot,
                "ab_action": arm.action,
                "ab_plans": True,
            }
        patch = build_action_patch(arm.action, current, cfg=cfg or self.cfg)
        patch["ab_plans"] = True
        patch["ab_slot"] = arm.slot
        patch["ab_action"] = arm.action
        patch["ab_diagnosis"] = arm.diagnosis
        note = str(patch.get("note") or "")
        tag = f"ab_plans:{arm.slot}:{arm.action}"
        patch["note"] = f"{note}; {tag}" if note else tag
        return patch

    def dashboard_blob(self) -> dict[str, Any]:
        a = self.state.arms.get("A")
        b = self.state.arms.get("B")
        return {
            "active_slot": self.state.active_slot,
            "trials": self.state.trials,
            "wins_a": self.state.wins_a,
            "wins_b": self.state.wins_b,
            "ties": self.state.ties,
            "consecutive_ties": self.state.consecutive_ties,
            "pause_until_cycle": self.state.pause_until_cycle,
            "paused": self.paused(),
            "last_winner": self.state.last_winner,
            "last_result": self.state.last_result,
            "last_score_a": self.state.last_score_a,
            "last_score_b": self.state.last_score_b,
            "A": a.to_dict() if a else {},
            "B": b.to_dict() if b else {},
            "elo_eval_count": self.state.elo_eval_count,
            "recent_challengers": list(self.state.recent_challengers[-6:]),
        }
