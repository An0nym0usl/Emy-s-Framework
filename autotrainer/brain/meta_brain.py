"""
Online meta-learning controller for AutoTrainer.

Contextual bandit / EMA preference learner over
discrete interventions — not a second full PPO agent. It watches the bot's
training state, picks mild override nudges, scores outcomes after N cycles,
and updates action preferences so AutoTrainer itself improves across runs.

HARD RULE: red-zone / entropy_death recovery ALWAYS wins. Meta may observe and
learn a negative outcome, but it cannot override fail-closed recovery.
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


STATE_FILENAME = "meta_brain_state.json"

# Discrete interventions available via runtime_overrides (mild, safety-clamped later).
ACTION_NAMES: tuple[str, ...] = (
    "hold",
    "entropy_nudge_up",
    "entropy_nudge_down",
    "reward_touch_mix",
    "reward_goal_mix",
    "league_ease",
    "league_pressure",
    "lr_nudge_down",
    "lr_nudge_up",
    "gpu_reset_diversify",
    "pause_op_toys",
)


def meta_learn_enabled(cfg: dict[str, Any] | None = None) -> bool:
    """Master switch: config meta_learn.enabled (default True) + env opt-out."""
    if os.environ.get("GIGA_NO_META_LEARN", "").strip() in ("1", "true", "TRUE", "yes"):
        return False
    ml = (cfg or {}).get("meta_learn") or {}
    return bool(ml.get("enabled", True))


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


def _entropy_band(h: float) -> str:
    if math.isnan(h) or h < 0.10:
        return "dead"
    if h < 0.18:
        return "low"
    if h < 0.50:
        return "ok"
    if h < 1.50:
        return "healthy"
    return "hot"


def _reward_trend(delta_pct: float | None) -> str:
    if delta_pct is None or math.isnan(float(delta_pct)):
        return "flat"
    d = float(delta_pct)
    if d <= -15.0:
        return "crash"
    if d < -3.0:
        return "down"
    if d > 5.0:
        return "up"
    return "flat"


def _sps_band(sps: float) -> str:
    if math.isnan(sps) or sps <= 0:
        return "unk"
    if sps < 50_000:
        return "slow"
    if sps < 200_000:
        return "mid"
    return "fast"


def context_key(
    *,
    zone: str,
    entropy: float,
    reward_delta_pct: float | None,
    sps: float,
    phase: int,
    skill: float | None = None,
    viz_suffix: str = "",
) -> str:
    """Bucketed context for the bandit (compact string key).

    ``viz_suffix`` is optional soft features from live --render watch
    (e.g. ``|V=viz_no_touch+viz_reward_low``) so meta can learn from watch quality.
    """
    sk = "noskill"
    if skill is not None and not math.isnan(float(skill)):
        s = float(skill)
        if s < 0.2:
            sk = "skill_low"
        elif s < 0.5:
            sk = "skill_mid"
        else:
            sk = "skill_hi"
    base = (
        f"z={zone}|H={_entropy_band(entropy)}|R={_reward_trend(reward_delta_pct)}"
        f"|S={_sps_band(sps)}|P={int(phase)}|{sk}"
    )
    vs = (viz_suffix or "").strip()
    if vs and not vs.startswith("|"):
        vs = "|" + vs
    return base + vs


def score_outcome(
    *,
    before: dict[str, float],
    after: dict[str, float],
    zone_after: str,
    crashed: bool = False,
    reward_variance: float | None = None,
) -> float:
    """Scalar meta-reward: Elo delta + H health + reward *stability* (not twitchy %).

    Primary axes (in order of weight):
      1. Elo / skill delta (credit over longer horizons)
      2. Entropy health band
      3. Reward stability (low variance) + mild level progress
    Twitchy ±200% single-cycle reward deltas are heavily damped.
    """
    h0 = float(before.get("entropy", float("nan")))
    h1 = float(after.get("entropy", float("nan")))
    r0 = float(before.get("reward", float("nan")))
    r1 = float(after.get("reward", float("nan")))
    reward = 0.0

    # --- Elo / skill (primary) ---
    e0 = float(before.get("elo", float("nan")))
    e1 = float(after.get("elo", float("nan")))
    if math.isnan(e0):
        e0 = float(before.get("skill", float("nan")))
    if math.isnan(e1):
        e1 = float(after.get("skill", float("nan")))
    elo_delta = float(after.get("elo_delta", float("nan")))
    if not math.isnan(elo_delta):
        reward += max(-0.7, min(0.7, elo_delta / 15.0))
    elif not math.isnan(e0) and not math.isnan(e1):
        reward += max(-0.7, min(0.7, (e1 - e0) / 15.0))

    # --- Entropy health ---
    if not math.isnan(h1):
        if h1 < 0.10:
            reward -= 1.5
        elif h1 < 0.18:
            reward -= 0.4
        elif 0.18 <= h1 <= 2.0:
            reward += 0.40
            if not math.isnan(h0) and h1 > h0 + 0.02:
                reward += 0.12
        if not math.isnan(h0) and h0 < 0.18 and h1 >= 0.18:
            reward += 0.4  # recovered

    # --- Reward: stability first, progress second (damped) ---
    if reward_variance is not None and not math.isnan(float(reward_variance)):
        # Low variance → bonus; high variance → penalty
        v = float(reward_variance)
        if v < 0.01:
            reward += 0.25
        elif v < 0.04:
            reward += 0.10
        elif v > 0.12:
            reward -= 0.35
        elif v > 0.06:
            reward -= 0.15

    if not math.isnan(r0) and not math.isnan(r1) and abs(r0) > 1e-9:
        pct = (r1 - r0) / abs(r0)
        # Dampen twitchy swings — clamp harder than before (±0.35 vs ±0.8)
        reward += max(-0.35, min(0.35, pct * 0.6))
    elif not math.isnan(r1) and not math.isnan(r0):
        reward += max(-0.2, min(0.2, (r1 - r0) * 2.0))

    if crashed or zone_after == "red":
        reward -= 1.0
    if zone_after == "green":
        reward += 0.1

    return max(-2.0, min(2.0, reward))


# Opposite actions — ban oscillation after a scored trial
_ACTION_OPPOSITES: dict[str, str] = {
    "league_ease": "league_pressure",
    "league_pressure": "league_ease",
    "entropy_nudge_up": "entropy_nudge_down",
    "entropy_nudge_down": "entropy_nudge_up",
    "reward_touch_mix": "reward_goal_mix",
    "reward_goal_mix": "reward_touch_mix",
    "lr_nudge_up": "lr_nudge_down",
    "lr_nudge_down": "lr_nudge_up",
}
def build_action_patch(
    action: str,
    current: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map discrete action → mild override patch (never recovery-critical alone)."""
    cur = dict(current or {})
    ml = (cfg or {}).get("meta_learn") or {}
    ent_step = float(ml.get("entropy_nudge", 0.003))
    lr_factor = float(ml.get("lr_nudge_factor", 0.08))
    league_step = float(ml.get("league_nudge", 0.03))
    rw_step = float(ml.get("reward_nudge", 0.08))

    if action == "hold" or action not in ACTION_NAMES:
        return {}

    out: dict[str, Any] = {"meta_action": action, "meta_learn": True}

    if action == "entropy_nudge_up":
        base = float(cur.get("entropy_scale", 0.022) or 0.022)
        out["entropy_scale"] = min(0.04, base + ent_step)
    elif action == "entropy_nudge_down":
        base = float(cur.get("entropy_scale", 0.022) or 0.022)
        out["entropy_scale"] = max(0.008, base - ent_step)
    elif action == "reward_touch_mix":
        rw = dict(cur.get("reward_weights") or {})
        for k, mult in (
            ("TouchReward", 1.0 + rw_step),
            ("VelocityPlayerToBallReward", 1.0 + rw_step * 0.6),
            ("GoalReward", max(0.05, 1.0 - rw_step * 0.5)),
        ):
            rw[k] = float(rw.get(k, 1.0)) * mult
        out["reward_weights"] = rw
    elif action == "reward_goal_mix":
        rw = dict(cur.get("reward_weights") or {})
        for k, mult in (
            ("GoalReward", 1.0 + rw_step),
            ("StrongTouchReward", 1.0 + rw_step * 0.5),
            ("TouchReward", max(0.5, 1.0 - rw_step * 0.3)),
        ):
            rw[k] = float(rw.get(k, 1.0)) * mult
        out["reward_weights"] = rw
    elif action == "league_ease":
        pool = float(cur.get("opponent_pool_chance", 0.05) or 0.05)
        old = float(cur.get("train_against_old_chance", 0.15) or 0.15)
        out["opponent_pool_chance"] = max(0.0, pool - league_step)
        out["train_against_old_chance"] = max(0.05, old - league_step)
    elif action == "league_pressure":
        pool = float(cur.get("opponent_pool_chance", 0.05) or 0.05)
        old = float(cur.get("train_against_old_chance", 0.15) or 0.15)
        out["opponent_pool_chance"] = min(0.25, pool + league_step)
        out["train_against_old_chance"] = min(0.40, old + league_step)
    elif action == "lr_nudge_down":
        plr = float(cur.get("policy_lr", 1e-4) or 1e-4)
        clr = float(cur.get("critic_lr", 1e-4) or 1e-4)
        out["policy_lr"] = max(1e-5, plr * (1.0 - lr_factor))
        out["critic_lr"] = max(1e-5, clr * (1.0 - lr_factor))
    elif action == "lr_nudge_up":
        plr = float(cur.get("policy_lr", 1e-4) or 1e-4)
        clr = float(cur.get("critic_lr", 1e-4) or 1e-4)
        out["policy_lr"] = min(5e-4, plr * (1.0 + lr_factor))
        out["critic_lr"] = min(5e-4, clr * (1.0 + lr_factor))
    elif action == "gpu_reset_diversify":
        out["gpu_reset_kickoff"] = 0.30
        out["gpu_reset_fuzzed"] = 0.40
        out["gpu_reset_aerial"] = 0.30
    elif action == "pause_op_toys":
        # Soft exploratory mute — NOT the same as HARD recovery; safety can still override.
        out["es_noise_scale"] = 0.0
        out["priority_sampling"] = False
        out["meta_pause_op"] = True

    out["note"] = f"meta_learn:{action}"
    return out


def format_meta_update_banner(
    *,
    meta_reward: float,
    action: str,
    pref_before: float,
    pref_after: float,
    episodes: int,
    context: str = "",
) -> str:
    """Leak-iteration-style report for meta preference updates."""
    sign = "+" if meta_reward >= 0 else ""
    lines = [
        "========================================",
        "[AutoTrainer Meta] Update",
        f"  Meta Reward: {sign}{meta_reward:.2f}",
        f"  Action: {action}",
        f"  Preference: {pref_before:.2f} -> {pref_after:.2f}",
        f"  Episodes: {episodes}",
    ]
    if context:
        lines.append(f"  Context: {context}")
    lines.append("========================================")
    return "\n".join(lines)


@dataclass
class PendingTrial:
    action: str
    context: str
    cycle_started: int
    before: dict[str, float] = field(default_factory=dict)
    preference_before: float = 0.0
    # Long credit assignment anchors
    timesteps_started: int = 0
    elo_evals_started: int = 0
    features: list[float] = field(default_factory=list)
    source: str = "bandit"  # bandit | mlp

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "context": self.context,
            "cycle_started": self.cycle_started,
            "before": dict(self.before),
            "preference_before": self.preference_before,
            "timesteps_started": self.timesteps_started,
            "elo_evals_started": self.elo_evals_started,
            "features": [float(x) for x in self.features],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PendingTrial | None:
        if not d or not d.get("action"):
            return None
        return cls(
            action=str(d["action"]),
            context=str(d.get("context") or ""),
            cycle_started=int(d.get("cycle_started") or 0),
            before={k: float(v) for k, v in (d.get("before") or {}).items()},
            preference_before=float(d.get("preference_before") or 0.0),
            timesteps_started=int(d.get("timesteps_started") or 0),
            elo_evals_started=int(d.get("elo_evals_started") or 0),
            features=[float(x) for x in (d.get("features") or [])],
            source=str(d.get("source") or "bandit"),
        )


@dataclass
class MetaBrainState:
    """Persisted online learner: EMA preferences per (context, action)."""

    preferences: dict[str, dict[str, float]] = field(default_factory=dict)
    episodes: int = 0
    cycles: int = 0
    pending: PendingTrial | None = None
    last_action: str = "hold"
    last_context: str = ""
    last_meta_reward: float = 0.0
    blocked_by_safety: int = 0
    seed: int = 42
    banned_until_cycle: dict[str, int] = field(default_factory=dict)
    reward_history: list[float] = field(default_factory=list)
    last_major_action_cycle: int = -999
    elo_eval_count: int = 0
    last_selector: str = "bandit"  # bandit | mlp

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "kind": "contextual_bandit_ema+meta_mlp",
            "updated": datetime.now(timezone.utc).isoformat(),
            "preferences": {
                ctx: {a: float(v) for a, v in acts.items()}
                for ctx, acts in self.preferences.items()
            },
            "episodes": self.episodes,
            "cycles": self.cycles,
            "pending": self.pending.to_dict() if self.pending else None,
            "last_action": self.last_action,
            "last_context": self.last_context,
            "last_meta_reward": self.last_meta_reward,
            "blocked_by_safety": self.blocked_by_safety,
            "seed": self.seed,
            "banned_until_cycle": {
                str(k): int(v) for k, v in self.banned_until_cycle.items()
            },
            "reward_history": [float(x) for x in self.reward_history[-24:]],
            "last_major_action_cycle": self.last_major_action_cycle,
            "elo_eval_count": self.elo_eval_count,
            "last_selector": self.last_selector,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> MetaBrainState:
        if not d:
            return cls()
        prefs: dict[str, dict[str, float]] = {}
        for ctx, acts in (d.get("preferences") or {}).items():
            prefs[str(ctx)] = {str(a): float(v) for a, v in (acts or {}).items()}
        return cls(
            preferences=prefs,
            episodes=int(d.get("episodes") or 0),
            cycles=int(d.get("cycles") or 0),
            pending=PendingTrial.from_dict(d.get("pending")),
            last_action=str(d.get("last_action") or "hold"),
            last_context=str(d.get("last_context") or ""),
            last_meta_reward=float(d.get("last_meta_reward") or 0.0),
            blocked_by_safety=int(d.get("blocked_by_safety") or 0),
            seed=int(d.get("seed") or 42),
            banned_until_cycle={
                str(k): int(v) for k, v in (d.get("banned_until_cycle") or {}).items()
            },
            reward_history=[float(x) for x in (d.get("reward_history") or [])][-24:],
            last_major_action_cycle=int(
                d.get("last_major_action_cycle")
                if d.get("last_major_action_cycle") is not None
                else -999
            ),
            elo_eval_count=int(d.get("elo_eval_count") or 0),
            last_selector=str(d.get("last_selector") or "bandit"),
        )

    def prefs_for(self, context: str) -> dict[str, float]:
        if context not in self.preferences:
            # Optimistic start — slight preference for hold to avoid thrash
            self.preferences[context] = {a: (0.15 if a == "hold" else 0.0) for a in ACTION_NAMES}
        # Ensure all actions exist
        row = self.preferences[context]
        for a in ACTION_NAMES:
            row.setdefault(a, 0.0)
        return row


class MetaBrainController:
    """Contextual bandit (+ optional meta MLP) that nudges overrides and learns."""

    def __init__(self, watch_dir: Path, cfg: dict[str, Any] | None = None) -> None:
        self.watch_dir = Path(watch_dir)
        self.cfg = cfg or {}
        self.state_path = self.watch_dir / STATE_FILENAME
        self.state = MetaBrainState.from_dict(read_json(self.state_path))
        ml = self.cfg.get("meta_learn") or {}
        if "seed" in ml:
            self.state.seed = int(ml["seed"])
        self._rng = random.Random(self.state.seed + self.state.episodes)
        self.mlp = None
        try:
            from .meta_mlp import MetaMLPController, meta_mlp_enabled

            if meta_mlp_enabled(self.cfg):
                self.mlp = MetaMLPController(self.watch_dir, self.cfg)
        except Exception:
            self.mlp = None

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state.to_dict())
        if self.mlp is not None:
            try:
                self.mlp.save()
            except Exception:
                pass

    def note_elo_eval(self) -> None:
        """Call when a periodic Elo eval completes — advances long credit."""
        self.state.elo_eval_count += 1
        self.save()

    def _ml(self) -> dict[str, Any]:
        return self.cfg.get("meta_learn") or {}

    def _alpha(self) -> float:
        return float(self._ml().get("learning_rate", 0.25))

    def _epsilon(self) -> float:
        """ε-greedy with decay over episodes (less random thrash as we learn)."""
        ml = self._ml()
        base = float(ml.get("epsilon", 0.12))
        eps_min = float(ml.get("epsilon_min", 0.02))
        decay = float(ml.get("epsilon_decay", 0.97))
        # Prefer intelligence.* when present
        intel = self.cfg.get("intelligence") or {}
        if intel.get("enabled", True) and intel.get("diagnose_plan_act", True):
            if "epsilon_start" in intel:
                base = float(intel["epsilon_start"])
            if "epsilon_min" in intel:
                eps_min = float(intel["epsilon_min"])
            if "epsilon_decay" in intel:
                decay = float(intel["epsilon_decay"])
        eps = base * (decay ** max(0, self.state.episodes))
        return max(eps_min, min(1.0, eps))

    def _score_every(self) -> int:
        """Long credit: many AT cycles (not 3 ticks). Default ~16 under intelligence."""
        ml = self._ml()
        default = 3
        intel = self.cfg.get("intelligence") or {}
        if intel.get("enabled", True) and intel.get("diagnose_plan_act", True):
            default = int(intel.get("score_horizon_cycles", 16))
        return max(1, int(ml.get("score_every_cycles", default)))

    def _score_horizon_steps(self) -> int:
        intel = self.cfg.get("intelligence") or {}
        ml = self._ml()
        default = 25_000_000
        if "score_horizon_steps" in intel:
            default = int(intel["score_horizon_steps"])
        return max(1_000_000, int(ml.get("score_horizon_steps", default)))

    def _min_elo_evals(self) -> int:
        intel = self.cfg.get("intelligence") or {}
        ml = self._ml()
        default = 1
        if "score_min_elo_evals" in intel:
            default = int(intel["score_min_elo_evals"])
        return max(1, int(ml.get("score_min_elo_evals", default)))

    def _trial_due(self, pending: PendingTrial, *, timesteps: int = 0, force: bool = False) -> bool:
        """Long credit: score after cycles OR Elo evals OR ~25–50M steps."""
        if force:
            return True
        elapsed_c = self.state.cycles - pending.cycle_started
        if elapsed_c >= self._score_every():
            return True
        elo_delta = self.state.elo_eval_count - int(pending.elo_evals_started or 0)
        if elo_delta >= self._min_elo_evals():
            return True
        if timesteps > 0 and pending.timesteps_started > 0:
            if timesteps - pending.timesteps_started >= self._score_horizon_steps():
                return True
        return False

    def _temperature(self) -> float:
        return max(0.05, float(self._ml().get("softmax_temperature", 0.45)))

    def _action_cooldown(self) -> int:
        intel = self.cfg.get("intelligence") or {}
        if intel.get("enabled", True) and intel.get("diagnose_plan_act", True):
            return max(1, int(intel.get("cooldown_cycles", 8)))
        return max(1, int(self._ml().get("action_cooldown_cycles", 3)))

    def _material_threshold(self) -> float:
        intel = self.cfg.get("intelligence") or {}
        if "score_material_threshold" in intel:
            return float(intel["score_material_threshold"])
        return float(self._ml().get("score_material_threshold", 0.15))

    def _reward_variance(self) -> float:
        xs = self.state.reward_history[-12:]
        if len(xs) < 3:
            return 0.0
        mean = sum(xs) / len(xs)
        return sum((x - mean) ** 2 for x in xs) / len(xs)

    def _allowed_actions(self) -> list[str]:
        out: list[str] = []
        for a in ACTION_NAMES:
            until = int(self.state.banned_until_cycle.get(a) or 0)
            if self.state.cycles < until:
                continue
            out.append(a)
        return out or ["hold"]

    def _ban_opposite(self, action: str) -> None:
        ml = self._ml()
        intel = self.cfg.get("intelligence") or {}
        ban = ml.get("ban_oscillation", intel.get("ban_oscillation", True))
        if not ban:
            return
        opp = _ACTION_OPPOSITES.get(action)
        if not opp:
            return
        ban_for = max(
            4,
            int(intel.get("oscillation_ban_cycles") or ml.get("oscillation_ban_cycles", 12)),
        )
        self.state.banned_until_cycle[opp] = self.state.cycles + ban_for

    def _snapshot_metrics(
        self,
        metrics: dict[str, Any] | None,
        reward_delta_pct: float | None = None,
        elo_signal: dict[str, float] | None = None,
    ) -> dict[str, float]:
        snap = {
            "entropy": _f(metrics, "Policy Entropy", "policy_entropy", default=float("nan")),
            "reward": _f(metrics, "Average Step Reward", "avg_reward", default=float("nan")),
            "sps": _f(metrics, "Overall Steps/Second", "SPS", default=float("nan")),
            "reward_delta_pct": float(reward_delta_pct)
            if reward_delta_pct is not None and not math.isnan(float(reward_delta_pct))
            else float("nan"),
            "elo": float("nan"),
            "elo_delta": float("nan"),
            "skill": _f(metrics, "Rating/1v1", "Rating/2v2", "Skill Rating", "skill_rating", default=float("nan")),
        }
        sig = elo_signal or {}
        if "elo" in sig:
            snap["elo"] = float(sig["elo"])
        if "elo_delta" in sig:
            snap["elo_delta"] = float(sig["elo_delta"])
        if "skill" in sig and math.isnan(snap["skill"]):
            snap["skill"] = float(sig["skill"])
        return snap

    def _select_action_bandit(self, context: str) -> tuple[str, float]:
        prefs = self.state.prefs_for(context)
        allowed = self._allowed_actions()
        # Respect major-action cooldown (prefer hold)
        cd = self._action_cooldown()
        if self.state.cycles - self.state.last_major_action_cycle < cd:
            if "hold" in allowed:
                return "hold", float(prefs.get("hold", 0.0))

        if self._rng.random() < self._epsilon():
            action = self._rng.choice(allowed)
            return action, float(prefs.get(action, 0.0))

        # Softmax over allowed preferences only
        temp = self._temperature()
        xs = [float(prefs[a]) / temp for a in allowed]
        m = max(xs)
        exps = [math.exp(x - m) for x in xs]
        z = sum(exps) + 1e-12
        probs = [e / z for e in exps]
        r = self._rng.random()
        cum = 0.0
        action = allowed[0]
        for a, p in zip(allowed, probs):
            cum += p
            if r <= cum:
                action = a
                break
        return action, float(prefs.get(action, 0.0))

    def _select_action(
        self,
        context: str,
        *,
        features: Any = None,
    ) -> tuple[str, float, str]:
        """Return (action, preference_or_prob, selector) where selector is mlp|bandit."""
        prefs = self.state.prefs_for(context)
        allowed = self._allowed_actions()
        cd = self._action_cooldown()
        force_hold = self.state.cycles - self.state.last_major_action_cycle < cd

        # Prefer meta MLP when available; bandit remains fallback
        if self.mlp is not None and features is not None:
            try:
                import numpy as np
                from .meta_mlp import format_mlp_banner

                feats = np.asarray(features, dtype=np.float32)
                action, pd, _ld = self.mlp.select_action(
                    feats,
                    allowed=allowed,
                    epsilon=self._epsilon(),
                    force_hold=force_hold and "hold" in allowed,
                )
                self.state.last_selector = "mlp"
                # Occasional banner (material non-hold)
                if action != "hold" and self.state.episodes % 3 == 0:
                    top = sorted(pd.items(), key=lambda kv: kv[1], reverse=True)
                    print(
                        format_mlp_banner(
                            action=action,
                            backend=self.mlp.state.backend,
                            top_probs=top,
                            features_highlight=self.mlp.state.last_features,
                        )
                    )
                return action, float(pd.get(action, 0.0)), "mlp"
            except Exception:
                if self.mlp is not None:
                    self.mlp.state.fallback_bandit += 1

        action, pref = self._select_action_bandit(context)
        self.state.last_selector = "bandit"
        return action, pref, "bandit"

    def _update_preference(self, context: str, action: str, meta_reward: float) -> tuple[float, float]:
        prefs = self.state.prefs_for(context)
        before = float(prefs.get(action, 0.0))
        alpha = self._alpha()
        after = (1.0 - alpha) * before + alpha * meta_reward
        prefs[action] = after
        self.state.preferences[context] = prefs
        return before, after

    def _safety_blocks(self, zone: str, patch: dict[str, Any] | None, gov_recovery: bool) -> bool:
        """True when meta must not intervene (fail-closed)."""
        if gov_recovery:
            return True
        if str(zone).lower() == "red":
            return True
        p = patch or {}
        if p.get("entropy_death_recovery") or p.get("hard_recovery") or p.get("freeze_op_chaos"):
            return True
        return False

    def maybe_score_pending(
        self,
        *,
        metrics: dict[str, Any] | None,
        zone: str,
        reward_delta_pct: float | None = None,
        force: bool = False,
        crashed: bool = False,
        elo_signal: dict[str, float] | None = None,
        timesteps: int = 0,
    ) -> str | None:
        """If a trial finished (long credit / force), score it and update bandit+MLP."""
        pending = self.state.pending
        if not pending:
            return None
        if not self._trial_due(pending, timesteps=timesteps, force=force):
            return None

        after = self._snapshot_metrics(metrics, reward_delta_pct, elo_signal)
        var = self._reward_variance()
        # Death / red while trial open → strong negative
        if zone == "red" or crashed:
            mr = score_outcome(
                before=pending.before,
                after=after,
                zone_after=zone,
                crashed=True,
                reward_variance=var,
            )
            mr = min(mr, -0.8)
        else:
            mr = score_outcome(
                before=pending.before,
                after=after,
                zone_after=zone,
                crashed=False,
                reward_variance=var,
            )

        pref_b, pref_a = self._update_preference(pending.context, pending.action, mr)
        # Train meta MLP from the same long-horizon outcome
        if self.mlp is not None and pending.features:
            try:
                import numpy as np

                self.mlp.train_from_outcome(
                    np.asarray(pending.features, dtype=np.float32),
                    pending.action,
                    mr,
                )
            except Exception:
                pass

        self.state.episodes += 1
        self.state.last_meta_reward = mr
        self.state.last_action = pending.action
        self.state.last_context = pending.context
        if pending.action != "hold":
            self._ban_opposite(pending.action)
        src = pending.source
        self.state.pending = None

        # Quiet: skip Meta Update spam for useless holds / tiny score drifts
        material = abs(mr) >= self._material_threshold()
        pref_moved = abs(pref_a - pref_b) >= 0.05
        interesting = pending.action != "hold" and (material or pref_moved)
        # Holds: almost never banner (only strong negative / death scores)
        hold_loud = pending.action == "hold" and mr <= -0.5
        if interesting or hold_loud:
            banner = format_meta_update_banner(
                meta_reward=mr,
                action=pending.action,
                pref_before=pref_b,
                pref_after=pref_a,
                episodes=self.state.episodes,
                context=f"{pending.context}|src={src}",
            )
            print(banner)
            self.save()
            return banner
        self.save()
        return None

    def step(
        self,
        *,
        metrics: dict[str, Any] | None,
        status: dict[str, Any] | None,
        zone: str,
        current_overrides: dict[str, Any] | None,
        safety_patch: dict[str, Any] | None = None,
        gov_recovery: bool = False,
        reward_delta_pct: float | None = None,
        reward_crashed: bool = False,
        elo_signal: dict[str, float] | None = None,
        viz_suffix: str = "",
        viz_state: dict[str, Any] | None = None,
        suppress: bool = False,
        green_h_streak: int = 0,
        unmute_level: float = 0.0,
        elo_flat_evals: int = 0,
    ) -> dict[str, Any] | None:
        """
        One meta cycle: score pending trial if due, then maybe pick a new action.

        Returns a mild patch to merge, or None when blocked / holding / disabled.
        ``viz_suffix`` soft-tags live --render watch quality into the context key.
        ``suppress`` (intelligence owns the cycle): still score pending, no new nudge.
        """
        if not meta_learn_enabled(self.cfg):
            return None

        self.state.cycles += 1
        # Track reward for stability scoring
        r_now = _f(metrics, "Average Step Reward", "avg_reward")
        if not math.isnan(r_now):
            self.state.reward_history.append(float(r_now))
            self.state.reward_history = self.state.reward_history[-24:]

        st = status or {}
        ts_now = int(st.get("total_timesteps") or 0)

        self.maybe_score_pending(
            metrics=metrics,
            zone=zone,
            reward_delta_pct=reward_delta_pct,
            force=reward_crashed or zone == "red",
            crashed=reward_crashed,
            elo_signal=elo_signal,
            timesteps=ts_now,
        )

        if self._safety_blocks(zone, safety_patch, gov_recovery):
            self.state.blocked_by_safety += 1
            self.save()
            return None

        if suppress:
            self.save()
            return None

        # Only start a new trial when none pending
        if self.state.pending is not None:
            self.save()
            return None

        phase = int(st.get("curriculum_phase") or st.get("phase") or 0)
        snap = self._snapshot_metrics(metrics, reward_delta_pct, elo_signal)
        skill = snap.get("skill", float("nan"))
        if math.isnan(skill) and not math.isnan(snap.get("elo", float("nan"))):
            skill = snap["elo"]
        ctx = context_key(
            zone=zone,
            entropy=snap["entropy"],
            reward_delta_pct=reward_delta_pct,
            sps=snap["sps"],
            phase=phase,
            skill=None if math.isnan(skill) else skill,
            viz_suffix=viz_suffix or "",
        )

        # Build MLP features (includes viz soft signals)
        feat_list: list[float] = []
        try:
            from .meta_mlp import extract_features

            feats = extract_features(
                metrics=metrics,
                status=st,
                zone=zone,
                reward_delta_pct=reward_delta_pct,
                reward_variance=self._reward_variance(),
                elo_signal=elo_signal,
                viz_state=viz_state,
                green_h_streak=green_h_streak,
                unmute_level=unmute_level,
                elo_flat_evals=elo_flat_evals,
                pending_age_cycles=0,
                score_horizon=self._score_every(),
            )
            feat_list = [float(x) for x in feats.tolist()]
        except Exception:
            feats = None

        action, pref, selector = self._select_action(ctx, features=feats if feat_list else None)
        self.state.last_action = action
        self.state.last_context = ctx
        self.state.last_selector = selector

        trial = PendingTrial(
            action=action,
            context=ctx,
            cycle_started=self.state.cycles,
            before=snap,
            preference_before=pref,
            timesteps_started=ts_now,
            elo_evals_started=self.state.elo_eval_count,
            features=feat_list,
            source=selector,
        )

        if action == "hold":
            # Still count as a trial so hold preference can be updated
            self.state.pending = trial
            self.save()
            return None

        patch = build_action_patch(action, current_overrides, cfg=self.cfg)
        if selector == "mlp":
            patch["meta_mlp"] = True
            patch["meta_selector"] = "mlp"
        self.state.pending = trial
        self.state.last_major_action_cycle = self.state.cycles
        self.save()
        return patch


def merge_meta_patch(
    base: dict[str, Any],
    meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge meta patch into post-safety overrides without touching recovery locks."""
    if not meta:
        return dict(base)
    out = dict(base)
    # Never let meta clear or fight HARD recovery flags
    if (
        out.get("entropy_death_recovery")
        or out.get("hard_recovery")
        or out.get("freeze_op_chaos")
        or str(out.get("safety_zone") or "").lower() == "red"
    ):
        return out

    locked = {
        "entropy_death_recovery",
        "hard_recovery",
        "freeze_op_chaos",
        "op_destructive_paused",
        "safety_zone",
        "pbt_paused",
        "es_enabled",
    }
    for k, v in meta.items():
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
