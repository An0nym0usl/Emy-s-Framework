"""Hard safety gate for AutoTrainer.

Full control is allowed, but safety veto wins always:
  - entropy_death / NaN / KL explode -> CRITICAL RECOVERY + mute OP toys
  - SafetyGovernor hysteresis (red/yellow/green) freezes PBT/ES/meta/rotate
  - clamp_patch raises entropy ceiling only while recovering

Aggressive safety is ON by default.
Opt-out: safety.aggressive: false  or  GIGA_AT_SAFETY_OFF=1
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

from .event_driven import (
    ENTROPY_CLEAR_EXIT,
    ENTROPY_DEATH_CRITICAL,
    ENTROPY_DEATH_HARD,
    ENTROPY_DEATH_SOFT,
    ENTROPY_FULLY_RECOVERED,
    ENTROPY_HEALTHY,
    entropy_death_recovery_patch,
    entropy_tier,
    is_entropy_dead,
    recovery_ok_cycles_needed,
    taper_entropy_scale,
)


def aggressive_safety_enabled(safety: dict | None) -> bool:
    if os.environ.get("GIGA_AT_SAFETY_OFF", "").strip() in ("1", "true", "TRUE", "yes"):
        return False
    return bool((safety or {}).get("aggressive", True))


def critical_recovery_banner(detail: str = "") -> str:
    msg = "[AutoTrainer] CRITICAL RECOVERY entropy_death"
    if detail:
        return f"{msg} -> {detail}"
    return msg


@dataclass
class SafetyGovernor:
    """Hysteresis governor: red freezes destructive OP; green restores toys."""

    zone: str = "green"  # red | yellow | green
    red_cycles: int = 0
    ok_streak: int = 0
    last_entropy: float = 0.5
    last_log: str = ""
    destructive_paused: bool = False
    in_recovery: bool = False

    # Thresholds (overridable from config.safety)
    red_threshold: float = ENTROPY_DEATH_SOFT  # <0.10 -> enter red
    yellow_threshold: float = 0.12
    recover_ok: float = ENTROPY_HEALTHY  # ≥0.18 soft exit / yellow
    green_threshold: float = ENTROPY_CLEAR_EXIT  # ≥0.25 clear -> green
    recover_cycles: int = 3  # consecutive ok cycles to leave red (fewer if H>0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "red_cycles": self.red_cycles,
            "ok_streak": self.ok_streak,
            "last_entropy": self.last_entropy,
            "last_log": self.last_log,
            "destructive_paused": self.destructive_paused,
            "in_recovery": self.in_recovery,
            "red_threshold": self.red_threshold,
            "yellow_threshold": self.yellow_threshold,
            "recover_ok": self.recover_ok,
            "green_threshold": self.green_threshold,
            "recover_cycles": self.recover_cycles,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SafetyGovernor:
        if not d:
            return cls()
        g = cls()
        for k in (
            "zone",
            "last_log",
        ):
            if k in d and d[k] is not None:
                setattr(g, k, str(d[k]))
        for k in (
            "red_cycles",
            "ok_streak",
            "recover_cycles",
        ):
            if k in d and d[k] is not None:
                setattr(g, k, int(d[k]))
        for k in (
            "last_entropy",
            "red_threshold",
            "yellow_threshold",
            "recover_ok",
            "green_threshold",
        ):
            if k in d and d[k] is not None:
                setattr(g, k, float(d[k]))
        g.destructive_paused = bool(d.get("destructive_paused"))
        g.in_recovery = bool(d.get("in_recovery"))
        return g

    def observe(
        self,
        policy_entropy: float,
        *,
        force_red: bool = False,
        event_entropy_death: bool = False,
    ) -> None:
        """Update zone hysteresis.

        force_red / event_entropy_death must ONLY mean "entropy is actually dead".
        Callers must not pass them merely because sticky recovery is still armed —
        that pinned zone=red forever and zeroed ok_streak every cycle.
        Healthy H (≥ recover_ok) always takes the exit path, even if force_red was set.
        """
        h = float(policy_entropy)
        if math.isnan(h) or math.isinf(h):
            h = 0.0
            force_red = True
        self.last_entropy = h

        # H >= 0.5 while in red/yellow → force exit green immediately
        if (self.in_recovery or self.destructive_paused or self.zone in ("red", "yellow")) and (
            h >= ENTROPY_FULLY_RECOVERED
        ):
            self.zone = "green"
            self.destructive_paused = False
            self.in_recovery = False
            self.red_cycles = 0
            self.ok_streak = 0
            self.last_log = (
                f"[AutoTrainer] RECOVERY COMPLETE H={h:.4f} -> zone=green "
                f"(force-exit H>={ENTROPY_FULLY_RECOVERED})"
            )
            return

        # Enter hard-red only when entropy is actually bad.
        # Ignore stale force_red if H has already recovered past soft exit.
        enter_red = h < self.red_threshold or (
            (force_red or event_entropy_death) and h < self.recover_ok
        )
        if enter_red:
            self.zone = "red"
            self.red_cycles += 1
            self.ok_streak = 0
            self.destructive_paused = True
            self.in_recovery = True
            tier = entropy_tier(h)
            self.last_log = (
                f"zone=red tier={tier} H={h:.4e} red_cycles={self.red_cycles} "
                f"OP_toys=MUTED"
            )
            return

        # Healthy enough to exit — accumulate ok_streak (never zero it here)
        if self.in_recovery or self.destructive_paused:
            needed = recovery_ok_cycles_needed(h, default=self.recover_cycles)
            if h >= self.recover_ok:
                self.ok_streak += 1
            else:
                self.ok_streak = 0
            if self.ok_streak >= needed and h >= self.green_threshold:
                self.zone = "green"
                self.destructive_paused = False
                self.in_recovery = False
                self.red_cycles = 0
                self.last_log = (
                    f"[AutoTrainer] RECOVERY COMPLETE H={h:.4f} -> zone=green "
                    f"ok_streak>={needed}"
                )
                self.ok_streak = 0
            elif self.ok_streak >= needed and h >= self.recover_ok:
                # Soft clear at ≥0.18: leave red mute, stay yellow briefly
                self.zone = "yellow"
                self.destructive_paused = False
                self.in_recovery = False
                self.red_cycles = 0
                self.last_log = (
                    f"zone=yellow H={h:.4f} soft_exit ok_streak>={needed}"
                )
                self.ok_streak = 0
            elif h >= self.recover_ok:
                self.zone = "yellow"
                self.destructive_paused = True
                self.in_recovery = True
                self.last_log = (
                    f"zone=yellow H={h:.4f} ok_streak={self.ok_streak}/"
                    f"{needed} (exit taper)"
                )
            else:
                self.zone = "red"
                self.destructive_paused = True
                self.in_recovery = True
                self.red_cycles += 1
                self.last_log = (
                    f"zone=red H={h:.4e} still_dead red_cycles={self.red_cycles}"
                )
            return

        if h < self.yellow_threshold:
            self.zone = "yellow"
            self.in_recovery = True
            self.destructive_paused = True
            self.last_log = f"zone=yellow H={h:.4f} preemptive_mute"
        else:
            self.zone = "green"
            self.in_recovery = False
            self.destructive_paused = False
            self.last_log = f"zone=green H={h:.4f}"


def _metric_f(metrics: dict[str, Any] | None, *keys: str, default: float = 0.0) -> float:
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


def metrics_are_toxic(
    metrics: dict[str, Any] | None,
    safety: dict | None = None,
    *,
    timesteps: int = 0,
) -> tuple[bool, str]:
    from .event_driven import entropy_metric_unreliable, read_policy_entropy

    cfg = safety or {}
    entropy_opt = read_policy_entropy(metrics)
    # Missing / boot-zero Policy Entropy is NOT toxic (first cycles after fresh start)
    if entropy_metric_unreliable(entropy_opt, timesteps=timesteps):
        entropy = float("nan")  # skip death check below
    else:
        entropy = float(entropy_opt)  # type: ignore[arg-type]
        if math.isnan(entropy):
            return True, "Policy Entropy is NaN"
    reward = _metric_f(metrics, "Average Step Reward", "avg_reward", default=0.0)
    if math.isnan(reward):
        return True, "Average Step Reward is NaN"
    kl = _metric_f(metrics, "KL Div Loss", "kl", "Mean KL Divergence", default=0.0)
    if math.isnan(kl):
        return True, "KL is NaN"
    kl_boom = float(cfg.get("kl_explosion_threshold", 0.12))
    if kl > kl_boom:
        return True, f"KL exploded ({kl:.4f} > {kl_boom})"
    if not math.isnan(entropy) and is_entropy_dead(entropy):
        return True, f"entropy_death H={entropy:.4e} tier={entropy_tier(entropy)}"
    return False, ""


def govern_patch(
    patch: dict[str, Any],
    gov: SafetyGovernor,
    *,
    safety: dict | None = None,
) -> dict[str, Any]:
    """Fail-closed: whitelist recovery knobs in red/yellow; raise entropy ceiling."""
    cfg = dict(safety or {})
    out = dict(patch)

    if not aggressive_safety_enabled(cfg):
        return clamp_patch(out, cfg)

    exiting = bool(out.get("recovery_exiting")) or (
        gov.zone == "yellow" and gov.last_entropy >= ENTROPY_HEALTHY
    )
    healthy_h = gov.last_entropy >= ENTROPY_HEALTHY
    fully_ok = gov.last_entropy >= ENTROPY_FULLY_RECOVERED
    late_yellow = gov.zone == "yellow" and (
        fully_ok or (gov.last_entropy >= ENTROPY_CLEAR_EXIT and gov.ok_streak >= 2)
    )
    green = gov.zone == "green" and not gov.in_recovery

    if green and not out.get("entropy_death_recovery"):
        # Fully clear — do not re-apply hard recovery mutes; keep exit taper value
        out["safety_zone"] = "green"
        out.pop("hard_recovery", None)
        out.pop("freeze_op_chaos", None)
        # Do NOT force another full taper to baseline here — exit patch already stepped down
        return clamp_patch(out, cfg)

    if gov.in_recovery or gov.destructive_paused or out.get("entropy_death_recovery"):
        out["safety_zone"] = gov.zone
        out["freeze_op_chaos"] = not (exiting or late_yellow)
        out["hard_recovery"] = not (exiting or healthy_h)
        out["entropy_death_recovery"] = not (exiting and fully_ok)
        out["es_noise_scale"] = 0.0
        out["priority_sampling"] = False
        # Skill-eval: red = off; late yellow / green path = on
        if gov.zone == "red" and not healthy_h:
            out["skill_tracker_enabled"] = False
        elif late_yellow or green or (exiting and fully_ok):
            out["skill_tracker_enabled"] = True
        else:
            out["skill_tracker_enabled"] = False
        out["es_enabled"] = bool(late_yellow or green)
        out["pbt_paused"] = not (late_yellow or green)
        out["op_destructive_paused"] = gov.zone == "red" and not healthy_h
        out["truncation_pbt_paused"] = not (late_yellow or green)
        out["meta_gradients_paused"] = gov.zone == "red" and not healthy_h
        out["env_architect_paused"] = gov.zone == "red" and not healthy_h
        out["compete_rotate_paused"] = gov.zone == "red" and not healthy_h
        out["epochs"] = 1 if gov.zone == "red" and not healthy_h else int(out.get("epochs", 1) or 1)
        # Cap opponent pressure while still muted
        if gov.zone == "red" and not healthy_h:
            if "opponent_pool_chance" in out:
                out["opponent_pool_chance"] = min(float(out["opponent_pool_chance"]), 0.01)
            if "train_against_old_chance" in out:
                out["train_against_old_chance"] = min(
                    float(out["train_against_old_chance"]), 0.15
                )
            for k in list(out.keys()):
                if str(k).startswith("opponent_weight_"):
                    out[k] = 0.0

        # Entropy floor: only spike when still dead; taper when exiting
        try:
            cur = float(out.get("entropy_scale") or 0.0)
        except (TypeError, ValueError):
            cur = 0.0
        baseline = float(cfg.get("baseline_entropy_scale", 0.022))
        if healthy_h or exiting:
            # Decline toward baseline — never re-pin at 0.35
            tapered = taper_entropy_scale(
                max(cur, baseline),
                baseline,
                ok_streak=max(1, gov.ok_streak),
                taper_cycles=4,
            )
            out["entropy_scale"] = tapered
        elif is_entropy_dead(gov.last_entropy):
            # Critical/hard/soft death: pin at recovery target so LKG 0.017 can never win
            # tier=ok never enters here (no HARD RECOVERY from tiny coef churn)
            if gov.last_entropy < ENTROPY_DEATH_CRITICAL:
                floor = 0.35
            elif gov.last_entropy < ENTROPY_DEATH_HARD:
                floor = 0.35
            else:
                floor = 0.10
            if cur < floor:
                rec, _ = entropy_death_recovery_patch(
                    policy_entropy=gov.last_entropy,
                    baseline_entropy_scale=baseline,
                    prev_coef=baseline,
                    cycle=gov.red_cycles,
                )
                if rec:
                    out["entropy_scale"] = max(floor, float(rec["entropy_scale"]))
                    out["policy_lr"] = rec.get("policy_lr", out.get("policy_lr"))
                    out["critic_lr"] = rec.get("critic_lr", out.get("critic_lr"))
                    for gk in (
                        "gpu_reset_kickoff",
                        "gpu_reset_fuzzed",
                        "gpu_reset_aerial",
                        "kickoff_weight",
                        "fuzzed_weight",
                        "aerial_weight",
                        "var_max",
                        "var_min",
                        "clip_range",
                        "mask_entropy",
                        "ssl_guide_post_apex",
                    ):
                        if gk in rec and gk not in out:
                            out[gk] = rec[gk]
        out.pop("_pbt_commands", None)
        out.pop("model_arch_overrides", None)
        out.pop("shared_replay_warmstart", None)
        out.pop("safety_governor", None)
        note = str(out.get("note") or "")
        if "ROLLBACK+HARD_RECOVERY" not in note:
            if healthy_h or exiting:
                out["note"] = (
                    f"[AutoTrainer] RECOVERY EXIT zone={gov.zone} "
                    f"coef->{out.get('entropy_scale')} H={gov.last_entropy:.4f}"
                )
            else:
                out["note"] = critical_recovery_banner(
                    f"zone={gov.zone} coef->{out.get('entropy_scale')} H={gov.last_entropy:.4e}"
                )

    return clamp_patch(out, cfg)


def apply_safety_veto(
    patch: dict[str, Any],
    metrics: dict[str, Any] | None,
    safety: dict | None = None,
    *,
    lkg_overrides: dict[str, Any] | None = None,
    baseline_entropy_scale: float = 0.022,
    gov: SafetyGovernor | None = None,
    timesteps: int = 0,
) -> tuple[dict[str, Any], list[str]]:
    """Hard gate used by engine.apply — safety veto wins always."""
    from .event_driven import entropy_metric_unreliable, read_policy_entropy
    from .rollback import _sanitize_recovery_detail

    cfg = dict(safety or {})
    out = dict(patch)
    reasons: list[str] = []

    if not aggressive_safety_enabled(cfg):
        return clamp_patch(out, cfg), reasons

    entropy_opt = read_policy_entropy(metrics)
    unreliable = entropy_metric_unreliable(entropy_opt, timesteps=timesteps)
    if unreliable:
        entropy = float("nan")
        dead = False
    else:
        entropy = float(entropy_opt)  # type: ignore[arg-type]
        dead = is_entropy_dead(entropy)
    toxic, toxic_why = metrics_are_toxic(metrics, cfg, timesteps=timesteps)

    # Strip false HARD RECOVERY flags armed from boot-zero / missing H
    if unreliable:
        out.pop("entropy_death_recovery", None)
        out.pop("hard_recovery", None)
        note0 = str(out.get("note") or "")
        if "HARD RECOVERY" in note0 or "CRITICAL RECOVERY" in note0:
            out["note"] = "boot:entropy_unreliable(no HARD_RECOVERY)"

    if out.get("entropy_recovery_rollback") or (
        lkg_overrides and (toxic or dead) and out.get("note", "").startswith("ROLLBACK")
    ):
        # Never restore LKG entropy_scale as-is — merge rewards, HARD RECOVERY wins
        from .rollback import merge_lkg_keep_recovery, stage_safe_recovery_template

        h = 0.0 if (math.isnan(entropy) or unreliable) else entropy
        if unreliable:
            # Boot: do not invent ROLLBACK+HARD_RECOVERY from missing entropy
            reasons.append("safety_veto:skip_rollback(boot_entropy_unreliable)")
        elif lkg_overrides:
            out = merge_lkg_keep_recovery(
                dict(lkg_overrides),
                metrics,
                reason=_sanitize_recovery_detail(toxic_why or "rollback"),
                baseline_entropy_scale=baseline_entropy_scale,
                current_overrides=out,
            )
            out["entropy_recovery_rollback"] = True
            reasons.append(
                f"safety_veto:ROLLBACK+HARD_RECOVERY ent={out.get('entropy_scale')} "
                f"({_sanitize_recovery_detail(toxic_why or 'rollback')})"
            )
        else:
            out = {**out, **stage_safe_recovery_template(h, baseline_entropy_scale=baseline_entropy_scale)}
            out["entropy_recovery_rollback"] = True
            reasons.append(
                f"safety_veto:ROLLBACK+HARD_RECOVERY ent={out.get('entropy_scale')} "
                f"({_sanitize_recovery_detail(toxic_why or 'rollback')})"
            )
    elif toxic or dead or (
        out.get("entropy_death_recovery")
        and not unreliable
        and (gov is None or gov.last_entropy < ENTROPY_HEALTHY)
    ) or (
        gov
        and gov.in_recovery
        and not unreliable
        and gov.last_entropy < ENTROPY_HEALTHY
    ):
        if gov is None:
            gov = SafetyGovernor()
            gov.observe(0.0 if math.isnan(entropy) else entropy, force_red=True)
        out = govern_patch(out, gov, safety=cfg)
        # Single reason line (engine prints one cycle summary) — never nest ROLLBACK banners
        if gov.last_entropy >= ENTROPY_HEALTHY:
            reasons.append(
                f"safety_veto:RECOVERY_EXIT ent={out.get('entropy_scale')} "
                f"zone={gov.zone}"
            )
        else:
            why = _sanitize_recovery_detail(toxic_why or "entropy_death")
            reasons.append(
                f"safety_veto:HARD_RECOVERY ent={out.get('entropy_scale')} ({why})"
            )
    elif gov and (gov.in_recovery or out.get("recovery_exiting") or out.get("entropy_death_recovery")):
        # Healthy H but still exiting — taper via govern, don't re-spike
        out = govern_patch(out, gov, safety=cfg)
        reasons.append(
            f"safety_veto:RECOVERY_EXIT ent={out.get('entropy_scale')} zone={gov.zone}"
        )

    return clamp_patch(out, cfg), reasons


def clamp_patch(patch: dict[str, Any], safety: dict) -> dict[str, Any]:
    out = dict(patch)
    exiting = bool(out.get("recovery_exiting"))
    zone = str(out.get("safety_zone") or "")
    recovering = bool(
        out.get("entropy_death_recovery")
        or out.get("hard_recovery")
        or out.get("freeze_op_chaos")
        or out.get("op_destructive_paused")
    ) and not exiting and zone != "green"
    # Late yellow exit: treat as soft-recovering for clamps but allow skill-eval
    soft_exit = exiting or zone == "yellow"
    smax = float(
        safety.get(
            "max_entropy_scale_recovery" if (recovering or soft_exit) else "max_entropy_scale",
            0.35 if (recovering or soft_exit) else 0.04,
        )
    )
    if recovering or soft_exit:
        smax = max(smax, float(safety.get("max_entropy_scale_recovery", 0.35)))
    smin = float(safety.get("min_entropy_scale", 0.005))
    if "entropy_scale" in out:
        out["entropy_scale"] = max(smin, min(smax, float(out["entropy_scale"])))
    if "var_max" in out:
        out["var_max"] = max(0.1, min(float(safety.get("max_var_max", 1.2)), float(out["var_max"])))
    if "var_min" in out:
        out["var_min"] = max(0.05, min(0.5, float(out["var_min"])))
    if "clip_range" in out:
        out["clip_range"] = max(0.05, min(0.4, float(out["clip_range"])))
    if "gae_gamma" in out:
        out["gae_gamma"] = max(0.9, min(0.999, float(out["gae_gamma"])))
    if "gae_lambda" in out:
        out["gae_lambda"] = max(0.8, min(0.99, float(out["gae_lambda"])))
    if "policy_lr" in out:
        out["policy_lr"] = max(1e-5, min(1e-3, float(out["policy_lr"])))
    if "critic_lr" in out:
        out["critic_lr"] = max(1e-5, min(1e-3, float(out["critic_lr"])))
    if "epochs" in out:
        out["epochs"] = int(max(1, min(4, int(out["epochs"]))))
    if "es_noise_scale" in out:
        if recovering:
            out["es_noise_scale"] = 0.0
        else:
            out["es_noise_scale"] = max(0.0, min(0.12, float(out["es_noise_scale"])))
    if "event_advantage_boost" in out:
        out["event_advantage_boost"] = max(1.0, min(3.0, float(out["event_advantage_boost"])))
    if "priority_sampling" in out:
        out["priority_sampling"] = False if recovering else bool(out["priority_sampling"])
    if "mask_entropy" in out:
        out["mask_entropy"] = bool(out["mask_entropy"])
    if "active_agent_id" in out:
        out["active_agent_id"] = int(out["active_agent_id"])
    if "skill_tracker_enabled" in out:
        if recovering and not soft_exit:
            out["skill_tracker_enabled"] = False
        else:
            out["skill_tracker_enabled"] = bool(out["skill_tracker_enabled"])

    out.pop("_pbt_commands", None)
    out.pop("op_mode", None)
    out.pop("ssl_geometry", None)
    out.pop("dpp_report", None)
    out.pop("shared_replay_warmstart", None)
    out.pop("ssl_hardened_skipped", None)
    out.pop("ssl_stagnation_escape", None)
    out.pop("safety_governor", None)
    for meta_k in ("ssl_league_current", "ssl_league_old", "ssl_league_expert"):
        out.pop(meta_k, None)
    # Keep reward_weights_replace through merge_overrides; strip before C++ write in engine.

    if "reward_weights" in out and isinstance(out["reward_weights"], dict):
        rw = {}
        rmin = float(safety.get("min_reward_multiplier", 0.01))
        rmax = float(safety.get("max_reward_multiplier", 3.0))
        for k, v in out["reward_weights"].items():
            rw[k] = max(rmin, min(rmax, float(v)))
        out["reward_weights"] = rw

    for key in ("opponent_pool_chance", "train_against_old_chance"):
        if key in out:
            cap = 0.06 if recovering else 0.5
            out[key] = max(0.0, min(cap, float(out[key])))

    for key in (
        "kickoff_weight",
        "fuzzed_weight",
        "aerial_weight",
        "gpu_reset_kickoff",
        "gpu_reset_fuzzed",
        "gpu_reset_aerial",
        "ball_chase_weight",
        "random_state_weight",
    ):
        if key in out:
            out[key] = max(0.0, min(1.0, float(out[key])))

    for k in list(out.keys()):
        if str(k).startswith("opponent_weight_"):
            out[k] = 0.0 if recovering else max(0.0, min(3.0, float(out[k])))

    for bk in (
        "ssl_guide_post_apex",
        "autotrainer_full_control",
        "full_control",
        "ssl_autonomy",
        "save_policy_versions",
        "entropy_death_recovery",
        "hard_recovery",
        "freeze_op_chaos",
        "es_enabled",
        "pbt_paused",
        "op_destructive_paused",
        "truncation_pbt_paused",
        "meta_gradients_paused",
        "env_architect_paused",
        "compete_rotate_paused",
        "entropy_recovery_rollback",
    ):
        if bk in out:
            out[bk] = bool(out[bk])

    if "max_episode_duration" in out:
        out["max_episode_duration"] = max(0.5, min(8.0, float(out["max_episode_duration"])))
    if "ts_per_save" in out:
        out["ts_per_save"] = int(max(250_000, min(50_000_000, int(out["ts_per_save"]))))
    if "ts_per_version" in out:
        out["ts_per_version"] = int(max(500_000, min(100_000_000, int(out["ts_per_version"]))))
    if "no_touch_seconds" in out:
        out["no_touch_seconds"] = max(1.0, min(30.0, float(out["no_touch_seconds"])))
    if "skill_tracker_interval" in out:
        out["skill_tracker_interval"] = int(max(1, min(512, int(out["skill_tracker_interval"]))))
    if "max_grad_norm" in out:
        out["max_grad_norm"] = max(0.1, min(2.5, float(out["max_grad_norm"])))
    if "opponent_beat_bonus" in out:
        out["opponent_beat_bonus"] = max(0.0, min(200.0, float(out["opponent_beat_bonus"])))
    if "opponent_concede_penalty" in out:
        out["opponent_concede_penalty"] = max(-100.0, min(0.0, float(out["opponent_concede_penalty"])))
    for key in ("w_icm", "w_rnd"):
        if key in out:
            out[key] = max(0.0, min(0.5, float(out[key])))
    if "entropy_death_tier" in out:
        out["entropy_death_tier"] = str(out["entropy_death_tier"])
    if "safety_zone" in out:
        out["safety_zone"] = str(out["safety_zone"])

    out.pop("ssl_graduation", None)
    out.pop("ssl_track_id", None)
    out.pop("ssl_track_label", None)
    out.pop("ssl_sparring_warmup", None)
    out.pop("ssl_play_hours", None)
    out.pop("ssl_tick_skip", None)
    out.pop("legacy_roadmap_phase", None)
    out.pop("wazne_roadmap_phase", None)  # back-compat key

    return out
