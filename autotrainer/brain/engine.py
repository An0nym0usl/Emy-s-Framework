"""
AutoTrainer decision engine - replaces manual curriculum tuning.

One change per cooldown window; snapshots before each patch; rollback on degradation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .full_control import (
    env_readonly,
    full_control_enabled,
    log_rewards_now,
    post_warmup_sparring,
    stamp_full_control,
)
from .gates import evaluate_phase_gates
from .io_utils import (
    append_jsonl,
    load_yaml,
    merge_overrides,
    read_json,
    write_json_atomic,
    write_overrides,
)
from .commands import apply_pbt_commands
from .llm_advisor import llm_advise
from .op_stack import OPStack
from .reward_tuner import tune_rewards
from .event_driven import is_entropy_dead
from .reward_watch import (
    format_cycle_line,
    observe_reward_swing,
    reward_delta_pct_from_state,
)
from .rollback import (
    discard_poisoned_lkg,
    is_healthy_for_lkg,
    load_last_known_good,
    safe_rollback_patch,
    save_last_known_good,
    save_snapshot,
    should_rollback,
)
from .meta_brain import (
    MetaBrainController,
    merge_meta_patch,
    meta_learn_enabled,
)
from .intelligence import (
    IntelligenceController,
    intelligence_enabled,
    merge_intelligence_patch,
)
from .periodic_eval import (
    elo_signal_for_meta,
    maybe_run_periodic_eval,
)
from .teachers import merge_teachers_patch, teachers_patch
from .best_skill import flush_pending_best_skill, maybe_save_best_skill
from .error_pressure import error_pressure_patch, merge_error_pressure
from .coach_summary import maybe_write_coach_summary
from .safety import SafetyGovernor, apply_safety_veto, clamp_patch, critical_recovery_banner
from .wandb_client import fetch_latest_metrics

# Dashboard / visualizer / legacy roadmap hooks are no-ops in this tree.


def write_dashboard(*_a, **_k):  # noqa: ANN001, ANN002
    return None


def ingest_visualizer_insight(*_a, **_k):  # noqa: ANN001, ANN002
    return None


def maybe_log_visualizer(*_a, **_k):  # noqa: ANN001, ANN002
    return None


def meta_viz_suffix(_state=None):  # noqa: ANN001
    return ""


def apply_legacy_roadmap(patch, **_k):  # noqa: ANN001
    """No-op stub (personal roadmap schedules not shipped)."""
    return patch, "roadmap_off"


# Back-compat alias for older call sites / tests
apply_wazne_roadmap = apply_legacy_roadmap


PHASE_NAMES = {0: "chase", 1: "foundation", 2: "advanced"}


def _metric_avg_reward(metrics: dict[str, Any] | None) -> float | None:
    m = metrics or {}
    for k in ("Average Step Reward", "avg_reward"):
        if k in m and m[k] is not None:
            try:
                v = float(m[k])
                if v == v and abs(v) != float("inf"):  # not NaN/Inf
                    return v
            except (TypeError, ValueError):
                pass
    return None


@dataclass
class Decision:
    action: str
    reason: str
    patch: dict[str, Any] = field(default_factory=dict)
    source: str = "rules"

    def to_log(self) -> dict:
        return {
            "time": datetime.now(timezone.utc).isoformat(),
            "action": self.action,
            "reason": self.reason,
            "source": self.source,
            "patch": self.patch,
        }


class AutoTrainerEngine:
    def __init__(
        self,
        watch_dir: Path,
        profiles_dir: Path,
        profile: dict,
        cfg: dict,
    ) -> None:
        self.watch_dir = watch_dir
        self.profiles_dir = profiles_dir
        self.profile = profile
        self.cfg = cfg
        self.state_path = watch_dir / "orchestrator_state.json"
        self.state = read_json(self.state_path)
        self.op = OPStack(cfg, watch_dir=watch_dir)
        self.ssl_cfg = cfg.get("ssl_guide") or {}
        self.meta = (
            MetaBrainController(watch_dir, cfg) if meta_learn_enabled(cfg) else None
        )
        self.intel = (
            IntelligenceController(watch_dir, cfg)
            if intelligence_enabled(cfg)
            else None
        )

    def _emerge(self) -> dict:
        return self.profile.get("skill_emergence") or {}

    def _save_state(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def _wandb_metrics(self, status: dict) -> dict:
        wb = self.cfg.get("wandb") or {}
        if not wb.get("enabled", True):
            return {}
        run_id = status.get("run_id")
        if not run_id:
            return {}
        return fetch_latest_metrics(
            wb.get("project", "gigalearncpp"),
            run_id,
            wb.get("entity"),
        )

    def _cooldown_ok(self, ts: int) -> bool:
        cooldown = int(self._emerge().get("cooldown_after_change_steps") or 100_000_000)
        last = int(self.state.get("last_change_timesteps") or 0)
        return ts - last >= cooldown

    def _eval_ok(self, ts: int) -> bool:
        interval = int(self._emerge().get("eval_interval_steps") or 50_000_000)
        last = int(self.state.get("last_eval_timesteps") or 0)
        return ts - last >= interval

    def _current_overrides(self, status: dict) -> dict:
        from .ssl_guide import from_scratch_context

        active = status.get("active_overrides") or {}
        if active:
            out = dict(active)
        else:
            out = read_json(self.watch_dir / "runtime_overrides.json")
        # From-scratch: ignore poisoned on-disk / status reward_weights so the
        # first apply can force-replace with Stage-1 starter.
        if from_scratch_context() and isinstance(out, dict):
            ts = int(status.get("total_timesteps") or 0)
            if ts < 100_000_000 and "reward_weights" in out:
                out = dict(out)
                out.pop("reward_weights", None)
        return out

    def _check_pending_rollback(self, status: dict) -> Decision | None:
        pending = self.state.get("pending_rollback")
        if not pending:
            return None
        after_ts = int(pending.get("check_after_timesteps") or 0)
        ts = int(status.get("total_timesteps") or 0)
        if ts < after_ts:
            return None

        before_m = pending.get("before_metrics") or {}
        after_m = status.get("last_metrics") or {}
        do_rb, reason = should_rollback(self.cfg, before_m, after_m)
        self.state.pop("pending_rollback", None)

        if do_rb:
            snap_path = pending.get("snapshot")
            if snap_path and Path(snap_path).exists():
                snap = json.loads(Path(snap_path).read_text(encoding="utf-8"))
                current = self._current_overrides(status)
                gov = SafetyGovernor.from_dict(self.state.get("safety_governor"))
                try:
                    h = float(
                        after_m.get("Policy Entropy", after_m.get("policy_entropy", 0.5))
                    )
                except (TypeError, ValueError):
                    h = 0.0
                in_rec = bool(
                    gov.in_recovery
                    or gov.zone in ("red", "yellow")
                    or is_entropy_dead(h)
                    or current.get("hard_recovery")
                    or current.get("entropy_death_recovery")
                    or current.get("freeze_op_chaos")
                )
                # Loud crash line when rollback is reward-driven (observe_reward_swing
                # also announces; this covers the eval-window path immediately).
                before_r = _metric_avg_reward(before_m)
                after_r = _metric_avg_reward(after_m)
                if (
                    before_r is not None
                    and after_r is not None
                    and before_r > 0
                    and after_r < before_r
                    and "Reward" in reason
                ):
                    pct = (before_r - after_r) / abs(before_r) * 100.0
                    print(
                        f"[AutoTrainer] REWARD CRASH: {before_r:.2f} -> {after_r:.2f} "
                        f"(-{pct:.0f}%)"
                    )
                    mon = dict(self.state.get("reward_monitor") or {})
                    mon["in_crash"] = True
                    mon["in_drop"] = True
                    mon["crash_from"] = before_r
                    mon["crash_floor"] = after_r
                    mon["drop_from"] = before_r
                    mon["drop_to"] = after_r
                    mon["last_avg"] = after_r
                    self.state["reward_monitor"] = mon
                # Discard poisoned LKG on disk so future restores cannot re-poison
                discard_poisoned_lkg(self.watch_dir)
                patch, note = safe_rollback_patch(
                    snap,
                    after_m,
                    reason=reason,
                    in_hard_recovery=in_rec,
                    current_overrides=current,
                )
                if in_rec and patch.get("reward_weights"):
                    print(
                        "[AutoTrainer] ROLLBACK: restored rewards only "
                        f"(entropy_scale={float(patch.get('entropy_scale') or 0):.3f} held)"
                    )
                self.state["last_good_snapshot"] = snap_path
                return Decision("rollback", note or reason, patch, source="rollback")

        self.state["last_good_snapshot"] = pending.get("snapshot")
        return None

    def decide(self, status: dict) -> Decision | None:
        if not status:
            return None

        if env_readonly():
            return Decision("noop", "GIGA_AT_READONLY=1 - observer mode", {}, source="readonly")

        rb = self._check_pending_rollback(status)
        if rb:
            return rb

        ts = int(status.get("total_timesteps") or 0)
        phase = int(status.get("curriculum_phase") or 0)
        phase_name = PHASE_NAMES.get(phase, "chase")

        # OP stack (PBT + ES + event-driven) runs on its own event clock - not the slow eval tick.
        if self.op.enabled:
            op_patch, op_reason, self.state = self.op.decide(status, self.state)
            self._save_state()
            if op_patch:
                op_patch = stamp_full_control(op_patch, self.cfg)
                # Curriculum gates still own phase transitions; OP owns hyperparams / exploration.
                return Decision("op_stack", op_reason, op_patch, source="op")

        if not self._eval_ok(ts):
            return Decision("noop", f"Waiting eval interval ({ts} steps)", {}, source="timer")

        wandb_m = self._wandb_metrics(status)
        gate = evaluate_phase_gates(self.profile, phase_name, status, wandb_m)
        manifest = read_json(self.watch_dir / "reward_manifest.json")
        current_rw = status.get("reward_multipliers") or {}

        emerge = self._emerge()
        chase_end = int(emerge.get("chase_min_steps") or 150_000_000)
        foundation_end = int(emerge.get("foundation_min_steps") or 800_000_000)
        advanced_min = int(emerge.get("advanced_min_steps") or 2_000_000_000)

        decision: Decision | None = None

        if phase == 0:
            if ts >= chase_end and gate.passed:
                decision = Decision(
                    "advance_curriculum",
                    f"Chase gates passed (score={gate.score:.2f})",
                    {
                        "foundation_end_steps": foundation_end,
                        "opponent_pool_chance": 0.05,
                        "entropy_scale": 0.016,
                    },
                )
            elif ts >= chase_end and not gate.passed:
                rw = tune_rewards(
                    self.profile, phase, gate.failures, manifest, current_rw, "boost",
                    timesteps=ts, ssl_cfg=self.ssl_cfg,
                )
                decision = Decision(
                    "extend_chase",
                    "Chase gates not met: " + "; ".join(gate.failures),
                    {
                        "chase_end_steps": ts + int(emerge.get("eval_interval_steps") or 50_000_000),
                        "entropy_scale": 0.018,
                        "reward_weights": rw,
                    },
                )
            elif not gate.passed and self._cooldown_ok(ts):
                rw = tune_rewards(
                    self.profile, phase, gate.failures, manifest, current_rw, "nudge",
                    timesteps=ts, ssl_cfg=self.ssl_cfg,
                )
                decision = Decision(
                    "chase_nudge",
                    "Early chase tuning: " + "; ".join(gate.failures[:2]),
                    {"reward_weights": rw, "entropy_scale": 0.017},
                )

        elif phase == 1:
            if ts >= foundation_end and gate.passed:
                decision = Decision(
                    "enter_advanced",
                    f"Foundation gates passed (score={gate.score:.2f})",
                    {
                        "foundation_end_steps": ts,
                        "opponent_pool_chance": 0.12,
                        "train_against_old_chance": 0.15,
                        "skill_tracker_enabled": True,
                        "var_max": 0.55,
                        "epochs": 2,
                        "ssl_guide_post_apex": True,
                    },
                )
            elif ts >= foundation_end // 2 and not gate.passed:
                rw = tune_rewards(
                    self.profile, phase, gate.failures, manifest, current_rw, "boost",
                    timesteps=ts, ssl_cfg=self.ssl_cfg,
                )
                decision = Decision(
                    "boost_foundation",
                    "Foundation weak: " + "; ".join(gate.failures),
                    {"var_max": 0.7, "entropy_scale": 0.022, "reward_weights": rw},
                )

        elif phase == 2:
            if ts >= advanced_min and gate.passed:
                decision = Decision(
                    "advanced_maintain",
                    f"Advanced gates OK (score={gate.score:.2f}) - sparring up",
                    {
                        "opponent_pool_chance": 0.10,
                        "train_against_old_chance": 0.333,
                        "epochs": 2,
                        "ssl_guide_post_apex": True,
                    },
                )
            elif not gate.passed:
                rw = tune_rewards(
                    self.profile, phase, gate.failures, manifest, current_rw, "boost",
                    timesteps=ts, ssl_cfg=self.ssl_cfg,
                )
                decision = Decision(
                    "advanced_tune",
                    "Advanced gates below target: " + "; ".join(gate.failures),
                    {
                        "opponent_pool_chance": 0.10,
                        "train_against_old_chance": 0.30,
                        "epochs": 2,
                        "reward_weights": rw,
                        "ssl_guide_post_apex": True,
                    },
                )

        # Optional LLM override (same cooldown)
        llm_cfg = self.cfg.get("llm") or {}
        eval_count = int(self.state.get("eval_count") or 0) + 1
        if (
            llm_cfg.get("enabled")
            and decision
            and decision.action not in ("noop",)
            and eval_count % int(llm_cfg.get("review_every_evals") or 4) == 0
        ):
            llm = llm_advise(self.cfg, self.profile, status, gate.failures, manifest)
            if llm and isinstance(llm.get("patch"), dict) and llm["patch"]:
                decision = Decision(
                    llm.get("action", "llm_tune"),
                    llm.get("reason", "LLM recommendation"),
                    llm["patch"],
                    source="llm",
                )

        if decision and decision.action == "noop":
            return decision

        # Full control: shorter curriculum cooldown so brain can steer alone.
        cooldown_ok = self._cooldown_ok(ts)
        if full_control_enabled(self.cfg):
            last = int(self.state.get("last_change_timesteps") or 0)
            cooldown_ok = ts - last >= min(
                int(self._emerge().get("cooldown_after_change_steps") or 100_000_000),
                25_000_000,
            )

        if decision and not cooldown_ok and decision.action not in ("rollback", "op_stack"):
            return Decision(
                "noop",
                f"Cooldown active until +{int(self._emerge().get('cooldown_after_change_steps', 0))} steps",
                {},
            )

        if decision and decision.patch:
            decision.patch = stamp_full_control(decision.patch, self.cfg)

        return decision

    def apply(self, decision: Decision, status: dict) -> None:
        if decision.action == "noop" or not decision.patch:
            append_jsonl(self.watch_dir / "decisions.jsonl", decision.to_log())
            if decision.source != "op":
                self.state["last_eval_timesteps"] = status.get("total_timesteps", 0)
                self.state["eval_count"] = int(self.state.get("eval_count") or 0) + 1
            self._save_state()
            return

        if env_readonly():
            append_jsonl(
                self.watch_dir / "decisions.jsonl",
                {**decision.to_log(), "skipped": "GIGA_AT_READONLY"},
            )
            return

        # Competitive PBT weight transfer commands (handled by C++ AutoTrainerBridge).
        # Safety veto: never apply PBT weight swaps while entropy_death / hard recovery.
        raw_patch = dict(decision.patch)
        patch_had_rewards = isinstance(raw_patch.get("reward_weights"), dict)
        recovering = bool(
            raw_patch.get("entropy_death_recovery")
            or raw_patch.get("hard_recovery")
            or raw_patch.get("op_destructive_paused")
            or raw_patch.get("freeze_op_chaos")
        )
        pbt_cmds = raw_patch.pop("_pbt_commands", None)
        if pbt_cmds and not recovering:
            apply_pbt_commands(self.watch_dir, pbt_cmds)
        elif pbt_cmds and recovering:
            print(
                critical_recovery_banner(
                    "PBT commands vetoed (entropy_death / hard recovery active)"
                )
            )

        safety = self.cfg.get("safety") or {}
        metrics = status.get("last_metrics") or {}
        discard_poisoned_lkg(self.watch_dir)
        lkg = load_last_known_good(self.watch_dir)
        lkg_ov = (lkg or {}).get("overrides") if lkg else None
        gov = SafetyGovernor.from_dict(self.state.get("safety_governor"))
        from .event_driven import entropy_metric_unreliable, read_policy_entropy

        ts_apply = int(status.get("total_timesteps") or 0)
        h_opt = read_policy_entropy(metrics)
        if entropy_metric_unreliable(h_opt, timesteps=ts_apply):
            h = 0.5
            force_red = False
        else:
            h = float(h_opt)  # type: ignore[arg-type]
            force_red = is_entropy_dead(h)
        # Only force red on actual death - never because the patch still carries
        # recovery flags (that deadlock pinned zone=red after H recovered).
        gov.observe(h, force_red=force_red)

        # Same-cycle LKG rollback: NEVER blind-merge LKG entropy over recovery.
        # safe path keeps rewards from LKG but HARD RECOVERY owns entropy/OP.
        if raw_patch.get("entropy_recovery_rollback"):
            from .rollback import safe_rollback_patch as _safe_rb

            snap_like = {"overrides": dict(lkg_ov or {})}
            merged_rb, _ = _safe_rb(
                snap_like,
                metrics,
                reason="entropy_recovery_rollback",
                in_hard_recovery=True,
                current_overrides=raw_patch,
            )
            # Prefer recovery knobs already on raw_patch, then safe merge
            raw_patch = {**merged_rb, **{k: v for k, v in raw_patch.items() if k != "note"}}
            raw_patch["entropy_recovery_rollback"] = True
            if "note" not in raw_patch or not raw_patch.get("note"):
                raw_patch["note"] = merged_rb.get("note")

        patch, veto_reasons = apply_safety_veto(
            stamp_full_control(raw_patch, self.cfg),
            metrics,
            safety,
            lkg_overrides=lkg_ov,
            gov=gov,
            timesteps=int(status.get("total_timesteps") or 0),
        )

        # Arm sparring/skill-eval after warmup - NEVER while recovering (safety > warmup).
        recovering_now = bool(
            patch.get("entropy_death_recovery")
            or patch.get("hard_recovery")
            or gov.in_recovery
        )
        if full_control_enabled(self.cfg) and not recovering_now:
            ts_now = int(status.get("total_timesteps") or 0)
            phase_now = int(status.get("phase") or status.get("curriculum_phase") or 0)
            for k, v in post_warmup_sparring(ts_now, self.cfg, phase=phase_now).items():
                if k not in patch:
                    patch[k] = v
        # Final clamp after optional sparring fill
        patch = clamp_patch(patch, safety)

        # Teachers (light expert distill) + Phase-1 error-pressure - after safety, before intel/meta
        ts_now = int(status.get("total_timesteps") or 0)
        t_patch = teachers_patch(
            timesteps=ts_now,
            metrics=metrics,
            zone=gov.zone,
            recovering=recovering_now,
            cfg=self.cfg,
            current=self._current_overrides(status),
            watch_dir=self.watch_dir,
            status=status,
            elo_signal=elo_signal_for_meta(self.state),
        )
        patch = merge_teachers_patch(patch, t_patch, recovering=recovering_now)
        e_patch = error_pressure_patch(
            metrics=metrics,
            state=self.state,
            zone=gov.zone,
            recovering=recovering_now,
            timesteps=ts_now,
            cfg=self.cfg,
        )
        patch = merge_error_pressure(patch, e_patch, recovering=recovering_now)
        patch = clamp_patch(patch, safety)

        # Intelligence (Diagnose->Plan->Act) THEN meta explore, THEN optional roadmap stub.
        # Priority: HARD_RECOVERY/safety > roadmap stub > diagnose/plan > SSL > meta
        current = self._current_overrides(status)
        patch = self._intelligence_then_meta(patch, status, metrics, gov, current)
        recovering_now2 = bool(
            recovering_now
            or patch.get("entropy_death_recovery")
            or patch.get("hard_recovery")
            or gov.in_recovery
            or str(gov.zone).lower() == "red"
        )
        patch, roadmap_reason = apply_legacy_roadmap(
            patch,
            timesteps=ts_now,
            root_cfg=self.cfg,
            state=self.state,
            recovering=recovering_now2,
        )
        if roadmap_reason and not roadmap_reason.startswith("roadmap_off"):
            # Keep reason visible on decision note when a real schedule is active
            note = str(patch.get("note") or "")
            if roadmap_reason not in note:
                patch["note"] = (note + "; " if note else "") + roadmap_reason
        # Re-assert HARD_RECOVERY after roadmap stub (safety wins)
        if recovering_now2 and (
            is_entropy_dead(h) or (gov.in_recovery and h < 0.18)
        ):
            from .rollback import stage_safe_recovery_template

            tmpl = stage_safe_recovery_template(h)
            try:
                cur_es = float(patch.get("entropy_scale") or 0.0)
            except (TypeError, ValueError):
                cur_es = 0.0
            need = float(tmpl["entropy_scale"])
            if cur_es < need:
                patch["entropy_scale"] = need
            patch["entropy_death_recovery"] = True
            patch["hard_recovery"] = True
            if "epochs" in tmpl:
                patch["epochs"] = tmpl["epochs"]
        patch = clamp_patch(patch, safety)
        merged = merge_overrides(current, patch)
        # Python-only flag - never persist / send to C++ RuntimeRewardRegistry
        merged.pop("reward_weights_replace", None)

        # After ANY rollback during entropy death: re-apply HARD RECOVERY in same cycle
        if decision.action == "rollback" and (
            is_entropy_dead(h) or (gov.in_recovery and h < 0.18)
        ):
            from .rollback import merge_lkg_keep_recovery

            rb_fix = merge_lkg_keep_recovery(
                dict(merged),
                metrics,
                reason=decision.reason,
                current_overrides=merged,
            )
            merged.update(rb_fix)
            merged = clamp_patch(merged, safety)
        elif is_entropy_dead(h):
            # Belt-and-suspenders: never leave a dead cycle with LKG-low entropy_scale
            from .rollback import stage_safe_recovery_template

            tmpl = stage_safe_recovery_template(h)
            try:
                cur_es = float(merged.get("entropy_scale") or 0.0)
            except (TypeError, ValueError):
                cur_es = 0.0
            need = float(tmpl["entropy_scale"])
            if cur_es < need or cur_es < 0.08:
                merged["entropy_scale"] = max(cur_es, need)
                for k in (
                    "entropy_death_recovery",
                    "hard_recovery",
                    "freeze_op_chaos",
                    "es_noise_scale",
                    "priority_sampling",
                    "skill_tracker_enabled",
                    "epochs",
                    "pbt_paused",
                    "op_destructive_paused",
                    "es_enabled",
                ):
                    if k in tmpl:
                        merged[k] = tmpl[k]
                merged = clamp_patch(merged, safety)

        snap = save_snapshot(
            self.watch_dir,
            merged,
            status,
            tag=decision.action,
        )

        # Prefer recovery note over bare decision.reason (avoids confusing LKG dumps)
        apply_note = str(merged.get("note") or decision.reason or "")
        write_overrides(self.watch_dir, merged, apply_note)
        # Expose final overrides on the decision so print_decision is not a lie
        decision.patch = dict(merged)
        decision.reason = apply_note or decision.reason

        # One unified compact line per cycle
        reward = _metric_avg_reward(metrics)
        try:
            sps = float(
                metrics.get("Overall Steps/Second")
                or metrics.get("SPS")
                or float("nan")
            )
        except (TypeError, ValueError):
            sps = float("nan")
        muted = ""
        if merged.get("op_destructive_paused") or merged.get("freeze_op_chaos"):
            muted = "muted=OP"
        elif merged.get("intelligence_unmute") or merged.get("unmute_level") is not None:
            try:
                lvl = float(merged.get("unmute_level", 0) or 0)
            except (TypeError, ValueError):
                lvl = 0.0
            if 0.0 < lvl < 1.0:
                muted = f"unmute_OP={lvl:.2f}"
            elif lvl >= 1.0:
                muted = "OP=live"
        print(
            format_cycle_line(
                zone=gov.zone,
                entropy=h,
                entropy_scale=merged.get("entropy_scale"),
                reward=reward if reward is not None else float("nan"),
                reward_delta_pct=reward_delta_pct_from_state(self.state),
                action=decision.action,
                sps=sps,
                extra=muted,
            ),
            flush=True,
        )
        # Full reward dump on any apply that writes/changes rewards (never truncated).
        rw = merged.get("reward_weights") or {}
        prev_rw = (current.get("reward_weights") or {}) if isinstance(current, dict) else {}
        note_l = apply_note.lower()
        force_rw = bool(
            decision.action == "rollback" and rw
            or "adjusted reward" in note_l
            or "restored reward" in note_l
        )
        if rw and log_rewards_now(
            rw,
            prev=prev_rw,
            patch_had_rewards=patch_had_rewards,
            note=apply_note,
            force=force_rw,
        ):
            self.state["last_logged_reward_weights"] = {
                str(k): float(v) for k, v in rw.items()
            }
        append_jsonl(self.watch_dir / "decisions.jsonl", decision.to_log())

        # Public leaderboard for dashboard / humans
        board = self.state.get("pbt_leaderboard")
        if board:
            write_json_atomic(self.watch_dir / "pbt_leaderboard.json", board)

        # Refresh LKG only on healthy green cycles (never snapshot death / recovery)
        if is_healthy_for_lkg(
            metrics,
            self.cfg,
            overrides=merged,
            in_recovery=bool(gov.in_recovery or merged.get("hard_recovery")),
            safety_zone=gov.zone,
        ):
            saved = save_last_known_good(self.watch_dir, merged, status)
            self.state["last_good_snapshot"] = str(saved)

        self.state["safety_governor"] = gov.to_dict()

        eval_interval = int(self._emerge().get("eval_interval_steps") or 50_000_000)
        # OP stack updates often; don't lock curriculum cooldown / rollback every time.
        if decision.source != "op":
            self.state["last_change_timesteps"] = status.get("total_timesteps", 0)
            self.state["last_eval_timesteps"] = status.get("total_timesteps", 0)
            self.state["eval_count"] = int(self.state.get("eval_count") or 0) + 1
            self.state["pending_rollback"] = {
                "snapshot": str(snap),
                "before_metrics": status.get("last_metrics") or {},
                "check_after_timesteps": int(status.get("total_timesteps") or 0) + eval_interval,
                "action": decision.action,
            }
        else:
            self.state["last_op_apply_timesteps"] = status.get("total_timesteps", 0)
        self._save_state()

    def _intelligence_then_meta(
        self,
        patch: dict[str, Any],
        status: dict,
        metrics: dict[str, Any],
        gov: SafetyGovernor,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        """Diagnose->Plan->Act owns priorities; meta explores only when intel allows."""
        suppress_meta = False
        mon = self.state.get("reward_monitor") or {}
        crashed = bool(mon.get("in_crash"))
        delta = reward_delta_pct_from_state(self.state)
        elo_sig = elo_signal_for_meta(self.state)
        viz_st = self.state.get("visualizer_insight") or {}
        safety_owns = bool(
            gov.in_recovery
            or gov.destructive_paused
            or patch.get("entropy_death_recovery")
            or patch.get("hard_recovery")
            or str(gov.zone).lower() == "red"
        )

        if self.intel is not None and intelligence_enabled(self.cfg):
            result = self.intel.step(
                metrics=metrics,
                status=status,
                zone=gov.zone,
                current_overrides=current,
                gov_recovery=bool(gov.in_recovery or gov.destructive_paused),
                reward_delta_pct=delta,
                reward_crashed=crashed,
                elo_signal=elo_sig,
                viz_state=viz_st if isinstance(viz_st, dict) else {},
                safety_owns=safety_owns,
            )
            for line in result.log_lines:
                print(line)
            if result.patch:
                patch = merge_intelligence_patch(patch, result.patch)
                if result.acted and result.plan.action != "hold":
                    note = str(patch.get("note") or "")
                    tag = f"intelligence:{result.plan.diagnosis}"
                    if tag not in note:
                        patch["note"] = (note + "; " if note else "") + tag
            suppress_meta = bool(result.suppress_meta)
            # Mirror compact intel summary into orchestrator state for dashboard
            self.state["intelligence"] = {
                "diagnosis": result.plan.diagnosis,
                "plan": result.plan.plan_text,
                "action": result.plan.action,
                "acted": result.acted,
                "suppress_meta": suppress_meta,
                "unmute_level": float(
                    (result.patch or {}).get("unmute_level")
                    or getattr(self.intel.state, "unmute_level", 0.0)
                ),
                "green_h_streak": int(getattr(self.intel.state, "green_h_streak", 0)),
                "elo_flat_evals": int(getattr(result.observation, "elo_flat_evals", 0)),
            }
            if self.intel.ab is not None:
                self.state["ab_plans"] = self.intel.ab.dashboard_blob()

        return self._meta_after_safety(
            patch,
            status,
            metrics,
            gov,
            current,
            suppress=suppress_meta,
        )

    def _meta_after_safety(
        self,
        patch: dict[str, Any],
        status: dict,
        metrics: dict[str, Any],
        gov: SafetyGovernor,
        current: dict[str, Any],
        *,
        suppress: bool = False,
    ) -> dict[str, Any]:
        """Run contextual bandit / meta MLP after safety (+ intel); merge if allowed."""
        if self.meta is None or not meta_learn_enabled(self.cfg):
            return patch
        mon = self.state.get("reward_monitor") or {}
        crashed = bool(mon.get("in_crash"))
        delta = reward_delta_pct_from_state(self.state)
        intel_st = self.state.get("intelligence") or {}
        viz_st = self.state.get("visualizer_insight") or {}
        meta_patch = self.meta.step(
            metrics=metrics,
            status=status,
            zone=gov.zone,
            current_overrides=current,
            safety_patch=patch,
            gov_recovery=bool(gov.in_recovery or gov.destructive_paused),
            reward_delta_pct=delta,
            reward_crashed=crashed,
            elo_signal=elo_signal_for_meta(self.state),
            viz_suffix=meta_viz_suffix(self.state),
            viz_state=viz_st if isinstance(viz_st, dict) else {},
            suppress=suppress,
            green_h_streak=int(intel_st.get("green_h_streak") or 0),
            unmute_level=float(intel_st.get("unmute_level") or 0.0),
            elo_flat_evals=int(intel_st.get("elo_flat_evals") or 0),
        )
        if self.meta.mlp is not None:
            self.state["meta_mlp"] = self.meta.mlp.dashboard_blob()
        if not meta_patch:
            return patch
        merged = merge_meta_patch(patch, meta_patch)
        # Tag decision source lightly for cycle logs when meta actually nudged
        if meta_patch.get("meta_action") and meta_patch.get("meta_action") != "hold":
            note = str(merged.get("note") or "")
            sel = meta_patch.get("meta_selector") or "bandit"
            tag = f"meta_learn:{meta_patch['meta_action']}"
            if sel == "mlp":
                tag = f"meta_mlp:{meta_patch['meta_action']}"
            if "meta_learn:" not in note and "meta_mlp:" not in note:
                merged["note"] = (note + "; " if note else "") + tag
        return merged

    def _meta_observe_noop(self, status: dict) -> dict[str, Any]:
        """Run Diagnose->Plan->Act + meta on noop; return soft patch (e.g. unmute).

        Previously returned early when meta_learn was off, which skipped intelligence
        entirely after the first OP tick - console went silent and unmute never
        reached runtime_overrides.json.
        """
        metrics = status.get("last_metrics") or {}
        gov = SafetyGovernor.from_dict(self.state.get("safety_governor"))
        try:
            h = float(metrics.get("Policy Entropy", metrics.get("policy_entropy", 0.5)))
        except (TypeError, ValueError):
            h = 0.0
        gov.observe(h, force_red=is_entropy_dead(h))
        self.state["safety_governor"] = gov.to_dict()
        current = self._current_overrides(status)
        # Synthetic empty post-safety patch - intel/meta may still score / hold / unmute
        patch = self._intelligence_then_meta({}, status, metrics, gov, current)
        return patch if isinstance(patch, dict) else {}

    def _apply_soft_observe_patch(self, status: dict, patch: dict[str, Any]) -> bool:
        """Persist soft intel/meta keys on noop without arming curriculum rollback."""
        if env_readonly() or not patch:
            return False
        # Ignore bookkeeping-only keys
        skip = {
            "note",
            "updated_at",
            "meta_selector",
            "meta_action",
            "intelligence",
            "intelligence_diagnosis",
            "dpp_report",
            "shared_replay_warmstart",
            "ssl_geometry",
        }
        current = self._current_overrides(status)
        meaningful = False
        for k, v in patch.items():
            if k in skip:
                continue
            if current.get(k) != v:
                meaningful = True
                break
        if not meaningful:
            return False
        safety = self.cfg.get("safety") or {}
        clean = stamp_full_control(clamp_patch(dict(patch), safety), self.cfg)
        merged = merge_overrides(current, clean)
        merged.pop("reward_weights_replace", None)
        note = str(merged.get("note") or "intel_soft_observe")
        write_overrides(self.watch_dir, merged, note)
        return True

    def _print_cycle_heartbeat(
        self,
        status: dict,
        decision: Decision | None,
        *,
        applied_hard: bool,
    ) -> None:
        """Always emit one line per poll so HOLD/noop never looks frozen."""
        if applied_hard:
            return  # apply() already printed the rich cycle line
        metrics = status.get("last_metrics") or {}
        gov = SafetyGovernor.from_dict(self.state.get("safety_governor"))
        try:
            h = float(metrics.get("Policy Entropy", metrics.get("policy_entropy", 0.5)))
        except (TypeError, ValueError):
            h = float("nan")
        reward = _metric_avg_reward(metrics)
        try:
            sps = float(
                metrics.get("Overall Steps/Second")
                or metrics.get("SPS")
                or float("nan")
            )
        except (TypeError, ValueError):
            sps = float("nan")
        ov = self._current_overrides(status)
        intel = self.state.get("intelligence") or {}
        action = "idle"
        extra_bits: list[str] = []
        if decision is not None:
            action = decision.action or "idle"
            if decision.action == "noop" and decision.reason:
                extra_bits.append(str(decision.reason)[:72])
        diag = intel.get("diagnosis")
        if diag:
            extra_bits.append(f"diag={diag}")
        plan = intel.get("plan")
        if plan and str(plan).startswith("HOLD"):
            extra_bits.append("HOLD")
        try:
            um = float(intel.get("unmute_level") or 0.0)
            if 0.0 < um < 1.0:
                extra_bits.append(f"unmute_OP={um:.2f}")
            elif um >= 1.0:
                extra_bits.append("OP=live")
        except (TypeError, ValueError):
            pass
        cycle = int(self.state.get("tick_count") or 0)
        extra_bits.append(f"poll=#{cycle}")
        print(
            format_cycle_line(
                zone=gov.zone,
                entropy=h,
                entropy_scale=ov.get("entropy_scale"),
                reward=reward if reward is not None else float("nan"),
                reward_delta_pct=reward_delta_pct_from_state(self.state),
                action=action,
                sps=sps,
                extra=" ".join(extra_bits),
            ),
            flush=True,
        )

    def tick(self) -> Decision | None:
        status = read_json(self.watch_dir / "trainer_status.json")
        if not status:
            return None
        self.state["tick_count"] = int(self.state.get("tick_count") or 0) + 1
        metrics = status.get("last_metrics") or {}
        log_cfg = self.cfg.get("logging") or {}
        crash_thr = float(
            log_cfg.get("reward_crash_drop")
            or log_cfg.get("reward_crash_threshold")
            or 0.30
        )
        observe_reward_swing(self.state, metrics, crash_drop=crash_thr)

        # Ingest --render visualizer telemetry BEFORE decide/meta so bandit
        # context and soft GREEN hints see live watch quality (never Learn).
        try:
            gov_pre = SafetyGovernor.from_dict(self.state.get("safety_governor"))
            try:
                h_pre = float(
                    metrics.get("Policy Entropy", metrics.get("policy_entropy", 0.5))
                )
            except (TypeError, ValueError):
                h_pre = 0.0
            recovering_pre = bool(gov_pre.in_recovery or is_entropy_dead(h_pre))
            ingest_visualizer_insight(
                self.watch_dir,
                self.state,
                metrics=metrics,
                zone=gov_pre.zone,
                recovering=recovering_pre,
                cfg=self.cfg,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[AutoTrainer] visualizer insight failed: {exc}", flush=True)

        # Periodic Elo eval (interval only - never every training iter)
        eval_summary = maybe_run_periodic_eval(
            self.watch_dir, status, self.state, cfg=self.cfg
        )
        if eval_summary and eval_summary.get("ok"):
            maybe_save_best_skill(
                self.watch_dir,
                status,
                self.state,
                cfg=self.cfg,
                eval_summary=eval_summary,
            )
            # Long credit: advance Elo-eval counters for meta MLP / A/B
            if self.meta is not None:
                try:
                    self.meta.note_elo_eval()
                except Exception:
                    pass
            if self.intel is not None:
                try:
                    self.intel.note_elo_eval()
                except Exception:
                    pass
        flush_pending_best_skill(self.watch_dir, status, self.state, cfg=self.cfg)

        decision = self.decide(status)
        applied_hard = False
        if decision:
            self.apply(decision, status)
            applied_hard = bool(decision.action != "noop" and decision.patch)
            # noop / empty patch skips meta inside apply - still observe + soft-apply
            if decision.action == "noop" or not decision.patch:
                soft = self._meta_observe_noop(status)
                self._apply_soft_observe_patch(status, soft)
        else:
            soft = self._meta_observe_noop(status)
            self._apply_soft_observe_patch(status, soft)

        # Legacy compact visualizer line (insight logs the rich line when fresh)
        try:
            maybe_log_visualizer(self.watch_dir, self.state, cfg=self.cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[AutoTrainer] visualizer watch failed: {exc}", flush=True)

        # Dashboard JSON (offline HTML) - every tick
        try:
            gov = SafetyGovernor.from_dict(self.state.get("safety_governor"))
            ov = self._current_overrides(status)
            write_dashboard(
                self.watch_dir,
                status,
                self.state,
                cfg=self.cfg,
                zone=gov.zone,
                entropy_scale=ov.get("entropy_scale"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[AutoTrainer] dashboard write failed: {exc}", flush=True)

        # Offline coach summary (heuristic; optional LLM) - never blocks
        try:
            gov_c = SafetyGovernor.from_dict(self.state.get("safety_governor"))
            maybe_write_coach_summary(
                self.watch_dir,
                self.state,
                status=status,
                metrics=metrics,
                zone=gov_c.zone,
                cfg=self.cfg,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[AutoTrainer] coach summary failed: {exc}", flush=True)

        # Side systems on noop: teachers / error-pressure still need a write path
        if decision is None or decision.action == "noop" or not (decision.patch if decision else None):
            self._apply_side_systems_noop(status)

        # Heartbeat: HOLD / waiting-eval must never look like a one-shot hang
        self._print_cycle_heartbeat(status, decision, applied_hard=applied_hard)

        self._save_state()
        return decision

    def _apply_side_systems_noop(self, status: dict) -> None:
        """Apply teachers + error-pressure when decide() returned empty/noop."""
        if env_readonly():
            return
        metrics = status.get("last_metrics") or {}
        gov = SafetyGovernor.from_dict(self.state.get("safety_governor"))
        try:
            h = float(metrics.get("Policy Entropy", metrics.get("policy_entropy", 0.5)))
        except (TypeError, ValueError):
            h = 0.0
        gov.observe(h, force_red=is_entropy_dead(h))
        recovering = bool(gov.in_recovery or is_entropy_dead(h))
        current = self._current_overrides(status)
        ts_now = int(status.get("total_timesteps") or 0)
        patch: dict[str, Any] = {}
        t_patch = teachers_patch(
            timesteps=ts_now,
            metrics=metrics,
            zone=gov.zone,
            recovering=recovering,
            cfg=self.cfg,
            current=current,
            watch_dir=self.watch_dir,
            status=status,
            elo_signal=elo_signal_for_meta(self.state),
        )
        patch = merge_teachers_patch(patch, t_patch, recovering=recovering)
        e_patch = error_pressure_patch(
            metrics=metrics,
            state=self.state,
            zone=gov.zone,
            recovering=recovering,
            timesteps=ts_now,
            cfg=self.cfg,
        )
        patch = merge_error_pressure(patch, e_patch, recovering=recovering)
        if not patch:
            self.state["safety_governor"] = gov.to_dict()
            return
        # Skip write if nothing meaningful changed vs current
        meaningful = False
        for k, v in patch.items():
            if k == "note":
                continue
            if current.get(k) != v:
                meaningful = True
                break
        if not meaningful:
            self.state["safety_governor"] = gov.to_dict()
            return
        safety = self.cfg.get("safety") or {}
        patch = stamp_full_control(clamp_patch(patch, safety), self.cfg)
        merged = merge_overrides(current, patch)
        merged.pop("reward_weights_replace", None)
        merged = clamp_patch(merged, safety)
        merged.pop("reward_weights_replace", None)
        side_note = str(merged.get("note") or "side_systems")
        write_overrides(self.watch_dir, merged, side_note)
        if isinstance(merged.get("reward_weights"), dict) and log_rewards_now(
            merged["reward_weights"],
            prev=current.get("reward_weights") or {},
            patch_had_rewards="reward_weights" in patch,
            note=side_note,
        ):
            self.state["last_logged_reward_weights"] = {
                str(k): float(v) for k, v in merged["reward_weights"].items()
            }
        self.state["safety_governor"] = gov.to_dict()

    @classmethod
    def from_paths(
        cls,
        watch_dir: Path,
        profile_name: str,
        config_path: Path | None = None,
    ) -> AutoTrainerEngine:
        root = Path(__file__).resolve().parent.parent
        profiles_dir = root / "profiles"
        cfg_path = config_path or (root / "config.default.yaml")
        cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        profile_path = profiles_dir / f"{profile_name}.yaml"
        profile = load_yaml(profile_path)
        return cls(watch_dir, profiles_dir, profile, cfg)


def print_banner(profile: dict, watch_dir: Path, cfg: dict | None = None) -> None:
    fc = full_control_enabled(cfg or {})
    ro = env_readonly()
    meta_on = meta_learn_enabled(cfg or {})
    print("\n" + "-" * 64)
    print("  GigaLearnRL AutoTrainer")
    print("  Rewards / PPO / league / GPU resets / skill-eval")
    print("-" * 64)
    print(f"  Profile : {profile.get('name', '?')}")
    print(f"  Watch   : {watch_dir}")
    print(f"  full_control: {fc}  readonly: {ro}  meta_learn: {meta_on}")
    print("  Opt-out: GIGA_NO_SSL_AUTONOMY=1 | GIGA_NO_META_LEARN=1")
    print("-" * 64 + "\n")


def print_decision(decision: Decision, status: dict, *, verbose: bool = False) -> None:
    ts = status.get("total_timesteps", "?")
    phase = status.get("curriculum_phase", "?")
    # apply() already printed the unified cycle line - keep this quiet by default
    quiet = bool(
        not verbose
        or decision.action == "rollback"
        or (decision.patch or {}).get("hard_recovery")
        or (decision.patch or {}).get("entropy_death_recovery")
        or (decision.patch or {}).get("recovery_exiting")
    )
    if quiet:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] steps={ts} phase={phase} "
            f"[{decision.source}] {decision.action}"
        )
        return
    bar = "-" * 60  # ASCII - Windows cp1252 consoles choke on box-drawing chars
    print(f"\n{bar}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] steps={ts} phase={phase} [{decision.source}]")
    print(f"  {decision.action}: {decision.reason}")
    if decision.patch:
        print(f"  patch: {json.dumps(decision.patch, indent=2)}")
    print(f"{bar}\n")
