"""
OP Stack — Competitive 4-agent PBT + ES + event-driven (quality-first).

What the strongest systems do:
  1. Event-driven: react when metrics move, not on a fixed tick clock.
  2. Competitive PBT: 4 real agents with separate weight slots — losers copy winners.
  3. ES + PPO: Evolution Strategies steer hyperparameters; PPO does gradient descent.

All free, all local, no paid APIs.
"""

from __future__ import annotations

from typing import Any

from .competitive_league import (
    CompetitiveLeague,
    get_agent,
    init_league,
    league_public_status,
    rotate_and_compete,
    score_agent,
    should_rotate,
)
from .es_optimizer import ESState, observe_fitness, propose_candidates, seed_es
from .event_driven import (
    EventState,
    detect_events,
    merge_event_suggestions,
    should_fire,
    snapshot_metrics,
    update_entropy_recovery_state,
)
from .four_level_orchestrator import FourLevelOrchestrator
from .pbt import genome_to_overrides
from .full_control import full_control_enabled, stamp_full_control
from .safety import SafetyGovernor, critical_recovery_banner, govern_patch
from .ssl_guide import build_autonomy_patch, hardened_experts_available


class OPStack:
    def __init__(self, cfg: dict[str, Any] | None = None, watch_dir: Any = None) -> None:
        # Full config (may include ssl_guide at root) vs op_mode nested block
        self.root_cfg = cfg or {}
        self.cfg = self.root_cfg.get("op_mode") or self.root_cfg
        self.enabled = bool(self.cfg.get("enabled", True))
        self.es_enabled = bool(self.cfg.get("es_enabled", True))
        self.event_enabled = bool(self.cfg.get("event_driven", True))
        self.competitive = bool(self.cfg.get("competitive_pbt", True))
        self.n_agents = int(self.cfg.get("competitive_agents", 4))
        self.blend = float(self.cfg.get("blend", 0.35))  # lean on competitive agent genome
        self.population_size = int(self.cfg.get("population_size", 8))  # legacy virtual PBT
        fl_cfg = self.cfg.get("four_level") or {}
        self.four_level = FourLevelOrchestrator(fl_cfg, watch_dir=watch_dir)
        ssl = dict(self.root_cfg.get("ssl_guide") or {})
        if self.root_cfg.get("ssl_autonomy"):
            ssl["_root_autonomy"] = self.root_cfg.get("ssl_autonomy")
        self.four_level.set_ssl_cfg(ssl)
        self.four_level.set_root_cfg(self.root_cfg)
        self.watch_dir = watch_dir
        safety = self.root_cfg.get("safety") or {}
        self._safety_cfg = dict(safety)

    def load(
        self, state: dict[str, Any]
    ) -> tuple[CompetitiveLeague, ESState, EventState, SafetyGovernor]:
        op = state.get("op_stack") or {}
        league = CompetitiveLeague.from_dict(op.get("league"))
        if self.competitive and not league.agents:
            league = init_league(
                n_agents=self.n_agents,
                seed=int(self.cfg.get("seed", 42)),
                rotate_every_steps=int(self.cfg.get("rotate_every_steps", 10_000_000)),
            )
        elif league.agents and "rotate_every_steps" in self.cfg:
            league.rotate_every_steps = int(self.cfg["rotate_every_steps"])

        es = ESState.from_dict(op.get("es"))
        if self.es_enabled and es.generation == 0 and not es.pending and not es.best_theta:
            es = seed_es(cfg=self.cfg.get("es") or {})
        ev = EventState.from_dict(op.get("events"))
        if "event_cooldown_steps" in self.cfg:
            ev.cooldown_steps = int(self.cfg["event_cooldown_steps"])
        gov = SafetyGovernor.from_dict(state.get("safety_governor") or op.get("safety_governor"))
        # Allow config to override hysteresis thresholds
        s = self._safety_cfg
        if "entropy_red" in s:
            gov.red_threshold = float(s["entropy_red"])
        if "entropy_yellow" in s:
            gov.yellow_threshold = float(s["entropy_yellow"])
        if "entropy_recover_ok" in s:
            gov.recover_ok = float(s["entropy_recover_ok"])
        if "entropy_green" in s:
            gov.green_threshold = float(s["entropy_green"])
        if "recover_ok_cycles" in s:
            gov.recover_cycles = int(s["recover_ok_cycles"])
        return league, es, ev, gov

    def save(
        self,
        state: dict[str, Any],
        league: CompetitiveLeague,
        es: ESState,
        ev: EventState,
        gov: SafetyGovernor | None = None,
    ) -> None:
        state["op_stack"] = {
            "league": league.to_dict(),
            "es": es.to_dict(),
            "events": ev.to_dict(),
            "leaderboard": league_public_status(league),
        }
        if gov is not None:
            state["op_stack"]["safety_governor"] = gov.to_dict()
            state["safety_governor"] = gov.to_dict()

    def _blend(
        self,
        agent_g: dict[str, float],
        es_g: dict[str, float],
        event_patch: dict[str, Any],
    ) -> dict[str, Any]:
        keys = set(agent_g) | set(es_g)
        blended: dict[str, Any] = {}
        b = self.blend
        for k in keys:
            if k in ("active_agent_id", "train_against_old_chance", "opponent_pool_chance",
                     "skill_tracker_enabled", "priority_sampling", "note", "op_mode"):
                continue
            try:
                a = float(agent_g.get(k, es_g.get(k, 0)))
                c = float(es_g.get(k, a))
                blended[k] = (1 - b) * a + b * c
            except (TypeError, ValueError):
                blended[k] = agent_g.get(k, es_g.get(k))
        if "epochs" in blended:
            blended["epochs"] = int(round(float(blended["epochs"])))
        out = genome_to_overrides(blended)
        # Preserve competitive extras from agent package
        for k in (
            "active_agent_id",
            "train_against_old_chance",
            "opponent_pool_chance",
            "skill_tracker_enabled",
            "priority_sampling",
            "ssl_guide_post_apex",
            "fuzzed_weight",
            "aerial_weight",
            "gpu_reset_kickoff",
            "gpu_reset_fuzzed",
            "gpu_reset_aerial",
        ):
            if k in agent_g:
                out[k] = agent_g[k]
        for k, v in agent_g.items():
            if str(k).startswith("opponent_weight_"):
                out[k] = v
        for k, v in event_patch.items():
            out[k] = v
        out["op_mode"] = True
        out["note"] = f"OP: 4-agent competitive PBT + ES + events (agent {out.get('active_agent_id', '?')})"
        return out

    def decide(
        self,
        status: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
        if not self.enabled:
            return None, "op_disabled", state

        league, es, ev, gov = self.load(state)
        ts = int(status.get("total_timesteps") or 0)
        phase = int(status.get("curriculum_phase") or 0)
        metrics = status.get("last_metrics") or {}
        from .event_driven import (
            entropy_metric_unreliable,
            read_policy_entropy,
        )

        h_real = read_policy_entropy(metrics)
        if entropy_metric_unreliable(h_real, timesteps=ts):
            policy_entropy = 0.5  # unknown → green-friendly; do not enter red
        else:
            policy_entropy = float(h_real)  # type: ignore[arg-type]

        # Always score the live agent — quality signal for the league.
        if self.competitive:
            score_agent(league, metrics, phase, ts)

        events = (
            detect_events(metrics, ev.last_metrics, phase=phase, timesteps=ts)
            if self.event_enabled
            else []
        )
        event_patch: dict[str, Any] = {}
        reason_parts: list[str] = []
        pbt_commands: dict[str, Any] = {}

        force_death = any(e.name == "entropy_death" for e in events)
        if entropy_metric_unreliable(h_real, timesteps=ts):
            force_death = False

        # Sticky entropy recovery — arm before fire gate so death always schedules an OP tick
        track_hint = str((state.get("ssl_autonomy") or {}).get("track_id") or "aerial")
        # From-scratch boot uses mechanical baseline; don't invent aerial mid-run coefs
        if ts < 100_000_000:
            track_hint = str(
                (state.get("ssl_autonomy") or {}).get("track_id") or "mechanical"
            )
        baseline_ent = {
            "mechanical": 0.024,
            "aerial": 0.022,
            "mid": 0.019,
            "late": 0.016,
            "ssl_pressure": 0.014,
        }.get(track_hint, 0.022)
        prolonged_n = int(self._safety_cfg.get("prolonged_red_cycles", 4))
        recovery_early = update_entropy_recovery_state(
            ev,
            metrics,
            timesteps=ts,
            baseline_entropy_scale=baseline_ent,
            force_arm=force_death,
            prolonged_cycles=prolonged_n,
            tick_cycle=True,
        )
        # Governor observes AFTER sticky arm/clear.
        # NEVER force_red just because recovery is still armed — that zeroed ok_streak
        # every cycle and pinned zone=red after H had fully recovered.
        gov.observe(
            policy_entropy,
            force_red=force_death,
            event_entropy_death=force_death,
        )
        red_zone = (gov.zone == "red" and gov.destructive_paused and policy_entropy < 0.18) or (
            ev.entropy_recovery_active and policy_entropy < 0.18
        )
        # Expose sticky flag to four_level / ssl_guide before they compose overrides
        if ev.entropy_recovery_active:
            state["entropy_recovery"] = {
                "active": True,
                "coef": ev.entropy_recovery_coef,
                "prev_coef": ev.entropy_recovery_prev_coef,
                "log": ev.entropy_recovery_log,
                "started_ts": ev.entropy_recovery_started_ts,
                "cycles": ev.entropy_recovery_cycles,
                "ok_streak": ev.entropy_ok_streak,
            }
        else:
            state.pop("entropy_recovery", None)
        state["safety_governor"] = gov.to_dict()

        fired = False
        if self.event_enabled and should_fire(ev, ts, events):
            fired = True
            event_patch = merge_event_suggestions(events)
            if recovery_early:
                event_patch.update(recovery_early)
            ev.last_event_timesteps = ts
            ev.event_count += 1
            ev.history.append(
                {
                    "t": ts,
                    "events": [
                        {"name": e.name, "severity": e.severity, "detail": e.detail}
                        for e in events[:5]
                    ],
                }
            )
            reason_parts.append("events:" + ",".join(e.name for e in events[:3]))
            if ev.entropy_recovery_active and ev.entropy_recovery_log:
                reason_parts.append(ev.entropy_recovery_log)

        # Competitive rotation — HARD BLOCKED in red zone (collapse amplifier)
        rotating = False
        agent_package: dict[str, Any] = {}
        if red_zone and self.competitive and should_rotate(league, ts):
            reason_parts.append("compete_rotate_BLOCKED_red_zone")
        elif self.competitive and should_rotate(league, ts):
            fired = True
            rotating = True
            league, agent_package, pbt_commands = rotate_and_compete(
                league,
                ts,
                run_tourney=True,
                seed=ts ^ (league.generation * 17),
            )
            meta = pbt_commands.pop("_meta", {})
            champ = (meta.get("tourney") or {}).get("champion")
            reason_parts.append(
                f"compete_rotate->agent{league.active_id}"
                + (f"_champ{champ}" if champ is not None else "")
            )

        force_interval = int(self.cfg.get("force_interval_steps", 15_000_000))
        last_op = int(state.get("last_op_timesteps") or -1)
        if not fired and last_op < 0 and full_control_enabled(self.root_cfg):
            fired = True
            reason_parts.append("autonomy_bootstrap")
        elif not fired and ts - max(0, last_op) >= force_interval:
            fired = True
            reason_parts.append("scheduled_op_tick")
        elif not fired and (ev.entropy_recovery_active or recovery_early or red_zone):
            fired = True
            reason_parts.append("entropy_death_sticky")
            if recovery_early:
                event_patch.update(recovery_early)

        if not fired:
            ev.last_metrics = snapshot_metrics(metrics)
            self.save(state, league, es, ev, gov)
            return None, "op_waiting", state

        if red_zone:
            reason_parts.insert(
                0,
                critical_recovery_banner(
                    f"zone={gov.zone} H={policy_entropy:.4e} "
                    f"red_cycles={gov.red_cycles} OP_toys=MUTED"
                ),
            )

        # If not rotating, keep current agent's genome as the PBT contribution.
        if not rotating and self.competitive:
            ag = get_agent(league)
            if ag:
                agent_package = genome_to_overrides(ag.genome)
                agent_package["active_agent_id"] = ag.id
                agent_package["priority_sampling"] = True

        es_genome: dict[str, float] = {}
        # ES — HARD MUTED in red zone (do not propose / step / blend ES genomes)
        if red_zone:
            reason_parts.append("es_MUTED_red_zone")
            es_genome = {}
        elif self.es_enabled:
            if not es.pending:
                propose_candidates(es, seed=ts + es.generation * 101)
                reason_parts.append(f"es_propose_g{es.generation}")
            nxt = observe_fitness(es, metrics, phase)
            if nxt:
                es_genome = {k: float(v) for k, v in nxt.items() if isinstance(v, (int, float))}
                reason_parts.append(f"es_step_g{es.generation}")
            elif es.pending:
                es_genome = {
                    k: float(v)
                    for k, v in es.pending[0]["genome"].items()
                    if isinstance(v, (int, float))
                }
                reason_parts.append("es_eval_candidate")
            else:
                es_genome = {
                    k: float(v) for k, v in es.theta.items() if isinstance(v, (int, float))
                }

        if not agent_package:
            agent_package = dict(es_genome)
        if not es_genome:
            es_genome = {
                k: float(v)
                for k, v in agent_package.items()
                if isinstance(v, (int, float))
            }

        # In red zone: do not blend agent/ES hypers that fight recovery — recovery owns surface
        if red_zone:
            patch = dict(event_patch) if event_patch else dict(recovery_early or {})
            if self.competitive:
                ag = get_agent(league)
                if ag:
                    patch["active_agent_id"] = ag.id
            patch["op_mode"] = True
            # Strip any residual PBT weight-transfer commands
            pbt_commands = {}
        else:
            patch = self._blend(agent_package, es_genome, event_patch)
            if pbt_commands:
                patch["_pbt_commands"] = {
                    k: v for k, v in pbt_commands.items() if not str(k).startswith("_")
                }

        # Four-level: in red zone only run SSL autonomy (rewards/league ease), never
        # meta_fd / truncation_pbt / env_plr / NAS mutate.
        if self.four_level.enabled and not red_zone:
            fl_patch, fl_reason, state = self.four_level.on_op_tick(
                status=status,
                state=state,
                league_agents=list(league.agents),
                active_id=int(league.active_id),
                rotating=rotating,
                pbt_commands=pbt_commands,
            )
            if fl_patch:
                for k, v in fl_patch.items():
                    if k in ("dpp_report", "shared_replay_warmstart", "ssl_geometry"):
                        patch[k] = v
                    elif k == "reward_weights" and isinstance(v, dict):
                        rw = dict(patch.get("reward_weights") or {})
                        rw.update(v)
                        patch["reward_weights"] = rw
                    elif k == "model_arch_overrides":
                        patch[k] = v
                    else:
                        patch[k] = v
                reason_parts.append(fl_reason)
                roles = (state.get("four_level") or {}).get("roles") or {}
                for ag in league.agents:
                    if str(ag.id) in roles:
                        ag.role = roles[str(ag.id)]
        else:
            if red_zone:
                reason_parts.append("four_level_MUTED_red_zone")
            ssl_cfg = self.root_cfg.get("ssl_guide") or {}
            ssl_hist = list((state.get("ssl_guide") or {}).get("reward_history") or [])
            ssl_ent = list((state.get("ssl_guide") or {}).get("entropy_history") or [])
            manifest = {}
            if self.watch_dir:
                try:
                    from pathlib import Path
                    from .io_utils import read_json

                    manifest = read_json(Path(self.watch_dir) / "reward_manifest.json")
                except OSError:
                    pass
            # First autonomy tick / from-scratch: ignore live multipliers so LKG
            # aerial leftovers cannot poison the Stage-1 starter recipe.
            from .ssl_guide import from_scratch_context

            live_mult = dict(status.get("reward_multipliers") or {})
            boot_seed = (
                last_op < 0
                or "autonomy_bootstrap" in reason_parts
                or from_scratch_context()
            )
            # From-scratch: also ignore poisoned on-disk / status reward maps
            if boot_seed or from_scratch_context():
                live_mult = {}
            ssl_patch, ssl_reason = build_autonomy_patch(
                timesteps=ts,
                phase=phase,
                metrics=metrics,
                current_mult=live_mult,
                manifest=manifest,
                ssl=ssl_cfg,
                root_cfg=self.root_cfg,
                reward_history=ssl_hist,
                entropy_history=ssl_ent,
                hardened_available=hardened_experts_available(status, ssl_cfg),
                sps_safe=bool((metrics or {}).get("Curriculum/SpsSafe")),
                state=state,
            )
            replace_rw = bool(
                ssl_patch.get("reward_weights_replace")
                or boot_seed
                or from_scratch_context()
            )
            if replace_rw and ssl_patch.get("reward_weights"):
                ssl_patch["reward_weights_replace"] = True
            if ssl_patch:
                # In red zone, only take reward_weights + recovery-safe league keys from SSL
                if red_zone:
                    for k in (
                        "reward_weights",
                        "reward_weights_replace",
                        "train_against_old_chance",
                        "opponent_pool_chance",
                        "skill_tracker_enabled",
                        "skill_tracker_interval",
                        "ssl_guide_post_apex",
                        "gpu_reset_kickoff",
                        "gpu_reset_fuzzed",
                        "gpu_reset_aerial",
                        "kickoff_weight",
                        "fuzzed_weight",
                        "aerial_weight",
                    ):
                        if k in ssl_patch:
                            if k == "reward_weights" and isinstance(ssl_patch[k], dict):
                                if replace_rw:
                                    patch["reward_weights"] = dict(ssl_patch[k])
                                    patch["reward_weights_replace"] = True
                                else:
                                    rw = dict(patch.get("reward_weights") or {})
                                    rw.update(ssl_patch[k])
                                    patch["reward_weights"] = rw
                            elif k != "reward_weights":
                                patch[k] = ssl_patch[k]
                    for k, v in ssl_patch.items():
                        if str(k).startswith("opponent_weight_"):
                            patch[k] = v
                    # Prefer recovery entropy over SSL hyper_schedule
                    if "entropy_scale" in ssl_patch and not patch.get("entropy_death_recovery"):
                        pass
                else:
                    starter_rw = ssl_patch.pop("reward_weights", None)
                    replace_flag = bool(ssl_patch.pop("reward_weights_replace", False))
                    geo = ssl_patch.pop("ssl_geometry", None)
                    # Drop any agent/ES reward_weights before applying starter replace
                    if replace_flag or replace_rw:
                        patch.pop("reward_weights", None)
                    patch.update(ssl_patch)
                    if isinstance(starter_rw, dict) and starter_rw:
                        if replace_flag or replace_rw:
                            patch["reward_weights"] = dict(starter_rw)
                            patch["reward_weights_replace"] = True
                        else:
                            rw = dict(patch.get("reward_weights") or {})
                            rw.update(starter_rw)
                            patch["reward_weights"] = rw
                    if geo is not None:
                        patch["ssl_geometry"] = geo
                reason_parts.append(ssl_reason)
            try:
                ssl_hist.append(float((metrics or {}).get("Average Step Reward") or 0))
            except (TypeError, ValueError):
                pass
            try:
                ssl_ent.append(float((metrics or {}).get("Policy Entropy") or 0))
            except (TypeError, ValueError):
                pass
            state["ssl_guide"] = {
                "reward_history": ssl_hist[-32:],
                "entropy_history": ssl_ent[-32:],
                "last_reason": ssl_reason if ssl_patch else "ssl_noop",
            }

        patch = stamp_full_control(patch, self.root_cfg)

        # CRITICAL: SSL aerial hyper_schedule must not overwrite entropy_death recovery.
        # Use scheduled baseline (not the already-overwritten patch value) for ×N spike.
        # tick_cycle=False — already counted on recovery_early this OP tick.
        # force_arm ONLY on fresh death events — never because sticky is still active.
        recovery_final = update_entropy_recovery_state(
            ev,
            metrics,
            timesteps=ts,
            baseline_entropy_scale=baseline_ent,
            force_arm=force_death,
            prolonged_cycles=prolonged_n,
            tick_cycle=False,
        )
        # Re-sync governor after final sticky update (may have cleared / tapered)
        gov.observe(
            policy_entropy,
            force_red=force_death,
            event_entropy_death=force_death,
        )
        red_zone = (gov.zone == "red" and gov.destructive_paused and policy_entropy < 0.18) or (
            ev.entropy_recovery_active and policy_entropy < 0.18
        )
        if recovery_final:
            patch.update(recovery_final)
            if recovery_final.get("note"):
                reason_parts.append(str(recovery_final["note"]))
        elif event_patch.get("entropy_death_recovery"):
            for k, v in event_patch.items():
                if k.startswith("opponent_weight_") or k in (
                    "entropy_scale",
                    "entropy_death_recovery",
                    "safety_zone",
                    "epochs",
                    "es_noise_scale",
                    "es_enabled",
                    "pbt_paused",
                    "op_destructive_paused",
                    "truncation_pbt_paused",
                    "meta_gradients_paused",
                    "env_architect_paused",
                    "compete_rotate_paused",
                    "var_max",
                    "var_min",
                    "clip_range",
                    "opponent_pool_chance",
                    "train_against_old_chance",
                    "skill_tracker_enabled",
                    "skill_tracker_interval",
                    "policy_lr",
                    "critic_lr",
                    "event_advantage_boost",
                    "priority_sampling",
                    "mask_entropy",
                    "gpu_reset_kickoff",
                    "gpu_reset_fuzzed",
                    "gpu_reset_aerial",
                    "kickoff_weight",
                    "fuzzed_weight",
                    "aerial_weight",
                    "reward_weights",
                    "w_icm",
                    "w_rnd",
                    "note",
                ):
                    patch[k] = v

        # Fail-closed governor: whitelist + raise recovery entropy ceiling past 0.04 clamp
        patch = govern_patch(patch, gov, safety=self._safety_cfg)
        # Re-stamp flags after govern (strips safety_governor nested dict for C++)
        patch = stamp_full_control(patch, self.root_cfg)
        if gov.in_recovery and gov.zone == "red" and policy_entropy < 0.18:
            patch["entropy_death_recovery"] = True
            patch["safety_zone"] = gov.zone
            patch.update(
                {
                    "es_noise_scale": 0.0,
                    "es_enabled": False,
                    "pbt_paused": True,
                    "op_destructive_paused": True,
                    "truncation_pbt_paused": True,
                    "compete_rotate_paused": True,
                    "epochs": 1,
                    "skill_tracker_enabled": False,
                }
            )
        elif gov.zone == "yellow" or patch.get("recovery_exiting"):
            patch["safety_zone"] = gov.zone
            # Late yellow / exit: allow skill-eval; keep ES muted until green
            if policy_entropy >= 0.25 and gov.ok_streak >= 2:
                patch["skill_tracker_enabled"] = True
            elif policy_entropy >= 0.5:
                patch["skill_tracker_enabled"] = True
            else:
                patch.setdefault("skill_tracker_enabled", False)
            patch["es_noise_scale"] = min(float(patch.get("es_noise_scale", 0) or 0), 0.02)
            patch["pbt_paused"] = True
            patch["op_destructive_paused"] = True

        note_bits = [
            f"OP: zone={gov.zone} agent {patch.get('active_agent_id', '?')}"
        ]
        if patch.get("recovery_exiting") and patch.get("note"):
            note_bits.append(str(patch["note"]))
        elif patch.get("entropy_death_recovery") and patch.get("note"):
            # Suppress HARD RECOVERY spam once H has rebound
            note = str(patch["note"])
            if policy_entropy >= 0.5 and "HARD RECOVERY" in note:
                note = note.replace("HARD RECOVERY entropy_death", "RECOVERY EXIT")
            note_bits.append(note)
        elif gov.in_recovery and policy_entropy < 0.18:
            note_bits.append(critical_recovery_banner(gov.last_log))
        elif gov.in_recovery:
            note_bits.append(f"[AutoTrainer] RECOVERY EXIT {gov.last_log}")
        patch["note"] = " | ".join(note_bits)

        ev.last_metrics = snapshot_metrics(metrics)
        state["last_op_timesteps"] = ts
        state["pbt_leaderboard"] = league_public_status(league)
        if ev.entropy_recovery_active:
            state["entropy_recovery"] = {
                "active": True,
                "coef": ev.entropy_recovery_coef,
                "prev_coef": ev.entropy_recovery_prev_coef,
                "log": ev.entropy_recovery_log,
                "started_ts": ev.entropy_recovery_started_ts,
                "cycles": ev.entropy_recovery_cycles,
                "ok_streak": ev.entropy_ok_streak,
            }
        else:
            state.pop("entropy_recovery", None)
        state["safety_governor"] = gov.to_dict()
        self.save(state, league, es, ev, gov)

        return patch, " | ".join(reason_parts) or "op_update", state
