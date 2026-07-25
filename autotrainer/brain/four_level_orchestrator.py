"""
Four-level async hierarchy orchestrator — wires Levels 1–4 into OP stack ticks.

LEVEL 1 Operational Executor  — meta-gradients → LR/γ/ε overrides
LEVEL 2 Dynamic Evolutionist  — truncation PBT 20% + NAS mutate
LEVEL 3 Strategic General     — AlphaStar roles + Elo + freeze snapshots
LEVEL 4 Environment Architect — POET/PLR + ICM/RND weights

Also: DPP diversity penalty, shared prioritized replay warm-start on clone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dpp_diversity import diversity_adjusted_fitness, population_diversity_report
from .dynamically_mutatable_transformer import NASGenome, mutate_nas
from .env_architect import EnvArchitectState, step_env_architect
from .meta_gradients import MetaGradState, meta_update
from .pbt import GENOME_KEYS, _clamp_genome, _perturb, genome_to_overrides
from .shared_replay import SharedPrioritizedReplay, default_replay
from .ssl_guide import build_autonomy_patch, hardened_experts_available


ROLE_MAIN = "main"
ROLE_MAIN_EXPLOITER = "main_exploiter"
ROLE_LEAGUE_EXPLOITER = "league_exploiter"
ROLES = (ROLE_MAIN, ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER)


@dataclass
class FourLevelState:
    meta: MetaGradState = field(default_factory=MetaGradState)
    env: EnvArchitectState = field(default_factory=EnvArchitectState)
    nas_by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)  # agent_id -> role
    freeze_snapshots: list[dict[str, Any]] = field(default_factory=list)
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "env": self.env.to_dict(),
            "nas_by_agent": dict(self.nas_by_agent),
            "roles": dict(self.roles),
            "freeze_snapshots": list(self.freeze_snapshots[-64:]),
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FourLevelState:
        if not d:
            return cls()
        return cls(
            meta=MetaGradState.from_dict(d.get("meta")),
            env=EnvArchitectState.from_dict(d.get("env")),
            nas_by_agent=dict(d.get("nas_by_agent") or {}),
            roles={str(k): str(v) for k, v in (d.get("roles") or {}).items()},
            freeze_snapshots=list(d.get("freeze_snapshots") or []),
            generation=int(d.get("generation") or 0),
        )


def assign_roles(agent_ids: list[int], rng: random.Random) -> dict[str, str]:
    """Asymmetric AlphaStar-style roles, reassigned each PBT iteration."""
    ids = list(agent_ids)
    rng.shuffle(ids)
    out: dict[str, str] = {}
    if not ids:
        return out
    # At least one Main; cycle remaining as exploiters
    out[str(ids[0])] = ROLE_MAIN
    for i, aid in enumerate(ids[1:]):
        out[str(aid)] = ROLES[1 + (i % 2)]
    return out


def truncation_selection(
    agents: list[Any],
    *,
    bottom_frac: float = 0.2,
    top_frac: float = 0.2,
    mutate_strength: float = 0.75,
    seed: int | None = None,
    dpp_lam: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Bottom 20% wipe/clone Actor+Critic+meta from top 20% + explore mutate.
    Returns (transfers, diversity_report_list).
    """
    if len(agents) < 2:
        return [], []
    rng = random.Random(seed)
    genomes = [dict(a.genome) for a in agents]
    keys = list(GENOME_KEYS)
    # Adjust fitness with DPP before ranking
    for a, g in zip(agents, genomes):
        a.fitness = diversity_adjusted_fitness(float(a.fitness), genomes, keys, lam=dpp_lam)

    ranked = sorted(agents, key=lambda a: (a.fitness, getattr(a, "elo", 0)), reverse=True)
    import math

    n = len(ranked)
    n_top = max(1, int(math.ceil(n * top_frac)))
    n_bot = max(1, int(math.ceil(n * bottom_frac)))
    top = ranked[:n_top]
    bottom = ranked[-n_bot:]
    transfers: list[dict[str, Any]] = []
    for loser in bottom:
        donor = rng.choice(top)
        loser.genome = _clamp_genome({**donor.genome})
        for k in GENOME_KEYS:
            loser.genome[k] = _perturb(float(loser.genome[k]), k, mutate_strength, rng)
        loser.genome = _clamp_genome(loser.genome)
        if hasattr(loser, "generation"):
            loser.generation += 1
        loser.fitness *= 0.5
        transfers.append({"loser": loser.id, "donor": donor.id, "mode": "truncation_20"})
    return transfers, [population_diversity_report(genomes, keys)]


def role_sparring_overrides(role: str, *, ssl_base: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    AlphaStar-style role bias on top of SSL 60/30/10 league mix when provided.
    """
    base = dict(ssl_base or {})
    old = float(base.get("train_against_old_chance", 0.22))
    opp = float(base.get("opponent_pool_chance", 0.10))

    if role == ROLE_MAIN:
        return {
            **base,
            "train_against_old_chance": old,
            "opponent_pool_chance": opp,
            "skill_tracker_enabled": True,
            "league_role": ROLE_MAIN,
        }
    if role == ROLE_MAIN_EXPLOITER:
        return {
            **base,
            "train_against_old_chance": max(0.05, old * 0.4),
            "opponent_pool_chance": min(0.35, opp * 2.2),  # punish live mains / experts
            "skill_tracker_enabled": True,
            "event_advantage_boost": 2.0,
            "league_role": ROLE_MAIN_EXPLOITER,
        }
    # league exploiter — historical mixture / frozen
    return {
        **base,
        "train_against_old_chance": min(0.45, old * 1.35),
        "opponent_pool_chance": max(0.04, opp * 0.7),
        "skill_tracker_enabled": True,
        "league_role": ROLE_LEAGUE_EXPLOITER,
    }


class FourLevelOrchestrator:
    def __init__(self, cfg: dict[str, Any] | None = None, watch_dir: Path | None = None) -> None:
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.watch_dir = Path(watch_dir) if watch_dir else None
        self.dpp_lam = float(self.cfg.get("dpp_lambda", 0.15))
        self.nas_prob = float(self.cfg.get("nas_mutation_prob", 0.35))
        self.meta_enabled = bool(self.cfg.get("meta_gradients", True))
        self.env_enabled = bool(self.cfg.get("env_architect", True))
        self.shared_replay_enabled = bool(self.cfg.get("shared_replay", True))
        self.shared_replay_capacity = int(self.cfg.get("shared_replay_capacity", 8192))
        # ssl_guide lives on root config; four_level.cfg may be nested — parent passes via set_ssl
        self.ssl_cfg: dict[str, Any] = {}
        self.root_cfg: dict[str, Any] = {}

    def set_ssl_cfg(self, ssl: dict[str, Any] | None) -> None:
        self.ssl_cfg = dict(ssl or {})

    def set_root_cfg(self, cfg: dict[str, Any] | None) -> None:
        self.root_cfg = dict(cfg or {})

    def load(self, state: dict[str, Any]) -> FourLevelState:
        return FourLevelState.from_dict(state.get("four_level"))

    def save(self, state: dict[str, Any], fl: FourLevelState) -> None:
        state["four_level"] = fl.to_dict()

    def on_op_tick(
        self,
        *,
        status: dict[str, Any],
        state: dict[str, Any],
        league_agents: list[Any],
        active_id: int,
        rotating: bool,
        pbt_commands: dict[str, Any],
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """
        Returns (patch_extras, reason, updated_state).
        Call from OPStack.decide after competitive scoring / rotation.
        """
        if not self.enabled:
            return {}, "four_level_off", state

        fl = self.load(state)
        metrics = status.get("last_metrics") or {}
        phase = int(status.get("curriculum_phase") or 0)
        ts = int(status.get("total_timesteps") or 0)
        reasons: list[str] = []
        patch: dict[str, Any] = {}

        # --- LEVEL 1: meta-gradients ---
        if self.meta_enabled:
            ov = status.get("active_overrides") or {}
            fl.meta, meta_patch, meta_reason = meta_update(fl.meta, metrics, current_overrides=ov)
            patch.update(meta_patch)
            # Dynamic grad clip as evolvable hyperparam
            if "max_grad_norm" not in patch:
                base_gn = float(ov.get("max_grad_norm", 0.5))
                # nudge from KL
                try:
                    kl = float(metrics.get("Mean KL Divergence") or metrics.get("KL Div Loss") or 0)
                except (TypeError, ValueError):
                    kl = 0.0
                patch["max_grad_norm"] = max(0.2, min(2.0, base_gn * (1.15 if kl > 0.05 else 0.95)))
            reasons.append(meta_reason)

        # Shared replay push (always-on when four_level.shared_replay=true)
        replay: SharedPrioritizedReplay | None = None
        if self.shared_replay_enabled and self.watch_dir:
            replay = default_replay(self.watch_dir, capacity=self.shared_replay_capacity)
            replay.push_from_metrics(metrics, phase=phase, agent_id=active_id, steps=ts)

        # --- LEVEL 2+3 on rotation ---
        if rotating and league_agents:
            fl.generation += 1
            ids = [a.id for a in league_agents]
            rng = random.Random(ts ^ (fl.generation * 9133))
            fl.roles = assign_roles(ids, rng)
            transfers, div_rep = truncation_selection(
                league_agents,
                bottom_frac=0.2,
                top_frac=0.2,
                seed=ts + fl.generation,
                dpp_lam=self.dpp_lam,
            )
            if div_rep:
                patch["dpp_report"] = div_rep[0]
            # Freeze historical snapshot (Pareto / anti-cycle)
            champ = max(league_agents, key=lambda a: (a.fitness, getattr(a, "elo", 0)))
            fl.freeze_snapshots.append(
                {
                    "t": ts,
                    "agent_id": champ.id,
                    "elo": getattr(champ, "elo", 0),
                    "fitness": champ.fitness,
                    "genome": dict(champ.genome),
                    "generation": fl.generation,
                }
            )
            # NAS mutate losers
            for t in transfers:
                lid = str(t["loser"])
                nas = NASGenome.from_dict(fl.nas_by_agent.get(lid))
                if rng.random() < self.nas_prob:
                    nas = mutate_nas(nas, rng, strength=0.7)
                    fl.nas_by_agent[lid] = nas.to_dict()
                    patch["model_arch_overrides"] = nas.to_cpp_overrides()
                    patch["nas_agent_id"] = t["loser"]
                if replay:
                    hint = replay.clone_warmstart_hint(t["donor"], t["loser"])
                    patch["shared_replay_warmstart"] = hint
                    # Keep priority sampling hot after clone
                    patch["priority_sampling"] = True
                    patch["event_advantage_boost"] = max(
                        1.75, float(patch.get("event_advantage_boost", 1.75))
                    )
            reasons.append(f"truncation_pbt_x{len(transfers)}")
            reasons.append("roles:" + ",".join(f"{k}:{v}" for k, v in fl.roles.items()))

        # --- SSL guide (§3–§5): league mix, reward decay, scenarios, stagnation ---
        ssl_hist = list((state.get("ssl_guide") or {}).get("reward_history") or [])
        ssl_ent = list((state.get("ssl_guide") or {}).get("entropy_history") or [])
        current_rw = status.get("reward_multipliers") or {}
        manifest = {}
        if self.watch_dir:
            try:
                from .io_utils import read_json

                manifest = read_json(Path(self.watch_dir) / "reward_manifest.json")
            except OSError:
                manifest = {}
        hardened = hardened_experts_available(status, self.ssl_cfg)
        sps_safe = bool(status.get("sps_safe") or (metrics or {}).get("Curriculum/SpsSafe"))
        ssl_patch, ssl_reason = build_autonomy_patch(
            timesteps=ts,
            phase=phase,
            metrics=metrics,
            current_mult=dict(current_rw),
            manifest=manifest,
            ssl=self.ssl_cfg,
            root_cfg=self.root_cfg or {"ssl_guide": self.ssl_cfg, "ssl_autonomy": {"enabled": True, "full_control": True}},
            reward_history=ssl_hist,
            entropy_history=ssl_ent,
            hardened_available=hardened,
            sps_safe=sps_safe,
            state=state,
        )
        # Track histories for next stagnation check
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
            "last_reason": ssl_reason,
            "track": (state.get("ssl_autonomy") or {}).get("track_id"),
            "track_label": (state.get("ssl_autonomy") or {}).get("track_label"),
            "play_hours": (state.get("ssl_autonomy") or {}).get("play_hours"),
        }
        ssl_league_base = {
            k: ssl_patch[k]
            for k in (
                "train_against_old_chance",
                "opponent_pool_chance",
                "ssl_guide_post_apex",
                "skill_tracker_enabled",
                "skill_tracker_interval",
            )
            if k in ssl_patch
        }
        for k, v in ssl_patch.items():
            if k.startswith("opponent_weight_"):
                ssl_league_base[k] = v

        # Role-specific sparring for active agent (biased SSL mix).
        # Respect autonomy warmup: do not force skill-eval ON while still warming.
        role = fl.roles.get(str(active_id), ROLE_MAIN)
        if ssl_league_base.get("skill_tracker_enabled") is False:
            patch.update({
                "skill_tracker_enabled": False,
                "skill_tracker_interval": ssl_league_base.get("skill_tracker_interval", 256),
                "train_against_old_chance": float(ssl_league_base.get("train_against_old_chance", 0.0)),
                "opponent_pool_chance": float(ssl_league_base.get("opponent_pool_chance", 0.0)),
                "ssl_guide_post_apex": True,
            })
        else:
            patch.update(role_sparring_overrides(role, ssl_base=ssl_league_base or None))

        # NAS genome for active → arch file hint
        nas_active = NASGenome.from_dict(fl.nas_by_agent.get(str(active_id)))
        if nas_active.use_memory or nas_active.residual_skips:
            patch["model_arch_overrides"] = nas_active.to_cpp_overrides()

        # --- LEVEL 4: env architect ---
        if self.env_enabled and (rotating or fl.generation == 0):
            fit = 0.0
            try:
                fit = float((metrics or {}).get("Average Step Reward") or 0)
            except (TypeError, ValueError):
                pass
            fl.env, env_patch, env_reason = step_env_architect(
                fl.env, fit, random.Random(ts + 7)
            )
            # merge reward_weights carefully
            rw = dict(patch.get("reward_weights") or {})
            rw.update(env_patch.pop("reward_weights", {}) or {})
            patch.update(env_patch)
            if rw:
                patch["reward_weights"] = rw
            # PBT-mutable intrinsic coeffs also on agent genome surface
            patch["w_icm"] = fl.env.genome.w_icm
            patch["w_rnd"] = fl.env.genome.w_rnd
            reasons.append(env_reason)

        # SSL reward/scenario/stagnation on top (scenarios may override env kickoff mix)
        if ssl_patch:
            rw = dict(patch.get("reward_weights") or {})
            rw.update(ssl_patch.pop("reward_weights", {}) or {})
            # Diagnostic nested dict — keep out of C++ float clamps via safety.pop later
            geo = ssl_patch.pop("ssl_geometry", None)
            patch.update(ssl_patch)
            if rw:
                patch["reward_weights"] = rw
            if geo is not None:
                patch["ssl_geometry"] = geo
            reasons.append(ssl_reason)

        # Persist arch overrides beside commands for C++ / reload tooling
        if self.watch_dir and "model_arch_overrides" in patch:
            arch_path = Path(self.watch_dir) / "model_arch_overrides.json"
            try:
                import json

                arch_path.write_text(
                    json.dumps(patch["model_arch_overrides"], indent=2), encoding="utf-8"
                )
            except OSError:
                pass

        self.save(state, fl)
        return patch, " | ".join(reasons) or "four_level", state
