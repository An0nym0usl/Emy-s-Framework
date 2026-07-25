"""
Competitive Population-Based Training — 4 real agents fighting for quality.

PBT-style league:
  - 4 agents, each with its own hyperparameters + weight slot on disk
  - They take turns on the live trainer (one GPU), scored by fitness/Elo
  - Losers copy the winner's weights + mutate hyperparameters (exploit + explore)
  - Competitive, quality-first — not just random hyperparam noise
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from .pbt import (
    BOUNDS,
    DEFAULT_GENOME,
    GENOME_KEYS,
    _clamp_genome,
    _perturb,
    fitness_from_metrics,
    genome_to_overrides,
)


@dataclass
class LeagueAgent:
    id: int
    genome: dict[str, float]
    elo: float = 1000.0
    fitness: float = 0.0
    steps_trained: int = 0
    wins: int = 0
    losses: int = 0
    generation: int = 0
    has_checkpoint: bool = False
    role: str = "main"  # main | main_exploiter | league_exploiter

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "genome": dict(self.genome),
            "elo": self.elo,
            "fitness": self.fitness,
            "steps_trained": self.steps_trained,
            "wins": self.wins,
            "losses": self.losses,
            "generation": self.generation,
            "has_checkpoint": self.has_checkpoint,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LeagueAgent:
        return cls(
            id=int(d["id"]),
            genome=_clamp_genome(d.get("genome") or DEFAULT_GENOME),
            elo=float(d.get("elo") or 1000),
            fitness=float(d.get("fitness") or 0),
            steps_trained=int(d.get("steps_trained") or 0),
            wins=int(d.get("wins") or 0),
            losses=int(d.get("losses") or 0),
            generation=int(d.get("generation") or 0),
            has_checkpoint=bool(d.get("has_checkpoint")),
            role=str(d.get("role") or "main"),
        )


@dataclass
class CompetitiveLeague:
    agents: list[LeagueAgent] = field(default_factory=list)
    active_id: int = 0
    generation: int = 0
    last_rotate_timesteps: int = 0
    rotate_every_steps: int = 10_000_000
    pending_save: int | None = None
    pending_load: int | None = None
    last_tournament: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "active_id": self.active_id,
            "generation": self.generation,
            "last_rotate_timesteps": self.last_rotate_timesteps,
            "rotate_every_steps": self.rotate_every_steps,
            "pending_save": self.pending_save,
            "pending_load": self.pending_load,
            "last_tournament": dict(self.last_tournament),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> CompetitiveLeague:
        if not d:
            return cls()
        return cls(
            agents=[LeagueAgent.from_dict(a) for a in (d.get("agents") or [])],
            active_id=int(d.get("active_id") or 0),
            generation=int(d.get("generation") or 0),
            last_rotate_timesteps=int(d.get("last_rotate_timesteps") or 0),
            rotate_every_steps=int(d.get("rotate_every_steps") or 10_000_000),
            pending_save=d.get("pending_save"),
            pending_load=d.get("pending_load"),
            last_tournament=dict(d.get("last_tournament") or {}),
        )


def init_league(
    n_agents: int = 4,
    seed: int = 42,
    rotate_every_steps: int = 10_000_000,
    base: dict[str, float] | None = None,
) -> CompetitiveLeague:
    """Create 4 diversified agents — different hyperparam niches competing."""
    rng = random.Random(seed)
    base_g = _clamp_genome({**DEFAULT_GENOME, **(base or {})})
    # Distinct niches so competition explores different regimes.
    niches = [
        {},  # baseline
        {"entropy_scale": 0.022, "es_noise_scale": 0.04, "var_max": 0.7},  # explorer
        {"entropy_scale": 0.010, "policy_lr": 6e-5, "clip_range": 0.15, "epochs": 2},  # exploiter
        {"event_advantage_boost": 1.8, "gae_lambda": 0.95, "var_max": 0.55, "es_noise_scale": 0.025},  # event-hunter
    ]
    league = CompetitiveLeague(rotate_every_steps=rotate_every_steps)
    for i in range(n_agents):
        g = copy.deepcopy(base_g)
        niche = niches[i % len(niches)]
        g.update(niche)
        if i > 0:
            for k in GENOME_KEYS:
                if rng.random() < 0.5:
                    g[k] = _perturb(float(g[k]), k, strength=0.7, rng=rng)
        g = _clamp_genome(g)
        league.agents.append(LeagueAgent(id=i, genome=g, elo=1000.0 + i))
    league.active_id = 0
    return league


def get_agent(league: CompetitiveLeague, agent_id: int | None = None) -> LeagueAgent | None:
    aid = league.active_id if agent_id is None else agent_id
    for a in league.agents:
        if a.id == aid:
            return a
    return league.agents[0] if league.agents else None


def _skill_bonus(metrics: dict[str, Any]) -> float:
    """Pull skill/Elo-like signals from trainer metrics if present."""
    bonus = 0.0
    for k, v in (metrics or {}).items():
        lk = str(k).lower()
        if "rating" in lk or "elo" in lk or "skill" in lk:
            try:
                bonus += float(v) * 0.01
            except (TypeError, ValueError):
                pass
    return bonus


def score_agent(
    league: CompetitiveLeague,
    metrics: dict[str, Any],
    phase: int,
    timesteps: int,
) -> LeagueAgent | None:
    agent = get_agent(league)
    if not agent:
        return None
    fit = fitness_from_metrics(metrics, phase) + _skill_bonus(metrics)
    # EMA so one noisy iteration doesn't dominate
    agent.fitness = 0.7 * agent.fitness + 0.3 * fit if agent.steps_trained > 0 else fit
    agent.steps_trained = timesteps
    return agent


def _elo_update(winner: LeagueAgent, loser: LeagueAgent, k: float = 24.0) -> None:
    exp_w = 1.0 / (1.0 + 10 ** ((loser.elo - winner.elo) / 400.0))
    winner.elo += k * (1.0 - exp_w)
    loser.elo += k * (0.0 - (1.0 - exp_w))
    winner.wins += 1
    loser.losses += 1


def run_tournament(league: CompetitiveLeague, seed: int | None = None) -> dict[str, Any]:
    """
    Pairwise compete by fitness; update Elo.
    Truncation selection: bottom 20% clone top 20% (weights via pending_load + mutate).
    Reassigns AlphaStar-style roles each generation.
    """
    if len(league.agents) < 2:
        return {"action": "noop"}

    import math

    rng = random.Random(seed if seed is not None else league.generation * 7919)
    ranked = sorted(league.agents, key=lambda a: (a.fitness, a.elo), reverse=True)

    # Round-robin-ish Elo from ranking: each beats everyone below.
    for i, stronger in enumerate(ranked):
        for weaker in ranked[i + 1 :]:
            _elo_update(stronger, weaker)

    n = len(ranked)
    n_top = max(1, int(math.ceil(n * 0.2)))
    n_bot = max(1, int(math.ceil(n * 0.2)))
    top = ranked[:n_top]
    bottom = ranked[-n_bot:]

    # Asymmetric league roles (Main / MainExploiter / LeagueExploiter)
    role_cycle = ["main", "main_exploiter", "league_exploiter", "main"]
    shuffled = list(league.agents)
    rng.shuffle(shuffled)
    for i, ag in enumerate(shuffled):
        ag.role = role_cycle[i % len(role_cycle)]

    transfers: list[dict[str, Any]] = []
    for loser in bottom:
        donor = rng.choice(top)
        # Exploit: copy winner genome, then explore (mutate)
        loser.genome = copy.deepcopy(donor.genome)
        for k in GENOME_KEYS:
            loser.genome[k] = _perturb(float(loser.genome[k]), k, strength=0.75, rng=rng)
        loser.genome = _clamp_genome(loser.genome)
        loser.generation += 1
        # Soft Elo pull toward donor (not full reset — keeps ranking memory)
        loser.elo = 0.85 * loser.elo + 0.15 * donor.elo
        loser.fitness *= 0.5
        transfers.append(
            {
                "loser": loser.id,
                "donor": donor.id,
                "donor_elo": donor.elo,
                "donor_fitness": donor.fitness,
                "mode": "truncation_20",
            }
        )

    league.generation += 1
    summary = {
        "generation": league.generation,
        "ranking": [
            {"id": a.id, "fitness": a.fitness, "elo": a.elo, "role": a.role}
            for a in ranked
        ],
        "transfers": transfers,
        "champion": ranked[0].id,
        "champion_elo": ranked[0].elo,
        "roles": {str(a.id): a.role for a in league.agents},
    }
    league.last_tournament = summary
    return summary


def should_rotate(league: CompetitiveLeague, timesteps: int) -> bool:
    if not league.agents:
        return False
    return timesteps - league.last_rotate_timesteps >= league.rotate_every_steps


def rotate_and_compete(
    league: CompetitiveLeague,
    timesteps: int,
    *,
    run_tourney: bool = True,
    seed: int | None = None,
) -> tuple[CompetitiveLeague, dict[str, Any], dict[str, Any]]:
    """
    End of an agent's turn:
      1. Save current agent weights to its slot
      2. Optional tournament (exploit/explore among the 4)
      3. Activate next agent; load its weights (or donor weights if just exploited)

    Returns (league, genome_overrides, pbt_commands).
    """
    cmds: dict[str, Any] = {}
    active = get_agent(league)
    if active:
        cmds["save_agent_slot"] = active.id
        active.has_checkpoint = True

    tourney = {}
    if run_tourney and league.generation % 1 == 0:
        tourney = run_tournament(league, seed=seed)
        # If the NEXT agent was a loser, load the donor's weights into them.
        # We set pending_load after we know next_id.

    # Rotate to next agent
    ids = [a.id for a in league.agents]
    if not ids:
        return league, {}, cmds
    try:
        idx = ids.index(league.active_id)
    except ValueError:
        idx = 0
    next_id = ids[(idx + 1) % len(ids)]

    # If next agent was just exploited, load donor checkpoint instead of their own.
    load_id = next_id
    if tourney.get("transfers"):
        for t in tourney["transfers"]:
            if t["loser"] == next_id:
                donor = t["donor"]
                donor_agent = get_agent(league, donor)
                if donor_agent and donor_agent.has_checkpoint:
                    load_id = donor
                    break

    next_agent = get_agent(league, next_id)
    if next_agent and (next_agent.has_checkpoint or load_id != next_id):
        donor_ok = get_agent(league, load_id)
        if donor_ok and donor_ok.has_checkpoint:
            cmds["load_agent_slot"] = load_id

    league.active_id = next_id
    league.last_rotate_timesteps = timesteps
    league.pending_save = cmds.get("save_agent_slot")
    league.pending_load = cmds.get("load_agent_slot")

    genome = genome_to_overrides(next_agent.genome if next_agent else DEFAULT_GENOME)
    genome["active_agent_id"] = next_id
    # Do not force skill tracker / heavy sparring during early competitive turns —
    # those crush SPS. OP hyperparams still apply; curriculum owns phase gates.
    genome["priority_sampling"] = True
    genome["event_advantage_boost"] = max(1.5, float(genome.get("event_advantage_boost", 1.5)))

    reason_meta = {
        "tourney": tourney,
        "active": next_id,
        "load": cmds.get("load_agent_slot"),
        "save": cmds.get("save_agent_slot"),
    }
    return league, genome, {**cmds, "_meta": reason_meta}


def league_public_status(league: CompetitiveLeague) -> dict[str, Any]:
    ranked = sorted(league.agents, key=lambda a: a.elo, reverse=True)
    return {
        "active_id": league.active_id,
        "generation": league.generation,
        "rotate_every_steps": league.rotate_every_steps,
        "leaderboard": [
            {
                "id": a.id,
                "elo": round(a.elo, 1),
                "fitness": round(a.fitness, 3),
                "wins": a.wins,
                "losses": a.losses,
                "has_checkpoint": a.has_checkpoint,
                "role": a.role,
                "entropy_scale": a.genome.get("entropy_scale"),
                "policy_lr": a.genome.get("policy_lr"),
                "max_grad_norm": a.genome.get("max_grad_norm"),
                "w_icm": a.genome.get("w_icm"),
                "w_rnd": a.genome.get("w_rnd"),
            }
            for a in ranked
        ],
        "last_tournament": league.last_tournament,
    }
