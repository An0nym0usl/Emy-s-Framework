"""
Population-Based Training (PBT) — DeepMind-style exploit + explore over PPO hyperparams.

Virtual population lives in orchestrator state (single trainer process). Each "member"
is a hyperparameter genome scored from live metrics; losers copy winners and mutate.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any


# Search space for one PBT member (maps 1:1 to runtime_overrides keys).
GENOME_KEYS = (
    "entropy_scale",
    "policy_lr",
    "critic_lr",
    "gae_gamma",
    "gae_lambda",
    "var_max",
    "clip_range",
    "epochs",
    "es_noise_scale",
    "event_advantage_boost",
    "max_grad_norm",
    "w_icm",
    "w_rnd",
)

DEFAULT_GENOME: dict[str, float] = {
    "entropy_scale": 0.015,
    "policy_lr": 1e-4,
    "critic_lr": 1e-4,
    "gae_gamma": 0.99,
    "gae_lambda": 0.975,
    "var_max": 0.45,
    "clip_range": 0.2,
    "epochs": 1,
    "es_noise_scale": 0.0,
    "event_advantage_boost": 1.0,
    "max_grad_norm": 0.5,
    "w_icm": 0.05,
    "w_rnd": 0.03,
}

BOUNDS: dict[str, tuple[float, float]] = {
    "entropy_scale": (0.005, 0.04),
    "policy_lr": (1e-5, 5e-4),
    "critic_lr": (1e-5, 5e-4),
    "gae_gamma": (0.97, 0.999),
    "gae_lambda": (0.9, 0.99),
    "var_max": (0.15, 1.2),
    "clip_range": (0.1, 0.35),
    "epochs": (1, 3),
    "es_noise_scale": (0.0, 0.08),
    "event_advantage_boost": (1.0, 2.5),
    "max_grad_norm": (0.2, 2.0),
    "w_icm": (0.0, 0.3),
    "w_rnd": (0.0, 0.25),
}


@dataclass
class PBTMember:
    id: int
    genome: dict[str, float]
    fitness: float = 0.0
    steps_at_score: int = 0
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "genome": dict(self.genome),
            "fitness": self.fitness,
            "steps_at_score": self.steps_at_score,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PBTMember:
        return cls(
            id=int(d["id"]),
            genome={k: float(v) for k, v in (d.get("genome") or {}).items()},
            fitness=float(d.get("fitness") or 0),
            steps_at_score=int(d.get("steps_at_score") or 0),
            generation=int(d.get("generation") or 0),
        )


@dataclass
class PBTPopulation:
    members: list[PBTMember] = field(default_factory=list)
    active_id: int = 0
    generation: int = 0
    next_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "members": [m.to_dict() for m in self.members],
            "active_id": self.active_id,
            "generation": self.generation,
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PBTPopulation:
        if not d:
            return cls()
        return cls(
            members=[PBTMember.from_dict(m) for m in (d.get("members") or [])],
            active_id=int(d.get("active_id") or 0),
            generation=int(d.get("generation") or 0),
            next_id=int(d.get("next_id") or 0),
        )


def _clamp_genome(g: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in GENOME_KEYS:
        lo, hi = BOUNDS[k]
        v = float(g.get(k, DEFAULT_GENOME[k]))
        if k == "epochs":
            out[k] = float(int(round(max(lo, min(hi, v)))))
        else:
            out[k] = max(lo, min(hi, v))
    return out


def _perturb(value: float, key: str, strength: float, rng: random.Random) -> float:
    lo, hi = BOUNDS[key]
    if key in ("policy_lr", "critic_lr"):
        # Log-space multiply (classic PBT)
        factor = 1.2 if rng.random() < 0.5 else 0.8
        if rng.random() < strength:
            factor *= 1.0 + rng.uniform(-0.15, 0.15)
        return value * factor
    span = hi - lo
    delta = rng.uniform(-1, 1) * span * (0.08 + 0.25 * strength)
    return value + delta


def fitness_from_metrics(metrics: dict[str, Any], phase: int) -> float:
    """Scalar fitness: reward + skill signals. Higher is better."""
    m = metrics or {}

    def g(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in m and m[k] is not None:
                try:
                    return float(m[k])
                except (TypeError, ValueError):
                    pass
        return default

    reward = g("Average Step Reward", "avg_reward")
    entropy = g("Policy Entropy", "policy_entropy", default=0.5)
    kl = g("KL Div Loss", "kl", default=0.0)
    sps = g("Overall Steps/Second", default=0.0)
    touch = g("Touch Ratio", "touch_ratio", "Episode Touch Ratio")
    goal = g("Goal Rate", "goal_rate", "Goals Per Episode")

    # Prefer learning signal + skill, lightly penalize collapsed entropy / huge KL.
    score = (
        reward * 10.0
        + touch * 5.0
        + goal * 8.0
        + math.log1p(max(0.0, sps)) * 0.15
        - max(0.0, 0.15 - entropy) * 20.0
        - max(0.0, kl - 0.05) * 30.0
    )
    # Later phases care more about goals / sparring outcomes.
    if phase >= 1:
        score += goal * 4.0
    if phase >= 2:
        score += touch * 3.0
    return score


def init_population(size: int, seed: int = 42, base: dict[str, float] | None = None) -> PBTPopulation:
    rng = random.Random(seed)
    base_g = _clamp_genome({**DEFAULT_GENOME, **(base or {})})
    pop = PBTPopulation(next_id=size)
    for i in range(size):
        g = copy.deepcopy(base_g)
        if i > 0:
            for k in GENOME_KEYS:
                g[k] = _perturb(g[k], k, strength=0.9, rng=rng)
            g = _clamp_genome(g)
        pop.members.append(PBTMember(id=i, genome=g, generation=0))
    pop.active_id = 0
    return pop


def active_member(pop: PBTPopulation) -> PBTMember | None:
    for m in pop.members:
        if m.id == pop.active_id:
            return m
    return pop.members[0] if pop.members else None


def score_active(pop: PBTPopulation, metrics: dict, phase: int, timesteps: int) -> PBTMember | None:
    m = active_member(pop)
    if not m:
        return None
    m.fitness = fitness_from_metrics(metrics, phase)
    m.steps_at_score = timesteps
    return m


def exploit_explore(
    pop: PBTPopulation,
    *,
    bottom_frac: float = 0.2,
    top_frac: float = 0.2,
    mutate_strength: float = 0.6,
    seed: int | None = None,
) -> tuple[PBTPopulation, dict[str, float], str]:
    """
    Classic PBT step: if active is in bottom quantile, copy a top member and mutate.
    Always advances generation and rotates active to keep diversity over time.
    Returns (pop, genome_patch, reason).
    """
    if len(pop.members) < 2:
        m = active_member(pop)
        g = _clamp_genome(m.genome if m else DEFAULT_GENOME)
        return pop, g, "pbt_single_member"

    rng = random.Random(seed if seed is not None else pop.generation * 9973 + pop.active_id)
    ranked = sorted(pop.members, key=lambda x: x.fitness, reverse=True)
    n = len(ranked)
    n_top = max(1, int(math.ceil(n * top_frac)))
    n_bot = max(1, int(math.ceil(n * bottom_frac)))
    top = ranked[:n_top]
    bottom_ids = {m.id for m in ranked[-n_bot:]}

    active = active_member(pop)
    reason = "pbt_hold"
    genome = dict(active.genome) if active else dict(DEFAULT_GENOME)

    if active and active.id in bottom_ids:
        donor = rng.choice(top)
        genome = copy.deepcopy(donor.genome)
        for k in GENOME_KEYS:
            genome[k] = _perturb(genome[k], k, mutate_strength, rng)
        genome = _clamp_genome(genome)
        active.genome = genome
        active.generation = pop.generation + 1
        active.fitness = donor.fitness * 0.5  # reset optimism
        reason = f"pbt_exploit_from_{donor.id}_explore"

    # Rotate active member each generation so the live trainer samples the population.
    pop.generation += 1
    idx = next((i for i, m in enumerate(pop.members) if m.id == pop.active_id), 0)
    next_idx = (idx + 1) % len(pop.members)
    pop.active_id = pop.members[next_idx].id
    genome = _clamp_genome(pop.members[next_idx].genome)
    reason = f"{reason};activate_{pop.active_id}"
    return pop, genome, reason


def genome_to_overrides(genome: dict[str, float]) -> dict[str, Any]:
    g = _clamp_genome(genome)
    return {
        "entropy_scale": g["entropy_scale"],
        "policy_lr": g["policy_lr"],
        "critic_lr": g["critic_lr"],
        "gae_gamma": g["gae_gamma"],
        "gae_lambda": g["gae_lambda"],
        "var_max": g["var_max"],
        "clip_range": g["clip_range"],
        "epochs": int(g["epochs"]),
        "es_noise_scale": g["es_noise_scale"],
        "event_advantage_boost": g["event_advantage_boost"],
        "max_grad_norm": g["max_grad_norm"],
        "w_icm": g["w_icm"],
        "w_rnd": g["w_rnd"],
    }
