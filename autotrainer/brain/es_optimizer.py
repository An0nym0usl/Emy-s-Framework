"""
Evolution Strategies (ES) meta-optimizer over PPO hyperparameters.

OpenAI-style antithetic ES: sample noise ε, evaluate fitness of θ±σε in a
virtual population buffer, then take a gradient-free update of θ.
PPO keeps doing local gradient descent; ES steers the hyperparameter manifold.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .pbt import BOUNDS, DEFAULT_GENOME, GENOME_KEYS, _clamp_genome, fitness_from_metrics


@dataclass
class ESState:
    theta: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GENOME))
    sigma: float = 0.12
    learning_rate: float = 0.08
    population: int = 8
    generation: int = 0
    best_fitness: float = float("-inf")
    best_theta: dict[str, float] = field(default_factory=dict)
    pending: list[dict[str, Any]] = field(default_factory=list)  # queued +/- evaluations
    completed: list[dict[str, Any]] = field(default_factory=list)
    history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta": dict(self.theta),
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "population": self.population,
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "best_theta": dict(self.best_theta),
            "pending": list(self.pending),
            "completed": list(self.completed),
            "history": list(self.history[-64:]),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ESState:
        if not d:
            return cls()
        return cls(
            theta=_clamp_genome(d.get("theta") or DEFAULT_GENOME),
            sigma=float(d.get("sigma") or 0.12),
            learning_rate=float(d.get("learning_rate") or 0.08),
            population=int(d.get("population") or 8),
            generation=int(d.get("generation") or 0),
            best_fitness=float(d.get("best_fitness") if d.get("best_fitness") is not None else float("-inf")),
            best_theta=dict(d.get("best_theta") or {}),
            pending=list(d.get("pending") or []),
            completed=list(d.get("completed") or []),
            history=list(d.get("history") or []),
        )


def _encode(theta: dict[str, float]) -> list[float]:
    """Map genome to unit-ish vector in [0,1] for isotropic noise."""
    vec = []
    for k in GENOME_KEYS:
        lo, hi = BOUNDS[k]
        v = float(theta.get(k, DEFAULT_GENOME[k]))
        if k in ("policy_lr", "critic_lr"):
            # log-normalize
            lv, llo, lhi = math.log(max(v, 1e-12)), math.log(lo), math.log(hi)
            vec.append((lv - llo) / max(1e-12, lhi - llo))
        else:
            vec.append((v - lo) / max(1e-12, hi - lo))
    return vec


def _decode(vec: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, k in enumerate(GENOME_KEYS):
        lo, hi = BOUNDS[k]
        t = max(0.0, min(1.0, float(vec[i])))
        if k in ("policy_lr", "critic_lr"):
            llo, lhi = math.log(lo), math.log(hi)
            out[k] = math.exp(llo + t * (lhi - llo))
        elif k == "epochs":
            out[k] = float(int(round(lo + t * (hi - lo))))
        else:
            out[k] = lo + t * (hi - lo)
    return _clamp_genome(out)


def _add_noise(theta: dict[str, float], noise: list[float], sigma: float) -> dict[str, float]:
    base = _encode(theta)
    return _decode([b + sigma * n for b, n in zip(base, noise)])


def seed_es(base: dict[str, float] | None = None, cfg: dict[str, Any] | None = None) -> ESState:
    c = cfg or {}
    return ESState(
        theta=_clamp_genome({**DEFAULT_GENOME, **(base or {})}),
        sigma=float(c.get("sigma", 0.12)),
        learning_rate=float(c.get("learning_rate", 0.08)),
        population=int(c.get("population", 8)),
    )


def propose_candidates(state: ESState, seed: int) -> list[dict[str, Any]]:
    """
    Build antithetic noise pairs for the next ES generation.
    Each candidate is {noise, sign, genome} — trainer evaluates genomes sequentially via overrides.
    """
    rng = random.Random(seed)
    half = max(2, state.population // 2)
    pending: list[dict[str, Any]] = []
    for _ in range(half):
        noise = [rng.gauss(0, 1) for _ in GENOME_KEYS]
        for sign in (1.0, -1.0):
            genome = _add_noise(state.theta, [sign * n for n in noise], state.sigma)
            pending.append({"noise": noise, "sign": sign, "genome": genome})
    state.pending = pending
    return pending


def observe_fitness(state: ESState, metrics: dict, phase: int) -> dict[str, float] | None:
    """
    Record fitness for the head of the pending queue.
    When the queue drains, apply the ES update and return the new center theta as overrides.
    """
    if not state.pending:
        return None

    fit = fitness_from_metrics(metrics, phase)
    head = state.pending.pop(0)
    head["fitness"] = fit
    state.history.append(fit)

    if fit >= state.best_fitness:
        state.best_fitness = fit
        state.best_theta = dict(head["genome"])

    state.completed.append(head)

    if state.pending:
        # Still evaluating — push next candidate genome.
        return dict(state.pending[0]["genome"])

    # Generation complete → ES gradient estimate.
    completed: list[dict[str, Any]] = list(state.completed)
    state.completed = []
    if not completed:
        return dict(state.theta)

    # Group antithetic pairs by noise identity (same noise list).
    # Average fitness-weighted noise direction.
    dim = len(GENOME_KEYS)
    grad = [0.0] * dim
    # Normalize fitness
    fits = [float(c["fitness"]) for c in completed]
    mean_f = sum(fits) / len(fits)
    std_f = math.sqrt(sum((f - mean_f) ** 2 for f in fits) / max(1, len(fits))) or 1.0

    for c in completed:
        f_norm = (float(c["fitness"]) - mean_f) / std_f
        sign = float(c["sign"])
        noise = c["noise"]
        for i in range(dim):
            grad[i] += f_norm * sign * float(noise[i])

    n = max(1, len(completed))
    alpha = state.learning_rate / (n * max(1e-6, state.sigma))
    center = _encode(state.theta)
    updated = [center[i] + alpha * grad[i] for i in range(dim)]
    state.theta = _decode(updated)
    state.generation += 1

    # Mild sigma anneal toward exploitation.
    state.sigma = max(0.04, state.sigma * 0.995)

    # Blend toward best-so-far (elitist ES).
    if state.best_theta:
        bt = _encode(state.best_theta)
        th = _encode(state.theta)
        state.theta = _decode([0.85 * th[i] + 0.15 * bt[i] for i in range(dim)])

    return dict(state.theta)


def peek_next_genome(state: ESState) -> dict[str, float]:
    if state.pending:
        return dict(state.pending[0]["genome"])
    return dict(state.theta)
