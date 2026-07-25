"""
LEVEL 4 — Environment Architect (POET/PLR + automatic reward shaping).

Mutates curriculum / state-setter weights and intrinsic coefficients (w_icm, w_rnd).
Python meta-signals push into runtime_overrides; C++ applies no_touch / state weights
when wired in TrainingCurriculum + ExampleMain.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvGenome:
    ball_chase_weight: float = 0.85
    random_state_weight: float = 0.10
    kickoff_weight: float = 0.05
    fuzzed_weight: float = 0.40
    aerial_weight: float = 0.10
    no_touch_seconds: float = 4.0
    players_per_team: int = 1  # chase default; foundation/advanced override by phase
    w_icm: float = 0.05
    w_rnd: float = 0.03
    complexity: float = 1.0  # 0=easy subgoals … 1=full

    def to_dict(self) -> dict[str, Any]:
        return {
            "ball_chase_weight": self.ball_chase_weight,
            "random_state_weight": self.random_state_weight,
            "kickoff_weight": self.kickoff_weight,
            "fuzzed_weight": self.fuzzed_weight,
            "aerial_weight": self.aerial_weight,
            "no_touch_seconds": self.no_touch_seconds,
            "players_per_team": self.players_per_team,
            "w_icm": self.w_icm,
            "w_rnd": self.w_rnd,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> EnvGenome:
        if not d:
            return cls()
        return cls(**{k: d[k] for k in cls().to_dict() if k in d})

    def normalized_state_weights(self) -> tuple[float, float, float]:
        s = max(1e-6, self.ball_chase_weight + self.random_state_weight + self.kickoff_weight)
        return (
            self.ball_chase_weight / s,
            self.random_state_weight / s,
            self.kickoff_weight / s,
        )

    def gpu_reset_weights(self) -> tuple[float, float, float]:
        """kickoff / fuzzed / aerial — drives RocketSimCuda::SetResetCurriculum."""
        s = max(1e-6, self.kickoff_weight + self.fuzzed_weight + self.aerial_weight)
        return (
            self.kickoff_weight / s,
            self.fuzzed_weight / s,
            self.aerial_weight / s,
        )

    def to_overrides(self) -> dict[str, Any]:
        bc, rs, ko = self.normalized_state_weights()
        gk, gf, ga = self.gpu_reset_weights()
        return {
            "ball_chase_weight": bc,
            "random_state_weight": rs,
            "kickoff_weight": ko,
            "fuzzed_weight": float(gf),
            "aerial_weight": float(ga),
            "gpu_reset_kickoff": float(gk),
            "gpu_reset_fuzzed": float(gf),
            "gpu_reset_aerial": float(ga),
            "no_touch_seconds": float(self.no_touch_seconds),
            "w_icm": float(self.w_icm),
            "w_rnd": float(self.w_rnd),
            "env_complexity": float(self.complexity),
            # Intrinsic shaping as reward multipliers (Python→C++ RuntimeRewardRegistry)
            "reward_weights": {
                "VelocityPlayerToBallReward": 1.0 + 0.5 * self.w_icm,
                "TouchBallReward": 1.0 + 0.4 * self.w_rnd,
                "VelocityPlayerToBallReward": 1.0 + 0.3 * self.w_icm,
            },
        }


def mutate_env(genome: EnvGenome, rng: random.Random, *, stagnating: bool = False) -> EnvGenome:
    g = copy.deepcopy(genome)
    if stagnating:
        # SSL §4: force scenario diversity (more fuzzed/aerial) while easing complexity
        g.complexity = max(0.35, g.complexity * 0.85)
        g.ball_chase_weight = min(0.95, g.ball_chase_weight + 0.05)
        g.kickoff_weight = max(0.02, g.kickoff_weight * 0.75)
        g.fuzzed_weight = min(0.55, g.fuzzed_weight + 0.08)
        g.aerial_weight = min(0.50, g.aerial_weight + 0.06)
        g.no_touch_seconds = max(3.0, g.no_touch_seconds - 0.5)
        g.w_icm = min(0.25, g.w_icm * 1.15)
        g.w_rnd = min(0.2, g.w_rnd * 1.1)
        return g

    g.ball_chase_weight = min(0.95, max(0.4, g.ball_chase_weight * rng.uniform(0.9, 1.1)))
    g.random_state_weight = min(0.4, max(0.05, g.random_state_weight * rng.uniform(0.85, 1.15)))
    g.kickoff_weight = min(0.3, max(0.02, g.kickoff_weight * rng.uniform(0.85, 1.15)))
    g.fuzzed_weight = min(0.55, max(0.15, g.fuzzed_weight * rng.uniform(0.85, 1.15)))
    g.aerial_weight = min(0.55, max(0.05, g.aerial_weight * rng.uniform(0.85, 1.2)))
    g.no_touch_seconds = min(12.0, max(3.0, g.no_touch_seconds + rng.uniform(-1.0, 1.0)))
    g.w_icm = min(0.3, max(0.0, g.w_icm * rng.choice([0.8, 1.0, 1.25])))
    g.w_rnd = min(0.25, max(0.0, g.w_rnd * rng.choice([0.8, 1.0, 1.25])))
    g.complexity = min(1.0, max(0.35, g.complexity + rng.uniform(-0.08, 0.08)))
    return g


def detect_stagnation(history: list[float], window: int = 5, eps: float = 1e-3) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    return (max(recent) - min(recent)) < eps


@dataclass
class EnvArchitectState:
    genome: EnvGenome = field(default_factory=EnvGenome)
    fitness_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"genome": self.genome.to_dict(), "fitness_history": list(self.fitness_history[-32:])}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> EnvArchitectState:
        if not d:
            return cls()
        return cls(
            genome=EnvGenome.from_dict(d.get("genome")),
            fitness_history=list(d.get("fitness_history") or []),
        )


def step_env_architect(
    state: EnvArchitectState,
    fitness: float,
    rng: random.Random,
) -> tuple[EnvArchitectState, dict[str, Any], str]:
    state.fitness_history.append(float(fitness))
    stagnating = detect_stagnation(state.fitness_history)
    state.genome = mutate_env(state.genome, rng, stagnating=stagnating)
    reason = "env_poet_ease" if stagnating else "env_plr_mutate"
    return state, state.genome.to_overrides(), reason
