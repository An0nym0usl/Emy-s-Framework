"""
LEVEL 2 — Evolutionary NAS genome (topological macro-mutations).

Hyperpower keeps discrete MLP fast path; this genome is the optional Gated/memory
mutation slot applied on agent slot reload via model_arch_overrides.json.
Hidden dims stay multiples of 32 for Tensor Cores.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any


def _round32(n: int) -> int:
    return max(32, int(round(n / 32.0)) * 32)


@dataclass
class NASGenome:
    """Compact architecture chromosome for Actor/Critic MLP (+ optional memory)."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    use_memory: bool = False  # GRU / gated residual slot
    memory_dim: int = 128
    residual_skips: bool = False
    shared_head_dim: int = 256
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_dims": list(self.hidden_dims),
            "use_memory": self.use_memory,
            "memory_dim": int(self.memory_dim),
            "residual_skips": self.residual_skips,
            "shared_head_dim": int(self.shared_head_dim),
            "generation": int(self.generation),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> NASGenome:
        if not d:
            return cls()
        dims = [_round32(int(x)) for x in (d.get("hidden_dims") or [256, 256, 256])]
        return cls(
            hidden_dims=dims,
            use_memory=bool(d.get("use_memory", False)),
            memory_dim=_round32(int(d.get("memory_dim") or 128)),
            residual_skips=bool(d.get("residual_skips", False)),
            shared_head_dim=_round32(int(d.get("shared_head_dim") or 256)),
            generation=int(d.get("generation") or 0),
        )

    def to_cpp_overrides(self) -> dict[str, Any]:
        """Keys consumed by AutoTrainerBridge / Learner on slot reload."""
        return {
            "policy_layer_sizes": list(self.hidden_dims),
            "critic_layer_sizes": list(self.hidden_dims),
            "shared_head_layer_sizes": [self.shared_head_dim],
            "use_memory_gate": self.use_memory,
            "memory_dim": self.memory_dim,
            "residual_skips": self.residual_skips,
            "nas_generation": self.generation,
        }


MUTATION_KINDS = (
    "widen",
    "narrow",
    "add_layer",
    "remove_layer",
    "toggle_memory",
    "toggle_residual",
    "memory_dim",
)


def mutate_nas(genome: NASGenome, rng: random.Random, strength: float = 0.7) -> NASGenome:
    """Topological macro-mutation + width changes (×32)."""
    g = copy.deepcopy(genome)
    g.generation += 1
    kind = rng.choice(MUTATION_KINDS)

    if kind == "widen" and g.hidden_dims:
        i = rng.randrange(len(g.hidden_dims))
        g.hidden_dims[i] = _round32(min(1024, int(g.hidden_dims[i] * (1.25 + 0.25 * strength))))
    elif kind == "narrow" and g.hidden_dims:
        i = rng.randrange(len(g.hidden_dims))
        g.hidden_dims[i] = _round32(max(64, int(g.hidden_dims[i] * (0.75 - 0.1 * strength))))
    elif kind == "add_layer" and len(g.hidden_dims) < 6:
        base = g.hidden_dims[-1] if g.hidden_dims else 256
        g.hidden_dims.append(_round32(base))
    elif kind == "remove_layer" and len(g.hidden_dims) > 2:
        del g.hidden_dims[rng.randrange(len(g.hidden_dims))]
    elif kind == "toggle_memory":
        g.use_memory = not g.use_memory
        if g.use_memory:
            g.memory_dim = _round32(rng.choice([64, 128, 256]))
    elif kind == "toggle_residual":
        g.residual_skips = not g.residual_skips
    elif kind == "memory_dim":
        g.memory_dim = _round32(rng.choice([64, 96, 128, 192, 256]))

    g.shared_head_dim = _round32(min(512, max(64, g.shared_head_dim)))
    return g


def crossover_nas(a: NASGenome, b: NASGenome, rng: random.Random) -> NASGenome:
    child = NASGenome(
        hidden_dims=list(a.hidden_dims if rng.random() < 0.5 else b.hidden_dims),
        use_memory=a.use_memory if rng.random() < 0.5 else b.use_memory,
        memory_dim=a.memory_dim if rng.random() < 0.5 else b.memory_dim,
        residual_skips=a.residual_skips if rng.random() < 0.5 else b.residual_skips,
        shared_head_dim=a.shared_head_dim if rng.random() < 0.5 else b.shared_head_dim,
        generation=max(a.generation, b.generation) + 1,
    )
    return child
