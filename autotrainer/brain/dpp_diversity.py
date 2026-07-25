"""
Geometric Diversity Regularization (DPP-style).

K_ij = exp(-D_KL(π_i || π_j)) approximated from hyperparam / behavior fingerprints
when full policy KL is unavailable. Fitness penalty: -λ * log det(K + εI).
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def _fingerprint(genome: dict[str, float], keys: Sequence[str]) -> list[float]:
    vec: list[float] = []
    for k in keys:
        v = float(genome.get(k, 0.0))
        if k in ("policy_lr", "critic_lr"):
            vec.append(math.log10(max(v, 1e-8)))
        else:
            vec.append(v)
    return vec


def _kl_proxy(a: list[float], b: list[float]) -> float:
    """Symmetric squared distance as stand-in for D_KL when logits unavailable."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a))


def similarity_kernel(
    genomes: Sequence[dict[str, float]],
    keys: Sequence[str],
    *,
    temperature: float = 1.0,
) -> list[list[float]]:
    fps = [_fingerprint(g, keys) for g in genomes]
    n = len(fps)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = _kl_proxy(fps[i], fps[j])
            K[i][j] = math.exp(-d / max(1e-6, temperature))
    return K


def _det_2x2_or_cholesky_logdet(K: list[list[float]], eps: float = 1e-4) -> float:
    """log det(K + εI) via Cholesky (stable for small n)."""
    n = len(K)
    if n == 0:
        return 0.0
    A = [[K[i][j] + (eps if i == j else 0.0) for j in range(n)] for i in range(n)]
    # Cholesky
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 1e-12:
                    return -1e6  # near-singular → heavy penalty already applied via caller
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return 2.0 * sum(math.log(max(L[i][i], 1e-12)) for i in range(n))


def dpp_penalty(
    genomes: Sequence[dict[str, float]],
    keys: Sequence[str],
    *,
    lam: float = 0.15,
    temperature: float = 1.0,
) -> float:
    """
    Returns a non-negative penalty to SUBTRACT from population fitness
    when members collapse (small logdet → large penalty).
    """
    if len(genomes) < 2:
        return 0.0
    K = similarity_kernel(genomes, keys, temperature=temperature)
    logdet = _det_2x2_or_cholesky_logdet(K)
    # Higher logdet = more diverse = lower penalty
    # Target: well-conditioned K has logdet roughly in [0, n]
    n = len(genomes)
    target = 0.5 * n
    deficit = max(0.0, target - logdet)
    return lam * deficit


def diversity_adjusted_fitness(
    base_fitness: float,
    genomes: Sequence[dict[str, float]],
    keys: Sequence[str],
    *,
    lam: float = 0.15,
) -> float:
    return base_fitness - dpp_penalty(genomes, keys, lam=lam)


def population_diversity_report(
    genomes: Sequence[dict[str, float]],
    keys: Sequence[str],
) -> dict[str, Any]:
    pen = dpp_penalty(genomes, keys)
    K = similarity_kernel(genomes, keys)
    off = [K[i][j] for i in range(len(K)) for j in range(i + 1, len(K))]
    mean_sim = sum(off) / max(1, len(off))
    return {
        "dpp_penalty": round(pen, 4),
        "mean_pairwise_similarity": round(mean_sim, 4),
        "n_agents": len(genomes),
    }
