"""
LEVEL 1 — Online Meta-Gradient Tuning (bi-level).

Outer objective: validation / next-step proxy loss impact of α=(LR, γ, ε_clip).
Inner: PPO already runs in C++. We approximate ∂L_val/∂α via finite differences
on live metric proxies and push overrides through runtime_overrides.json.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


META_KEYS = ("policy_lr", "critic_lr", "gae_gamma", "clip_range")

# Finite-diff relative step (multiplicative for LRs, additive for γ / clip).
EPS: dict[str, float] = {
    "policy_lr": 0.15,
    "critic_lr": 0.15,
    "gae_gamma": 0.002,
    "clip_range": 0.02,
}

BOUNDS: dict[str, tuple[float, float]] = {
    "policy_lr": (1e-5, 5e-4),
    "critic_lr": (1e-5, 5e-4),
    "gae_gamma": (0.97, 0.999),
    "clip_range": (0.1, 0.35),
}


@dataclass
class MetaGradState:
    """Tracks last proxy loss and last applied α for FD meta-updates."""

    alpha: dict[str, float] = field(default_factory=dict)
    last_proxy: float = 0.0
    last_grad: dict[str, float] = field(default_factory=dict)
    updates: int = 0
    meta_lr: float = 0.25  # outer step size on normalized grads

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": dict(self.alpha),
            "last_proxy": self.last_proxy,
            "last_grad": dict(self.last_grad),
            "updates": self.updates,
            "meta_lr": self.meta_lr,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> MetaGradState:
        if not d:
            return cls()
        return cls(
            alpha={k: float(v) for k, v in (d.get("alpha") or {}).items()},
            last_proxy=float(d.get("last_proxy") or 0),
            last_grad={k: float(v) for k, v in (d.get("last_grad") or {}).items()},
            updates=int(d.get("updates") or 0),
            meta_lr=float(d.get("meta_lr") or 0.25),
        )


def proxy_loss(metrics: dict[str, Any]) -> float:
    """
    Surrogate for L_val: lower is better.
    Combines KL blow-up, entropy collapse, and negative reward.
    """
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
    kl = g("KL Div Loss", "Mean KL Divergence", "kl", default=0.0)
    # L_proxy ≈ KL + entropy_collapse - reward
    return (
        max(0.0, kl) * 40.0
        + max(0.0, 0.12 - entropy) * 25.0
        - reward * 8.0
    )


def _clamp(key: str, v: float) -> float:
    lo, hi = BOUNDS[key]
    return max(lo, min(hi, v))


def seed_alpha(base: dict[str, Any] | None = None) -> dict[str, float]:
    b = base or {}
    return {
        "policy_lr": float(b.get("policy_lr", 1e-4)),
        "critic_lr": float(b.get("critic_lr", 1e-4)),
        "gae_gamma": float(b.get("gae_gamma", 0.99)),
        "clip_range": float(b.get("clip_range", 0.2)),
    }


def meta_update(
    state: MetaGradState,
    metrics: dict[str, Any],
    *,
    current_overrides: dict[str, Any] | None = None,
) -> tuple[MetaGradState, dict[str, float], str]:
    """
    Practical bi-level FD step:

        g_α ≈ (L_proxy(t) - L_proxy(t-1)) / (α(t) - α(t-1) + ε)
        α ← α - η_meta * normalize(g_α)

    When α did not move, fall back to a heuristic gradient from KL / entropy.
    """
    if not state.alpha:
        state.alpha = seed_alpha(current_overrides)

    L = proxy_loss(metrics)
    reason = "meta_hold"
    patch: dict[str, float] = {}

    if state.updates == 0:
        state.last_proxy = L
        state.updates = 1
        return state, dict(state.alpha), "meta_seed"

    dL = L - state.last_proxy
    grads: dict[str, float] = {}

    # Finite-diff along last Δα when available; else heuristic from metrics.
    for k in META_KEYS:
        prev = float(state.alpha.get(k, seed_alpha()[k]))
        cur_ov = current_overrides or {}
        cur = float(cur_ov.get(k, prev)) if k in (cur_ov or {}) else prev
        da = cur - prev
        if abs(da) > 1e-12:
            grads[k] = dL / da
        else:
            # Heuristic: if KL high → shrink LR / clip; if entropy low → raise clip slightly
            kl = 0.0
            ent = 0.5
            try:
                kl = float((metrics or {}).get("Mean KL Divergence") or (metrics or {}).get("KL Div Loss") or 0)
                ent = float((metrics or {}).get("Policy Entropy") or 0.5)
            except (TypeError, ValueError):
                pass
            if k in ("policy_lr", "critic_lr"):
                grads[k] = 1.0 if kl > 0.04 else (-0.5 if ent < 0.2 else 0.1 * math.tanh(dL))
            elif k == "clip_range":
                grads[k] = 1.0 if kl > 0.05 else (-0.3 if ent < 0.15 else 0.0)
            else:  # gae_gamma
                grads[k] = 0.2 * math.tanh(dL)

    # Normalize grads
    norm = math.sqrt(sum(g * g for g in grads.values())) + 1e-8
    new_alpha = dict(state.alpha)
    for k, g in grads.items():
        step = state.meta_lr * (g / norm)
        if k in ("policy_lr", "critic_lr"):
            # Multiplicative on log-scale
            factor = math.exp(-0.35 * step)
            new_alpha[k] = _clamp(k, new_alpha[k] * factor)
        else:
            new_alpha[k] = _clamp(k, new_alpha[k] - 0.05 * step)

    state.last_grad = grads
    state.last_proxy = L
    state.alpha = new_alpha
    state.updates += 1
    patch = dict(new_alpha)
    reason = f"meta_fd_dL={dL:.4f}"
    return state, patch, reason
