"""Metric gate evaluation against bot profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GateResult:
    phase: str
    passed: bool
    score: float
    failures: list[str]
    details: dict[str, float | None]


def metric_value(status: dict, wandb_metrics: dict, key: str) -> float | None:
    lm = status.get("last_metrics") or {}
    if key in lm:
        return float(lm[key])
    if key in wandb_metrics:
        return float(wandb_metrics[key])
    return None


def _gate_ok(gate: dict, value: float | None) -> bool:
    if value is None:
        return False
    if "min" in gate and value < gate["min"]:
        return False
    if "max" in gate and value > gate["max"]:
        return False
    return True


def evaluate_phase_gates(
    profile: dict,
    phase_name: str,
    status: dict,
    wandb_metrics: dict,
) -> GateResult:
    gates = (profile.get("gates") or {}).get(phase_name) or []
    failures: list[str] = []
    details: dict[str, float | None] = {}
    weighted_score = 0.0
    total_weight = 0.0

    for g in gates:
        key = g["metric"]
        val = metric_value(status, wandb_metrics, key)
        details[key] = val
        w = float(g.get("weight", 1.0))
        total_weight += w
        ok = _gate_ok(g, val)
        if ok:
            weighted_score += w
        else:
            failures.append(
                f"{key}={val} (need min={g.get('min')} max={g.get('max')})"
            )

    passed = len(failures) == 0 and total_weight > 0
    score = weighted_score / total_weight if total_weight else 0.0
    return GateResult(phase_name, passed, score, failures, details)
