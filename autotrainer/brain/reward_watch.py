"""Rolling Average Step Reward crash / recover announcements.

Always log loudly when reward collapses or rebounds — even if AutoTrainer
takes no corrective action (soft rollback / noop).

Also formats the unified compact cycle status line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# Default: mid of the 25–40% crash band vs rolling baseline
_DEFAULT_CRASH_DROP = 0.30
_DEFAULT_BASELINE_WINDOW = 8
_DEFAULT_RECOVER_FRACTION = 0.5


def _f(metrics: dict[str, Any] | None, *keys: str, default: float = float("nan")) -> float:
    m = metrics or {}
    for k in keys:
        if k in m and m[k] is not None:
            try:
                v = float(m[k])
                if math.isnan(v) or math.isinf(v):
                    return float("nan")
                return v
            except (TypeError, ValueError):
                pass
    return default


def _fmt_reward(v: float) -> str:
    """Compact reward display (2 decimals when |v|>=0.1 else 4)."""
    if math.isnan(v):
        return "nan"
    av = abs(v)
    if av >= 10:
        return f"{v:.2f}"
    if av >= 0.1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def format_cycle_line(
    *,
    zone: str | None = None,
    entropy: float | None = None,
    entropy_scale: float | None = None,
    reward: float | None = None,
    reward_delta_pct: float | None = None,
    action: str | None = None,
    sps: float | None = None,
    extra: str = "",
) -> str:
    """One compact status line per OP cycle."""
    bits: list[str] = ["[AutoTrainer]"]
    if zone is not None:
        bits.append(f"zone={zone}")
    if entropy is not None and not math.isnan(float(entropy)):
        h = float(entropy)
        bits.append(f"H={h:.4f}" if h >= 0.01 else f"H={h:.4e}")
    if entropy_scale is not None:
        try:
            bits.append(f"entropy_scale={float(entropy_scale):.4f}")
        except (TypeError, ValueError):
            bits.append(f"entropy_scale={entropy_scale}")
    if reward is not None and not math.isnan(float(reward)):
        bits.append(f"reward={_fmt_reward(float(reward))}")
    if reward_delta_pct is not None and not math.isnan(float(reward_delta_pct)):
        d = float(reward_delta_pct)
        sign = "+" if d >= 0 else ""
        bits.append(f"reward_delta={sign}{d:.1f}%")
    if sps is not None and not math.isnan(float(sps)):
        bits.append(f"SPS={float(sps):.0f}")
    if action:
        bits.append(f"action={action}")
    if extra:
        bits.append(extra.strip())
    return " ".join(bits)


@dataclass
class RewardWatch:
    """Track reward vs rolling baseline; emit REWARD CRASH / REWARD RECOVERED."""

    last_avg: float = float("nan")
    baseline: float = float("nan")
    in_crash: bool = False
    crash_from: float = float("nan")
    crash_floor: float = float("nan")
    samples: int = 0
    # Fraction drop vs rolling baseline that counts as a crash (25–40% band)
    crash_drop: float = _DEFAULT_CRASH_DROP
    # Reclaim this fraction of lost ground to count as recovered
    recover_fraction: float = _DEFAULT_RECOVER_FRACTION
    baseline_window: int = _DEFAULT_BASELINE_WINDOW
    history: list[float] = field(default_factory=list)
    # Last computed delta% vs baseline (for cycle line; 0 on first sample)
    last_delta_pct: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_avg": self.last_avg,
            "baseline": self.baseline,
            "in_crash": self.in_crash,
            # Back-compat aliases used by older engine/rollback paths
            "in_drop": self.in_crash,
            "drop_from": self.crash_from,
            "drop_to": self.crash_floor,
            "crash_from": self.crash_from,
            "crash_floor": self.crash_floor,
            "samples": self.samples,
            "crash_drop": self.crash_drop,
            "drop_threshold": self.crash_drop,
            "recover_fraction": self.recover_fraction,
            "baseline_window": self.baseline_window,
            "last_delta_pct": self.last_delta_pct,
            "history": list(self.history[-32:]),
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any] | None,
        *,
        crash_drop: float | None = None,
    ) -> RewardWatch:
        w = cls()
        if d:
            for k in (
                "last_avg",
                "baseline",
                "crash_from",
                "crash_floor",
                "crash_drop",
                "recover_fraction",
                "last_delta_pct",
            ):
                if k in d and d[k] is not None:
                    try:
                        setattr(w, k, float(d[k]))
                    except (TypeError, ValueError):
                        pass
            if "last_reward" in d and math.isnan(w.last_avg):
                try:
                    w.last_avg = float(d["last_reward"])
                except (TypeError, ValueError):
                    pass
            if "drop_from" in d and math.isnan(w.crash_from):
                try:
                    w.crash_from = float(d["drop_from"])
                except (TypeError, ValueError):
                    pass
            if "drop_to" in d and math.isnan(w.crash_floor):
                try:
                    w.crash_floor = float(d["drop_to"])
                except (TypeError, ValueError):
                    pass
            if "drop_threshold" in d and crash_drop is None:
                # Prefer explicit crash_drop; only fall back if unset
                try:
                    thr = float(d["drop_threshold"])
                    # Rollback uses 0.15 — ignore that for crash logs if we have crash_drop later
                    if "crash_drop" not in d:
                        w.crash_drop = thr
                except (TypeError, ValueError):
                    pass
            if "baseline_window" in d and d["baseline_window"] is not None:
                try:
                    w.baseline_window = max(2, int(d["baseline_window"]))
                except (TypeError, ValueError):
                    pass
            w.in_crash = bool(d.get("in_crash") or d.get("in_drop"))
            w.samples = int(d.get("samples") or 0)
            hist = d.get("history") or []
            w.history = [float(x) for x in hist if x is not None][-32:]
        if crash_drop is not None:
            w.crash_drop = float(crash_drop)
        return w

    def _rolling_baseline(self) -> float:
        """Mean of recent history (pre-append of current sample)."""
        win = max(2, int(self.baseline_window))
        hist = self.history[-win:]
        if not hist:
            return float("nan")
        return sum(hist) / len(hist)

    def observe(self, metrics: dict[str, Any] | None) -> list[str]:
        """Update state; return loud log lines (may be empty)."""
        logs: list[str] = []
        r = _f(metrics, "Average Step Reward", "avg_reward")
        if math.isnan(r):
            return logs

        baseline = self._rolling_baseline()
        if math.isnan(baseline) and not math.isnan(self.last_avg):
            baseline = self.last_avg
        if not math.isnan(baseline) and abs(baseline) > 1e-12:
            self.last_delta_pct = 100.0 * (r - baseline) / abs(baseline)
        elif not math.isnan(self.last_avg) and abs(self.last_avg) > 1e-12:
            self.last_delta_pct = 100.0 * (r - self.last_avg) / abs(self.last_avg)
        else:
            self.last_delta_pct = 0.0

        self.baseline = baseline if not math.isnan(baseline) else self.last_avg
        self.last_avg = r
        self.samples += 1
        self.history.append(r)
        self.history = self.history[-32:]

        if math.isnan(baseline):
            return logs

        thr = max(0.15, min(0.50, float(self.crash_drop)))
        # Prefer mid-band default when someone passes rollback's 0.15 by mistake
        if thr < 0.25:
            thr = max(thr, 0.25)

        if (not self.in_crash) and baseline > 0 and r < baseline * (1.0 - thr):
            pct = 100.0 * (baseline - r) / abs(baseline)
            self.in_crash = True
            self.crash_from = baseline
            self.crash_floor = r
            logs.append(
                f"[AutoTrainer] REWARD CRASH: {_fmt_reward(baseline)} -> {_fmt_reward(r)} "
                f"(-{pct:.0f}%)"
            )
        elif self.in_crash:
            crash_from = self.crash_from if not math.isnan(self.crash_from) else baseline
            crash_floor = self.crash_floor if not math.isnan(self.crash_floor) else baseline
            if r < crash_floor:
                self.crash_floor = r
                crash_floor = r
            regained = crash_from - crash_floor
            target = crash_floor + regained * float(self.recover_fraction)
            # Also accept climbing back near baseline (≥70% of pre-crash baseline)
            near_baseline = r >= crash_from * 0.70
            bounced = (
                (r >= target or near_baseline)
                and r > crash_floor
            )
            if bounced:
                logs.append(
                    f"[AutoTrainer] REWARD RECOVERED: {_fmt_reward(crash_floor)} -> {_fmt_reward(r)}"
                )
                self.in_crash = False
                self.crash_from = float("nan")
                self.crash_floor = float("nan")

        return logs


def observe_reward_swing(
    state: dict[str, Any],
    metrics: dict[str, Any] | None,
    *,
    crash_drop: float = _DEFAULT_CRASH_DROP,
    drop_threshold: float | None = None,
) -> list[str]:
    """Convenience: load/save RewardWatch on state['reward_monitor'] and print logs.

    ``drop_threshold`` is accepted as an alias for ``crash_drop`` (older call sites).
    """
    thr = float(crash_drop if drop_threshold is None else drop_threshold)
    # Don't let rollback's 0.15 silently become the crash threshold
    if thr < 0.25:
        thr = _DEFAULT_CRASH_DROP
    watch = RewardWatch.from_dict(
        state.get("reward_monitor") or state.get("reward_watch"),
        crash_drop=thr,
    )
    logs = watch.observe(metrics)
    for line in logs:
        print(line)
    state["reward_monitor"] = watch.to_dict()
    return logs


def reward_delta_pct_from_state(state: dict[str, Any] | None) -> float:
    mon = (state or {}).get("reward_monitor") or {}
    try:
        v = float(mon.get("last_delta_pct"))
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except (TypeError, ValueError):
        return float("nan")
