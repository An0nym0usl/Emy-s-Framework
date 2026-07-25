"""
True meta-model: small MLP mapping state features → action logits.

Trains online from scored outcomes (REINFORCE-style advantage update).
Uses PyTorch when available; falls back to a NumPy MLP. If both fail or
``GIGA_NO_META_MLP=1``, callers fall back to the contextual bandit.

Tiny online policy over discrete AutoTrainer
interventions — not overnight SSL magic, not a second full PPO agent.
Safety veto / entropy_death recovery ALWAYS win over MLP suggestions.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import read_json, write_json_atomic
from .meta_brain import ACTION_NAMES

WEIGHTS_TORCH = "meta_mlp.pt"
WEIGHTS_NUMPY = "meta_mlp.npz"
META_JSON = "meta_mlp_state.json"

# Fixed feature layout (order matters for persisted weights)
FEATURE_NAMES: tuple[str, ...] = (
    "entropy_norm",
    "reward",
    "reward_delta_pct",
    "reward_var",
    "sps_norm",
    "zone_red",
    "zone_yellow",
    "zone_green",
    "elo_norm",
    "elo_delta_norm",
    "skill_norm",
    "phase_norm",
    "timesteps_norm",
    "viz_touch",
    "viz_air",
    "viz_no_touch",
    "viz_hack_risk",
    "viz_reward",
    "green_h_streak_norm",
    "unmute_level",
    "elo_flat",
    "pending_age_norm",
    "bias",
)

FEATURE_DIM = len(FEATURE_NAMES)
N_ACTIONS = len(ACTION_NAMES)
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_NAMES)}


def meta_mlp_enabled(cfg: dict[str, Any] | None = None) -> bool:
    if os.environ.get("GIGA_NO_META_MLP", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    if os.environ.get("GIGA_AT_LEGACY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    mm = (cfg or {}).get("meta_mlp") or {}
    if mm.get("enabled") is False:
        return False
    intel = (cfg or {}).get("intelligence") or {}
    # Default ON when intelligence layer is on
    if intel.get("enabled", True) and mm.get("enabled") is None:
        return True
    return bool(mm.get("enabled", True))


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _f(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def extract_features(
    *,
    metrics: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    zone: str = "unk",
    reward_delta_pct: float | None = None,
    reward_variance: float = 0.0,
    elo_signal: dict[str, float] | None = None,
    viz_state: dict[str, Any] | None = None,
    green_h_streak: int = 0,
    unmute_level: float = 0.0,
    elo_flat_evals: int = 0,
    pending_age_cycles: int = 0,
    score_horizon: int = 16,
) -> np.ndarray:
    """Build a fixed-size float32 feature vector for the meta MLP."""
    m = metrics or {}
    st = status or {}
    sig = elo_signal or {}
    viz = viz_state or {}

    h = _f(m.get("Policy Entropy", m.get("policy_entropy")), 0.5)
    reward = _f(m.get("Average Step Reward", m.get("avg_reward")), 0.0)
    sps = _f(m.get("Overall Steps/Second", m.get("SPS")), 0.0)
    z = str(zone or "unk").lower()
    elo = _f(sig.get("elo"), float("nan"))
    elo_d = _f(sig.get("elo_delta"), 0.0)
    skill = _f(sig.get("skill"), float("nan"))
    if math.isnan(skill):
        skill = elo
    phase = _f(st.get("curriculum_phase", st.get("phase")), 0.0)
    ts = _f(st.get("total_timesteps"), 0.0)

    soft = list(viz.get("soft_features") or [])
    viz_touch = _f(viz.get("touch"), 0.0)
    viz_air = _f(viz.get("air"), 0.0)
    viz_r = _f(viz.get("viz_reward", viz.get("reward")), 0.0)
    viz_no = 1.0 if ("viz_no_touch" in soft or str(viz.get("tag") or "") in ("no_contact", "viz_idle")) else 0.0
    viz_hack = 1.0 if (
        "viz_reward_hack_risk" in soft
        or str(viz.get("viz_hint") or "") == "reward_hack_watch"
    ) else 0.0

    delta = _f(reward_delta_pct, 0.0)
    # squash large % swings
    delta_n = max(-1.0, min(1.0, delta / 50.0))

    feats = {
        "entropy_norm": _clip01(h / 2.5),
        "reward": max(-1.0, min(1.0, reward)),
        "reward_delta_pct": delta_n,
        "reward_var": _clip01(float(reward_variance) * 8.0),
        "sps_norm": _clip01(sps / 400_000.0),
        "zone_red": 1.0 if z == "red" else 0.0,
        "zone_yellow": 1.0 if z == "yellow" else 0.0,
        "zone_green": 1.0 if z == "green" else 0.0,
        "elo_norm": _clip01((_f(elo, 1000.0) - 800.0) / 800.0),
        "elo_delta_norm": max(-1.0, min(1.0, elo_d / 30.0)),
        "skill_norm": _clip01((_f(skill, 1000.0) - 800.0) / 800.0),
        "phase_norm": _clip01(phase / 5.0),
        "timesteps_norm": _clip01(ts / 2_000_000_000.0),
        "viz_touch": _clip01(viz_touch),
        "viz_air": _clip01(viz_air),
        "viz_no_touch": viz_no,
        "viz_hack_risk": viz_hack,
        "viz_reward": max(-1.0, min(1.0, viz_r)),
        "green_h_streak_norm": _clip01(green_h_streak / 10.0),
        "unmute_level": _clip01(float(unmute_level)),
        "elo_flat": _clip01(elo_flat_evals / 4.0),
        "pending_age_norm": _clip01(pending_age_cycles / max(1, score_horizon)),
        "bias": 1.0,
    }
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float32)


def feature_dict(vec: np.ndarray) -> dict[str, float]:
    """Named view of a feature vector (dashboard / coach)."""
    out: dict[str, float] = {}
    for i, name in enumerate(FEATURE_NAMES):
        if i < len(vec):
            out[name] = float(vec[i])
    return out


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(1e-3, float(temperature))
    x = logits.astype(np.float64) / t
    x = x - np.max(x)
    e = np.exp(x)
    return (e / (e.sum() + 1e-12)).astype(np.float32)


class _NumpyMLP:
    """Two-layer MLP: in → hidden → actions (ReLU)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        scale1 = 1.0 / math.sqrt(in_dim)
        scale2 = 1.0 / math.sqrt(hidden)
        self.w1 = rng.randn(in_dim, hidden).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.randn(hidden, out_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def train_reinforce(
        self,
        x: np.ndarray,
        action_idx: int,
        advantage: float,
        lr: float,
    ) -> float:
        """One REINFORCE step; returns loss proxy."""
        logits = self.forward(x)
        probs = _softmax(logits)
        # dL/dlogit ≈ (probs - onehot) * (-adv) for maximize E[adv*log p]
        grad_logits = probs.copy()
        grad_logits[action_idx] -= 1.0
        grad_logits *= -float(advantage)

        h = np.maximum(0.0, x @ self.w1 + self.b1)
        # backprop
        gw2 = np.outer(h, grad_logits)
        gb2 = grad_logits
        gh = self.w2 @ grad_logits
        gh = gh * (h > 0).astype(np.float32)
        gw1 = np.outer(x, gh)
        gb1 = gh

        self.w2 -= lr * gw2.astype(np.float32)
        self.b2 -= lr * gb2.astype(np.float32)
        self.w1 -= lr * gw1.astype(np.float32)
        self.b1 -= lr * gb1.astype(np.float32)
        # entropy bonus proxy
        return float(-math.log(max(1e-8, float(probs[action_idx]))) * abs(advantage))

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.w1 = np.asarray(d["w1"], dtype=np.float32)
        self.b1 = np.asarray(d["b1"], dtype=np.float32)
        self.w2 = np.asarray(d["w2"], dtype=np.float32)
        self.b2 = np.asarray(d["b2"], dtype=np.float32)


class _TorchMLP:
    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 42) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = float(lr)

    def forward(self, x: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            logits = self.net(t.tensor(x, dtype=t.float32).unsqueeze(0))
        return logits.squeeze(0).cpu().numpy().astype(np.float32)

    def train_reinforce(
        self,
        x: np.ndarray,
        action_idx: int,
        advantage: float,
        lr: float,
    ) -> float:
        t = self.torch
        self.set_lr(lr)
        self.opt.zero_grad()
        logits = self.net(t.tensor(x, dtype=t.float32).unsqueeze(0)).squeeze(0)
        log_probs = t.nn.functional.log_softmax(logits, dim=-1)
        loss = -log_probs[action_idx] * float(advantage)
        # mild entropy bonus so we don't collapse
        probs = t.softmax(logits, dim=-1)
        ent = -(probs * t.log(probs + 1e-8)).sum()
        loss = loss - 0.01 * ent
        loss.backward()
        self.opt.step()
        return float(loss.detach().cpu().item())

    def save(self, path: Path) -> None:
        self.torch.save(self.net.state_dict(), path)

    def load(self, path: Path) -> None:
        sd = self.torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(sd)


@dataclass
class MetaMLPState:
    backend: str = "none"
    updates: int = 0
    last_action: str = "hold"
    last_loss: float = 0.0
    last_features: dict[str, float] = field(default_factory=dict)
    last_logits: dict[str, float] = field(default_factory=dict)
    last_probs: dict[str, float] = field(default_factory=dict)
    fallback_bandit: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "kind": "meta_mlp",
            "updated": datetime.now(timezone.utc).isoformat(),
            "backend": self.backend,
            "updates": self.updates,
            "last_action": self.last_action,
            "last_loss": self.last_loss,
            "last_features": dict(self.last_features),
            "last_logits": dict(self.last_logits),
            "last_probs": dict(self.last_probs),
            "fallback_bandit": self.fallback_bandit,
            "feature_names": list(FEATURE_NAMES),
            "actions": list(ACTION_NAMES),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> MetaMLPState:
        if not d:
            return cls()
        return cls(
            backend=str(d.get("backend") or "none"),
            updates=int(d.get("updates") or 0),
            last_action=str(d.get("last_action") or "hold"),
            last_loss=float(d.get("last_loss") or 0.0),
            last_features={str(k): float(v) for k, v in (d.get("last_features") or {}).items()},
            last_logits={str(k): float(v) for k, v in (d.get("last_logits") or {}).items()},
            last_probs={str(k): float(v) for k, v in (d.get("last_probs") or {}).items()},
            fallback_bandit=int(d.get("fallback_bandit") or 0),
        )


class MetaMLPController:
    """Online meta MLP over ACTION_NAMES with bandit fallback signaling."""

    def __init__(self, watch_dir: Path, cfg: dict[str, Any] | None = None) -> None:
        self.watch_dir = Path(watch_dir)
        self.cfg = cfg or {}
        self.state_path = self.watch_dir / META_JSON
        self.state = MetaMLPState.from_dict(read_json(self.state_path))
        mm = self.cfg.get("meta_mlp") or {}
        self.hidden = int(mm.get("hidden", 32))
        self.lr = float(mm.get("learning_rate", 1e-3))
        self.temperature = float(mm.get("temperature", 0.8))
        self.seed = int(mm.get("seed", 42))
        self._rng = random.Random(self.seed + self.state.updates)
        self._model: Any = None
        self._backend = "none"
        self._init_model()

    def _mm(self) -> dict[str, Any]:
        return self.cfg.get("meta_mlp") or {}

    def _init_model(self) -> None:
        prefer = str(self._mm().get("backend", "auto")).lower()
        use_torch = prefer in ("auto", "torch", "pytorch") and _torch_available()
        if prefer == "numpy":
            use_torch = False
        if use_torch:
            try:
                self._model = _TorchMLP(FEATURE_DIM, self.hidden, N_ACTIONS, seed=self.seed)
                pt = self.watch_dir / WEIGHTS_TORCH
                if pt.exists():
                    self._model.load(pt)
                self._backend = "torch"
            except Exception:
                use_torch = False
        if not use_torch:
            self._model = _NumpyMLP(FEATURE_DIM, self.hidden, N_ACTIONS, seed=self.seed)
            npz = self.watch_dir / WEIGHTS_NUMPY
            if npz.exists():
                try:
                    data = np.load(npz)
                    self._model.load_state_dict({k: data[k] for k in ("w1", "b1", "w2", "b2")})
                except Exception:
                    pass
            self._backend = "numpy"
        self.state.backend = self._backend

    def save(self) -> None:
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.state_path, self.state.to_dict())
        try:
            if self._backend == "torch" and self._model is not None:
                self._model.save(self.watch_dir / WEIGHTS_TORCH)
            elif self._backend == "numpy" and self._model is not None:
                np.savez(self.watch_dir / WEIGHTS_NUMPY, **self._model.state_dict())
        except OSError:
            pass

    def logits(self, features: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return np.asarray(self._model.forward(features), dtype=np.float32)

    def probs(self, features: np.ndarray, *, temperature: float | None = None) -> np.ndarray:
        return _softmax(self.logits(features), temperature or self.temperature)

    def select_action(
        self,
        features: np.ndarray,
        *,
        allowed: list[str] | None = None,
        epsilon: float = 0.05,
        force_hold: bool = False,
    ) -> tuple[str, dict[str, float], dict[str, float]]:
        """Return (action, probs_dict, logits_dict)."""
        if force_hold:
            action = "hold"
            logits = self.logits(features)
            probs = _softmax(logits, self.temperature)
            pd = {a: float(probs[i]) for i, a in enumerate(ACTION_NAMES)}
            ld = {a: float(logits[i]) for i, a in enumerate(ACTION_NAMES)}
            self.state.last_action = action
            self.state.last_features = feature_dict(features)
            self.state.last_probs = pd
            self.state.last_logits = ld
            return action, pd, ld

        allow = allowed or list(ACTION_NAMES)
        logits = self.logits(features)
        # Mask disallowed
        masked = logits.copy()
        for i, a in enumerate(ACTION_NAMES):
            if a not in allow:
                masked[i] = -1e9
        probs = _softmax(masked, self.temperature)
        if self._rng.random() < max(0.0, float(epsilon)):
            action = self._rng.choice(allow)
        else:
            # sample from masked softmax
            r = self._rng.random()
            cum = 0.0
            action = allow[0]
            for i, a in enumerate(ACTION_NAMES):
                if a not in allow:
                    continue
                cum += float(probs[i])
                if r <= cum:
                    action = a
                    break
            else:
                action = ACTION_NAMES[int(np.argmax(probs))]
                if action not in allow:
                    action = allow[0]

        pd = {a: float(probs[i]) for i, a in enumerate(ACTION_NAMES)}
        ld = {a: float(logits[i]) for i, a in enumerate(ACTION_NAMES)}
        self.state.last_action = action
        self.state.last_features = feature_dict(features)
        self.state.last_probs = pd
        self.state.last_logits = ld
        return action, pd, ld

    def train_from_outcome(
        self,
        features: np.ndarray,
        action: str,
        meta_reward: float,
    ) -> float:
        """Online REINFORCE update from long-horizon meta reward."""
        if action not in ACTION_INDEX:
            return 0.0
        idx = ACTION_INDEX[action]
        # Center advantage lightly
        adv = max(-2.0, min(2.0, float(meta_reward)))
        loss = float(self._model.train_reinforce(features, idx, adv, self.lr))
        self.state.updates += 1
        self.state.last_loss = loss
        self.state.last_features = feature_dict(features)
        self.save()
        return loss

    def dashboard_blob(self) -> dict[str, Any]:
        return {
            "backend": self.state.backend,
            "updates": self.state.updates,
            "last_action": self.state.last_action,
            "last_loss": self.state.last_loss,
            "features": dict(self.state.last_features),
            "logits": dict(self.state.last_logits),
            "probs": dict(self.state.last_probs),
            "fallback_bandit": self.state.fallback_bandit,
            "feature_names": list(FEATURE_NAMES),
        }


def format_mlp_banner(
    *,
    action: str,
    backend: str,
    top_probs: list[tuple[str, float]],
    features_highlight: dict[str, float] | None = None,
) -> str:
    lines = [
        "========================================",
        "[AutoTrainer MetaMLP] Select",
        f"  Backend: {backend}",
        f"  Action: {action}",
        "  Top probs: "
        + ", ".join(f"{a}={p:.2f}" for a, p in top_probs[:4]),
    ]
    if features_highlight:
        bits = [f"{k}={v:.2f}" for k, v in list(features_highlight.items())[:6]]
        lines.append("  Features: " + ", ".join(bits))
    lines.append("========================================")
    return "\n".join(lines)
