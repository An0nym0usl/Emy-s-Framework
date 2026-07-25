"""Teachers after warmup — expert sparring + true soft BC distill when possible.

Paths:
  1) **BC guiding (true mid-run distill):** when H stable + warmup + not red,
     point ``guiding_policy_path`` at a **same-arch** teacher under the **active
     checkpoint box** next to the running exe (watch_dir.parent). C++
     ``SetSoftDistillTeacher`` enables PPO guiding loss. Light ``distill_coef``.
  2) **Expert sparring:** NextoTled / Requiem / Necto via opponent pool weights
     (different obs sizes — cannot load as guiding; sparring only).
  3) OFF during entropy death / red.
  4) **From-scratch / empty box:** soft distill stays OFF until the local
     same-box folder has at least one numbered (or best_skill) checkpoint.
     Never cross-path to another install tree (e.g. a CUDA mirror vs this repo).

Note: Requiem/NextoTled with mismatched obs ≠ mid-run StartTransferLearn BC.
Same-arch guiding = real TransferLearn-style distill inside PPO. Full
``--transfer-discrete`` boot path remains separate.

Opt-out: teachers.enabled: false  or  GIGA_NO_TEACHERS=1
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from .full_control import warmup_steps


def teachers_enabled(cfg: dict[str, Any] | None = None) -> bool:
    if os.environ.get("GIGA_NO_TEACHERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    t = (cfg or {}).get("teachers") or {}
    return bool(t.get("enabled", True))


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


def _h_stable(metrics: dict[str, Any] | None, cfg: dict[str, Any] | None) -> bool:
    t = (cfg or {}).get("teachers") or {}
    safety = (cfg or {}).get("safety") or {}
    min_h = float(t.get("min_entropy", safety.get("entropy_recover_ok", 0.18)))
    h = _f(metrics, "Policy Entropy", "policy_entropy")
    if math.isnan(h):
        return False
    return h >= min_h


def _is_ckpt(p: Path) -> bool:
    return (p / "POLICY.lt").exists() and (p / "SHARED_HEAD.lt").exists()


def _latest_numbered(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    if _is_ckpt(folder):
        return folder
    best_ts, best = -1, None
    try:
        for child in folder.iterdir():
            if child.is_dir() and child.name.isdigit() and _is_ckpt(child):
                ts = int(child.name)
                if ts > best_ts:
                    best_ts, best = ts, child
    except OSError:
        return None
    return best


def active_checkpoint_box(watch_dir: Path | str | None) -> Path | None:
    """Checkpoint box next to the running exe (= watch_dir parent / checkpoints)."""
    if watch_dir is None:
        return None
    return Path(watch_dir).resolve().parent / "checkpoints"


def local_same_box_teacher_ready(
    watch_dir: Path | str | None,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """True when the active (local) checkpoint box has a same-arch ckpt to distill from."""
    box = active_checkpoint_box(watch_dir)
    if box is None:
        return False
    if _is_ckpt(box):
        return True
    if _latest_numbered(box) is not None:
        return True
    bs = box / "best_skill"
    if _is_ckpt(bs) or _latest_numbered(bs) is not None:
        return True
    # Explicit config path only if it resolves under the same box
    t = (cfg or {}).get("teachers") or {}
    for key in ("best_skill_path", "bc_teacher_path"):
        raw = t.get(key)
        if not raw:
            continue
        p = Path(str(raw))
        try:
            p.resolve().relative_to(box.resolve())
        except (OSError, ValueError):
            continue
        if _is_ckpt(p) or _latest_numbered(p) is not None:
            return True
    return False


def resolve_best_skill_teacher(
    watch_dir: Path | None,
    cfg: dict[str, Any] | None = None,
) -> Path | None:
    """Prefer local ``checkpoints/best_skill`` next to the running exe."""
    t = (cfg or {}).get("teachers") or {}
    if t.get("best_skill_teacher") is False:
        return None
    if os.environ.get("GIGA_NO_BEST_SKILL_TEACHER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None

    box = active_checkpoint_box(watch_dir)
    candidates: list[Path] = []
    if t.get("best_skill_path"):
        p = Path(str(t["best_skill_path"]))
        # Only accept explicit path when it lives under the active box (or box missing)
        if box is None:
            candidates.append(p)
        else:
            try:
                p.resolve().relative_to(box.resolve())
                candidates.append(p)
            except (OSError, ValueError):
                pass
    if box is not None:
        candidates.append(box / "best_skill")

    for p in candidates:
        if _is_ckpt(p):
            return p
        latest = _latest_numbered(p)
        if latest:
            return latest
    return None


def _want_best_skill_teacher(
    *,
    cfg: dict[str, Any] | None,
    zone: str,
    elo_signal: dict[str, float] | None,
    metrics: dict[str, Any] | None,
) -> bool:
    """Continuous teacher when Elo drops or green+H stable (default ON)."""
    t = (cfg or {}).get("teachers") or {}
    if t.get("best_skill_teacher") is False:
        return False
    # Always prefer best_skill when present unless explicitly continuous=false
    if t.get("best_skill_continuous", True):
        z = str(zone).lower()
        if z == "red":
            return False
        sig = elo_signal or {}
        elo_delta = sig.get("elo_delta")
        drop_thr = float(t.get("best_skill_elo_drop", 3.0))
        if elo_delta is not None:
            try:
                if float(elo_delta) <= -drop_thr:
                    return True
            except (TypeError, ValueError):
                pass
        # Green stable: H ok → use best_skill as soft teacher
        if z == "green" and _h_stable(metrics, cfg):
            return True
        # Default: if best_skill exists, still use it lightly after warmup
        return bool(t.get("best_skill_always", True))
    return False


def resolve_bc_teacher_path(
    watch_dir: Path | None,
    cfg: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    *,
    prefer_best_skill: bool = False,
) -> Path | None:
    """
    Same-arch teacher for guiding BC — **local active box only**.

    Prefer: best_skill → explicit config path (same-box) → numbered ckpts /
    policy_versions / old_versions under exe dir. Never fall back to another
    install tree (mirror vs this repo).
    """
    t = (cfg or {}).get("teachers") or {}
    box = active_checkpoint_box(watch_dir)
    if box is None:
        return None

    if prefer_best_skill:
        bs = resolve_best_skill_teacher(watch_dir, cfg)
        if bs is not None:
            return bs

    if t.get("bc_teacher_path"):
        p = Path(str(t["bc_teacher_path"]))
        try:
            p.resolve().relative_to(box.resolve())
        except (OSError, ValueError):
            p = None  # type: ignore[assignment]
        if p is not None:
            if _is_ckpt(p):
                return p
            latest = _latest_numbered(p)
            if latest:
                return latest

    # Explicit allowlist of same-arch roots under the active exe dir only
    exe_parent = Path(watch_dir).resolve().parent
    roots: list[Path] = []
    for key in ("bc_teacher_roots", "same_arch_teachers"):
        raw = t.get(key) or []
        for item in raw:
            p = Path(str(item))
            try:
                p.resolve().relative_to(exe_parent.resolve())
                roots.append(p)
            except (OSError, ValueError):
                continue

    roots.extend(
        [
            box / "best_skill",
            box,
            exe_parent / "checkpoints_reconstructed",
            exe_parent / "policy_versions",
            exe_parent / "old_versions",
        ]
    )

    # Prefer an older numbered ckpt (not the absolute latest) as distill target
    for root in roots:
        if not root.exists():
            continue
        if _is_ckpt(root):
            return root
        if root.name.isdigit() and _is_ckpt(root):
            return root
        numbered: list[Path] = []
        try:
            for child in root.iterdir():
                if child.is_dir() and child.name.isdigit() and _is_ckpt(child):
                    numbered.append(child)
        except OSError:
            continue
        if not numbered:
            continue
        numbered.sort(key=lambda p: int(p.name), reverse=True)
        if len(numbered) >= 2:
            return numbered[1]
        return numbered[0]

    _ = status
    return None


def teachers_patch(
    *,
    timesteps: int,
    metrics: dict[str, Any] | None,
    zone: str,
    recovering: bool,
    cfg: dict[str, Any] | None = None,
    current: dict[str, Any] | None = None,
    watch_dir: Path | str | None = None,
    status: dict[str, Any] | None = None,
    elo_signal: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Expert pressure + optional same-arch BC distill flags.

    Soft distill requires local same-box checkpoints. Empty / from-scratch boxes
    get sparring-only (or warmup off) — never cross-install guiding.
    """
    if not teachers_enabled(cfg):
        return {}

    t = (cfg or {}).get("teachers") or {}
    auto = (cfg or {}).get("ssl_autonomy") or {}
    warm = int(t.get("warmup_steps", auto.get("sparring_warmup_steps", warmup_steps(cfg))))
    wd = Path(watch_dir) if watch_dir else None
    local_ready = local_same_box_teacher_ready(wd, cfg)

    # Entropy death / red → force teachers OFF
    if recovering or str(zone).lower() == "red" or not _h_stable(metrics, cfg):
        if timesteps < warm:
            return {}
        return {
            "teachers_active": False,
            "teachers_mode": "off_unstable",
            "distill_soft": False,
            "distill_off": True,
            "best_skill_teacher": False,
            "opponent_weight_nexto": 0.0,
            "opponent_weight_nexto_tled": 0.0,
            "opponent_weight_necto": 0.0,
            "opponent_weight_requiem": 0.0,
            "note": "teachers:off_unstable",
        }

    if timesteps < warm:
        return {
            "teachers_active": False,
            "teachers_mode": "warmup",
            "distill_soft": False,
            "distill_off": True,
            "best_skill_teacher": False,
            "note": "teachers:warmup",
        }

    # From-scratch / empty local box: never soft-distill from another tree
    if not local_ready:
        return {
            "teachers_active": False,
            "teachers_mode": "await_local_ckpt",
            "distill_soft": False,
            "distill_off": True,
            "best_skill_teacher": False,
            "note": "teachers:soft_distill_off(no local same-box ckpt)",
        }

    pool = float(t.get("pool_chance", 0.06))
    old = float(t.get("old_chance_floor", 0.10))
    intensity = float(t.get("intensity", 0.55))
    weights = dict(t.get("expert_weights") or {})
    defaults = {
        "nexto": 0.85,
        "nexto_tled": 0.90,
        "necto": 0.70,
        "requiem": 1.0,
    }
    for k, v in defaults.items():
        weights.setdefault(k, v)

    cur = current or {}
    cur_pool = float(cur.get("opponent_pool_chance", 0.0) or 0.0)
    cur_old = float(cur.get("train_against_old_chance", 0.0) or 0.0)

    # Soft BC schedule — light so SPS doesn't die
    bc_enabled = bool(t.get("bc_distill", True))
    distill_coef = float(t.get("distill_coef", 0.05)) * intensity
    # Cap BC strength
    distill_coef = max(0.0, min(0.12, distill_coef))

    prefer_bs = _want_best_skill_teacher(
        cfg=cfg, zone=zone, elo_signal=elo_signal, metrics=metrics
    )
    # Mildly bump distill when Elo dropped and best_skill is the teacher
    if prefer_bs:
        bump = float(t.get("best_skill_distill_bump", 1.25))
        distill_coef = max(0.0, min(0.12, distill_coef * bump))

    bc_path = (
        resolve_bc_teacher_path(wd, cfg, status, prefer_best_skill=prefer_bs)
        if bc_enabled and distill_coef > 0
        else None
    )
    used_best_skill = bool(
        prefer_bs
        and bc_path is not None
        and ("best_skill" in str(bc_path).replace("\\", "/"))
    )
    mode = "light_sparring"
    out: dict[str, Any] = {
        "teachers_active": True,
        "opponent_pool_chance": max(cur_pool, pool * intensity),
        "train_against_old_chance": max(cur_old, old),
        "ssl_guide_post_apex": True,
        "best_skill_teacher": used_best_skill,
    }
    for name, w in weights.items():
        out[f"opponent_weight_{name}"] = float(w) * intensity

    if bc_path is not None:
        mode = "best_skill_distill+sparring" if used_best_skill else "bc_guiding+sparring"
        out["distill_soft"] = True
        out["teachers_bc"] = True
        out["distill_coef"] = distill_coef
        out["guiding_strength"] = distill_coef
        out["guiding_policy_path"] = str(bc_path)
        out["distill_teacher_path"] = str(bc_path)
        tag = "best_skill" if used_best_skill else bc_path.name
        out["note"] = f"teachers:bc_guiding:{tag}"
    else:
        # Sparring-only for mismatched experts; still expose soft flag for logs
        mode = "expert_sparring"
        out["distill_soft"] = False
        out["distill_off"] = True
        out["note"] = "teachers:expert_sparring(no same-arch BC teacher yet)"

    out["teachers_mode"] = mode
    return out


def merge_teachers_patch(
    base: dict[str, Any],
    teachers: dict[str, Any] | None,
    *,
    recovering: bool,
) -> dict[str, Any]:
    """Merge teachers into overrides; never override recovery locks."""
    if not teachers:
        return dict(base)
    out = dict(base)
    if recovering or out.get("entropy_death_recovery") or out.get("hard_recovery"):
        for k, v in teachers.items():
            if str(k).startswith("opponent_weight_") and float(v or 0) == 0.0:
                out[k] = 0.0
        out["teachers_active"] = False
        out["teachers_mode"] = "off_recovery"
        out["distill_soft"] = False
        out["distill_off"] = True
        return out

    for k, v in teachers.items():
        if k == "note":
            prev = str(out.get("note") or "")
            out["note"] = f"{prev}; {v}" if prev else str(v)
        elif k == "reward_weights":
            continue
        elif k in ("opponent_pool_chance", "train_against_old_chance"):
            try:
                out[k] = max(float(out.get(k) or 0), float(v))
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out
