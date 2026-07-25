"""Periodic Elo / skill eval — interval only (never every training iter).

Runs a cheap multi-seed tournament (tools/eval_harness scoring) against
old-version checkpoints + Nexto / Requiem / Necto when present. Results feed
the meta-bandit and best-skill checkpoint tracker.

Opt-out: periodic_eval.enabled: false  or  GIGA_NO_PERIODIC_EVAL=1
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json_atomic


STATE_KEY = "periodic_eval"


def periodic_eval_enabled(cfg: dict[str, Any] | None = None) -> bool:
    if os.environ.get("GIGA_NO_PERIODIC_EVAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    pe = (cfg or {}).get("periodic_eval") or {}
    return bool(pe.get("enabled", True))


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


def live_skill_from_metrics(metrics: dict[str, Any] | None) -> float | None:
    """Prefer C++ in-sim Eval/Elo or skill-tracker Rating/* when present."""
    m = metrics or {}
    for k in (
        "Eval/Elo",
        "Rating/1v1",
        "Rating/2v2",
        "Skill Rating",
        "skill_rating",
        "Skill",
    ):
        v = _f(m, k)
        if not math.isnan(v):
            return v
    # Any Rating/* key
    for k, v in m.items():
        if str(k).startswith("Rating/"):
            try:
                fv = float(v)
                if not math.isnan(fv):
                    return fv
            except (TypeError, ValueError):
                pass
    return None


def live_eval_signal(metrics: dict[str, Any] | None) -> dict[str, float]:
    """Extract in-sim Eval/* metrics for meta / best_skill blending."""
    m = metrics or {}
    out: dict[str, float] = {}
    mapping = (
        ("Eval/Elo", "elo"),
        ("Eval/EloDelta", "elo_delta"),
        ("Eval/WinRate", "win_rate"),
        ("Eval/AvgReward", "avg_reward"),
    )
    for src, dst in mapping:
        v = _f(m, src)
        if not math.isnan(v):
            out[dst] = v
    return out


def _resolve_exe_dir(watch_dir: Path, cfg: dict[str, Any] | None) -> Path:
    pe = (cfg or {}).get("periodic_eval") or {}
    if pe.get("exe_dir"):
        return Path(str(pe["exe_dir"]))
    # watch_dir is typically build/Release/autotrainer
    parent = Path(watch_dir).resolve().parent
    if (parent / "GigaLearnBot.exe").exists() or (parent / "checkpoints").exists():
        return parent
    env_release = os.environ.get("GIGA_RELEASE_DIR")
    if env_release:
        p = Path(env_release)
        if p.exists():
            return p
    mirror = os.environ.get("GIGA_CUDA_MIRROR")
    if mirror:
        p = Path(mirror) / "build" / "Release"
        if p.exists():
            return p
    return parent


def _import_harness():
    """Lazy-import eval_harness from tools/."""
    roots = [
        Path(__file__).resolve().parents[2] / "tools",
    ]
    env_root = os.environ.get("GIGA_REPO_ROOT")
    if env_root:
        roots.append(Path(env_root) / "tools")
    mirror = os.environ.get("GIGA_CUDA_MIRROR")
    if mirror:
        roots.append(Path(mirror) / "tools")
    for r in roots:
        if (r / "eval_harness.py").exists():
            s = str(r)
            if s not in sys.path:
                sys.path.insert(0, s)
            import eval_harness as eh  # type: ignore

            return eh
    return None


def due_for_eval(
    state: dict[str, Any],
    timesteps: int,
    *,
    cfg: dict[str, Any] | None = None,
    cycle: int = 0,
) -> bool:
    if not periodic_eval_enabled(cfg):
        return False
    pe = (cfg or {}).get("periodic_eval") or {}
    interval_steps = int(pe.get("interval_steps", 25_000_000))
    every_cycles = int(pe.get("every_autotrainer_cycles", 0) or 0)
    st = dict(state.get(STATE_KEY) or {})
    last_ts = int(st.get("last_eval_timesteps") or 0)
    last_cycle = int(st.get("last_eval_cycle") or 0)
    if timesteps - last_ts >= interval_steps:
        return True
    if every_cycles > 0 and cycle > 0 and (cycle - last_cycle) >= every_cycles:
        return True
    return False


def run_periodic_eval(
    watch_dir: Path,
    status: dict[str, Any],
    state: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run cheap multi-seed tournament; write eval_report_periodic.json + update state.

    Prefer live in-sim Eval/* / Rating/* blended with harness Elo. Supports 5–9 seeds.
    Never blocks training SPS path — AutoTrainer-only, interval gated.
    """
    pe = (cfg or {}).get("periodic_eval") or {}
    watch_dir = Path(watch_dir)
    exe_dir = _resolve_exe_dir(watch_dir, cfg)
    metrics = status.get("last_metrics") or {}
    ts = int(status.get("total_timesteps") or 0)
    live = live_skill_from_metrics(metrics)
    in_sim = live_eval_signal(metrics)

    seeds_raw = pe.get("seeds") or [7, 11, 19, 23, 29, 31, 37]
    if isinstance(seeds_raw, str):
        seeds = [int(x.strip()) for x in seeds_raw.split(",") if x.strip()]
    else:
        seeds = [int(x) for x in seeds_raw]
    # Cap seed count for AutoTrainer latency (still multi-seed)
    max_seeds = int(pe.get("max_seeds", 9))
    seeds = seeds[: max(1, max_seeds)]

    eh = _import_harness()
    summary: dict[str, Any] = {
        "ok": False,
        "timesteps": ts,
        "live_skill": live,
        "in_sim_eval": in_sim or None,
        "elo": None,
        "elo_raw": None,
        "elo_std": None,
        "fitness": None,
        "champion_path": None,
        "n_candidates": 0,
        "n_seeds": len(seeds),
        "vs_experts": {},
        "mode": "harness_proxy",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Prefer fresh in-sim Elo when present (RocketSim arenas — not live RL client).
    if in_sim.get("elo") is not None and live is None:
        live = float(in_sim["elo"])
        summary["live_skill"] = live
    if in_sim.get("win_rate") is not None:
        summary["in_sim_win_rate"] = float(in_sim["win_rate"])
    if in_sim.get("avg_reward") is not None:
        summary["in_sim_avg_reward"] = float(in_sim["avg_reward"])
    if in_sim.get("elo_delta") is not None:
        summary["in_sim_elo_delta"] = float(in_sim["elo_delta"])

    if eh is None:
        summary["error"] = "eval_harness unavailable"
        if live is not None:
            summary["ok"] = True
            summary["elo"] = float(live)
            summary["mode"] = "live_skill_only"
        _persist(watch_dir, state, summary, ts, cfg)
        return summary

    try:
        ckpts = eh.discover(exe_dir)
        box = exe_dir / "checkpoints"
        latest_box = eh._latest(box) if box.exists() else None
        if latest_box and latest_box not in ckpts:
            ckpts.insert(0, latest_box)
        # Cap candidates for speed; keep local box + teachers first
        max_cand = int(pe.get("max_candidates", 12))
        if len(ckpts) > max_cand:
            ckpts = ckpts[:max_cand]

        if not ckpts:
            summary["error"] = "no checkpoints"
            if live is not None:
                summary["ok"] = True
                summary["elo"] = float(live)
                summary["mode"] = "live_skill_only"
            _persist(watch_dir, state, summary, ts, cfg)
            return summary

        per_seed: dict[str, list[float]] = {str(p): [] for p in ckpts}
        for seed in seeds:
            for p in ckpts:
                per_seed[str(p)].append(eh.score_checkpoint(p, seed))

        agents = []
        for p in ckpts:
            scores = per_seed[str(p)]
            mean = sum(scores) / len(scores)
            var = (
                sum((x - mean) ** 2 for x in scores) / max(1, len(scores) - 1)
                if len(scores) > 1
                else 0.0
            )
            agents.append(
                {
                    "id": p.name,
                    "path": str(p),
                    "fitness": mean,
                    "fitness_std": var ** 0.5,
                    "elo": 1000.0,
                    "wins": 0,
                    "losses": 0,
                }
            )
        # Multi-seed Elo: average rankings across tournament seeds
        elo_accum: dict[str, list[float]] = {a["path"]: [] for a in agents}
        for si, seed in enumerate(seeds):
            round_agents = []
            for a in agents:
                round_agents.append(
                    {
                        "id": a["id"],
                        "path": a["path"],
                        "fitness": per_seed[a["path"]][si],
                        "elo": 1000.0,
                        "wins": 0,
                        "losses": 0,
                    }
                )
            ranked_round = eh.elo_tournament(round_agents, seed=seed * 7919 + si)
            for a in ranked_round:
                elo_accum[a["path"]].append(float(a["elo"]))

        for a in agents:
            elos = elo_accum[a["path"]]
            a["elo"] = sum(elos) / len(elos)
            a["elo_std"] = (
                (sum((x - a["elo"]) ** 2 for x in elos) / max(1, len(elos) - 1)) ** 0.5
                if len(elos) > 1
                else 0.0
            )
        ranked = sorted(agents, key=lambda a: (a["elo"], a["fitness"]), reverse=True)

        our = None
        if latest_box is not None:
            for a in ranked:
                if Path(a["path"]).resolve() == latest_box.resolve():
                    our = a
                    break
        if our is None and ranked:
            our = ranked[0]

        vs: dict[str, float] = {}
        for a in ranked:
            name = str(a["id"]).lower()
            path_l = str(a["path"]).lower()
            tag = None
            if "nexto" in name or "nexto" in path_l:
                tag = "nexto"
            elif "requiem" in name or "requ" in path_l:
                tag = "requiem"
            elif "necto" in name or "necto" in path_l:
                tag = "necto"
            elif "old" in name or "version" in path_l:
                tag = "old_version"
            if tag and our is not None:
                vs[tag] = float(a["elo"]) - float(our["elo"])

        elo_raw = float(our["elo"]) if our else None
        elo = elo_raw
        # Prefer C++ in-sim Eval/Elo when present (RocketSim arenas). Harness is a
        # cheap proxy — weight live higher when Eval/* is available.
        has_in_sim = bool(in_sim.get("elo") is not None)
        default_blend = 0.70 if has_in_sim else 0.40
        blend = float(pe.get("live_skill_blend", default_blend))
        if live is not None and elo is not None:
            elo = (1.0 - blend) * elo + blend * float(live)
            summary["mode"] = "in_sim_blend" if has_in_sim else "harness_proxy"
        elif live is not None:
            elo = float(live)
            summary["mode"] = "live_skill_primary" if has_in_sim else "live_skill_only"

        summary.update(
            {
                "ok": True,
                "elo": elo,
                "elo_raw": elo_raw,
                "elo_std": float(our["elo_std"]) if our else None,
                "fitness": float(our["fitness"]) if our else None,
                "champion_path": our["path"] if our else None,
                "n_candidates": len(ckpts),
                "vs_experts": vs,
                "ranking_top": [
                    {
                        "id": a["id"],
                        "elo": a["elo"],
                        "elo_std": a.get("elo_std"),
                        "fitness": a["fitness"],
                        "path": a["path"],
                    }
                    for a in ranked[:8]
                ],
            }
        )
        elo_std_s = float((our or {}).get("elo_std") or 0.0)
        elo_s = float(elo) if elo is not None else float("nan")
        raw_s = float(elo_raw) if elo_raw is not None else elo_s
        print(
            f"[AutoTrainer] PERIODIC EVAL elo={elo_s:.1f} "
            f"(raw={raw_s:.1f}±{elo_std_s:.1f}) "
            f"seeds={len(seeds)} candidates={len(ckpts)} live_skill="
            f"{live if live is not None else 'n/a'} "
            f"vs={vs or '{}'}"
        )
    except Exception as exc:  # noqa: BLE001 — never crash AutoTrainer on eval
        summary["error"] = str(exc)
        if live is not None:
            summary["ok"] = True
            summary["elo"] = float(live)
            summary["mode"] = "live_skill_fallback"
        print(f"[AutoTrainer] PERIODIC EVAL failed: {exc}")

    _persist(watch_dir, state, summary, ts, cfg)
    return summary


def _persist(
    watch_dir: Path,
    state: dict[str, Any],
    summary: dict[str, Any],
    timesteps: int,
    cfg: dict[str, Any] | None,
) -> None:
    pe = (cfg or {}).get("periodic_eval") or {}
    st = dict(state.get(STATE_KEY) or {})
    prev_elo = st.get("last_elo")
    st["last_eval_timesteps"] = timesteps
    st["last_eval_cycle"] = int(st.get("cycles") or 0) + 1
    st["cycles"] = int(st.get("cycles") or 0) + 1
    st["last_summary"] = summary
    if summary.get("elo") is not None:
        st["last_elo"] = float(summary["elo"])
        st["elo_delta"] = (
            float(summary["elo"]) - float(prev_elo)
            if prev_elo is not None
            else 0.0
        )
    if summary.get("live_skill") is not None:
        st["last_live_skill"] = float(summary["live_skill"])
    state[STATE_KEY] = st

    out_name = str(pe.get("report_name") or "eval_report_periodic.json")
    write_json_atomic(Path(watch_dir) / out_name, summary)
    # Also drop next to exe when possible
    try:
        exe = _resolve_exe_dir(watch_dir, cfg)
        write_json_atomic(exe / out_name, summary)
    except OSError:
        pass


def elo_signal_for_meta(state: dict[str, Any] | None) -> dict[str, float]:
    """Compact skill/Elo fields for meta_brain scoring / context."""
    st = dict((state or {}).get(STATE_KEY) or {})
    out: dict[str, float] = {}
    if st.get("last_elo") is not None:
        try:
            out["elo"] = float(st["last_elo"])
        except (TypeError, ValueError):
            pass
    if st.get("elo_delta") is not None:
        try:
            out["elo_delta"] = float(st["elo_delta"])
        except (TypeError, ValueError):
            pass
    if st.get("last_live_skill") is not None:
        try:
            out["skill"] = float(st["last_live_skill"])
        except (TypeError, ValueError):
            pass
    return out


def maybe_run_periodic_eval(
    watch_dir: Path,
    status: dict[str, Any],
    state: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Gate + run. Returns summary or None if not due / disabled."""
    if not periodic_eval_enabled(cfg):
        return None
    ts = int(status.get("total_timesteps") or 0)
    cycle = int((state.get(STATE_KEY) or {}).get("cycles") or 0)
    if not due_for_eval(state, ts, cfg=cfg, cycle=cycle):
        return None
    # Skip while entropy death — do not burn cycles during recovery
    metrics = status.get("last_metrics") or {}
    h = _f(metrics, "Policy Entropy", "policy_entropy")
    if not math.isnan(h) and h < 0.10:
        return None
    return run_periodic_eval(watch_dir, status, state, cfg=cfg)
