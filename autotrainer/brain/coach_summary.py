"""
Optional offline coach summary for AutoTrainer.

Every N cycles (default 75) or on DIAGNOSE/PLAN change writes ``coach_summary.txt``
(and a small JSON) with a local
heuristic "coach" narrative. If an LLM API key is present and ``llm.enabled``,
may enrich the text — but **never blocks** training (timeouts/errors → heuristic).

Opt-out: ``coach.enabled: false`` or ``GIGA_NO_COACH=1``.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import write_json_atomic


SUMMARY_TXT = "coach_summary.txt"
SUMMARY_JSON = "coach_summary.json"


def coach_enabled(cfg: dict[str, Any] | None = None) -> bool:
    if os.environ.get("GIGA_NO_COACH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    c = (cfg or {}).get("coach") or {}
    if c.get("enabled") is False:
        return False
    # Default ON with intelligence
    intel = (cfg or {}).get("intelligence") or {}
    if intel.get("enabled", True) and c.get("enabled") is None:
        return True
    return bool(c.get("enabled", True))


def _f(metrics: dict[str, Any] | None, *keys: str, default: float = float("nan")) -> float:
    m = metrics or {}
    for k in keys:
        if k in m and m[k] is not None:
            try:
                v = float(m[k])
                if math.isnan(v) or math.isinf(v):
                    return default
                return v
            except (TypeError, ValueError):
                pass
    return default


def _fmt(v: float, digits: int = 2) -> str:
    if math.isnan(v):
        return "?"
    return f"{v:.{digits}f}"


def build_heuristic_summary(
    *,
    status: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    zone: str,
    intel: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    mlp: dict[str, Any] | None = None,
    ab: dict[str, Any] | None = None,
    teachers: dict[str, Any] | None = None,
    best_skill: dict[str, Any] | None = None,
    elo: dict[str, float] | None = None,
) -> str:
    """Local coach narrative — no network."""
    st = status or {}
    m = metrics or {}
    intel = intel or {}
    meta = meta or {}
    mlp = mlp or {}
    ab = ab or {}
    teachers = teachers or {}
    best_skill = best_skill or {}
    elo = elo or {}

    h = _f(m, "Policy Entropy", "policy_entropy")
    reward = _f(m, "Average Step Reward", "avg_reward")
    sps = _f(m, "Overall Steps/Second", "SPS")
    ts = int(st.get("total_timesteps") or 0)
    phase = st.get("curriculum_phase", st.get("phase", "?"))

    lines = [
        "=== AutoTrainer Coach Summary (local heuristic) ===",
        f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timesteps: {ts:,}  Phase: {phase}  Zone: {zone}",
        f"H={_fmt(h, 3)}  reward={_fmt(reward, 3)}  SPS={_fmt(sps, 0)}",
        f"Elo={_fmt(_f(elo.get('elo')), 1)}  delta={_fmt(_f(elo.get('elo_delta')), 1)}  "
        f"best_skill={_fmt(_f(best_skill.get('best_skill')), 1)}",
        "",
        "-- Diagnosis / Plan --",
        f"  DIAGNOSE: {intel.get('diagnosis', '?')}",
        f"  PLAN: {intel.get('plan', '?')}",
        f"  ACT: {intel.get('action', '?')}  (acted={intel.get('acted')})",
        f"  unmute_OP={_fmt(_f(intel.get('unmute_level')), 2)}",
        "",
        "-- Meta / MLP --",
        f"  bandit_action={meta.get('last_action', '?')}  episodes={meta.get('episodes', '?')}  "
        f"last_meta_reward={_fmt(_f(meta.get('last_meta_reward')), 2)}",
        f"  mlp_backend={mlp.get('backend', 'off')}  updates={mlp.get('updates', 0)}  "
        f"mlp_action={mlp.get('last_action', '?')}",
    ]
    feats = mlp.get("features") or {}
    if feats:
        top = sorted(feats.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:8]
        lines.append("  mlp_features: " + ", ".join(f"{k}={float(v):.2f}" for k, v in top))
    probs = mlp.get("probs") or {}
    if probs:
        top_p = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)[:4]
        lines.append("  mlp_probs: " + ", ".join(f"{k}={float(v):.2f}" for k, v in top_p))

    lines.extend(
        [
            "",
            "-- A/B plans --",
            f"  active={ab.get('active_slot', '?')}  trials={ab.get('trials', 0)}  "
            f"wins A/B={ab.get('wins_a', 0)}/{ab.get('wins_b', 0)}  "
            f"last_winner={ab.get('last_winner', '-')}",
            f"  scores A/B={_fmt(_f(ab.get('last_score_a')), 2)}/"
            f"{_fmt(_f(ab.get('last_score_b')), 2)}",
            "",
            "-- Teachers / distill --",
            f"  mode={teachers.get('teachers_mode', teachers.get('mode', '?'))}  "
            f"active={teachers.get('teachers_active', '?')}  "
            f"best_skill_teacher={teachers.get('best_skill_teacher', False)}",
            f"  guiding={teachers.get('guiding_policy_path') or teachers.get('distill_teacher_path') or '-'}",
            "",
            "-- Coach tips (heuristic) --",
        ]
    )

    tips: list[str] = []
    z = str(zone).lower()
    if z == "red" or (not math.isnan(h) and h < 0.12):
        tips.append("Entropy risk: do NOT chase Elo; let safety recovery own the run.")
    diag = str(intel.get("diagnosis") or "")
    if diag == "elo_stalled":
        tips.append("Elo flat — keep league pressure + touch mix; wait 1–2 Elo evals before judging.")
    if diag == "reward_hack_suspect":
        tips.append("Train reward may be ahead of real contact — trust viz/no-touch signals.")
    if diag in ("healthy", "sps_ok_learning"):
        tips.append("Learning looks healthy — prefer HOLD / gentle LR; avoid thrash.")
    if teachers.get("best_skill_teacher"):
        tips.append("best_skill soft-distill is active — good continuous teacher while green.")
    if ab.get("trials", 0) > 0:
        tips.append(
            f"A/B racing live (winner={ab.get('last_winner') or 'pending'}) — "
            "judge arms after long credit, not 3 ticks."
        )
    if mlp.get("backend") and mlp.get("backend") != "off":
        tips.append(
            f"Meta MLP ({mlp.get('backend')}) is selecting interventions online — "
            "bandit remains fallback."
        )
    if not tips:
        tips.append("Keep diagnose→plan→act; safety veto always wins; no overnight SSL magic.")
    for t in tips:
        lines.append(f"  * {t}")
    lines.append("")
    lines.append(
        "Coach text is a summary aid, not a second trainer. "
        "True SSL needs months of compute."
    )
    lines.append("=== end ===")
    return "\n".join(lines)


def _try_llm_enrich(cfg: dict[str, Any], heuristic: str) -> str | None:
    """Optional non-blocking LLM polish. Returns None on any failure."""
    llm = cfg.get("llm") or {}
    if not llm.get("enabled", False):
        return None
    if not (cfg.get("coach") or {}).get("use_llm_if_available", True):
        return None
    api_key = os.environ.get(llm.get("api_key_env", "OPENAI_API_KEY") or "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI(api_key=api_key, timeout=8.0)
        model = llm.get("model", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise Rocket League RL coach for GigaLearnRL AutoTrainer. "
                        "Rewrite the heuristic summary into 8–12 short bullet tips. "
                        "Do not invent metrics. Safety/entropy recovery always wins."
                    ),
                },
                {"role": "user", "content": heuristic[:6000]},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None
        return (
            "=== AutoTrainer Coach Summary (LLM-enriched) ===\n"
            + text
            + "\n\n--- heuristic base ---\n"
            + heuristic
        )
    except Exception:
        return None


def maybe_write_coach_summary(
    watch_dir: Path,
    state: dict[str, Any],
    *,
    status: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    zone: str,
    cfg: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Write coach_summary.txt every N AT cycles, or on DIAGNOSE/PLAN change.

    Default interval is sparse (75 cycles). Identical diagnose/plan fingerprints
    skip the write even when the interval fires — cuts log spam.
    """
    if not coach_enabled(cfg):
        return None
    cfg = cfg or {}
    c = cfg.get("coach") or {}
    every = max(1, int(c.get("every_cycles", 75)))
    only_on_change = bool(c.get("only_on_plan_change", True))
    st = dict(state.get("coach") or {})
    cycles = int(st.get("cycles") or 0) + 1
    st["cycles"] = cycles
    state["coach"] = st

    intel = state.get("intelligence") or {}
    fingerprint = (
        f"{intel.get('diagnosis')}|{intel.get('plan')}|{intel.get('action')}|"
        f"{zone}|{(state.get('ab_plans') or {}).get('last_result')}"
    )
    plan_changed = fingerprint != str(st.get("last_fingerprint") or "")
    interval_due = cycles % every == 0
    if not force:
        if only_on_change:
            if not plan_changed and not interval_due:
                return None
            # Interval due but same plan → skip (no spam); plan change always writes
            if interval_due and not plan_changed and st.get("last_path"):
                return None
        elif not interval_due:
            return None

    try:
        meta_path = Path(watch_dir) / "meta_brain_state.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            from .io_utils import read_json

            meta = read_json(meta_path) or {}
        mlp = state.get("meta_mlp") or {}
        ab = state.get("ab_plans") or {}
        ov = (status or {}).get("active_overrides") or {}
        teachers = {
            "teachers_mode": ov.get("teachers_mode"),
            "teachers_active": ov.get("teachers_active"),
            "best_skill_teacher": ov.get("best_skill_teacher"),
            "guiding_policy_path": ov.get("guiding_policy_path"),
            "distill_teacher_path": ov.get("distill_teacher_path"),
        }
        pe = state.get("periodic_eval") or {}
        elo = {
            "elo": pe.get("last_elo"),
            "elo_delta": pe.get("elo_delta"),
        }
        heuristic = build_heuristic_summary(
            status=status,
            metrics=metrics,
            zone=zone,
            intel=intel if isinstance(intel, dict) else {},
            meta=meta,
            mlp=mlp if isinstance(mlp, dict) else {},
            ab=ab if isinstance(ab, dict) else {},
            teachers=teachers,
            best_skill=state.get("best_skill") or {},
            elo={k: float(v) for k, v in elo.items() if v is not None},
        )
        text = _try_llm_enrich(cfg, heuristic) or heuristic
        wd = Path(watch_dir)
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / SUMMARY_TXT
        out.write_text(text, encoding="utf-8")
        write_json_atomic(
            wd / SUMMARY_JSON,
            {
                "updated": datetime.now(timezone.utc).isoformat(),
                "source": "llm" if text != heuristic else "heuristic",
                "cycles": cycles,
                "path": str(out),
                "fingerprint": fingerprint,
            },
        )
        st["last_write_s"] = time.time()
        st["last_path"] = str(out)
        st["last_fingerprint"] = fingerprint
        state["coach"] = st
        try:
            why = "plan-change" if plan_changed else f"every {every} cycles"
            print(f"[AutoTrainer] coach summary -> {out.name} ({why})")
        except Exception:
            pass
        return out
    except Exception as exc:  # noqa: BLE001 — never block training
        try:
            print(f"[AutoTrainer] coach summary skipped: {exc}")
        except Exception:
            pass
        return None
