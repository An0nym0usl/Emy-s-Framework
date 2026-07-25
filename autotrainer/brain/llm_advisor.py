"""Optional LLM advisor — recommendations only; engine applies with safety clamps."""

from __future__ import annotations

import json
import os
from typing import Any


SYSTEM_PROMPT = """You are an expert Rocket League RL training advisor for GigaLearnRL.
Given a bot profile, current metrics, curriculum phase, and recent gate failures,
suggest ONE conservative runtime_overrides patch as JSON.

Allowed keys only:
- chase_end_steps, foundation_end_steps (integers)
- opponent_pool_chance, train_against_old_chance (0.0-0.5)
- entropy_scale (0.005-0.04), var_max (0.1-1.2), epochs (1-3)
- skill_tracker_enabled (bool)
- reward_weights: { "RewardClassName": multiplier 0.05-3.0 }

Return ONLY valid JSON: {"action": "...", "reason": "...", "patch": {...}}
Never suggest more than 3 reward weight changes at once. Prefer small adjustments."""


def llm_advise(
    cfg: dict,
    profile: dict,
    status: dict,
    gate_failures: list[str],
    manifest: dict,
) -> dict[str, Any] | None:
    llm = cfg.get("llm") or {}
    if not llm.get("enabled", False):
        return None

    api_key = os.environ.get(llm.get("api_key_env", "OPENAI_API_KEY") or "")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    user_payload = {
        "profile_name": profile.get("name"),
        "description": profile.get("description"),
        "phase": status.get("curriculum_phase"),
        "timesteps": status.get("total_timesteps"),
        "metrics": status.get("last_metrics"),
        "gate_failures": gate_failures,
        "reward_names": [e.get("name") for e in (manifest.get("rewards") or [])],
        "current_overrides": status.get("active_overrides"),
    }

    model = llm.get("model", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(text)
    except Exception:
        return None
