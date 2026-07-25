"""Build or extend bot profiles from natural-language descriptions."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .io_utils import load_yaml, save_yaml


DEFAULT_GATES = {
    "chase": [
        {"metric": "Player/Ball Touch Ratio", "min": 0.12, "weight": 1.0},
        {"metric": "Kickoff/VelTowardBall", "min": 400, "weight": 0.5},
    ],
    "foundation": [
        {"metric": "Player/Ball Touch Ratio", "min": 0.18, "weight": 1.0},
        {"metric": "Player/Boost", "min": 0.25, "max": 0.75, "weight": 0.4},
    ],
    "advanced": [
        {"metric": "Player/Ball Touch Ratio", "min": 0.22, "weight": 0.8},
        {"metric": "Player/In Air Ratio", "min": 0.08, "weight": 1.0},
        {"metric": "Game/Goal Speed", "min": 800, "weight": 0.5},
    ],
}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "custom_bot"


def profile_from_text(
    text: str,
    name: str = "custom",
    profiles_dir: Path | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Create a YAML-ready profile dict from NL description."""
    base: dict[str, Any] = {
        "name": _slug(name),
        "description": text.strip(),
        "skill_emergence": {
            "chase_min_steps": 150_000_000,
            "foundation_min_steps": 800_000_000,
            "advanced_min_steps": 2_000_000_000,
            "eval_interval_steps": 50_000_000,
            "cooldown_after_change_steps": 100_000_000,
        },
        "gates": DEFAULT_GATES.copy(),
        "priorities": [],
        "exploit_resistance": [],
    }

    lower = text.lower()
    priorities = []

    if any(w in lower for w in ("speedflip", "kickoff", "kick off")):
        priorities.append({
            "id": "speedflip_kickoff",
            "label": "Speedflip / kickoff",
            "rewards": ["KickoffProximity", "VelocityPlayerToBallReward"],
        })
    if any(w in lower for w in ("aerial", "air dribble", "double tap", "air-dribble")):
        priorities.append({
            "id": "aerial_play",
            "label": "Aerial / air dribble / double tap",
            "rewards": ["AirReward"],
        })
    if any(w in lower for w in ("bump", "demo", "physical")):
        priorities.append({
            "id": "physical_play",
            "label": "Bump / demo pressure",
            "rewards": ["Bump", "Demo"],
        })
    if any(w in lower for w in ("2v2", "pass", "team", "sync")):
        priorities.append({
            "id": "team_sync",
            "label": "2v2 spacing and ball pressure",
            "rewards": ["VelocityBallToGoalReward", "FaceBallReward"],
        })
    if any(w in lower for w in ("read", "prediction", "react")):
        base["exploit_resistance"].append("high_tempo_unpredictable")

    base["priorities"] = priorities or [
        {
            "id": "fundamentals",
            "label": "Ball control and speed",
            "rewards": ["TouchBallReward", "VelocityPlayerToBallReward", "FaceBallReward"],
        }
    ]

    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Convert Rocket League bot description to JSON with keys: "
                            "priorities (list of {id, label, rewards}), gates (chase/foundation/advanced), "
                            "exploit_resistance (list of strings). Use realistic wandb metric gates."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            import json

            extra = json.loads(resp.choices[0].message.content or "{}")
            if extra.get("priorities"):
                base["priorities"] = extra["priorities"]
            if extra.get("gates"):
                base["gates"] = extra["gates"]
            if extra.get("exploit_resistance"):
                base["exploit_resistance"] = extra["exploit_resistance"]
        except Exception:
            pass

    if profiles_dir:
        profiles_dir.mkdir(parents=True, exist_ok=True)
        path = profiles_dir / f"{base['name']}.yaml"
        save_yaml(path, base)

    return base


def load_or_create_profile(profiles_dir: Path, name: str, text_path: Path | None = None) -> dict:
    path = profiles_dir / f"{name}.yaml"
    if path.exists():
        return load_yaml(path)
    if text_path and text_path.exists():
        return profile_from_text(text_path.read_text(encoding="utf-8"), name=name, profiles_dir=profiles_dir)
    raise FileNotFoundError(f"Profile not found: {path}")
