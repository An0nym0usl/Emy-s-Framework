"""Write control commands for the C++ training loop."""

from __future__ import annotations

import time
from pathlib import Path

from .io_utils import read_json, write_json_atomic


def pause_training(watch_dir: Path, seconds: int | None = None) -> None:
    cmd = read_json(watch_dir / "commands.json")
    cmd["paused"] = True
    if seconds:
        cmd["pause_until_unix"] = int(time.time()) + seconds
    write_json_atomic(watch_dir / "commands.json", cmd)


def resume_training(watch_dir: Path) -> None:
    cmd = read_json(watch_dir / "commands.json")
    cmd["paused"] = False
    cmd["clear_pause"] = True
    write_json_atomic(watch_dir / "commands.json", cmd)


def request_checkpoint(watch_dir: Path) -> None:
    cmd = read_json(watch_dir / "commands.json")
    cmd["save_checkpoint"] = True
    write_json_atomic(watch_dir / "commands.json", cmd)


def request_save_best_skill(watch_dir: Path) -> None:
    """Ask C++ to SavePolicyTo(checkpointFolder/best_skill) if bridge supports it."""
    cmd = read_json(watch_dir / "commands.json")
    cmd["save_best_skill"] = True
    write_json_atomic(watch_dir / "commands.json", cmd)


def request_save_agent_slot(watch_dir: Path, agent_id: int) -> None:
    """Save live policy weights into competitive PBT slot agent_{id}."""
    cmd = read_json(watch_dir / "commands.json")
    cmd["save_agent_slot"] = int(agent_id)
    write_json_atomic(watch_dir / "commands.json", cmd)


def request_load_agent_slot(watch_dir: Path, agent_id: int) -> None:
    """Load competitive PBT slot agent_{id} into the live trainer (weight transfer)."""
    cmd = read_json(watch_dir / "commands.json")
    cmd["load_agent_slot"] = int(agent_id)
    write_json_atomic(watch_dir / "commands.json", cmd)


def apply_pbt_commands(watch_dir: Path, pbt_commands: dict) -> None:
    """Merge save/load agent slot requests into commands.json."""
    if not pbt_commands:
        return
    cmd = read_json(watch_dir / "commands.json")
    if "save_agent_slot" in pbt_commands and pbt_commands["save_agent_slot"] is not None:
        cmd["save_agent_slot"] = int(pbt_commands["save_agent_slot"])
    if "load_agent_slot" in pbt_commands and pbt_commands["load_agent_slot"] is not None:
        cmd["load_agent_slot"] = int(pbt_commands["load_agent_slot"])
    write_json_atomic(watch_dir / "commands.json", cmd)
