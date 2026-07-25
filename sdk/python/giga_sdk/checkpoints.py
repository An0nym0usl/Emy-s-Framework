"""Pure-Python mirror of the C++ ``GGL::CheckpointManager``.

A checkpoint folder is named after its total timestep count and contains:
  POLICY.lt, CRITIC.lt, SHARED_HEAD.lt   (+ *_OPTIM.lt for resuming)
  RUNNING_STATS.json                      (training run metadata)

This module reads exactly that layout, with no third-party dependencies.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def default_checkpoint_dir() -> Path:
    """Best-effort default checkpoint folder.

    Resolution order:
      1. ``GIGA_CHECKPOINT_DIR`` environment variable.
      2. ``./checkpoints`` relative to the current working directory.
    """
    env = os.environ.get("GIGA_CHECKPOINT_DIR")
    if env:
        return Path(env)
    return Path.cwd() / "checkpoints"


@dataclass
class CheckpointInfo:
    path: Path
    timesteps: int = -1

    has_policy: bool = False
    has_critic: bool = False
    has_shared_head: bool = False
    has_optimizers: bool = False
    has_stats: bool = False

    total_timesteps: int = -1
    total_iterations: int = -1
    run_id: str = ""

    total_bytes: int = 0

    @property
    def is_valid_for_inference(self) -> bool:
        return self.has_policy and self.has_shared_head

    @property
    def is_valid_for_resume(self) -> bool:
        return self.has_policy and self.has_critic and self.has_shared_head and self.has_stats

    def as_dict(self) -> dict:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
        d["is_valid_for_inference"] = self.is_valid_for_inference
        d["is_valid_for_resume"] = self.is_valid_for_resume
        return d


def _file_nonempty(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


class CheckpointManager:
    """Lists, inspects, validates, exports and prunes the checkpoint box."""

    def __init__(self, folder: os.PathLike | str | None = None):
        self.folder = Path(folder) if folder is not None else default_checkpoint_dir()

    # ----- reading -----

    def read(self, directory: os.PathLike | str) -> CheckpointInfo:
        directory = Path(directory)
        try:
            timesteps = int(directory.name)
        except ValueError:
            timesteps = -1

        info = CheckpointInfo(path=directory, timesteps=timesteps)
        info.has_policy = _file_nonempty(directory / "POLICY.lt")
        info.has_critic = _file_nonempty(directory / "CRITIC.lt")
        info.has_shared_head = _file_nonempty(directory / "SHARED_HEAD.lt")
        info.has_optimizers = (
            _file_nonempty(directory / "POLICY_OPTIM.lt")
            and _file_nonempty(directory / "CRITIC_OPTIM.lt")
        )

        stats_path = directory / "RUNNING_STATS.json"
        if _file_nonempty(stats_path):
            try:
                data = json.loads(stats_path.read_text())
                info.has_stats = True
                info.total_timesteps = int(data.get("total_timesteps", -1))
                info.total_iterations = int(data.get("total_iterations", -1))
                info.run_id = str(data.get("run_id", ""))
            except (ValueError, OSError):
                info.has_stats = False

        try:
            info.total_bytes = sum(
                f.stat().st_size for f in directory.iterdir() if f.is_file()
            )
        except OSError:
            info.total_bytes = 0

        return info

    def _numbered_dirs(self, folder: Path) -> List[Path]:
        if not folder.is_dir():
            return []
        return [p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()]

    def list(self) -> List[CheckpointInfo]:
        infos = [self.read(p) for p in self._numbered_dirs(self.folder)]
        infos.sort(key=lambda c: c.timesteps)
        return infos

    def list_policy_versions(self) -> List[CheckpointInfo]:
        return CheckpointManager(self.folder / "policy_versions").list()

    def latest(self) -> Optional[CheckpointInfo]:
        infos = self.list()
        return infos[-1] if infos else None

    def latest_valid_for_inference(self) -> Optional[CheckpointInfo]:
        for info in reversed(self.list()):
            if info.is_valid_for_inference:
                return info
        return None

    def get(self, timesteps: int) -> Optional[CheckpointInfo]:
        for info in self.list():
            if info.timesteps == timesteps:
                return info
        return None

    # ----- operations -----

    def validate(self, directory: os.PathLike | str) -> tuple[bool, str]:
        directory = Path(directory)
        if not directory.exists():
            return False, f"Checkpoint directory does not exist: {directory}"
        info = self.read(directory)
        if not info.has_policy:
            return False, "Missing or empty POLICY.lt"
        if not info.has_shared_head:
            return False, "Missing or empty SHARED_HEAD.lt"
        if not info.has_stats:
            return False, "Missing or unreadable RUNNING_STATS.json"
        return True, ""

    def export(self, timesteps: int, dest_folder: os.PathLike | str) -> tuple[bool, str]:
        info = self.get(timesteps)
        if info is None:
            return False, f"No checkpoint with timesteps={timesteps}"
        if not info.is_valid_for_inference:
            return False, "Checkpoint is missing inference files (POLICY.lt / SHARED_HEAD.lt)"

        dest = Path(dest_folder)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("POLICY.lt", "SHARED_HEAD.lt", "RUNNING_STATS.json"):
            src = info.path / name
            if src.exists():
                shutil.copy2(src, dest / name)
        return True, ""

    def prune(self, keep_n: int) -> int:
        if keep_n < 0:
            return 0
        infos = self.list()  # ascending
        to_remove = len(infos) - keep_n
        removed = 0
        for info in infos[: max(0, to_remove)]:
            try:
                shutil.rmtree(info.path)
                removed += 1
            except OSError:
                pass
        return removed
