"""
Centralized Shared Episodic Replay — Windows-friendly mmap / filesystem backend.

When a PBT loser clones an elite, sample high-advantage trajectory meta-rows
so the new agent does not cold-start. Full tensors stay in C++; Python stores
compact prioritized summaries + optional numpy mmap shards.
"""

from __future__ import annotations

import json
import math
import os
import random
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HEADER_FMT = "<IId"  # magic, capacity, write_idx as double for simplicity
MAGIC = 0x47495250  # 'GIRP'
ROW_FMT = "<qfffff"  # steps, adv, reward, phase, agent_id, unused
ROW_SIZE = struct.calcsize(ROW_FMT)


@dataclass
class TrajectoryMeta:
    steps: int
    advantage: float
    reward: float
    phase: int
    agent_id: int
    ts: float = field(default_factory=time.time)

    def priority(self) -> float:
        return abs(self.advantage) + 0.1 * abs(self.reward) + 1e-3


class SharedPrioritizedReplay:
    """
    Ring buffer of trajectory metas + sidecar JSON index.
    Path layout:
      <root>/shared_replay/meta.mmap
      <root>/shared_replay/index.json
      <root>/shared_replay/elite_samples.jsonl
    """

    def __init__(self, root: Path, capacity: int = 8192) -> None:
        self.root = Path(root) / "shared_replay"
        self.capacity = max(64, int(capacity))
        self.root.mkdir(parents=True, exist_ok=True)
        self.mmap_path = self.root / "meta.mmap"
        self.index_path = self.root / "index.json"
        self.elite_path = self.root / "elite_samples.jsonl"
        self._ensure_store()

    def _ensure_store(self) -> None:
        need = 16 + self.capacity * ROW_SIZE
        if not self.mmap_path.exists() or self.mmap_path.stat().st_size < need:
            with open(self.mmap_path, "wb") as f:
                f.write(struct.pack("<II", MAGIC, self.capacity))
                f.write(struct.pack("<d", 0.0))  # write cursor
                f.write(b"\x00" * (self.capacity * ROW_SIZE))
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"count": 0, "cursor": 0}), encoding="utf-8")

    def _read_cursor(self) -> tuple[int, int]:
        with open(self.mmap_path, "rb") as f:
            magic, cap = struct.unpack("<II", f.read(8))
            if magic != MAGIC:
                raise RuntimeError("shared_replay mmap corrupt")
            cursor = int(struct.unpack("<d", f.read(8))[0]) % cap
        return cap, cursor

    def push(self, meta: TrajectoryMeta) -> None:
        cap, cursor = self._read_cursor()
        offset = 16 + cursor * ROW_SIZE
        blob = struct.pack(
            ROW_FMT,
            int(meta.steps),
            float(meta.advantage),
            float(meta.reward),
            float(meta.phase),
            float(meta.agent_id),
            0.0,
        )
        with open(self.mmap_path, "r+b") as f:
            f.seek(offset)
            f.write(blob)
            nxt = (cursor + 1) % cap
            f.seek(8)
            f.write(struct.pack("<d", float(nxt)))
        idx = {"count": min(cap, self._count() + 1), "cursor": nxt}
        self.index_path.write_text(json.dumps(idx), encoding="utf-8")

    def _count(self) -> int:
        try:
            return int(json.loads(self.index_path.read_text(encoding="utf-8")).get("count") or 0)
        except Exception:
            return 0

    def _all_rows(self) -> list[TrajectoryMeta]:
        cap, _ = self._read_cursor()
        rows: list[TrajectoryMeta] = []
        with open(self.mmap_path, "rb") as f:
            f.seek(16)
            data = f.read(cap * ROW_SIZE)
        for i in range(cap):
            chunk = data[i * ROW_SIZE : (i + 1) * ROW_SIZE]
            if len(chunk) < ROW_SIZE:
                break
            steps, adv, rew, phase, agent, _ = struct.unpack(ROW_FMT, chunk)
            if steps == 0 and adv == 0 and rew == 0:
                continue
            rows.append(
                TrajectoryMeta(
                    steps=int(steps),
                    advantage=float(adv),
                    reward=float(rew),
                    phase=int(phase),
                    agent_id=int(agent),
                )
            )
        return rows

    def push_from_metrics(self, metrics: dict[str, Any], *, phase: int, agent_id: int, steps: int) -> None:
        m = metrics or {}

        def g(*keys: str, default: float = 0.0) -> float:
            for k in keys:
                if k in m and m[k] is not None:
                    try:
                        return float(m[k])
                    except (TypeError, ValueError):
                        pass
            return default

        self.push(
            TrajectoryMeta(
                steps=steps,
                advantage=g("Mean Absolute Advantage", "avg_advantage", default=g("Average Step Reward")),
                reward=g("Average Step Reward", "avg_reward"),
                phase=phase,
                agent_id=agent_id,
            )
        )

    def sample_elite(self, k: int = 16, rng: random.Random | None = None) -> list[dict[str, Any]]:
        """Prioritized sample of high-|advantage| metas for clone warm-start hints."""
        rng = rng or random.Random()
        rows = self._all_rows()
        if not rows:
            return []
        weights = [r.priority() for r in rows]
        total = sum(weights) or 1.0
        probs = [w / total for w in weights]
        # Manual weighted sample without numpy
        out: list[TrajectoryMeta] = []
        for _ in range(min(k, len(rows))):
            r = rng.random()
            acc = 0.0
            pick = rows[-1]
            for row, p in zip(rows, probs):
                acc += p
                if r <= acc:
                    pick = row
                    break
            out.append(pick)
        payload = [
            {
                "steps": r.steps,
                "advantage": r.advantage,
                "reward": r.reward,
                "phase": r.phase,
                "agent_id": r.agent_id,
                "priority": r.priority(),
            }
            for r in out
        ]
        with open(self.elite_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "samples": payload}) + "\n")
        return payload

    def clone_warmstart_hint(self, donor_id: int, loser_id: int, k: int = 8) -> dict[str, Any]:
        samples = self.sample_elite(k=k)
        # Prefer donor trajectories
        donor_first = [s for s in samples if s.get("agent_id") == donor_id] + [
            s for s in samples if s.get("agent_id") != donor_id
        ]
        return {
            "donor_id": donor_id,
            "loser_id": loser_id,
            "elite_trajectories": donor_first[:k],
            "note": "C++ priority_sampling + event_advantage_boost should stay ON after clone",
        }


def default_replay(watch_dir: Path | str, capacity: int = 8192) -> SharedPrioritizedReplay:
    return SharedPrioritizedReplay(Path(watch_dir), capacity=capacity)
