"""Optional wandb API for richer metric history."""

from __future__ import annotations

from typing import Any


def fetch_latest_metrics(
    project: str,
    run_id: str | None,
    entity: str | None = None,
) -> dict[str, float]:
    if not run_id:
        return {}
    try:
        import wandb
    except ImportError:
        return {}

    api = wandb.Api()
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    try:
        run = api.run(path)
    except Exception:
        return {}

    out: dict[str, float] = {}
    try:
        hist = run.history(samples=1)
        if hist is not None and len(hist) > 0:
            row = hist.iloc[-1]
            for col in row.index:
                val = row[col]
                if isinstance(val, (int, float)):
                    out[str(col)] = float(val)
    except Exception:
        pass
    return out
