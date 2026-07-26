# Experimental & secondary flags

The product entry is **from-scratch / normal POWER train**:

```bat
GigaLearnBot.exe
GigaLearnBot.exe --from-scratch
```

Flags below are supported but **not** the default path. Prefer docs over startup banners.

## Secondary (still useful)

| Flag / env | Purpose |
|------------|---------|
| `--continue-leak` / `--like-leak` / `--max-learn` / `--resume-leak` | Resume Leak-style 1024+LN into POWER train |
| `GIGA_CONTINUE_LEAK=1` / `GIGA_MAX_LEARN=1` | Same as continue-leak when CLI mode unset |
| `--metrics` | Full Leak-parity console Report each iter |
| `--wandb` / `--no-wandb` | MetricSender wandb (off by default for unattended SPS) |
| `--verbose` | Extra per-iter SPS / curriculum banners |
| `--apex` / `--no-apex` / `GIGA_NO_APEX=1` | Opponent/old-version curriculum pressure |
| `--cpu` | Force CPU RocketSim (slow; exact CPU rewards) |
| `--render` | View-only RocketSimVis (no Collect/Learn) |
| `--rlbot` | RLBot deploy helper |
| `--lean-resume` | Resume old lean box (`checkpoints_default_lean`) |

## Benchmark / curriculum experiments

| Flag | Purpose |
|------|---------|
| `--pure80` | SPS-only 1-step truncate benchmark — **not** normal training |
| `--hyperpower` | Dense BallChase curriculum path (lower Overall than blank POWER) |
| `--legacy-hyperpower` | Older Cap-augmented gate path |

Pure80 knobs (env): `GIGA_PURE80_ARENAS`, `GIGA_PURE80_STEPS`, `GIGA_PURE80_NET`.

## Removed / unsupported

| Flag | Status |
|------|--------|
| `--quality` | Rejected (discrete-only; not wired) |
| `--continuous` / `--attention` | Rejected (not wired) |
| Private reward packs (slater/mkh/…) | Not in this public repo |

See [`CUSTOMIZE.md`](CUSTOMIZE.md) for rewards/arenas and [`README.md`](../README.md) for the default train path.
