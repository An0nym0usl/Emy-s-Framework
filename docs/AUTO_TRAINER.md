# AutoTrainer

Optional Python sidecar. Watches metrics, applies curriculum / PPO / reward overrides, rollback on bad changes.

Disable: [`README.md`](../README.md) §5 (`GIGA_NO_AUTOTRAINER=1`, `--no-autotrainer`, or omit `autotrainer\` from zip).

## Architecture

```
profiles/default.yaml
        |
python -m autotrainer  --runtime_overrides.json-->  GigaLearnBot.exe
        ^                                               |
        +--------- trainer_status.json -----------------+
        |
        decisions.jsonl, orchestrator_state.json, snapshots/
```

## Setup

```powershell
cd GigaLearnRL
pip install -e .
```

## Start

```bat
run_fresh_train.bat
start_autotrainer.bat default
run_with_autotrainer.bat
```

Or:

```powershell
python autotrainer/orchestrator.py --watch-dir build/Release/autotrainer
python -m autotrainer --watch-dir build/Release/autotrainer
tools\launch_autotrainer.bat
```

Prefer bats (UTF-8 console). Dashboard UI is not shipped; use AT console + bot Report.

Optional: `--render` on the exe if you use your own visualizer.

## Profile from text

```powershell
python -m autotrainer --profile default --profile-text descriptions/default.txt --import-profile --once
```

Optional OpenAI refine if configured (`OPENAI_API_KEY` / config `llm`).

## Overrides

| Key | Effect |
|-----|--------|
| `chase_end_steps`, `foundation_end_steps` | Curriculum timing |
| `force_phase` | Phase 0-2 |
| `opponent_pool_chance`, `train_against_old_chance` | Sparring |
| `entropy_scale`, `var_max`, `epochs`, `gae_*`, `policy_lr`, `critic_lr` | PPO |
| `clip_range`, `es_noise_scale`, `event_advantage_boost`, `priority_sampling` | OP stack |
| `skill_tracker_enabled` | Skill eval |
| `reward_weights` | Reward multipliers |

## OP mode

Optional PBT league + ES + event controller (`op_mode` in `config.default.yaml`). PPO gradients stay in C++.

## Safety

- Cooldown between substantive changes (default 100M steps)
- Eval interval (default 50M)
- Clamped patches
- Snapshot + rollback on metric drop
- Optional external suggest path uses the same clamps

## Control files

| File | Writer | Purpose |
|------|--------|---------|
| `commands.json` | User | pause / save |
| `runtime_overrides.json` | Brain | Hot state |
| `trainer_status.json` | Bot | Steps, metrics |
| `decisions.jsonl` | Brain | Audit |

```python
from autotrainer.brain.commands import pause_training, request_checkpoint
pause_training(Path("build/Release/autotrainer"), seconds=3600)
```

## Default profile

`autotrainer/profiles/default.yaml`. Edit for your bot.

## Limits

- Skill needs large step counts and a real reward stack
- Decisions use metrics, not pixels
- Short build path if `nvcc` fails on spaces

## Config

`autotrainer/config.default.yaml`: poll interval, wandb, rollback, optional suggest settings, clamps.

Opt-outs: `GIGA_NO_AUTOTRAINER=1`, `--no-autotrainer`, `GIGA_NO_SSL_AUTONOMY=1`, `GIGA_AT_READONLY=1`.
