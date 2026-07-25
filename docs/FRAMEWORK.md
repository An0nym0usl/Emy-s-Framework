# GigaLearnRL - Framework Overview

Version **1.0**. C++ RL stack for Rocket League bots.

Discrete PPO (`DefaultAction` + `AdvancedObs`). Continuous / attention not wired in `ExampleMain`. Stub rewards: Goal / Touch / VelBallToGoal / VelPlayerToBall. See `docs/CUSTOMIZE.md`.

Also: [ARCHITECTURE.md](ARCHITECTURE.md), [AUTO_TRAINER.md](AUTO_TRAINER.md), [CUDA_SIM.md](CUDA_SIM.md), [CUSTOMIZE.md](CUSTOMIZE.md), [AMD.md](AMD.md).

## Summary

Trains discrete PPO across parallel RocketSim arenas (CUDA or HIP), optional self-play / Apex / AutoTrainer, RLBot deploy with the same obs builder.

| Is | Is not |
|----|--------|
| Offline RocketSim training | Live Rocket League during train |
| Discrete `DefaultAction` | Continuous policy (default) |
| `AdvancedObs` + flat MLP | Entity attention as default |
| Stub rewards | Finished bot |
| Optional AutoTrainer | Required for train |

`GigaLearnCPP` may still contain continuous/attention types; `ExampleMain` does not enable them.

## Data flow

```
ExampleMain.cpp
    |
    v
Learner --> N RocketSim arenas (EnvSet)
              ObsBuilder (AdvancedObs)
              ActionParser (DefaultAction)
              Weighted rewards
    |
    +-- Rollout (GPU infer)
    +-- GAE
    +-- PPO update
    +-- checkpoints/
    +-- PolicyVersionManager (optional)
    +-- wandb (optional)
    +-- AutoTrainerBridge (optional)
```

## Layers

| Layer | Role |
|-------|------|
| RocketSim | Physics |
| RLGymCPP | Env, obs, rewards, parsers |
| GigaLearnCPP | Learner, PPO, SDK |
| libtorch | Train / infer |
| RLBotCPP | Deploy |

Entry: `GigaLearnBot.exe` (`src/ExampleMain.cpp`).

## Simulation

Many parallel arenas (`cfg.numGames`). Physics: CPU or RocketSimCuda / HIP - [CUDA_SIM.md](CUDA_SIM.md), [AMD.md](AMD.md).

## Defaults

- Obs: `AdvancedObs`, `useAttentionHead = false`
- Actions: `PolicyType::DISCRETE` + `DefaultAction`

## AutoTrainer

Optional. Off by default (`GIGA_NO_AUTOTRAINER=1` via `run_fresh_train.bat`). [AUTO_TRAINER.md](AUTO_TRAINER.md).

## Customize

[CUSTOMIZE.md](CUSTOMIZE.md) · [SHARE.md](../SHARE.md)
