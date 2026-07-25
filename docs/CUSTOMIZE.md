# Customize GigaLearnRL

Version **1.0**. Discrete PPO stubs. Edit rewards, opponents, checkpoints, AutoTrainer profile.

Continuous / attention are not wired.

Docs: [`README.md`](../README.md) · IT: [`INIZIO_RAPIDO_IT.md`](../INIZIO_RAPIDO_IT.md) · AMD: [`AMD.md`](AMD.md) · Zip: [`SHARE.md`](../SHARE.md)

## Change Goal / Touch weights

Edit CPU and GPU paths, then rebuild.

| Step | File | Change |
|---|---|---|
| 1 | `src/ExampleMain.cpp` -> `EnvCreateDefault` | `GoalReward` / `TouchBallReward` weights |
| 2 | `GigaLearnCPP/.../CudaEnvSet.cpp` -> `rewardProfile == 1` | Same weights |
| 3 | Rebuild | AMD: `tools\build_amd.bat` · NVIDIA: `tools\build_cuda.bat` ([`BUILD.md`](../BUILD.md)) |
| 4 | Train | `run_fresh_train.bat` |

```cpp
rewards.push_back({ new GoalReward(), 100.f });
rewards.push_back({ new ZeroSumReward(new TouchBallReward(), 0.0f), 5.0f });
```

```cpp
add(rsc::TrainingRewardID::GOAL_REWARD, 100.f);
add(rsc::TrainingRewardID::TOUCH_BALL, 5.f, true);
```

Arenas/bank: `namespace TrainingSize` in `ExampleMain.cpp`, then rebuild. Editing `.cpp` alone does nothing until rebuild.

---

## 0. Training size

| Knob | Default (`amd_win_20k` / RX 6600 XT 8 GB) | Meaning |
|---|---|---|
| `kProfileName` | `"amd_win_20k"` | AMD HIP profile from hw_probe / bat |
| `kTargetMinOverallSps` | `20000` | Overall target with HIP |
| `kArenas` | `2048` | Parallel arenas; 1536/1024 if Overall <20k |
| `kStepsPerItr` | `3` | Bank steps; `ts ≈ arenas × 2 × steps` |
| `kTsPerItr` | `0` (auto) | Pin timesteps/iter |
| `kPpoEpochs` | `1` | PPO epochs |
| `kMiniBatchSize` | `0` (auto) | Full batch when epochs==1 |
| `kMaxEpisodeSeconds` | `1.5` | Episode wall |
| `kCheckpointDirName` | `"checkpoints"` | Box under exe dir |
| `kUseHwProbeArenas` | `true` | Allow probe override |
| `kForceTrainingSize` | `false` | Beat probe and `GIGA_ENV_*` |

Precedence: `kForceTrainingSize` -> `GIGA_ENV_ARENAS` -> `amd_win_20k_next.env` -> hw_probe -> `kArenas`.

If Overall <20k with HIP: auto-lean + `amd_win_20k_next.env`. See `INIZIO_RAPIDO_IT.md` / `docs/AMD.md`.

## 1. Rewards

| Where | What |
|---|---|
| `ExampleMain.cpp` default env | CPU path |
| `CudaEnvSet.cpp` `rewardProfile == 1` | GPU path |
| `RLGymCPP/RewardCore/*` | Stub library only (`CommonRewards`, `ZeroSum`, registry) |
| `RLGymCPP/Rewards/` | **Empty** (`.gitkeep` only) — do not put sources here |

Default: Goal 100 · Touch 5 (ZS) · VelBallToGoal 10 (ZS) · VelPlayerToBall 2.

Do **not** commit custom reward packs. Keep proprietary headers outside the clone (or under a local-only path that is gitignored).

Rebuild after C++ changes. AutoTrainer can scale named rewards via `reward_weights` if names are in the manifest.

## 2. Opponents

| Where | What |
|---|---|
| `opponents/opponents.json` | Pool (`entries: []` = self-play) |
| `opponents/<name>/` | Models |

Community models are not shipped. Add your own; see `opponents/README.md`.

## 3. Checkpoints

`build/Release/checkpoints/` (from `kCheckpointDirName`). Empty for share zips.

## 4. AutoTrainer

| Where | What |
|---|---|
| `autotrainer/profiles/default.yaml` | Profile |
| `descriptions/default.txt` | Bot description text |
| `autotrainer/config.default.yaml` | Master config |

```bat
run_fresh_train.bat
run_with_autotrainer.bat
start_autotrainer.bat default
```

AT off by default. Details: README §5.

## 5. New PC

1. Unzip  
2. `SETUP_FIRST_RUN.bat`  
3. `run_fresh_train.bat`  
4. First-iter metrics  
5. Optional AT  

GPU: NVIDIA CUDA or AMD HIP - `docs/AMD.md`, `README.md`.
