# RocketSimCuda GPU backend

GigaLearnRL vendors **RocketSimCuda**, a GPU (CUDA) reimplementation of RocketSim that runs the
whole rollout pipeline on the device: batched physics for N arenas, observation building, rewards,
terminals and GAE. It is an **optional** backend - the default build uses the proven CPU RocketSim
inside RLGymCPP and is completely unaffected.

## Why

The CPU env loop steps arenas on a thread pool and copies observations/actions host<->device every
step. RocketSimCuda keeps everything in VRAM (sim -> obs -> rewards -> GAE) and parallelizes arenas
across CUDA cores, which removes the per-step transfer bottleneck and pairs directly with GPU
training (libtorch) and bf16 autocast. On a modern NVIDIA GPU this can be a large throughput win at
high arena counts.

## Enable

Requires the CUDA toolkit (tested with **CUDA 12.8** (other 12.x usually fine)). It builds nothing unless you opt in:

```bat
tools\build_cuda.bat
```

Or manually (same preset; `GIGA_USE_CUDA_SIM` is already ON in `windows-cuda-gpu`):

```bat
cmake --preset windows-cuda-gpu
cmake --build --preset windows-cuda-gpu
```

If **cmake not in PATH**, see [`BUILD.md`](../BUILD.md).

For **AMD HIP / ROCm** use `-DGIGA_GPU_BACKEND=hip` and presets `windows-hip` / `linux-rocm` â€” see **`docs/AMD.md`**.

This builds the `RocketSimCuda` static library, links it into `GigaLearnBot`, defines the
`GIGA_USE_CUDA_SIM` macro, and builds the `giga_cuda_hybrid` example.

Target GPU architectures default to `75;86;89` (Turing / Ampere / Ada) in
`RocketSimCuda/CMakeLists.txt`; adjust `CUDA_ARCHITECTURES` for your card.

### Known limitation: build path must not contain `'` or spaces

`nvcc` cannot handle source/include paths that contain an apostrophe or spaces (it breaks while
building its internal commands). If your share path has spaces (e.g. under a cloud-sync folder), **always
build CUDA from a clean mirror** such as `C:\GigaLearnRL` â€” set `GIGA_CUDA_MIRROR` for the bats.

To build the CUDA backend, put the repository at a clean path, e.g. `C:\GigaLearnRL`, and build
from there. This is purely an nvcc path limitation, not a code issue: the vendored library and the
`giga_cuda_rollout` example were verified to compile (sm_75/86/89) and run on the GPU from a clean
path (64 arenas / 128 players, obsRow=109, kickoff ball at z=93.2).

## Public API (`rsc::RocketSimCudaBatch`, `RocketSimCuda/include/RocketSimCuda.h`)

```cpp
rsc::BatchConfig cfg;
cfg.numArenas = 1024;
cfg.maxCarsPerArena = 4;          // e.g. 2v2
cfg.obsMode = rsc::ObsMode::ADVANCED; // 51 + 29*cars (== AdvancedObsPadded), or VANTA (366)

rsc::RocketSimCudaBatch batch;
batch.Init(cfg);
for (int a = 0; a < cfg.numArenas; a++) { batch.AddCar(a, BLUE); batch.AddCar(a, ORANGE); }
batch.ResetAllArenas();

batch.ConfigureTrainingRewards(rewardCfg);     // GPU-built rewards (see TrainingRewardID)
batch.ConfigureTrainingTerminals(terminalCfg); // goal-score / no-touch timeout

// Per env step:
batch.SetAllCarControls(controls);   // or write GetControlsDevicePtr() from a CUDA kernel
batch.SnapshotTrainingState();
batch.Step(tickSkip);
batch.BuildAdvancedObsAndDefaultMasks();       // or BuildVantaObs()
batch.BuildRewardsAndTerminals(tickSkip);
// Copy to host, or keep on device via GetBuilt*DevicePtr() and feed libtorch directly.
```

GAE can also run on device: `rsc::ComputeGAEOnDevice(...)` in `RocketSimCudaTraining.h`.

A runnable end-to-end example is `RocketSimCuda/examples/cuda_rollout.cpp` (target
`giga_cuda_rollout`).

## Fidelity

Upstream ships a CPU<->CUDA parity harness (`tests/cpu_parity_harness.cpp`,
`RSCUDA_BUILD_CPU_PARITY=ON`) that host-compiles the real kernels and diffs them against an
independent float64 reference. The bundled report passes on ballistics, drag, bounce, demo/respawn,
goal detection, AdvancedObs (1v1 / 2v2 / inverted / demo / near-goal) and continuous action
apply. Long-horizon trajectories diverge slightly (chaotic system) - fine for RL.

### Spawn orientation + ball mass (RocketSim 1:1)

- **Kickoff / fuzzed GPU resets** mirror orange like RocketSim `Arena::ResetToRandomKickoff`:
  `pos *= {-1,-1,1}` and `yaw += π`, with a **shared** shuffled spawn slot per team-local index.
  An older CUDA path only flipped Y and assigned slots by absolute car index, so orange corner/diagonal
  cars were sometimes facing the wrong way.
- **Demo respawn** still uses Y-only flip (`Car::Respawn`) — that is intentional and unchanged.
- **Ball mass** is `CAR_MASS / 6` (= `30`), matching RocketSim `BALL_MASS_BT`. Landing / corner bounce
  feel differences are not from a diverged mass constant; corner stickiness remains a minor mesh/solver
  approximation issue.

## Hybrid integration (implemented)

The chosen integration is **hybrid**: RocketSimCuda runs the physics on the GPU, the device state is
bridged back into `RLGC::GameState`, and the **entire existing CPU observation/reward/terminal
curriculum runs unchanged** on top. This keeps your local custom rewards / entity obs while
moving the simulation onto the GPU.

Pieces (built and verified):

- **GameState bridge** - `GameState::UpdateFromCudaBatch(batch, arenaIdx, actions, prev)` and a
  `GameState(batch, arenaIdx)` constructor in
  `GigaLearnCPP/RLGymCPP/src/RLGymCPP/Gamestates/GameState.{h,cpp}` (guarded by `GIGA_USE_CUDA_SIM`).
  Maps `rsc::CarState`/`BallState`/boost pads into a `GameState` field-for-field, mirroring
  `UpdateFromArena`.
- **CudaEnvSet** - a hybrid environment in
  `GigaLearnCPP/src/public/GigaLearnCPP/Sim/CudaEnvSet.{h,cpp}`: GPU physics for N arenas + CPU
  obs/rewards/terminals/reset. Exposes `obs` / `rewards_out` / `terminals` and `StepContinuous`,
  i.e. the same surface a PPO trainer drives.
- Hybrid env example target removed from CMake (not required for discrete train).

### Using it for training

`CudaEnvSet` is the drop-in environment. A PPO loop is the same as with the CPU `EnvSet`:

```cpp
CudaEnvSet env(numArenas, carsPerTeam, tickSkip, createFn); // createFn returns obs/rewards/terminals
// per iteration:
//   actions = policy(env.obs)                       // PPOLearner inference
//   env.StepContinuous(actions, actionDim);         // GPU physics + CPU obs/rewards
//   collect (env.obs, actions, env.rewards_out, env.terminals)
// then GAE + PPOLearner::Learn(...) exactly as today.
```

### Remaining (optional) work

- Wiring `CudaEnvSet` into the full `Learner::Start()` so the GPU env also gets the Learner's
  self-play / Elo / wandb metrics / checkpoint plumbing (today those assume the CPU `EnvSet`). The
  environment itself is complete; this is loop plumbing.
- For maximum throughput you can additionally keep obs on the device (`GetBuiltAdvancedObsDevicePtr`)
  and use `rsc::ComputeGAEOnDevice`, at the cost of using the GPU's fixed built-in reward enum
  instead of the CPU reward library.

### Build path note

The whole hybrid stack (bridge + CudaEnvSet + RocketSimCuda + libtorch) was verified to build and
run from a clean path (e.g. `C:\GigaLearnRL`). If the share tree lives under a synced/spaced path,
copy or clone to a clean short path, then build there (see the path note above). Set
`GIGA_CUDA_MIRROR` so the portable bats find that Release if you use a separate build folder.
