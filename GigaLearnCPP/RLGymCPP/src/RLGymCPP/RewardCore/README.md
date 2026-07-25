# RewardCore (public stubs)

Formerly `RLGymCPP/Rewards/`. That folder is intentionally **empty** (`.gitkeep` only).

Public framework ships **stub / infrastructure only** here:

- `Reward.h` — base class
- `CommonRewards.h` — Goal / Touch / VelBallToGoal / VelPlayerToBall / FaceBall / Boost / …
- `ZeroSumReward.*` + `RewardWrapper.h`
- `RuntimeRewardRegistry.*` — AutoTrainer weight overrides

Include as:

```cpp
#include <RLGymCPP/RewardCore/Reward.h>
#include <RLGymCPP/RewardCore/CommonRewards.h>
#include <RLGymCPP/RewardCore/ZeroSumReward.h>
```

## Custom rewards

Keep proprietary reward headers **out of git**. Prefer a path outside the clone, or any local include path you add yourself. Do not put custom packs under `Rewards/` (kept empty) or commit them under `RewardCore/`.

See `docs/CUSTOMIZE.md`.
