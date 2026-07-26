#pragma once

// =============================================================================
// TRAINING SIZE - edit these, then REBUILD (tools\build_amd.bat or cmake).
// Arenas / PPO bank / episode wall / ckpt folder.
// Reference PC: Win11 - A520-M - Ryzen 5 3600 - RX 6600 XT 8GB - see docs/AMD.md
// Locked profile: amd_win_20k -> normal train Overall >= 20,000 SPS with HIP working.
// =============================================================================
// Precedence (default POWER path: from-scratch / normal / continue-leak):
//   1) kForceTrainingSize == true  -> numbers below ALWAYS win (ignore hw_probe + GIGA_ENV_*).
//   2) Else GIGA_ENV_ARENAS / TS / EPOCHS / MAXEP / STEPS -> env wins (no rebuild).
//   3) Else kUseHwProbeArenas && hw_probe arenas -> probe wins for arenas only.
//   4) Else -> kArenas / kTsPerItr / kPpoEpochs / kMaxEpisodeSeconds / kStepsPerItr below.
// Tip: set kUseHwProbeArenas = false to lock arenas to kArenas without forcing env off.
// After ANY edit here you must REBUILD - changing this file alone does nothing to the exe.
// =============================================================================
namespace TrainingSize {
	// Profile name (hw_probe / run_fresh_train.bat on AMD HIP).
	constexpr const char* kProfileName = "amd_win_20k";
	// Target Overall for Win11 + RX 6600 XT + 3600 with HIP + CPU PPO.
	// Overall = 1/(1/Coll+1/Cons). Without HIP, 20k is NOT guaranteed.
	constexpr float kTargetMinOverallSps = 20000.f;
	// Secondary Overall target when HIP is healthy.
	constexpr float kStretchOverallSps = 25000.f;

	// true  = this block is the single source (beats hw_probe and GIGA_ENV_*).
	// false = allow probe/env overrides (good until you lock a size).
	constexpr bool kForceTrainingSize = false;

	// When false, ignore hw_probe arena recommendation and use kArenas
	// (GIGA_ENV_ARENAS still overrides unless kForceTrainingSize).
	constexpr bool kUseHwProbeArenas = true;

	// Parallel RocketSim arenas (LearnerConfig::numGames).
	// amd_win_20k: 2048 = Overall sweet spot on 8GB (1536 safer / 2560 Coll-heavy).
	// Do NOT jump to 8192 on 8GB VRAM - Learn+Infer on the 3600 will choke Overall.
	// Auto-downgrade ladder if Overall stays <20k: 2048 -> 1536 -> 1024 (next restart).
	constexpr int kArenas = 2048;

	// Env steps in the experience bank. Auto ts ~ arenas * 4 * steps (2v2).
	// 2048 x 4 x 3 = 24576 ts/iter.
	constexpr int kStepsPerItr = 3;

	// Timesteps per PPO iteration. 0 = auto (arenas * 4 * kStepsPerItr).
	constexpr int kTsPerItr = 0;

	// PPO epochs per iteration (1 = full-batch SPS path; 2 = Wazne roadmap).
	constexpr int kPpoEpochs = 2;

	// Minibatch size. 0 = auto (full batch when epochs==1; Leak-style when epochs>=2).
	constexpr int kMiniBatchSize = 0;

	// Episode wall-clock truncate (seconds). From-scratch needs room to drive;
	// iterBudget still banks when tsPerItr fills (effective horizon ≈ stepsPerItr).
	constexpr double kMaxEpisodeSeconds = 45.0;

	// Pure self-play until this many steps (opp/old/skill stay off - SPS tax).
	constexpr int64_t kSparringDeferTs = 50'000'000;

	// From-scratch checkpoint folder name under the exe dir (e.g. "checkpoints").
	constexpr const char* kCheckpointDirName = "checkpoints";
}

// CUDA POWER (NVIDIA normal / from-scratch) - gated separately from amd_win_20k.
// 5060 Ti 16GB class: 8192 arenas + lean512 + half Infer. Does NOT touch AMD HIP defaults.
// Sustained target: median Overall ~ smoke steady (~946k+), not a lucky peak.
namespace CudaPower {
	constexpr int kArenas = 8192;
	// 48 steps → ~2.4s GAE horizon before iterBudget truncate (was 16/~0.8s = jump-only).
	// ts ≈ 8192*4*48 = 1.57M — still Learn-ok on 16GB; override with GIGA_ENV_STEPS.
	constexpr int kStepsPerItr = 48;
	constexpr int kPpoEpochs = 2; // Wazne: epochs = 2
	// Wall-clock cap. Real traj length is min(maxEp, stepsPerItr) via iterBudgetFull.
	constexpr double kMaxEpisodeSeconds = 45.0;
	// Sustained Overall target (median/mean), not peak.
	constexpr float kTargetMinOverallSps = 900000.f;
	// Defer disk Save hitches (~every 50M) so long-run median stays near peak path.
	constexpr int64_t kTsPerSave = 50'000'000;
	// Pure self-play until this many steps - Apex must not re-arm opp/old/skill early.
	constexpr int64_t kSparringDeferTs = 50'000'000;
}
