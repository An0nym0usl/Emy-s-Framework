#pragma once

#include <GigaLearnCPP/Export.h>

#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>

namespace GGL {

	// Lightweight, torch-free description of a single checkpoint folder.
	// A checkpoint folder is named after its total timestep count and contains:
	//   POLICY.lt, CRITIC.lt, SHARED_HEAD.lt          (+ *_OPTIM.lt for resuming)
	//   RUNNING_STATS.json                            (training run metadata)
	struct RG_IMEXPORT CheckpointInfo {
		std::filesystem::path path;
		int64_t timesteps = -1; // Parsed from the folder name

		bool hasPolicy = false;
		bool hasCritic = false;
		bool hasSharedHead = false;
		bool hasOptimizers = false;
		bool hasStats = false;

		// Parsed from RUNNING_STATS.json (when present)
		int64_t totalTimesteps = -1;
		int64_t totalIterations = -1;
		std::string runId;

		uint64_t totalBytes = 0; // Sum of file sizes in the folder

		// Enough weights to run the bot (policy + shared head).
		bool IsValidForInference() const { return hasPolicy && hasSharedHead; }
		// Enough to resume training (weights + stats; optimizers optional but recommended).
		bool IsValidForResume() const { return hasPolicy && hasCritic && hasSharedHead && hasStats; }
	};

	// Manages the "checkpoint box": the on-disk checkpoints/ folder that training writes to and
	// that deployment / the SDK read from. Pure filesystem + JSON, no libtorch dependency, so it
	// can list/inspect/validate/export/prune checkpoints cheaply.
	class RG_IMEXPORT CheckpointManager {
	public:
		std::filesystem::path folder;

		explicit CheckpointManager(std::filesystem::path checkpointFolder);

		// Reads metadata for a single checkpoint directory (does not validate deeply).
		CheckpointInfo Read(const std::filesystem::path& dir) const;

		// All checkpoints in the folder, sorted ascending by timesteps.
		std::vector<CheckpointInfo> List() const;

		// Policy-version snapshots stored under checkpoints/policy_versions/.
		std::vector<CheckpointInfo> ListPolicyVersions() const;

		// Highest-timestep checkpoint, or an empty (timesteps==-1) info if none exist.
		CheckpointInfo Latest() const;

		// Highest-timestep checkpoint that has the files needed to run a bot.
		CheckpointInfo LatestValidForInference() const;

		// Checkpoint with an exact timestep count, or empty info if not found.
		CheckpointInfo Get(int64_t timesteps) const;

		// Validates a checkpoint directory; on failure fills outError with the reason.
		bool Validate(const std::filesystem::path& dir, std::string& outError) const;

		// Copies the inference files (POLICY.lt + SHARED_HEAD.lt, plus RUNNING_STATS.json if present)
		// of the given checkpoint into destFolder. Returns false and fills outError on failure.
		bool Export(int64_t timesteps, const std::filesystem::path& destFolder, std::string& outError) const;

		// Deletes the oldest checkpoints, keeping at most keepN. Returns the number removed.
		int Prune(int keepN) const;
	};
}
