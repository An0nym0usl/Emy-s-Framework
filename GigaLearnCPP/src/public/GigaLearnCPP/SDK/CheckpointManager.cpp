#include "CheckpointManager.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {
	// A checkpoint folder is named purely with digits (its total timestep count).
	bool IsNumberedDir(const fs::path& p) {
		if (!fs::is_directory(p))
			return false;
		std::string name = p.filename().string();
		if (name.empty())
			return false;
		for (char c : name)
			if (!std::isdigit((unsigned char)c))
				return false;
		return true;
	}

	int64_t ParseTimesteps(const fs::path& p) {
		try {
			return std::stoll(p.filename().string());
		} catch (...) {
			return -1;
		}
	}

	bool FileExistsNonEmpty(const fs::path& p) {
		std::error_code ec;
		return fs::exists(p, ec) && fs::file_size(p, ec) > 0;
	}
}

GGL::CheckpointManager::CheckpointManager(fs::path checkpointFolder)
	: folder(std::move(checkpointFolder)) {}

GGL::CheckpointInfo GGL::CheckpointManager::Read(const fs::path& dir) const {
	CheckpointInfo info = {};
	info.path = dir;
	info.timesteps = ParseTimesteps(dir);

	info.hasPolicy = FileExistsNonEmpty(dir / "POLICY.lt");
	info.hasCritic = FileExistsNonEmpty(dir / "CRITIC.lt");
	info.hasSharedHead = FileExistsNonEmpty(dir / "SHARED_HEAD.lt");
	info.hasOptimizers =
		FileExistsNonEmpty(dir / "POLICY_OPTIM.lt") &&
		FileExistsNonEmpty(dir / "CRITIC_OPTIM.lt");

	fs::path statsPath = dir / "RUNNING_STATS.json";
	if (FileExistsNonEmpty(statsPath)) {
		try {
			std::ifstream f(statsPath);
			json j; f >> j;
			info.hasStats = true;
			if (j.contains("total_timesteps")) info.totalTimesteps = j["total_timesteps"].get<int64_t>();
			if (j.contains("total_iterations")) info.totalIterations = j["total_iterations"].get<int64_t>();
			if (j.contains("run_id")) info.runId = j["run_id"].get<std::string>();
		} catch (...) {
			info.hasStats = false;
		}
	}

	std::error_code ec;
	for (auto& entry : fs::directory_iterator(dir, ec)) {
		if (entry.is_regular_file(ec))
			info.totalBytes += entry.file_size(ec);
	}

	return info;
}

std::vector<GGL::CheckpointInfo> GGL::CheckpointManager::List() const {
	std::vector<CheckpointInfo> result;
	std::error_code ec;
	if (!fs::exists(folder, ec))
		return result;

	for (auto& entry : fs::directory_iterator(folder, ec)) {
		if (IsNumberedDir(entry.path()))
			result.push_back(Read(entry.path()));
	}

	std::sort(result.begin(), result.end(),
		[](const CheckpointInfo& a, const CheckpointInfo& b) { return a.timesteps < b.timesteps; });
	return result;
}

std::vector<GGL::CheckpointInfo> GGL::CheckpointManager::ListPolicyVersions() const {
	CheckpointManager sub(folder / "policy_versions");
	return sub.List();
}

GGL::CheckpointInfo GGL::CheckpointManager::Latest() const {
	auto all = List();
	if (all.empty())
		return {};
	return all.back();
}

GGL::CheckpointInfo GGL::CheckpointManager::LatestValidForInference() const {
	auto all = List();
	for (auto it = all.rbegin(); it != all.rend(); ++it)
		if (it->IsValidForInference())
			return *it;
	return {};
}

GGL::CheckpointInfo GGL::CheckpointManager::Get(int64_t timesteps) const {
	for (auto& info : List())
		if (info.timesteps == timesteps)
			return info;
	return {};
}

bool GGL::CheckpointManager::Validate(const fs::path& dir, std::string& outError) const {
	std::error_code ec;
	if (!fs::exists(dir, ec)) {
		outError = "Checkpoint directory does not exist: " + dir.string();
		return false;
	}

	CheckpointInfo info = Read(dir);
	if (!info.hasPolicy) { outError = "Missing or empty POLICY.lt"; return false; }
	if (!info.hasSharedHead) { outError = "Missing or empty SHARED_HEAD.lt"; return false; }
	if (!info.hasStats) { outError = "Missing or unreadable RUNNING_STATS.json"; return false; }

	outError.clear();
	return true;
}

bool GGL::CheckpointManager::Export(int64_t timesteps, const fs::path& destFolder, std::string& outError) const {
	CheckpointInfo info = Get(timesteps);
	if (info.timesteps < 0) {
		outError = "No checkpoint with timesteps=" + std::to_string(timesteps);
		return false;
	}
	if (!info.IsValidForInference()) {
		outError = "Checkpoint is missing inference files (POLICY.lt / SHARED_HEAD.lt)";
		return false;
	}

	std::error_code ec;
	fs::create_directories(destFolder, ec);
	if (ec) {
		outError = "Failed to create destination folder: " + ec.message();
		return false;
	}

	const char* files[] = { "POLICY.lt", "SHARED_HEAD.lt", "RUNNING_STATS.json" };
	for (const char* name : files) {
		fs::path src = info.path / name;
		if (!fs::exists(src, ec))
			continue; // RUNNING_STATS.json is optional for deployment
		fs::copy_file(src, destFolder / name, fs::copy_options::overwrite_existing, ec);
		if (ec) {
			outError = std::string("Failed to copy ") + name + ": " + ec.message();
			return false;
		}
	}

	outError.clear();
	return true;
}

int GGL::CheckpointManager::Prune(int keepN) const {
	if (keepN < 0)
		return 0;

	auto all = List(); // ascending by timesteps
	int toRemove = (int)all.size() - keepN;
	if (toRemove <= 0)
		return 0;

	int removed = 0;
	std::error_code ec;
	for (int i = 0; i < toRemove; i++) {
		fs::remove_all(all[i].path, ec);
		if (!ec)
			removed++;
	}
	return removed;
}
