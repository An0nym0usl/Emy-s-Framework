#include "OpponentPool.h"

#include <private/GigaLearnCPP/FrameworkTorch.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/Util/Utils.h>
#include <RLGymCPP/ObsBuilders/NextoActionLookup.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <nlohmann/json.hpp>
#include <cctype>
#include <limits>
#include <torch/csrc/jit/serialization/import.h>

using namespace nlohmann;

namespace GGL {

	namespace {
		std::vector<int> ParseIntList(const json& item, const char* key) {
			std::vector<int> out;
			if (!item.contains(key) || !item[key].is_array())
				return out;
			for (const auto& v : item[key]) {
				if (v.is_number_integer())
					out.push_back(v.get<int>());
				else if (v.is_number())
					out.push_back((int)v.get<double>());
			}
			return out;
		}

		RLGC::ObsBuilder* MakeObsForSize(int obsSize) {
			if (obsSize == 225)
				return new RLGC::AdvancedObsPadded();
			if (obsSize == 167)
				return new RLGC::AdvancedObs();
			return nullptr;
		}
	}

	void OpponentPool::FreeEntry(OpponentEntry& e) {
		e.models.Free();
		delete e.obsBuilder;
		e.obsBuilder = nullptr;
		delete e.actionParser;
		e.actionParser = nullptr;
		e.jitModule.reset();
	}

	OpponentPool::~OpponentPool() {
		for (auto& e : entries)
			FreeEntry(e);
		entries.clear();
	}

	std::array<float, 8> OpponentPool::ActionToContinuous(const RLGC::Action& a) {
		return {
			a.throttle,
			a.steer,
			a.pitch,
			a.yaw,
			a.roll,
			a.jump > 0 ? 1.f : -1.f,
			a.boost > 0 ? 1.f : -1.f,
			a.handbrake > 0 ? 1.f : -1.f
		};
	}

	int OpponentPool::ContinuousToDefaultActionIndex(const std::array<float, 8>& controls) {
		// Static LUT — DefaultAction space is fixed for default 1v1 discrete students.
		static RLGC::DefaultAction lut;
		static bool ready = false;
		static std::vector<std::array<float, 8>> rows;
		if (!ready) {
			rows.resize(lut.actions.size());
			for (size_t i = 0; i < lut.actions.size(); i++)
				rows[i] = ActionToContinuous(lut.actions[i]);
			ready = true;
		}
		int best = 0;
		float bestD = 1e30f;
		for (int i = 0; i < (int)rows.size(); i++) {
			float d = 0.f;
			for (int k = 0; k < 8; k++) {
				float e = controls[k] - rows[i][k];
				d += e * e;
			}
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		return best;
	}

	void OpponentPool::ClearPrevActions() {
		_prevActions.clear();
	}

	void OpponentPool::SetPrevAction(int globalPlayerIdx, const std::array<float, 8>& action) {
		_prevActions[globalPlayerIdx] = action;
	}

	float OpponentPool::TotalWeight() const {
		float sum = 0.f;
		for (const auto& e : entries)
			sum += e.weight;
		return sum;
	}

	OpponentEntry* OpponentPool::Pick() {
		if (entries.empty())
			return nullptr;

		float total = TotalWeight();
		if (total <= 0.f)
			return &entries[RocketSim::Math::RandInt(0, (int)entries.size())];

		float r = RocketSim::Math::RandFloat() * total;
		float acc = 0.f;
		for (auto& e : entries) {
			acc += e.weight;
			if (r <= acc)
				return &e;
		}
		return &entries.back();
	}

	std::array<float, 8> OpponentPool::LookupNextoAction(int actionIdx) const {
		if (actionIdx < 0 || actionIdx >= (int)_nextoLookup.size())
			return {};
		return _nextoLookup[actionIdx];
	}

	void OpponentPool::ResolvePlayerArena(
		const std::vector<RLGC::GameState>& gameStates,
		const std::vector<int>& arenaPlayerStartIdx,
		int globalIdx,
		int& arenaIdx,
		int& localIdx) {

		arenaIdx = 0;
		localIdx = globalIdx;
		for (int a = 0; a < (int)arenaPlayerStartIdx.size(); a++) {
			int start = arenaPlayerStartIdx[a];
			int count = (int)gameStates[a].players.size();
			if (globalIdx >= start && globalIdx < start + count) {
				arenaIdx = a;
				localIdx = globalIdx - start;
				return;
			}
		}
	}

	std::array<float, 8> OpponentPool::ParseNectoControls(const int64_t* a) {
		float throttle = (float)(a[0] - 1);
		float steer = (float)(a[1] - 1);
		float jump = (float)a[2];
		float boost = (float)a[3];
		float handbrake = (float)a[4];
		return {
			throttle,
			steer,
			throttle,
			steer * (1.f - handbrake),
			steer * handbrake,
			jump,
			boost,
			handbrake
		};
	}

	void OpponentPool::Load(
		const OpponentPoolConfig& config,
		PPOLearner* ppo,
		at::Device device) {

		for (auto& e : entries)
			FreeEntry(e);
		entries.clear();
		_nextoLookup = RLGC::MakeNextoLookupTable();
		_device = device;

		if (!config.enabled)
			return;

		try {
			if (!std::filesystem::exists(config.folder))
				std::filesystem::create_directories(config.folder);

			if (std::filesystem::exists(config.manifest))
				LoadManifest(config.manifest, config, ppo);
			else
				TryAutoDiscover(config, ppo);
		} catch (const std::exception& ex) {
			// Uncaught filesystem errors (e.g. create_directories under system32) used to
			// std::terminate with 0xc0000409 right after PolicyVersionManager::LoadVersions.
			RG_LOG("OpponentPool: load aborted (" << ex.what() << ") folder=" << config.folder
				<< " manifest=" << config.manifest);
			for (auto& e : entries)
				FreeEntry(e);
			entries.clear();
			return;
		}

		if (!config.quiet) {
			RG_LOG("OpponentPool: loaded " << entries.size() << " opponent(s) from " << config.folder);
			for (const auto& e : entries)
				RG_LOG("  > " << e.name
					<< " (weight=" << e.weight
					<< (e.UsesOwnObs() ? (", arch-obs=" + std::to_string(e.obsSize)) : "")
					<< ")");
		}
	}

	static OpponentKind ParseKind(const std::string& typeStr) {
		if (typeStr == "gigalearn" || typeStr == "GIGALEARN")
			return OpponentKind::GIGALEARN;
		if (typeStr == "necto_jit" || typeStr == "NECTO_JIT")
			return OpponentKind::NECTO_JIT;
		return OpponentKind::NEXTO_JIT;
	}

	void OpponentPool::TryAutoDiscover(const OpponentPoolConfig& config, PPOLearner* ppo) {
		if (!std::filesystem::exists(config.folder))
			return;

		for (const auto& entry : std::filesystem::directory_iterator(config.folder)) {
			if (!entry.is_regular_file())
				continue;
			auto path = entry.path();
			auto fname = path.filename().string();
			if (fname == "nexto-model.pt" || fname == "necto-model.pt") {
				OpponentEntry e = {};
				e.name = path.stem().string();
				e.kind = (fname.find("necto") != std::string::npos && fname.find("nexto") == std::string::npos)
					? OpponentKind::NECTO_JIT : OpponentKind::NEXTO_JIT;
				e.weight = 1.f;
				e.beta = 1.f;
				e.jitModule = std::make_unique<torch::jit::script::Module>(torch::jit::load(path.string()));
				e.jitModule->to(_device);
				e.jitModule->eval();
				entries.push_back(std::move(e));
			}
		}

		for (const auto& entry : std::filesystem::directory_iterator(config.folder)) {
			if (!entry.is_directory())
				continue;
			auto path = entry.path();
			if (std::filesystem::exists(path / "POLICY.lt") && std::filesystem::exists(path / "SHARED_HEAD.lt")) {
				try {
					ModelSet models = ppo->GetPolicyModels().CloneAll();
					models.Load(path, false, false);
					OpponentEntry e = {};
					e.name = path.filename().string();
					e.kind = OpponentKind::GIGALEARN;
					e.weight = 1.f;
					e.models = models;
					entries.push_back(std::move(e));
				} catch (std::exception& ex) {
					RG_LOG("OpponentPool: skipped GigaLearn checkpoint " << path << " (" << ex.what() << ")");
				}
			}
			auto nextoPt = path / "nexto-model.pt";
			if (std::filesystem::exists(nextoPt)) {
				OpponentEntry e = {};
				e.name = path.filename().string();
				e.kind = OpponentKind::NEXTO_JIT;
				e.weight = 1.f;
				e.beta = 1.f;
				e.jitModule = std::make_unique<torch::jit::script::Module>(torch::jit::load(nextoPt.string()));
				e.jitModule->to(_device);
				e.jitModule->eval();
				entries.push_back(std::move(e));
			}
			auto nectoPt = path / "necto-model.pt";
			if (std::filesystem::exists(nectoPt)) {
				OpponentEntry e = {};
				e.name = path.filename().string();
				e.kind = OpponentKind::NECTO_JIT;
				e.weight = 1.f;
				e.beta = 1.f;
				e.jitModule = std::make_unique<torch::jit::script::Module>(torch::jit::load(nectoPt.string()));
				e.jitModule->to(_device);
				e.jitModule->eval();
				entries.push_back(std::move(e));
			}
		}
	}

	void OpponentPool::LoadManifest(const std::filesystem::path& manifestPath, const OpponentPoolConfig& config, PPOLearner* ppo) {
		std::ifstream fIn(manifestPath);
		if (!fIn.good()) {
			RG_LOG("OpponentPool: could not read manifest " << manifestPath);
			return;
		}

		json j = json::parse(fIn);
		if (!j.contains("entries") || !j["entries"].is_array()) {
			RG_LOG("OpponentPool: manifest missing 'entries' array");
			return;
		}

		auto baseDir = manifestPath.parent_path();

		for (const auto& item : j["entries"]) {
			OpponentEntry e = {};
			e.name = item.value("name", "unnamed");
			e.weight = item.value("weight", 1.f);
			e.beta = item.value("beta", 1.f);
			e.beatBonusScale = item.value("beat_bonus_scale", 1.f);
			e.concedePenaltyScale = item.value("concede_penalty_scale", 1.f);
			e.kind = ParseKind(item.value("type", "nexto_jit"));
			e.obsSize = item.value("obs_size", 0);

			std::filesystem::path modelPath = item.value("model", std::string{});
			if (modelPath.empty() && item.contains("path"))
				modelPath = item["path"].get<std::string>();
			if (!modelPath.empty() && modelPath.is_relative())
				modelPath = baseDir / modelPath;

			try {
				if (e.kind == OpponentKind::GIGALEARN) {
					if (!std::filesystem::exists(modelPath)) {
						RG_LOG("OpponentPool: gigalearn path missing, skip " << e.name << ": " << modelPath);
						continue;
					}
					// Latest numbered checkpoint folder if pointing at a box root.
					if (!std::filesystem::exists(modelPath / "POLICY.lt")) {
						int64_t bestTs = -1;
						std::filesystem::path best;
						if (std::filesystem::is_directory(modelPath)) {
							for (const auto& sub : std::filesystem::directory_iterator(modelPath)) {
								if (!sub.is_directory()) continue;
								auto name = sub.path().filename().string();
								bool digits = !name.empty();
								for (char c : name) {
									if (!std::isdigit((unsigned char)c)) { digits = false; break; }
								}
								if (!digits) continue;
								if (!std::filesystem::exists(sub.path() / "POLICY.lt")) continue;
								int64_t ts = std::stoll(name);
								if (ts > bestTs) { bestTs = ts; best = sub.path(); }
							}
						}
						if (!best.empty())
							modelPath = best;
					}
					if (!std::filesystem::exists(modelPath / "POLICY.lt")) {
						RG_LOG("OpponentPool: no POLICY.lt for " << e.name << " @ " << modelPath);
						continue;
					}

					auto sharedLayers = ParseIntList(item, "shared_head");
					auto policyLayers = ParseIntList(item, "policy");
					const bool archAware = e.obsSize > 0 && !sharedLayers.empty() && !policyLayers.empty();

					if (archAware) {
						e.obsBuilder = MakeObsForSize(e.obsSize);
						if (!e.obsBuilder) {
							RG_LOG("OpponentPool: unsupported obs_size=" << e.obsSize << " for " << e.name);
							continue;
						}
						e.actionParser = new RLGC::DefaultAction();
						e.policyType = PolicyType::DISCRETE;

						PartialModelConfig sharedCfg = {};
						sharedCfg.layerSizes = sharedLayers;
						sharedCfg.addLayerNorm = item.value("layer_norm", true);
						sharedCfg.addOutputLayer = false;
						sharedCfg.activationType = ModelActivationType::RELU;
						sharedCfg.optimType = ModelOptimType::ADAM;

						PartialModelConfig policyCfg = {};
						policyCfg.layerSizes = policyLayers;
						policyCfg.addLayerNorm = item.value("layer_norm", true);
						policyCfg.addOutputLayer = true;
						policyCfg.activationType = ModelActivationType::RELU;
						policyCfg.optimType = ModelOptimType::ADAM;

						int numActions = e.actionParser->GetActionAmount();
						PPOLearner::MakeModels(
							false, e.obsSize, numActions,
							sharedCfg, policyCfg, {},
							_device, e.models,
							PolicyType::DISCRETE, 8
						);
						e.models.Load(modelPath, false, false);
						if (!config.quiet)
							RG_LOG("OpponentPool: arch-aware gigalearn \"" << e.name
								<< "\" obs=" << e.obsSize << " from " << modelPath);
					} else {
						// Same-arch clone of student policy (legacy).
						ModelSet models = ppo->GetPolicyModels().CloneAll();
						models.Load(modelPath, false, false);
						e.models = models;
						e.obsSize = 0;
					}
				} else {
					if (!std::filesystem::exists(modelPath)) {
						RG_LOG("OpponentPool: model not found, skipping " << e.name << ": " << modelPath);
						continue;
					}
					e.jitModule = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath.string()));
					e.jitModule->to(_device);
					e.jitModule->eval();
				}
				entries.push_back(std::move(e));
			} catch (std::exception& ex) {
				RG_LOG("OpponentPool: failed to load " << e.name << ": " << ex.what());
				FreeEntry(e);
			}
		}

		if (entries.empty())
			TryAutoDiscover(config, ppo);
	}

	torch::Tensor OpponentPool::InferNectoBatch(
		OpponentEntry& opponent,
		const std::vector<RLGC::NectoObsTuple>& obsBatch,
		int batchSize) const {

		RG_ASSERT(opponent.jitModule != nullptr);
		RG_ASSERT(batchSize > 0);

		int nEntities = obsBatch[0].nEntities;
		std::vector<float> allQ, allKv, allM;
		allQ.reserve(batchSize * RLGC::NectoObsTuple::Q_SIZE);
		allKv.reserve(batchSize * nEntities * RLGC::NectoObsTuple::FEAT);
		allM.reserve(batchSize * nEntities);

		for (int i = 0; i < batchSize; i++) {
			allQ.insert(allQ.end(), obsBatch[i].q.begin(), obsBatch[i].q.end());
			allKv.insert(allKv.end(), obsBatch[i].kv.begin(), obsBatch[i].kv.end());
			allM.insert(allM.end(), obsBatch[i].m.begin(), obsBatch[i].m.end());
		}

		auto tQ = torch::tensor(allQ).reshape({ batchSize, 1, RLGC::NectoObsTuple::Q_SIZE }).to(_device);
		auto tKv = torch::tensor(allKv).reshape({ batchSize, nEntities, RLGC::NectoObsTuple::FEAT }).to(_device);
		auto tM = torch::tensor(allM).reshape({ batchSize, nEntities }).to(_device);

		std::vector<torch::jit::IValue> inputs = { tQ, tKv, tM };
		auto output = opponent.jitModule->forward(inputs);

		std::vector<torch::Tensor> heads;
		if (output.isTuple()) {
			auto tuple = output.toTuple();
			auto outVal = tuple->elements()[0];
			if (outVal.isTensor()) {
				heads.push_back(outVal.toTensor());
			} else if (outVal.isTensorList()) {
				for (const auto& t : outVal.toTensorList())
					heads.push_back(t);
			} else if (outVal.isTuple()) {
				for (const auto& el : outVal.toTuple()->elements()) {
				 if (el.isTensor())
						heads.push_back(el.toTensor());
				}
			}
		} else if (output.isTensor()) {
			heads.push_back(output.toTensor());
		}

		RG_ASSERT(!heads.empty());

		int64_t maxShape = 0;
		for (const auto& h : heads)
			maxShape = RS_MAX(maxShape, h.size(-1));

		float padVal = opponent.beta >= 0.f ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
		std::vector<torch::Tensor> padded;
		for (auto h : heads) {
			if (h.dim() == 1)
				h = h.unsqueeze(0);
			if (h.size(-1) < maxShape) {
				auto pad = torch::full({ h.size(0), maxShape - h.size(-1) }, padVal, h.options());
				h = torch::cat({ h, pad }, -1);
			}
			padded.push_back(h);
		}

		auto stacked = torch::stack(padded, 0).swapdims(0, 1);
		if (stacked.dim() > 2)
			stacked = stacked.squeeze(1);

		torch::Tensor actions;
		if (opponent.beta >= 0.999f) {
			actions = stacked.argmax(-1);
		} else if (opponent.beta <= -0.999f) {
			actions = stacked.argmin(-1);
		} else {
			auto scaled = stacked.clone();
			if (opponent.beta <= 0.001f) {
				scaled = torch::where(torch::isfinite(scaled), torch::zeros_like(scaled), scaled);
			} else {
				float factor = std::log((opponent.beta + 1.f) / (1.f - opponent.beta)) / std::log(3.f);
				scaled = scaled * factor;
			}
			actions = torch::multinomial(torch::softmax(scaled, -1), 1, true).squeeze(-1);
		}

		return actions.cpu().reshape({ batchSize, (int64_t)heads.size() });
	}

	torch::Tensor OpponentPool::InferNextoBatch(
		OpponentEntry& opponent,
		const std::vector<RLGC::NextoObsTuple>& obsBatch,
		int batchSize) const {

		RG_ASSERT(opponent.jitModule != nullptr);
		RG_ASSERT(batchSize > 0);

		int nEntities = obsBatch[0].nEntities;
		std::vector<float> allQ, allKv, allM;
		allQ.reserve(batchSize * RLGC::NextoObsTuple::Q_SIZE);
		allKv.reserve(batchSize * nEntities * RLGC::NextoObsTuple::FEAT);
		allM.reserve(batchSize * nEntities);

		for (int i = 0; i < batchSize; i++) {
			allQ.insert(allQ.end(), obsBatch[i].q.begin(), obsBatch[i].q.end());
			allKv.insert(allKv.end(), obsBatch[i].kv.begin(), obsBatch[i].kv.end());
			allM.insert(allM.end(), obsBatch[i].m.begin(), obsBatch[i].m.end());
		}

		auto tQ = torch::tensor(allQ).reshape({ batchSize, 1, RLGC::NextoObsTuple::Q_SIZE }).to(_device);
		auto tKv = torch::tensor(allKv).reshape({ batchSize, nEntities, RLGC::NextoObsTuple::FEAT }).to(_device);
		auto tM = torch::tensor(allM).reshape({ batchSize, nEntities }).to(_device);

		std::vector<torch::jit::IValue> inputs = { tQ, tKv, tM };
		auto output = opponent.jitModule->forward(inputs);

		torch::Tensor logits;
		if (output.isTuple()) {
			auto tuple = output.toTuple();
			logits = tuple->elements()[0].toTensor();
			if (logits.dim() == 3)
				logits = logits.select(1, 0);
		} else {
			logits = output.toTensor();
		}

		if (logits.dim() == 1)
			logits = logits.unsqueeze(0);

		// Pad logits to common width (Nexto agent.py)
		int64_t maxShape = logits.size(-1);
		if (logits.dim() >= 2) {
			for (int64_t b = 0; b < logits.size(0); b++) {
				auto row = logits[b];
				if (row.size(-1) < maxShape) {
					auto pad = torch::full({ maxShape - row.size(-1) }, -std::numeric_limits<float>::infinity(), row.options());
					logits[b] = torch::cat({ row, pad }, -1);
				}
			}
		}

		torch::Tensor actions;
		if (opponent.beta >= 0.999f) {
			actions = logits.argmax(-1);
		} else if (opponent.beta <= -0.999f) {
			actions = logits.argmin(-1);
		} else {
			auto scaled = logits.clone();
			if (opponent.beta <= 0.001f) {
				scaled = torch::where(torch::isfinite(scaled), torch::zeros_like(scaled), scaled);
			} else {
				float factor = std::log((opponent.beta + 1.f) / (1.f - opponent.beta)) / std::log(3.f);
				scaled = scaled * factor;
			}
			actions = torch::multinomial(torch::softmax(scaled, -1), 1, true).squeeze(-1);
		}

		return actions.cpu();
	}

	void OpponentPool::InferOpponentActions(
		OpponentEntry& opponent,
		PPOLearner* ppo,
		const std::vector<RLGC::GameState>& gameStates,
		const std::vector<int>& arenaPlayerStartIdx,
		const std::vector<int>& opponentPlayerIndices,
		torch::Tensor tStates,
		torch::Tensor tActionMasks,
		int contActionDim,
		torch::Tensor* outActions) {

		RG_NO_GRAD;
		const int n = (int)opponentPlayerIndices.size();
		const bool studentDiscrete = (ppo->config.policyType == PolicyType::DISCRETE);
		if (n == 0) {
			*outActions = studentDiscrete
				? torch::zeros({ 0 }, torch::kLong)
				: torch::zeros({ 0, contActionDim });
			return;
		}

		if (opponent.kind == OpponentKind::GIGALEARN) {
			if (opponent.UsesOwnObs()) {
				InferGigalearnOwnObs(
					opponent, ppo, gameStates, arenaPlayerStartIdx,
					opponentPlayerIndices, contActionDim, outActions);
				return;
			}
			auto tIdx = torch::tensor(opponentPlayerIndices);
			auto tdStates = tStates.index_select(0, tIdx).to(_device, true);
			torch::Tensor tActs;

			if (ppo->config.policyType == PolicyType::CONTINUOUS) {
				torch::Tensor tLogProbs;
				PPOLearner::SampleContinuousActions(
					opponent.models, tdStates, true, ppo->config.useHalfPrecision,
					ppo->config.varMin, ppo->config.varMax, &tActs, &tLogProbs);
				*outActions = tActs.reshape({ n, contActionDim }).cpu();
			} else {
				auto tdMasks = tActionMasks.index_select(0, tIdx).to(_device, true);
				PPOLearner::InferActionsFromModels(
					opponent.models, tdStates, tdMasks, true, ppo->config.policyTemperature,
					ppo->config.useHalfPrecision, &tActs, nullptr);
				// Discrete student expects 1D Long indices (match InferActions dtype).
				*outActions = tActs.to(torch::kLong).cpu().flatten();
			}
			return;
		}

		// Nexto / Necto JIT → continuous controls; map to discrete indices when student is discrete.
		if (opponent.kind == OpponentKind::NECTO_JIT) {
			std::vector<RLGC::NectoObsTuple> obsBatch;
			obsBatch.reserve(n);

			for (int gIdx : opponentPlayerIndices) {
				int arenaIdx = 0, localIdx = gIdx;
				ResolvePlayerArena(gameStates, arenaPlayerStartIdx, gIdx, arenaIdx, localIdx);
				auto& gs = gameStates[arenaIdx];
				auto& player = gs.players[localIdx];

				std::array<float, 8> prev = {};
				auto it = _prevActions.find(gIdx);
				if (it != _prevActions.end())
					prev = it->second;

				obsBatch.push_back(_nectoObsBuilders[arenaIdx].Build(player, gs, prev));
			}

			auto actionTensor = InferNectoBatch(opponent, obsBatch, n);
			auto flat = TENSOR_TO_VEC<int64_t>(actionTensor.flatten());

			if (studentDiscrete) {
				std::vector<int64_t> idxs(n);
				for (int i = 0; i < n; i++) {
					auto controls = ParseNectoControls(&flat[i * 5]);
					idxs[i] = (int64_t)ContinuousToDefaultActionIndex(controls);
					SetPrevAction(opponentPlayerIndices[i], controls);
				}
				*outActions = torch::tensor(idxs, torch::kLong);
				return;
			}

			std::vector<float> outFlat;
			outFlat.reserve(n * contActionDim);
			for (int i = 0; i < n; i++) {
				auto controls = ParseNectoControls(&flat[i * 5]);
				for (int d = 0; d < contActionDim; d++)
					outFlat.push_back(d < 8 ? controls[d] : 0.f);
				SetPrevAction(opponentPlayerIndices[i], controls);
			}
			*outActions = torch::tensor(outFlat).reshape({ n, contActionDim });
			return;
		}

		RLGC::NextoObsBuilder obsBuilder;
		std::vector<RLGC::NextoObsTuple> obsBatch;
		obsBatch.reserve(n);

		for (int gIdx : opponentPlayerIndices) {
			int arenaIdx = 0, localIdx = gIdx;
			ResolvePlayerArena(gameStates, arenaPlayerStartIdx, gIdx, arenaIdx, localIdx);
			auto& gs = gameStates[arenaIdx];
			auto& player = gs.players[localIdx];

			std::array<float, 8> prev = {};
			auto it = _prevActions.find(gIdx);
			if (it != _prevActions.end())
				prev = it->second;

			obsBatch.push_back(obsBuilder.Build(player, gs, prev));
		}

		auto actionIndices = InferNextoBatch(opponent, obsBatch, n);
		auto idxVec = TENSOR_TO_VEC<int64_t>(actionIndices.flatten());

		if (studentDiscrete) {
			std::vector<int64_t> idxs(n);
			for (int i = 0; i < n; i++) {
				auto controls = LookupNextoAction((int)idxVec[i]);
				idxs[i] = (int64_t)ContinuousToDefaultActionIndex(controls);
				SetPrevAction(opponentPlayerIndices[i], controls);
			}
			*outActions = torch::tensor(idxs, torch::kLong);
			return;
		}

		std::vector<float> flat;
		flat.reserve(n * contActionDim);
		for (int i = 0; i < n; i++) {
			auto controls = LookupNextoAction((int)idxVec[i]);
			for (int d = 0; d < contActionDim; d++)
				flat.push_back(d < 8 ? controls[d] : 0.f);
			SetPrevAction(opponentPlayerIndices[i], controls);
		}

		*outActions = torch::tensor(flat).reshape({ n, contActionDim });
	}

	void OpponentPool::InferGigalearnOwnObs(
		OpponentEntry& opponent,
		PPOLearner* ppo,
		const std::vector<RLGC::GameState>& gameStates,
		const std::vector<int>& arenaPlayerStartIdx,
		const std::vector<int>& opponentPlayerIndices,
		int contActionDim,
		torch::Tensor* outActions) const {

		RG_NO_GRAD;
		const int n = (int)opponentPlayerIndices.size();
		const bool studentDiscrete = (ppo->config.policyType == PolicyType::DISCRETE);
		if (n == 0) {
			*outActions = studentDiscrete
				? torch::zeros({ 0 }, torch::kLong)
				: torch::zeros({ 0, contActionDim });
			return;
		}

		std::vector<float> allObs;
		std::vector<uint8_t> allMasks;
		std::vector<std::pair<const RLGC::Player*, const RLGC::GameState*>> ctx;
		ctx.reserve(n);
		allObs.reserve(n * opponent.obsSize);
		allMasks.reserve(n * opponent.actionParser->GetActionAmount());

		for (int gIdx : opponentPlayerIndices) {
			int arenaIdx = 0, localIdx = gIdx;
			ResolvePlayerArena(gameStates, arenaPlayerStartIdx, gIdx, arenaIdx, localIdx);
			const auto& gs = gameStates[arenaIdx];
			const auto& player = gs.players[localIdx];
			ctx.push_back({ &player, &gs });

			FList curObs = opponent.obsBuilder->BuildObs(player, gs);
			if ((int)curObs.size() != opponent.obsSize) {
				RG_ERR_CLOSE("OpponentPool: obs size mismatch for \"" << opponent.name
					<< "\" expected " << opponent.obsSize << " got " << curObs.size());
			}
			allObs.insert(allObs.end(), curObs.begin(), curObs.end());
			auto mask = opponent.actionParser->GetActionMask(player, gs);
			allMasks.insert(allMasks.end(), mask.begin(), mask.end());
		}

		auto tObs = torch::tensor(allObs).reshape({ n, opponent.obsSize }).to(_device);
		auto tMasks = torch::tensor(allMasks).reshape({
			n, opponent.actionParser->GetActionAmount()
		}).to(_device);

		torch::Tensor tActs;
		PPOLearner::InferActionsFromModels(
			opponent.models, tObs, tMasks,
			true, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
			&tActs, nullptr
		);

		auto actionIndices = TENSOR_TO_VEC<int>(tActs.cpu());
		if (studentDiscrete) {
			// Prefer remapping through continuous→DefaultAction so arch-specific parsers
			// (different discrete sizes) still land in the student action space.
			std::vector<int64_t> studentIdxs(n);
			for (int i = 0; i < n; i++) {
				RLGC::Action act = opponent.actionParser->ParseAction(
					actionIndices[i], *ctx[i].first, *ctx[i].second);
				studentIdxs[i] = (int64_t)ContinuousToDefaultActionIndex(ActionToContinuous(act));
			}
			*outActions = torch::tensor(studentIdxs, torch::kLong);
			return;
		}

		std::vector<float> flat;
		flat.reserve(n * contActionDim);
		for (int i = 0; i < n; i++) {
			RLGC::Action act = opponent.actionParser->ParseAction(
				actionIndices[i], *ctx[i].first, *ctx[i].second);
			auto controls = ActionToContinuous(act);
			for (int d = 0; d < contActionDim; d++)
				flat.push_back(d < 8 ? controls[d] : 0.f);
		}
		*outActions = torch::tensor(flat).reshape({ n, contActionDim });
	}

}
