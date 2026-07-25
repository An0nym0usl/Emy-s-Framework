#include "AutoTrainerBridge.h"
#include "TrainingCurriculum.h"

#include <RLGymCPP/RewardCore/RuntimeRewardRegistry.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <sstream>
#include <thread>
#include <unordered_map>

#ifdef GIGA_USE_CUDA_SIM
#include <GigaLearnCPP/Sim/CudaEnvSet.h>
#endif

using json = nlohmann::json;

namespace GGL {

	namespace {
		std::filesystem::path Root(const Learner* learner) {
			return AutoTrainerBridge::RootDir(learner);
		}

		json ReadJson(const std::filesystem::path& path) {
			std::ifstream f(path);
			if (!f.good()) return json::object();
			try {
				json j; f >> j; return j;
			} catch (...) {
				return json::object();
			}
		}

		void WriteJsonAtomic(const std::filesystem::path& path, const json& j) {
			std::filesystem::create_directories(path.parent_path());
			auto tmp = path.string() + ".tmp";
			{
				std::ofstream f(tmp);
				f << j.dump(2);
				f.flush();
			}
			std::error_code ec;
			std::filesystem::remove(path, ec); // Windows rename fails if dest exists
			ec.clear();
			std::filesystem::rename(tmp, path, ec);
			if (ec) {
				ec.clear();
				std::filesystem::copy_file(tmp, path, std::filesystem::copy_options::overwrite_existing, ec);
				std::filesystem::remove(tmp, ec);
			}
		}

		int64_t UnixNow() {
			return std::chrono::duration_cast<std::chrono::seconds>(
				std::chrono::system_clock::now().time_since_epoch()).count();
		}

		bool EnvTruthy(const char* name) {
			const char* v = std::getenv(name);
			return v && v[0] && std::atoi(v) != 0;
		}

		bool IsReadOnly() {
			return EnvTruthy("GIGA_AT_READONLY");
		}

		bool WantsFullControl(const json& ov) {
			if (IsReadOnly() || EnvTruthy("GIGA_NO_SSL_AUTONOMY"))
				return false;
			if (ov.contains("ssl_autonomy") && ov["ssl_autonomy"].is_boolean()
				&& !ov["ssl_autonomy"].get<bool>())
				return false;
			if (ov.contains("autotrainer_full_control"))
				return ov["autotrainer_full_control"].get<bool>();
			if (ov.contains("full_control"))
				return ov["full_control"].get<bool>();
			// Default ON so from-scratch AutoTrainer wins over Apex without YAML theater.
			return true;
		}

		bool WantsPostApex(const json& ov) {
			if (IsReadOnly())
				return false;
			if (WantsFullControl(ov))
				return true;
			return ov.value("ssl_guide_post_apex", false) || TrainingCurriculum::config.sslGuidePostApex;
		}

		// Host registry always; GPU constant-memory upload only when weights actually change
		// and never while Collect||Learn async PPO owns the device (hang / deadlock).
		void ApplyRewardWeights(Learner* learner, const json& ov, bool logChanges) {
			if (!ov.contains("reward_weights") || !ov["reward_weights"].is_object())
				return;

			std::unordered_map<std::string, float> weights;
			std::ostringstream changed;
			bool first = true;
			for (auto& [k, v] : ov["reward_weights"].items()) {
				if (!v.is_number())
					continue;
				float w = v.get<float>();
				weights[k] = w;
				if (logChanges) {
					if (!first) changed << ", ";
					changed << k << "=" << w;
					first = false;
				}
			}
			RLGC::RuntimeRewardRegistry::Instance().SetAll(weights);

#ifdef GIGA_USE_CUDA_SIM
			static std::string s_lastGpuRewardSig;
			const std::string sig = ov["reward_weights"].dump();
			const bool dirty = (sig != s_lastGpuRewardSig);
			if (learner && learner->cudaEnvSet && dirty) {
				if (learner->AsyncLearnInFlight()) {
					learner->RequestDeferredCudaSurfaceApply();
				} else {
					learner->cudaEnvSet->ApplyRuntimeGpuRewards();
					s_lastGpuRewardSig = sig;
				}
			}
#else
			(void)learner;
#endif
			if (logChanges && !first)
				RG_LOG("[AutoTrainer] adjusted rewards: " << changed.str());
		}

		void ApplyOverrides(Learner* learner, const json& ov, bool postApex, bool logRewards) {
			if (!learner || ov.empty() || IsReadOnly())
				return;

			auto& tc = TrainingCurriculum::config;
			auto& ppo = learner->config.ppo;
			const bool full = WantsFullControl(ov);

			if (ov.contains("chase_end_steps"))
				tc.chaseEndSteps = ov["chase_end_steps"].get<int64_t>();
			if (ov.contains("foundation_end_steps"))
				tc.foundationEndSteps = ov["foundation_end_steps"].get<int64_t>();

			if (ov.contains("opponent_pool_chance")) {
				float c = ov["opponent_pool_chance"].get<float>();
				learner->config.opponentPool.chance = c;
				// Warmup may write 0 - keep pool object but disable draws
				learner->config.opponentPool.enabled = c > 0.f;
			}
			if (ov.contains("train_against_old_chance")) {
				float c = ov["train_against_old_chance"].get<float>();
				learner->config.trainAgainstOldChance = c;
				learner->config.trainAgainstOldVersions = c > 0.f;
			}
			if (ov.contains("opponent_beat_bonus"))
				learner->config.opponentPool.beatBonus = ov["opponent_beat_bonus"].get<float>();
			if (ov.contains("opponent_concede_penalty"))
				learner->config.opponentPool.concedePenalty = ov["opponent_concede_penalty"].get<float>();
			if (ov.contains("save_policy_versions"))
				learner->config.savePolicyVersions = ov["save_policy_versions"].get<bool>();
			if (ov.contains("ts_per_save"))
				learner->config.tsPerSave = ov["ts_per_save"].get<int64_t>();
			if (ov.contains("ts_per_version"))
				learner->config.tsPerVersion = ov["ts_per_version"].get<int64_t>();

			if (ov.contains("entropy_scale")) {
				float e = ov["entropy_scale"].get<float>();
				const float prevE = ppo.entropyScale;
				ppo.entropyScale = e;
				if (postApex || full)
					learner->SetEntropyScale(e);
				// Sticky AutoTrainer HARD RECOVERY - log loudly every few iters
				if (ov.value("entropy_death_recovery", false) && (postApex || full)) {
					static float s_lastRecEnt = -1.f;
					static int64_t s_lastRecLogIter = -1;
					bool changed = std::fabs(e - s_lastRecEnt) > 1e-6f;
					bool periodic = (learner->totalIterations - s_lastRecLogIter) >= 4;
					if (changed || periodic || s_lastRecLogIter < 0) {
						std::string tier = ov.value("entropy_death_tier", std::string("?"));
						RG_LOG("[AutoTrainer] HARD RECOVERY entropy_death -> coef "
							<< prevE << " -> " << e
							<< " tier=" << tier
							<< " epochs=" << (ov.contains("epochs") ? ov["epochs"].get<int>() : ppo.epochs)
							<< " polLR=" << ppo.policyLR
							<< " critLR=" << ppo.criticLR
							<< " es=" << ppo.esNoiseScale
							<< " prio=" << (ppo.prioritySampling ? 1 : 0)
							<< " opp=" << learner->config.opponentPool.chance
							<< " old=" << learner->config.trainAgainstOldChance
							<< " skillEval=" << (learner->config.skillTracker.enabled ? 1 : 0)
							<< " SetEntropyScale=OK");
						s_lastRecEnt = e;
						s_lastRecLogIter = learner->totalIterations;
					}
				}
			}
			if (ov.contains("var_max"))
				ppo.varMax = ov["var_max"].get<float>();
			if (ov.contains("var_min"))
				ppo.varMin = ov["var_min"].get<float>();
			if (ov.contains("epochs"))
				ppo.epochs = ov["epochs"].get<int>();
			if (ov.contains("gae_gamma"))
				ppo.gaeGamma = ov["gae_gamma"].get<float>();
			if (ov.contains("gae_lambda"))
				ppo.gaeLambda = ov["gae_lambda"].get<float>();
			if (ov.contains("policy_lr"))
				ppo.policyLR = ov["policy_lr"].get<float>();
			if (ov.contains("critic_lr"))
				ppo.criticLR = ov["critic_lr"].get<float>();
			if (ov.contains("clip_range"))
				ppo.clipRange = ov["clip_range"].get<float>();
			if (ov.contains("max_grad_norm"))
				ppo.maxGradNorm = ov["max_grad_norm"].get<float>();
			if (ov.contains("es_noise_scale"))
				ppo.esNoiseScale = ov["es_noise_scale"].get<float>();
			if (ov.contains("event_advantage_boost"))
				ppo.eventAdvantageBoost = ov["event_advantage_boost"].get<float>();
			if (ov.contains("priority_sampling"))
				ppo.prioritySampling = ov["priority_sampling"].get<bool>();
			if (ov.contains("mask_entropy"))
				ppo.maskEntropy = ov["mask_entropy"].get<bool>();
			if (ov.contains("max_episode_duration"))
				ppo.maxEpisodeDuration = ov["max_episode_duration"].get<double>();

			// Advanced PPO knobs (framework core - hot via SyncRuntimePPOConfig).
			if (ov.contains("target_kl"))
				ppo.targetKl = (std::max)(0.f, ov["target_kl"].get<float>());
			if (ov.contains("target_entropy"))
				ppo.targetEntropy = ov["target_entropy"].get<float>();
			if (ov.contains("advantage_norm_mode"))
				ppo.advantageNormMode = ov["advantage_norm_mode"].get<int>();
			if (ov.contains("ratio_log_clamp"))
				ppo.ratioLogClamp = (std::max)(0.f, ov["ratio_log_clamp"].get<float>());
			if (ov.contains("grad_accumulation_steps"))
				ppo.gradAccumulationSteps = (std::max)(1, ov["grad_accumulation_steps"].get<int>());
			if (ov.contains("guiding_use_kl"))
				ppo.guidingUseKL = ov["guiding_use_kl"].get<bool>();
			if (ov.contains("guiding_loss_clip"))
				ppo.guidingLossClip = (std::max)(0.f, ov["guiding_loss_clip"].get<float>());
			if (ov.contains("adaptive_kl_lr_scale"))
				ppo.adaptiveKlLrScale = ov["adaptive_kl_lr_scale"].get<float>();

			if (ov.contains("skill_tracker_enabled"))
				learner->config.skillTracker.enabled = ov["skill_tracker_enabled"].get<bool>();
			if (ov.contains("skill_tracker_interval"))
				learner->config.skillTracker.updateInterval = (std::max)(1, ov["skill_tracker_interval"].get<int>());
			if (ov.contains("skill_tracker_ts_per_eval"))
				learner->config.skillTracker.tsPerEval = (std::max)((int64_t)0, ov["skill_tracker_ts_per_eval"].get<int64_t>());
			if (ov.contains("in_sim_eval_ts"))
				learner->config.skillTracker.tsPerEval = (std::max)((int64_t)0, ov["in_sim_eval_ts"].get<int64_t>());

			// LEVEL 4 - Environment Architect overrides
			if (ov.contains("ball_chase_weight"))
				tc.ballChaseWeight = ov["ball_chase_weight"].get<float>();
			if (ov.contains("random_state_weight"))
				tc.randomStateWeight = ov["random_state_weight"].get<float>();
			if (ov.contains("kickoff_weight"))
				tc.kickoffWeight = ov["kickoff_weight"].get<float>();
			if (ov.contains("fuzzed_weight"))
				tc.fuzzedWeight = ov["fuzzed_weight"].get<float>();
			if (ov.contains("aerial_weight"))
				tc.aerialWeight = ov["aerial_weight"].get<float>();
			if (ov.contains("gpu_reset_kickoff"))
				tc.kickoffWeight = ov["gpu_reset_kickoff"].get<float>();
			if (ov.contains("gpu_reset_fuzzed"))
				tc.fuzzedWeight = ov["gpu_reset_fuzzed"].get<float>();
			if (ov.contains("gpu_reset_aerial"))
				tc.aerialWeight = ov["gpu_reset_aerial"].get<float>();
			if (ov.contains("no_touch_seconds")) {
				float nt = ov["no_touch_seconds"].get<float>();
				tc.noTouchSecondsChase = nt;
				tc.noTouchSecondsAdvanced = (std::max)(nt, nt + 2.f);
#ifdef GIGA_USE_CUDA_SIM
				if (learner->cudaEnvSet) {
					static float s_lastNtUploaded = -1.f;
					const bool ntDirty = std::fabs(nt - s_lastNtUploaded) > 1e-6f
						|| std::fabs(nt - learner->cudaEnvSet->noTouchSeconds) > 1e-6f;
					learner->cudaEnvSet->noTouchSeconds = nt;
					if (ntDirty) {
						if (learner->AsyncLearnInFlight()) {
							// Stash host value; Learner flushes ConfigureTrainingTerminals after Learn join.
							learner->RequestDeferredCudaSurfaceApply();
						} else {
							learner->cudaEnvSet->SetNoTouchSeconds(nt);
							s_lastNtUploaded = nt;
						}
					}
				}
#endif
			}
			if (ov.contains("w_icm"))
				tc.wIcm = ov["w_icm"].get<float>();
			if (ov.contains("w_rnd"))
				tc.wRnd = ov["w_rnd"].get<float>();
			if (ov.contains("ssl_guide_post_apex"))
				tc.sslGuidePostApex = ov["ssl_guide_post_apex"].get<bool>();
			if (full)
				tc.sslGuidePostApex = true;

			if (ov.contains("force_phase")) {
				int fp = ov["force_phase"].get<int>();
				if (fp >= 0 && fp <= 2)
					TrainingCurriculum::currentPhase = static_cast<TrainingPhase>(fp);
			}

			ApplyRewardWeights(learner, ov, logRewards);

			// SSL §5 - hardened expert weights (Nexto/Requiem/Necto) when present
			for (auto it = ov.begin(); it != ov.end(); ++it) {
				const std::string& key = it.key();
				static const std::string kPrefix = "opponent_weight_";
				if (key.size() > kPrefix.size() && key.compare(0, kPrefix.size(), kPrefix) == 0) {
					std::string name = key.substr(kPrefix.size());
					if (it.value().is_number())
						learner->SetOpponentWeight(name, it.value().get<float>());
				}
			}

			if (ov.contains("note") && !postApex)
				RG_LOG("AutoTrainer: " << ov["note"].get<std::string>());

			// --- Soft TransferLearn / BC distill (same-arch guiding teacher) ---
			{
				const bool distillOn = ov.value("distill_soft", false) || ov.value("teachers_bc", false);
				const bool distillOff = ov.value("distill_off", false)
					|| (ov.contains("teachers_active") && !ov["teachers_active"].get<bool>());
				float coef = ov.value("distill_coef", ov.value("guiding_strength", 0.f));
				std::filesystem::path guidePath;
				if (ov.contains("guiding_policy_path") && ov["guiding_policy_path"].is_string())
					guidePath = ov["guiding_policy_path"].get<std::string>();
				else if (ov.contains("distill_teacher_path") && ov["distill_teacher_path"].is_string())
					guidePath = ov["distill_teacher_path"].get<std::string>();

				static std::string s_lastGuide;
				static float s_lastCoef = -1.f;
				static bool s_guideOn = false;

				if (distillOff || (!distillOn && coef <= 0.f)) {
					if (s_guideOn) {
#ifdef GIGA_USE_CUDA_SIM
						if (learner->AsyncLearnInFlight()) {
							learner->RequestDeferredCudaSurfaceApply();
						} else
#endif
						{
							learner->ClearSoftDistillTeacher();
							s_guideOn = false;
							s_lastGuide.clear();
							s_lastCoef = -1.f;
						}
					}
				} else if (distillOn && coef > 0.f && !guidePath.empty()) {
					std::string key = guidePath.string();
					if (!s_guideOn || key != s_lastGuide || std::fabs(coef - s_lastCoef) > 1e-6f) {
#ifdef GIGA_USE_CUDA_SIM
						if (learner->AsyncLearnInFlight()) {
							// Loading teacher weights while async Learn runs can deadlock the CUDA context.
							learner->RequestDeferredCudaSurfaceApply();
						} else
#endif
						{
							bool ok = learner->SetSoftDistillTeacher(guidePath, coef);
							s_guideOn = ok;
							s_lastGuide = ok ? key : "";
							s_lastCoef = ok ? coef : -1.f;
							if (ok && !postApex) {
								RG_LOG("[AutoTrainer] soft distill BC ON path=" << guidePath
									<< " coef=" << coef);
							} else if (!ok && !postApex) {
								RG_LOG("[AutoTrainer] soft distill BC skipped (arch mismatch or missing): "
									<< guidePath);
							}
						}
					}
				}
			}

#ifdef GIGA_USE_CUDA_SIM
			// --- Error savestate ring / replay (−1.5s style) ---
			if (learner->cudaEnvSet) {
				// Hard opt-out: GIGA_STATE_RING=0 wins over stale runtime_overrides.json.
				const char* ringEnv = std::getenv("GIGA_STATE_RING");
				const bool ringEnvOff = ringEnv && std::atoi(ringEnv) == 0;
				const bool wantRing = !ringEnvOff && (
					ov.value("state_ring_enable", false)
					|| ov.value("error_pressure_replay", false)
					|| ov.value("error_pressure_active", false));
				if (ringEnvOff && learner->cudaEnvSet->StateRingEnabled()) {
					learner->ConfigureErrorStateRing(false, 10, 3, 0.05f, 64);
				} else if (!ringEnvOff && (wantRing || ov.contains("state_ring_enable"))) {
					bool enable = ov.contains("state_ring_enable")
						? ov["state_ring_enable"].get<bool>()
						: wantRing;
					int depth = ov.value("state_ring_depth", 10);
					int every = ov.value("state_ring_capture_every", 3);
					float frac = ov.value("state_ring_restore_frac",
						ov.value("error_replay_frac", 0.05f));
					if (frac > 0.05f) frac = 0.05f;
					int maxArenas = ov.value("error_replay_max_arenas", 64);
					const bool needEnable = enable && !learner->cudaEnvSet->StateRingEnabled();
					const bool needDisable = !enable && learner->cudaEnvSet->StateRingEnabled()
						&& ov.contains("state_ring_enable") && !ov["state_ring_enable"].get<bool>();
					if (needEnable || needDisable) {
						if (learner->AsyncLearnInFlight()) {
							learner->RequestDeferredCudaSurfaceApply();
						} else if (needEnable) {
							learner->ConfigureErrorStateRing(true, depth, every, frac, maxArenas);
						} else {
							learner->ConfigureErrorStateRing(false, depth, every, frac, maxArenas);
						}
					}
				}

				// Heavy rate-limit: <=max_per_window restores, >=min_steps between triggers.
				// Explicit trigger does NOT bypass cooldown (was thrashing 983 arenas).
				static int64_t s_lastReplayTs = -1;
				static int s_windowTriggers = 0;
				static bool s_wasActive = false;
				const bool active = ov.value("error_pressure_active", false);
				if (active && !s_wasActive) {
					s_windowTriggers = 0;
				}
				if (!active)
					s_wasActive = false;
				else
					s_wasActive = true;

				const bool wantReplay = ov.value("error_pressure_replay", true)
					&& (active || ov.value("error_replay_trigger", false)
						|| ov.value("error_pressure_replay_now", false));
				if (wantReplay && learner->cudaEnvSet->StateRingEnabled()
					&& !learner->cudaEnvSet->StateRingReplayDisabled()) {
					const int64_t ts = (int64_t)learner->totalTimesteps;
					const int64_t minSteps = (std::max)((int64_t)5'000'000,
						(int64_t)ov.value("error_replay_min_steps", 15'000'000));
					const int maxPerWindow = (std::max)(1, ov.value("error_replay_max_per_window", 3));
					// First fire must also respect minSteps (was: s_lastReplayTs<0 bypassed warmup
					// and ran state-replay + OpponentPool load at ~1M steps -> freeze ~iter 9-13).
					const bool warmupOk = ts >= minSteps;
					const bool cooldownOk = warmupOk
						&& ((s_lastReplayTs < 0) || (ts - s_lastReplayTs >= minSteps));
					const bool underCap = s_windowTriggers < maxPerWindow;
					// Sticky window may re-fire on cooldown only (never bypass cooldown).
					const bool arm = active
						|| ov.value("error_pressure_replay_now", false)
						|| ov.value("error_replay_trigger", false);
					if (arm && cooldownOk && underCap) {
						if (learner->AsyncLearnInFlight()) {
							// Same Collect||Learn race class as OpponentPool - flush in FlushDeferredCuda only.
							learner->RequestDeferredCudaSurfaceApply();
							static int64_t s_lastDeferLogTs = -1;
							if (s_lastDeferLogTs < 0 || ts - s_lastDeferLogTs >= 1'000'000) {
								RG_LOG("[AutoTrainer] error state-replay deferred (async Learn in flight, ts="
									<< ts << ")");
								s_lastDeferLogTs = ts;
							}
						} else {
							float frac = ov.value("error_replay_frac", 0.05f);
							if (frac > 0.05f) frac = 0.05f;
							if (frac < 0.01f) frac = 0.01f;
							int lookback = ov.value("error_replay_lookback", 7);
							float fuzz = ov.value("error_replay_fuzz", 1.f);
							int maxArenas = ov.value("error_replay_max_arenas", 64);
							if (maxArenas <= 0) maxArenas = 64;
							int n = learner->TriggerErrorStateReplay(frac, lookback, fuzz, maxArenas);
							s_lastReplayTs = ts;
							s_windowTriggers++;
							if (!postApex && n > 0)
								RG_LOG("[AutoTrainer] error state-replay restored=" << n
									<< " (cap=" << maxArenas << " frac=" << frac
									<< " window=" << s_windowTriggers << "/" << maxPerWindow << ")"
									<< " events=" << ov.value("error_pressure_events", json::array()).dump());
						}
					}
				}
			}
#endif

			if (ov.contains("policy_lr") || ov.contains("critic_lr")) {
				// Avoid redundant SetLearningRates spam when overrides reassert same LRs.
				// NEVER touch Adam while async Learn is mid-step (same class of hang as cudaMemcpy).
				static float s_lastPol = -1.f, s_lastCrit = -1.f;
				if (ppo.policyLR != s_lastPol || ppo.criticLR != s_lastCrit) {
#ifdef GIGA_USE_CUDA_SIM
					if (learner->AsyncLearnInFlight()) {
						learner->RequestDeferredCudaSurfaceApply();
					} else
#endif
					{
						learner->SetLearningRates(ppo.policyLR, ppo.criticLR);
						s_lastPol = ppo.policyLR;
						s_lastCrit = ppo.criticLR;
					}
				}
			}

#ifdef GIGA_USE_CUDA_SIM
			// config.ppo already updated above; defer live PPOLearner/Adam sync during overlap.
			if (learner->AsyncLearnInFlight()) {
				// Only mark dirty when flush must push live PPO knobs (LR already gated above).
				// Do NOT request defer on every iter - that forced cudaMemcpyToSymbol after every join.
			} else {
				learner->SyncRuntimePPOConfig();
			}
#else
			learner->SyncRuntimePPOConfig();
#endif

			// Log only when the live control surface actually moves (no every-N-iter spam).
			if (postApex && full) {
				static float s_ent = -1.f, s_pol = -1.f, s_crit = -1.f, s_opp = -1.f, s_old = -1.f;
				static int s_skill = -1;
				const int skillOn = learner->config.skillTracker.enabled ? 1 : 0;
				const bool moved =
					s_ent < 0.f
					|| std::fabs(ppo.entropyScale - s_ent) > 1e-6f
					|| std::fabs(ppo.policyLR - s_pol) > 1e-12f
					|| std::fabs(ppo.criticLR - s_crit) > 1e-12f
					|| std::fabs(learner->config.opponentPool.chance - s_opp) > 1e-6f
					|| std::fabs(learner->config.trainAgainstOldChance - s_old) > 1e-6f
					|| skillOn != s_skill;
				if (moved) {
					RG_LOG("[AutoTrainer] full_control"
						<< " ent=" << ppo.entropyScale
						<< " polLR=" << ppo.policyLR
						<< " critLR=" << ppo.criticLR
						<< " gamma=" << ppo.gaeGamma
						<< " eventBoost=" << ppo.eventAdvantageBoost
						<< " opp=" << learner->config.opponentPool.chance
						<< " old=" << learner->config.trainAgainstOldChance
						<< " skillEval=" << skillOn
						<< " gpuReset=" << tc.kickoffWeight << "/" << tc.fuzzedWeight << "/" << tc.aerialWeight);
					s_ent = ppo.entropyScale;
					s_pol = ppo.policyLR;
					s_crit = ppo.criticLR;
					s_opp = learner->config.opponentPool.chance;
					s_old = learner->config.trainAgainstOldChance;
					s_skill = skillOn;
				}
			}
		}
	}

	std::filesystem::path AutoTrainerBridge::RootDir(const Learner* learner) {
		if (learner && !learner->config.checkpointFolder.empty())
			return learner->config.checkpointFolder.parent_path() / "autotrainer";
		return "autotrainer";
	}

	void AutoTrainerBridge::ProcessCommands(Learner* learner) {
		if (IsReadOnly())
			return;
		auto cmdPath = Root(learner) / "commands.json";
		json cmd = ReadJson(cmdPath);
		if (cmd.empty())
			return;

#ifdef GIGA_USE_CUDA_SIM
		// Weight save/load hits CUDA/PyTorch - never during Collect||Learn async Learn.
		const bool gpuBusy = learner && learner->AsyncLearnInFlight();
#else
		const bool gpuBusy = false;
#endif

        if (cmd.value("save_checkpoint", false)) {
			if (gpuBusy) {
				// Keep flag; retry next iter after Learn join.
			} else {
				RG_LOG("AutoTrainer: save_checkpoint requested");
				if (!learner->config.checkpointFolder.empty())
					learner->Save();
				cmd["save_checkpoint"] = false;
			}
		}

		if (cmd.value("save_best_skill", false)) {
			if (gpuBusy) {
				// retry next iter
			} else if (!learner->config.checkpointFolder.empty()) {
				auto bestDir = learner->config.checkpointFolder / "best_skill";
				RG_LOG("AutoTrainer: save_best_skill -> " << bestDir);
				learner->SavePolicyTo(bestDir);
				json meta;
				meta["total_timesteps"] = learner->totalTimesteps;
				meta["total_iterations"] = learner->totalIterations;
				meta["kind"] = "best_skill";
				WriteJsonAtomic(bestDir / "BEST_SKILL.json", meta);
				cmd["save_best_skill"] = false;
			} else {
				cmd["save_best_skill"] = false;
			}
		}

		if (cmd.contains("save_agent_slot") && !cmd["save_agent_slot"].is_null()) {
			if (!gpuBusy) {
				int slot = cmd["save_agent_slot"].get<int>();
				if (slot >= 0 && !learner->config.checkpointFolder.empty()) {
					auto slotDir = learner->config.checkpointFolder / "pbt_agents" / ("agent_" + std::to_string(slot));
					RG_LOG("AutoTrainer: saving competitive agent slot " << slot << " -> " << slotDir);
					learner->SavePolicyTo(slotDir);
					json meta;
					meta["agent_id"] = slot;
					meta["total_timesteps"] = learner->totalTimesteps;
					meta["total_iterations"] = learner->totalIterations;
					WriteJsonAtomic(slotDir / "AGENT_META.json", meta);
				}
				cmd.erase("save_agent_slot");
			}
		}

		if (cmd.contains("load_agent_slot") && !cmd["load_agent_slot"].is_null()) {
			if (!gpuBusy) {
				int slot = cmd["load_agent_slot"].get<int>();
				if (slot >= 0 && !learner->config.checkpointFolder.empty()) {
					auto slotDir = learner->config.checkpointFolder / "pbt_agents" / ("agent_" + std::to_string(slot));
					if (std::filesystem::exists(slotDir / "POLICY.lt") || std::filesystem::exists(slotDir / "SHARED_HEAD.lt")) {
						RG_LOG("AutoTrainer: loading competitive agent slot " << slot << " <- " << slotDir);
						learner->LoadPolicyFrom(slotDir);
					} else {
						RG_LOG("AutoTrainer: agent slot " << slot << " empty - skip load");
					}
				}
				cmd.erase("load_agent_slot");
			}
		}

		if (cmd.contains("clear_pause") && cmd["clear_pause"].get<bool>()) {
			cmd["paused"] = false;
			cmd.erase("clear_pause");
		}

		WriteJsonAtomic(cmdPath, cmd);
	}

	bool AutoTrainerBridge::IsPaused(const Learner* learner) {
		json cmd = ReadJson(Root(learner) / "commands.json");
		if (cmd.value("paused", false))
			return true;
		int64_t until = cmd.value("pause_until_unix", (int64_t)0);
		return until > UnixNow();
	}

	void AutoTrainerBridge::WaitWhilePaused(Learner* learner) {
		while (IsPaused(learner)) {
			RG_LOG("AutoTrainer: training paused - waiting...");
			ProcessCommands(learner);
			std::this_thread::sleep_for(std::chrono::seconds(2));
		}
	}

	void AutoTrainerBridge::OnIterationStart(Learner* learner) {
		WaitWhilePaused(learner);
		ProcessCommands(learner);

		json ov = ReadJson(Root(learner) / "runtime_overrides.json");
		if (ov.empty() || IsReadOnly())
			return;

		// Log reward adjustments when the override file mtime/content changes (iteration start).
		static std::string s_lastRewardSig;
		std::string sig;
		if (ov.contains("reward_weights"))
			sig = ov["reward_weights"].dump();
		bool logRw = (sig != s_lastRewardSig);
		if (logRw)
			s_lastRewardSig = sig;

		ApplyOverrides(learner, ov, /*postApex=*/false, /*logRewards=*/logRw);
	}

	void AutoTrainerBridge::ReapplyPostApex(Learner* learner) {
		if (!learner || IsReadOnly())
			return;
		json ov = ReadJson(Root(learner) / "runtime_overrides.json");
		if (ov.empty())
			return;
		// Full control OR SSL guide: AutoTrainer overrides win over static Apex every iter.
		if (!WantsPostApex(ov))
			return;

		ApplyOverrides(learner, ov, /*postApex=*/true, /*logRewards=*/false);
	}

	void AutoTrainerBridge::FlushDeferredCuda(Learner* learner) {
		if (!learner || IsReadOnly())
			return;
#ifdef GIGA_USE_CUDA_SIM
		if (learner->ConsumeDeferredCudaSurfaceApply()) {
			json ov = ReadJson(Root(learner) / "runtime_overrides.json");
			if (ov.empty()) {
				if (learner->cudaEnvSet) {
					learner->cudaEnvSet->ApplyRuntimeGpuRewards();
					if (learner->cudaEnvSet->noTouchSeconds > 0.f)
						learner->cudaEnvSet->SetNoTouchSeconds(learner->cudaEnvSet->noTouchSeconds);
				}
			} else {
				// Safe: async Learn has joined. Re-apply full surface (rewards/terminals/distill/replay/LR).
				ApplyOverrides(learner, ov, /*postApex=*/true, /*logRewards=*/false);
			}
		}
		// Host PPO knobs may have been written during Collect||Learn; always push after join.
		learner->SyncRuntimePPOConfig();
		// Checkpoint Save deferred while Learn owned the GPU.
		if (learner->deferredCheckpointSave.exchange(false, std::memory_order_acq_rel)) {
			if (!learner->config.checkpointFolder.empty())
				learner->Save();
		}
		// Retry checkpoint/PBT commands that were deferred while Learn owned the GPU.
		ProcessCommands(learner);
		// OpponentPool Nexto JIT / PolicyVersion create / skill arenas - after surface apply
		// and error-replay so they never race Collect||Learn or mid-flush CUDA work.
		learner->FlushDeferredRuntimeSubsystems();
#else
		(void)learner;
#endif
	}

	void AutoTrainerBridge::WriteRewardManifest(
		const Learner* learner,
		const std::vector<RLGC::WeightedReward>& rewards) {

		json manifest = json::object();
		manifest["generated_at_unix"] = UnixNow();
		json entries = json::array();
		for (const auto& wr : rewards) {
			json e;
			e["name"] = wr.reward->GetName();
			e["base_weight"] = wr.weight;
			e["runtime_multiplier"] = 1.f;
			entries.push_back(e);
		}
		// Advertise blank GPU-native keys so AutoTrainer can tune the default template.
		// Extra community IDs remain available in RocketSimCuda if you add them in CudaEnvSet.
		static const char* kGpuBlank[] = {
			"GoalReward",
			"TouchBallReward",
			"VelocityBallToGoalReward",
			"VelocityPlayerToBallReward",
		};
		std::unordered_map<std::string, bool> seen;
		for (const auto& e : entries)
			seen[e["name"].get<std::string>()] = true;
		for (const char* name : kGpuBlank) {
			if (seen.count(name))
				continue;
			json e;
			e["name"] = name;
			e["base_weight"] = 1.f;
			e["runtime_multiplier"] = 1.f;
			e["gpu_native"] = true;
			entries.push_back(e);
		}
		manifest["rewards"] = entries;
		WriteJsonAtomic(Root(learner) / "reward_manifest.json", manifest);
	}

	void AutoTrainerBridge::AppendMetricsHistory(const Learner* learner, const Report& report) {
		auto path = Root(learner) / "metrics_history.jsonl";
		std::filesystem::create_directories(path.parent_path());
		json row;
		row["t"] = UnixNow();
		row["steps"] = learner->totalTimesteps;
		row["phase"] = (int)TrainingCurriculum::currentPhase;
		row["metrics"] = json::object();
		for (const auto& p : report.data)
			row["metrics"][p.first] = p.second;
		std::ofstream f(path, std::ios::app);
		f << row.dump() << "\n";
	}

	void AutoTrainerBridge::WriteStatus(Learner* learner, const Report& report) {
		auto root = Root(learner);
		json st;
		st["total_timesteps"] = learner->totalTimesteps;
		st["total_iterations"] = learner->totalIterations;
		st["curriculum_phase"] = (int)TrainingCurriculum::currentPhase;
		st["run_id"] = learner->runID;
		st["updated_at_unix"] = UnixNow();
		st["paused"] = IsPaused(learner);
		st["opponent_pool_loaded"] = learner->opponentPool ? 1 : 0;
		st["sps_safe"] = report.Has("Curriculum/SpsSafe")
			? (report.data.at("Curriculum/SpsSafe") > 0.5) : false;
		st["autotrainer_readonly"] = IsReadOnly();
#ifdef GIGA_USE_CUDA_SIM
		st["gpu_native"] = (learner->cudaEnvSet && learner->cudaEnvSet->gpuNative) ? 1 : 0;
		if (learner->cudaEnvSet) {
			st["state_ring_enabled"] = learner->cudaEnvSet->StateRingEnabled() ? 1 : 0;
			st["state_ring_captures"] = learner->cudaEnvSet->StateRingCaptures();
			st["state_ring_restores"] = learner->cudaEnvSet->StateRingRestores();
		}
#else
		st["gpu_native"] = 0;
#endif
		st["soft_distill_active"] = learner->HasSoftDistillTeacher() ? 1 : 0;

		json metrics = json::object();
		for (const auto& p : report.data)
			metrics[p.first] = p.second;
		st["last_metrics"] = metrics;

		st["active_overrides"] = ReadJson(root / "runtime_overrides.json");
		st["commands"] = ReadJson(root / "commands.json");
		st["pbt_leaderboard"] = ReadJson(root / "pbt_leaderboard.json");
		if (st["active_overrides"].contains("active_agent_id"))
			st["active_agent_id"] = st["active_overrides"]["active_agent_id"];

		json rw = json::object();
		for (const auto& [k, v] : RLGC::RuntimeRewardRegistry::Instance().Snapshot())
			rw[k] = v;
		st["reward_multipliers"] = rw;

		WriteJsonAtomic(root / "trainer_status.json", st);
		if (learner->totalIterations % 4 == 0)
			AppendMetricsHistory(learner, report);
	}

}
