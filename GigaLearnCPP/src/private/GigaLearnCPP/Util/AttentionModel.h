#pragma once
#include <RLGymCPP/Framework.h>
#include "../FrameworkTorch.h"
#include "Models.h"

#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/cloneable.h>

#include <GigaLearnCPP/Util/ModelConfig.h>
#include <GigaLearnCPP/Util/AttentionModelConfig.h>

namespace GGL {

	/////////////////////////////
	// Refinement Block
	/////////////////////////////

	class RefinementBlockImpl : public torch::nn::Cloneable<RefinementBlockImpl> {
	public:
		int nDims = 0;
		int nHeads = 0;
		int dimFeedforward = 0;

		torch::nn::MultiheadAttention attention{nullptr};
		torch::nn::Linear linear1{nullptr}, linear2{nullptr};
		torch::nn::LayerNorm norm_q{nullptr}, norm_kv{nullptr}, norm_ff{nullptr};

		RefinementBlockImpl() = default;

		RefinementBlockImpl(int n_dims, int n_heads, int dim_feedforward)
			: nDims(n_dims), nHeads(n_heads), dimFeedforward(dim_feedforward) {
			reset();
		}

		void reset() {
			attention = register_module("attention",
				torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(nDims, nHeads)));
			linear1 = register_module("linear1", torch::nn::Linear(nDims, dimFeedforward));
			linear2 = register_module("linear2", torch::nn::Linear(dimFeedforward, nDims));
			norm_q = register_module("norm_q",
				torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)nDims})));
			norm_kv = register_module("norm_kv",
				torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)nDims})));
			norm_ff = register_module("norm_ff",
				torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)nDims})));
		}

		torch::Tensor forward(torch::Tensor query, torch::Tensor key_value,
							  torch::Tensor mask = {}) {
			// Pre-LN cross-attention
			auto q_norm = norm_q->forward(query);
			auto kv_norm = norm_kv->forward(key_value);

			// MultiheadAttention without batch_first expects (seq, batch, dim)
			q_norm = q_norm.transpose(0, 1);
			kv_norm = kv_norm.transpose(0, 1);

			// MHA key_padding_mask uses masked_fill, which is not implemented for
			// BFloat16 under autocast on some PyTorch/CUDA builds — run attention in fp32.
			torch::Tensor attn_out;
			{
				const bool autocastWasOn = at::autocast::is_autocast_enabled(at::kCUDA);
				if (autocastWasOn)
					at::autocast::set_autocast_enabled(at::kCUDA, false);

				auto q_fp32 = q_norm.to(torch::kFloat);
				auto kv_fp32 = kv_norm.to(torch::kFloat);
				auto attn_result = attention->forward(q_fp32, kv_fp32, kv_fp32,
					/*key_padding_mask=*/mask);
				attn_out = std::get<0>(attn_result);

				if (autocastWasOn)
					at::autocast::set_autocast_enabled(at::kCUDA, true);
			}

			// Transpose back to (batch, seq, dim)
			attn_out = attn_out.transpose(0, 1).to(query.scalar_type());
			query = query + attn_out;

			// Pre-LN feedforward
			auto ff_norm = norm_ff->forward(query);
			auto ff_out = linear2->forward(torch::relu(linear1->forward(ff_norm)));
			query = query + ff_out;

			return query;
		}
	};
	TORCH_MODULE(RefinementBlock);

	/////////////////////////////
	// Think Block
	/////////////////////////////

	class ThinkBlockImpl : public torch::nn::Cloneable<ThinkBlockImpl> {
	public:
		int nDims = 0;
		int thinkDims = 0;

		torch::nn::Linear expand{nullptr}, contract{nullptr};
		torch::nn::LayerNorm norm{nullptr};

		ThinkBlockImpl() = default;

		ThinkBlockImpl(int n_dims, int think_dims)
			: nDims(n_dims), thinkDims(think_dims) {
			reset();
		}

		void reset() {
			expand = register_module("expand", torch::nn::Linear(nDims, thinkDims));
			contract = register_module("contract", torch::nn::Linear(thinkDims, nDims));
			norm = register_module("norm",
				torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)nDims})));
		}

		torch::Tensor forward(torch::Tensor x) {
			// Pre-LN think expansion: n_dims -> think_dims -> n_dims
			auto x_norm = norm->forward(x);
			auto out = contract->forward(torch::relu(expand->forward(x_norm)));
			return x + out;
		}
	};
	TORCH_MODULE(ThinkBlock);

	/////////////////////////////
	// Attention Model
	/////////////////////////////

	class AttentionModel : public Model {
	public:
		AttentionModelConfig attnConfig;

		// Preprocessing MLPs
		torch::nn::Sequential queryPreprocess, kvPreprocess;

		// Alternating blocks
		torch::nn::ModuleList refinementBlocks, thinkBlocks;

		// Postprocessing
		torch::nn::Sequential postprocess;

		// Half-precision copies
		torch::nn::Sequential queryPreprocessHalf, kvPreprocessHalf, postprocessHalf;
		torch::nn::ModuleList refinementBlocksHalf, thinkBlocksHalf;
		bool _attnHalfOutdated = true;

		AttentionModel() : Model() {} // Uninitialized

		AttentionModel(
			const char* modelName,
			AttentionModelConfig config,
			torch::Device device
		);

		// Forward: takes batched observations, internally splits for perceiver-style attention
		// For shared-head usage: obs is (batch, obs_size), treated as single query attending to itself
		torch::Tensor Forward(torch::Tensor input, bool halfPrec) override;

		// Perceiver-style forward with explicit query and key-value inputs
		torch::Tensor ForwardPerceiver(torch::Tensor query, torch::Tensor keyValue,
									   bool halfPrec, torch::Tensor mask = {});

		void Save(std::filesystem::path folder, bool saveOptim = true) override;
		void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim = true) override;
		torch::Tensor CopyParams() const override;

		void StepOptim() override {
			Model::StepOptim();
			_attnHalfOutdated = true;
		}

		Model* MakeEmptyClone() override {
			return new AttentionModel(modelName, attnConfig, device);
		}

		Model* MakeClone() override {
			RG_NO_GRAD;
			auto* clone = static_cast<AttentionModel*>(MakeEmptyClone());
			auto fromParams = this->parameters();
			auto toParams = clone->parameters();
			for (size_t i = 0; i < fromParams.size(); i++)
				toParams[i].copy_(fromParams[i], true);
			return clone;
		}

		uint64_t GetParamCount();
	};

} // namespace GGL
