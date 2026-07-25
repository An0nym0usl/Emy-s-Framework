#pragma once

#include "ModelConfig.h"

namespace GGL {
	struct AttentionModelConfig {
		int numInputs = -1;

		int numDims = 256;         // Refinement dimension (attention width)
		int thinkDims = 1024;      // Think expansion dimension (MLP capacity)
		int numBlocks = 2;         // Number of Refine->Think pairs
		int numHeads = 4;          // Attention heads
		int preprocessLayers = 1;  // Dense layers before attention
		int postprocessLayers = 0; // Dense layers after attention

		// Feedforward dim inside refinement blocks (0 = 2*numDims)
		int refinementFeedforward = 0;

		ModelActivationType activationType = ModelActivationType::RELU;
		ModelOptimType optimType = ModelOptimType::ADAM;

		// === True multi-entity attention ===
		// When enabled, the flat observation is reshaped into (numEntities, entityFeatSize) tokens
		// (e.g. one token per car + a ball/context token) and real cross-attention runs over the
		// entities, with padded/missing entities masked out. The last feature of every token is a
		// validity flag (1 = real entity, 0 = padding). Use EntityObsBuilder to produce this layout.
		// When disabled, the head behaves as a transformer over a single pooled token (legacy).
		bool entityMode = false;
		int numEntities = 0;     // Number of tokens per sample (cars + context token)
		int entityFeatSize = 0;  // Feature width of each token (including role one-hots + validity flag)

		bool IsValid() const {
			return numInputs > 0 && numDims > 0 && thinkDims > 0 && numBlocks > 0 && numHeads > 0;
		}

		// Per-token input width fed to the preprocess MLP.
		// In entity mode each token is entityFeatSize wide; otherwise the whole flat obs is one token.
		int GetTokenInputDims() const {
			return entityMode ? entityFeatSize : numInputs;
		}

		int GetOutputDims() const { return numDims; }
	};
}
