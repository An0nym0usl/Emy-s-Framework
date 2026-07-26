#pragma once
// Constant memory + aux state
// Extracted from RocketSimCuda.cu (modular split; same TU via include).

// Reserved per-car aux slot for optional future stateful rewards (public stubs use pad only).
struct GpuRewardAuxState {
	uint8_t _pad;
};


// ============================================================================
// Constant memory
// ============================================================================

__constant__ ArenaSurface c_surfaces[MAX_ARENA_SURFACES];
__constant__ int c_numSurfaces;
__constant__ float c_tickTime;
__constant__ float c_ballRadius;
__constant__ float c_ballDrag;
__constant__ float c_ballFriction;
__constant__ float c_ballRestitution;
__constant__ float c_ballMaxSpeed;
__constant__ float c_ballInvInertia;
__constant__ float c_ballWorldFrictionCombined;
__constant__ float c_ballWorldRestitutionCombined;
__constant__ float c_ballHitExtraForceScale;
__constant__ float c_bumpForceScale;
__constant__ float c_bumpCooldownTime;
__constant__ float c_boostUsedPerSecond;
__constant__ float c_boostAccelGround;
__constant__ float c_boostAccelAir;
__constant__ float c_boostPadCooldownBig;
__constant__ float c_boostPadCooldownSmall;
__constant__ float c_goalThresholdY;
__constant__ Vec3 c_gravity;
__constant__ int c_obsPadMap[NUM_BOOST_PADS];
__constant__ int c_obsPadMapInv[NUM_BOOST_PADS];
__constant__ uint8_t c_defaultGroundMask[DEFAULT_ACTION_COUNT];
__constant__ uint8_t c_defaultAirMask[DEFAULT_ACTION_COUNT];
__constant__ uint8_t c_defaultJumpMask[DEFAULT_ACTION_COUNT];
__constant__ uint8_t c_defaultBoostMask[DEFAULT_ACTION_COUNT];
__constant__ TrainingRewardEntry c_trainingRewardEntries[MAX_TRAINING_REWARD_ENTRIES];
__constant__ int c_numTrainingRewardEntries;
__constant__ TrainingTerminalConfig c_trainingTerminalConfig;
__constant__ MeshGridView c_meshGrid;  // real arena triangles (numTris==0 -> analytic fallback)
__constant__ Vec3 c_vantaPadLocations[VantaConst::BOOST_SIZE];  // RLGymCPP pad order
__constant__ int c_vantaMirrorPadMap[VantaConst::BOOST_SIZE];   // X-mirror permutation (RLGym order)
__constant__ int c_vantaPadMap[VantaConst::BOOST_SIZE];         // RLGym order -> GPU pad index

