#pragma once

#include <cstdint>
#include <vector>

// ============================================================================
// RocketSimCuda - GPU-accelerated batched Rocket League physics simulator
// Runs N independent arenas in parallel on CUDA, one thread per arena.
// ============================================================================

namespace rsc {

// ---- Basic types (shared host/device, trivial POD for CUDA compatibility) ----

struct Vec3 {
    float x, y, z;
};

struct RotMat {
    Vec3 forward;
    Vec3 right;
    Vec3 up;
};

// ---- Enums ----

enum class Team : uint8_t {
    BLUE = 0,
    ORANGE = 1,
};

enum class CarPreset : uint8_t {
    OCTANE = 0,
    DOMINUS,
    PLANK,
    BREAKOUT,
    HYBRID,
    MERC,
};

enum class GameMode : uint8_t {
    SOCCAR = 0,
    THE_VOID,
};

// ---- Controls ----

struct CarControls {
    float throttle = 0;  // -1 to 1
    float steer    = 0;  // -1 to 1
    float pitch    = 0;  // -1 to 1
    float yaw      = 0;  // -1 to 1
    float roll     = 0;  // -1 to 1
    bool  jump     = false;
    bool  boost    = false;
    bool  handbrake = false;
};

// ---- Car state (full) ----

struct CarState {
    Vec3 pos;
    RotMat rotMat;
    Vec3 vel;
    Vec3 angVel;

    float boost = 33.33f;
    float timeSpentBoosting = 0;

    bool isOnGround = true;
    bool hasJumped = false;
    bool hasDoubleJumped = false;
    bool hasFlipped = false;
    bool isFlipping = false;
    bool isJumping = false;

    Vec3 flipRelTorque;
    float jumpTime = 0;
    float flipTime = 0;
    float airTime = 0;
    float airTimeSinceJump = 0;

    bool isSupersonic = false;
    float supersonicTime = 0;
    float handbrakeVal = 0;

    bool isAutoFlipping = false;
    float autoFlipTimer = 0;
    float autoFlipTorqueScale = 0;

    bool isDemoed = false;
    float demoRespawnTimer = 0;

    bool worldContactHasContact = false;
    Vec3 worldContactNormal;
    uint32_t carContactOtherID = 0;
    float carContactCooldownTimer = 0;

    bool ballHitValid = false;
    uint64_t ballHitTickCount = ~0ULL;

    Team team = Team::BLUE;
    CarPreset preset = CarPreset::OCTANE;
    uint32_t id = 0;

    CarControls lastControls;
};

// ---- Ball state ----

struct BallState {
    Vec3 pos    = {0, 0, 93.15f};
    RotMat rotMat;
    Vec3 vel;
    Vec3 angVel;
};

// ---- Boost pad state ----

struct BoostPadState {
    bool isActive = true;
    float cooldown = 0;
};

// ---- Mutator config ----

struct MutatorConfig {
    Vec3 gravity = {0, 0, -650.f};
    float carMass = 180.f;
    float ballMass = 180.f / 6.f; // RocketSim BALL_MASS_BT = CAR_MASS_BT / 6 (= 30)
    float ballMaxSpeed = 6000.f;
    float ballDrag = 0.03f;
    float ballWorldFriction = 0.35f;
    float ballWorldRestitution = 0.6f;
    float ballRadius = 91.25f;

    float jumpAccel = 1458.33f;
    float jumpImmediateForce = 291.67f;
    float boostAccelGround = 991.67f;
    float boostAccelAir = 1058.33f;
    float boostUsedPerSecond = 33.33f;
    float carSpawnBoostAmount = 33.33f;

    float respawnDelay = 3.f;
    float bumpCooldownTime = 0.25f;
    float boostPadCooldown_Big = 10.f;
    float boostPadCooldown_Small = 4.f;

    float ballHitExtraForceScale = 1.f;
    float bumpForceScale = 1.f;
    float goalBaseThresholdY = 5124.25f;
};

// ---- Arena info (per arena) ----

struct ArenaInfo {
    uint64_t tickCount = 0;
    int numCars = 0;
    GameMode gameMode = GameMode::SOCCAR;
    bool goalScored = false;     // Set to true when a goal is detected
    int goalTeam = -1;           // 0 = blue scored, 1 = orange scored
};

// ---- Batch configuration ----

enum class ObsMode : uint8_t {
    ADVANCED = 0,  // AdvancedObs + DefaultAction masks
    VANTA = 1,     // VantaObs v3.5 (366 floats, no action masks)
};

struct BatchConfig {
    int numArenas = 1;
    int maxCarsPerArena = 6;
    float tickRate = 120.f;
    GameMode gameMode = GameMode::SOCCAR;
    MutatorConfig mutatorConfig;

    // Path to the collision_meshes folder (the one passed to RocketSim::Init).
    // When set, ball-world contacts and suspension raycasts use the real
    // arena triangle meshes (exact reference geometry). When null, analytic
    // approximations of the arena are used instead.
    const char* collisionMeshesPath = nullptr;

    // GPU observation builder selection + VantaObs parameters
    ObsMode obsMode = ObsMode::ADVANCED;
    uint8_t vantaPredictBall = 0;
    uint8_t vantaOnlyClosestOpp = 0;
    float vantaDodgeDeadzone = 0.5f;
    int vantaTickSkip = 8;  // drives the obs timer interval (tickSkip / 120)
};

static constexpr int VANTA_OBS_SIZE = 366;

static constexpr int NUM_BOOST_PADS = 34;  // 28 small + 6 big (SOCCAR)
static constexpr int ADVANCED_OBS_BASE_SIZE = 51;
static constexpr int ADVANCED_OBS_PER_CAR_SIZE = 29;
static constexpr int DEFAULT_ACTION_COUNT = 90;
static constexpr int MAX_TRAINING_REWARD_ENTRIES = 16;

enum class TrainingRewardID : uint8_t {
    UNKNOWN = 0,
    GOAL_REWARD,
    VELOCITY_BALL_TO_GOAL,
    VELOCITY_PLAYER_TO_BALL,
    FACE_BALL,
    TOUCH_BALL,
    TOUCH_ACCEL,
    STRONG_TOUCH,
    SPEED,
    VELOCITY,
    AIR,
    WAVEDASH,
    PICKUP_BOOST,
    SAVE_BOOST,
    BUMP,
    BUMPED_PENALTY,
    DEMO,
    DEMOED_PENALTY,
    KICKOFF_PROXIMITY,
    TEAMMATE_BUMP_PENALTY,
    PASS_INCENTIVE,
    TEAM_PASSING,
    MIXED_INITIATE,
    MIXED_DOUBLETAP,
    SPEED_AERIAL,
    // Default reward fidelity (GPU approximations of Leak CPU reward classes)
    PRESSURE_FLICK,
    MAWKZY_FLICK,
    AIR_CARRY,
    ANTI_BALL_STACK,
};

constexpr int TRAINING_REWARD_MAX_PARAMS = 10;

struct TrainingRewardEntry {
    TrainingRewardID id = TrainingRewardID::UNKNOWN;
    float weight = 0.f;
    float params[TRAINING_REWARD_MAX_PARAMS] = {};
    uint8_t isZeroSum = 0;
    float teamSpirit = 0.f;
    float opponentScale = 1.f;
};

struct TrainingRewardConfig {
    TrainingRewardEntry entries[MAX_TRAINING_REWARD_ENTRIES] = {};
    int numEntries = 0;
};

struct TrainingTerminalConfig {
    uint8_t useGoalScore = 0;
    float noTouchTimeoutSeconds = 0.f;
};

// ============================================================================
// Main API class
// ============================================================================

class RocketSimCudaBatch {
public:
    RocketSimCudaBatch();
    ~RocketSimCudaBatch();

    // Initialize GPU memory for the batch
    void Init(const BatchConfig& config);

    // Cleanup GPU memory
    void Destroy();

    // --- Arena setup ---

    // Add a car to an arena. Returns car index within that arena.
    int AddCar(int arenaIdx, Team team, CarPreset preset = CarPreset::OCTANE);

    // Reset an arena to kickoff state (or curriculum mix — see SetResetCurriculum)
    void ResetArena(int arenaIdx);

    // Reset all arenas
    void ResetAllArenas();

    // GPU-native state curriculum (kickoff / fuzzed / aerial-ish). Weights are relative.
    // Default is pure kickoff (1,0,0) — matches prior ResetArena behavior.
    void SetResetCurriculum(float kickoffW, float fuzzedW, float aerialW);

    // --- Per-tick operations ---

    // Set controls for a specific car
    void SetCarControls(int arenaIdx, int carIdx, const CarControls& controls);

    // Set controls for all cars in bulk (array of numArenas * maxCarsPerArena)
    void SetAllCarControls(const CarControls* controls);

    // Upload discrete action LUT once (CarControls[actionCount]) for GPU expand.
    void UploadDiscreteActionLut(const CarControls* lut, int actionCount);

    // Expand discrete action indices -> d_controls via device LUT (H2D indices + kernel).
    // count must be numArenas * maxCarsPerArena (packed player order).
    void SetDiscreteActionsFromIndices(const int* indices, int count);

    // Expand discrete action indices already on device (int32[count]) — no H2D.
    void SetDiscreteActionsFromDevice(const int* deviceIndices, int count);

    // When false, Step / BuildObs / BuildRewards do not cudaDeviceSynchronize.
    // Caller must Synchronize() (or rely on stream ordering) before host reads.
    // Throughput path: chain kernels then sync once before Infer / D2H.
    void SetKernelSync(bool enabled) { m_kernelSync = enabled; }
    bool GetKernelSync() const { return m_kernelSync; }

    // Step all arenas by the given number of ticks
    void Step(int ticksToSimulate = 1);

    // --- Training interop (AdvancedObs + DefaultAction masks) ---

    // Builds dense per-player AdvancedObs rows and DefaultAction masks on the GPU.
    // This path assumes every arena has exactly maxCarsPerArena active cars.
    void BuildAdvancedObsAndDefaultMasks();

    // Builds dense per-player VantaObs rows (366 floats) on the GPU, including
    // the ball-prediction slices (rolled with the same exact ball physics).
    // Requires obsMode == ObsMode::VANTA. No action masks (continuous policy).
    void BuildVantaObs();

    // Active obs row size for the configured obs mode.
    int GetObsRowSize() const {
        return (m_config.obsMode == ObsMode::VANTA) ? VANTA_OBS_SIZE : GetAdvancedObsSize();
    }

    // Configure GPU-built rewards/terminals for training.
    void ConfigureTrainingRewards(const TrainingRewardConfig& config);
    void ConfigureTrainingTerminals(const TrainingTerminalConfig& config);

    // Snapshot the current sim state for delta rewards before the env step advances.
    void SnapshotTrainingState();

    // Builds dense per-player rewards and per-arena terminals on the GPU.
    // `stepTicks` should match the full env step length that the reward/terminal
    // logic sees from one GameState update to the next (for example `tickSkip`).
    void BuildRewardsAndTerminals(int stepTicks);

    // Copy the built training buffers to host memory.
    void CopyBuiltAdvancedObs(float* outObs) const;
    void CopyBuiltDefaultActionMasks(uint8_t* outMasks) const;
    void CopyBuiltRewards(float* outRewards) const;
    void CopyBuiltTerminals(uint8_t* outTerminals) const;
    void Synchronize() const;

    // --- State access (GPU -> CPU copy) ---

    // Get state for a specific car
    CarState GetCarState(int arenaIdx, int carIdx) const;

    // Get all car states (flat array: numArenas * maxCarsPerArena)
    void GetAllCarStates(CarState* outStates) const;

    // Get ball state for a specific arena
    BallState GetBallState(int arenaIdx) const;

    // Get all ball states
    void GetAllBallStates(BallState* outStates) const;

    // Get boost pad states for a specific arena
    void GetBoostPadStates(int arenaIdx, BoostPadState* outStates) const;
    void GetAllBoostPadStates(BoostPadState* outStates) const;

    // Get arena info
    ArenaInfo GetArenaInfo(int arenaIdx) const;
    void GetAllArenaInfos(ArenaInfo* outInfos) const;

    // Debug: raw copy of the internal GpuCarState (layout in src/GpuTypes.cuh).
    // Returns the number of bytes copied (min of maxBytes and sizeof state).
    int DebugCopyCarInternal(int arenaIdx, int carIdx, void* dst, int maxBytes) const;

    // --- State setting (CPU -> GPU copy) ---

    void SetCarState(int arenaIdx, int carIdx, const CarState& state);
    void SetBallState(int arenaIdx, const BallState& state);
    // Restore boost pads for one arena (pair with GetBoostPadStates for savestate replay).
    void SetBoostPadStates(int arenaIdx, const BoostPadState* states);
    // Bulk restore (flat: numArenas * NUM_BOOST_PADS).
    void SetAllBoostPadStates(const BoostPadState* states);

    // --- Direct GPU access ---
    // `GetControlsDevicePtr()` points to a flat `CarControls[numArenas * maxCarsPerArena]` buffer.
    // The state pointers expose RocketSimCuda's internal GPU layouts rather than the public
    // `CarState` / `BallState` ABI, so they are intended for custom CUDA kernels, not direct
    // host-side decoding as public structs.

    void* GetCarStatesDevicePtr() const { return d_carStates; }
    void* GetBallStatesDevicePtr() const { return d_ballStates; }
    void* GetBoostPadStatesDevicePtr() const { return d_padStates; }
    void* GetArenaInfosDevicePtr() const { return d_arenaInfos; }
    void* GetControlsDevicePtr() const { return d_controls; }
    void* GetBuiltAdvancedObsDevicePtr() const { return d_trainingObs; }
    void* GetBuiltDefaultActionMasksDevicePtr() const { return d_trainingActionMasks; }
    void* GetBuiltRewardsDevicePtr() const { return d_trainingRewards; }
    void* GetBuiltTerminalsDevicePtr() const { return d_trainingTerminals; }

    // --- Getters ---

    int GetNumArenas() const { return m_config.numArenas; }
    int GetMaxCarsPerArena() const { return m_config.maxCarsPerArena; }
    int GetNumPlayers() const { return m_config.numArenas * m_config.maxCarsPerArena; }
    int GetAdvancedObsSize() const { return ADVANCED_OBS_BASE_SIZE + ADVANCED_OBS_PER_CAR_SIZE * m_config.maxCarsPerArena; }
    float GetTickTime() const { return 1.f / m_config.tickRate; }
    const BatchConfig& GetConfig() const { return m_config; }
    const TrainingRewardConfig& GetTrainingRewardConfig() const { return m_trainingRewardConfig; }
    const TrainingTerminalConfig& GetTrainingTerminalConfig() const { return m_trainingTerminalConfig; }
    bool HasTrainingRewardConfig() const { return m_trainingRewardConfig.numEntries > 0; }
    bool HasTrainingTerminalConfig() const {
        return m_trainingTerminalConfig.useGoalScore || m_trainingTerminalConfig.noTouchTimeoutSeconds > 0.f;
    }

private:
    BatchConfig m_config;
    TrainingRewardConfig m_trainingRewardConfig;
    TrainingTerminalConfig m_trainingTerminalConfig;
    bool m_initialized = false;
    bool m_kernelSync = true;
    // Relative weights for ResetArena variety (learning curriculum on gpuNative).
    float m_resetKickoffW = 1.f;
    float m_resetFuzzedW = 0.f;
    float m_resetAerialW = 0.f;
    uint32_t m_resetRng = 0xC0FFEEu;

    // GPU device pointers (opaque to the public header)
    void* d_carStates    = nullptr;
    void* d_ballStates   = nullptr;
    void* d_padStates    = nullptr;
    void* d_controls     = nullptr;
    void* d_actionLut    = nullptr;
    void* d_actionIndices = nullptr;
    int   m_actionLutCount = 0;
    int   m_actionIndicesCapacity = 0;
    void* d_arenaInfos   = nullptr;
    void* d_mutatorConfig = nullptr;
    void* d_meshTris     = nullptr;
    void* d_meshCellStart = nullptr;
    void* d_meshCellTris = nullptr;
    void* d_trainingObs = nullptr;
    void* d_trainingActionMasks = nullptr;
    void* d_prevCarStates = nullptr;
    void* d_prevBallStates = nullptr;
    void* d_trainingRewards = nullptr;
    void* d_trainingTerminals = nullptr;
    void* d_noTouchTimers = nullptr;
    void* d_rewardAuxStates = nullptr;
    void* d_vantaTimers = nullptr;
    void* d_vantaArena = nullptr;

    // Host staging buffers for bulk operations
    std::vector<CarState> h_carStates;
    std::vector<BallState> h_ballStates;
    std::vector<ArenaInfo> h_arenaInfos;
};

} // namespace rsc
