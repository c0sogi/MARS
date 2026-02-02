import os

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for Idea 25 (Cross-Modal Consistency Dual-Stream)
WORKING_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"

# Ensure writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input File Paths
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata File Paths (Generated previously)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SEED = 42
N_JOBS = 12
USE_GPU = True

# Sampling Configuration
# Strategy: Target Majority Undersampling
# Retain 100% positive, subsample negative to 10:1 ratio
NEG_POS_RATIO = 10.0

# =============================================================================
# MODEL HYPERPARAMETERS (XGBoost)
# =============================================================================

# Common XGBoost parameters
XGB_COMMON_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "gpu_hist" if USE_GPU else "hist",
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "n_estimators": 3000,  # High upper bound, relying on early stopping
    "early_stopping_rounds": 50,
}

# Stream A: Interaction Model (Player-Player)
# Logic: Standard depth to model complex cross-modal interactions (Visual vs Tracking)
STREAM_A_PARAMS = XGB_COMMON_PARAMS.copy()
STREAM_A_PARAMS.update(
    {
        "max_depth": 6,
    }
)

# Stream B: Impact Model (Player-Ground)
# Logic: Robust depth to prevent overfitting sensor noise, relying on Ego-Jerk/Sway-Energy
STREAM_B_PARAMS = XGB_COMMON_PARAMS.copy()
STREAM_B_PARAMS.update(
    {
        "max_depth": 8,
    }
)

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

FEATURE_CONFIG = {
    # Temporal Window sizes (lags in steps) for finite difference calculations
    # Tracking data is 10Hz. [1, 2, 4, 8] corresponds to 0.1s, 0.2s, 0.4s, 0.8s
    "lags": [1, 2, 4, 8],
    # Stream A: Interaction (Player vs Player)
    # Focus: Consistency & Convergence
    "stream_a": {
        "use_visuals": True,
        "use_absolute_coords": False,  # Use relative metrics
        "features": [
            # Base Relational Primitives
            "distance",
            "speed_rel",
            "accel_rel",
            # System Energy (Flattened Pyramids)
            "speed_p1",
            "speed_p2",
            "accel_p1",
            "accel_p2",
            # Visual Consensus
            "iou_sideline",
            "iou_endzone",
            "iou_diff",  # |Sideline - Endzone|
            # Cross-Modal Alignment (Structural Innovation)
            "visual_looming_rate",  # d(IoU)/dt
            "physical_closure_rate",  # d(Dist)/dt
            "looming_closure_product",  # Do they agree?
            "looming_closure_ratio",  # Magnitude mismatch (Phantom Approach detection)
            "view_disagreement_trend",  # d(iou_diff)/dt
        ],
    },
    # Stream B: Impact (Player vs Ground)
    # Focus: Rotational-Difference Dynamics
    "stream_b": {
        "use_visuals": False,
        "use_absolute_coords": False,  # Strict Invariance
        "features": [
            # Invariant Baseline
            "speed",
            "acceleration",
            "sa",  # Signed acceleration
            # Rotational Dynamics (Structural Innovation)
            "v_surge",  # Velocity projected on orientation
            "v_sway",  # Velocity projected orthogonal to orientation
            "energy_surge",  # 0.5 * v_surge^2
            "energy_sway",  # 0.5 * v_sway^2 (Distinguish controlled cut vs tumble)
            "ego_jerk",  # Derivative of acceleration magnitude
        ],
    },
}
