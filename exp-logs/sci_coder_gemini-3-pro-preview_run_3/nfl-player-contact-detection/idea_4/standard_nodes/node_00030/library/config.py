import os
import numpy as np

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data Paths
TRACKING_PATH_TRAIN = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TRACKING_PATH_TEST = os.path.join(INPUT_DIR, "test_player_tracking.csv")
HELMETS_PATH_TRAIN = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
HELMETS_PATH_TEST = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
WINDOW_SIZE = 9  # t-4 to t+4
LAG_STEPS = 4  # Number of steps before and after the current step
SAMPLING_RATIO = 10.0  # Negative to Positive ratio for undersampling

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# XGBoost Parameters for GPU training
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "gpu_hist",  # Use GPU
    "predictor": "gpu_predictor",
    "n_jobs": -1,
    "random_state": SEED,
    "early_stopping_rounds": 50,
    "verbose": 0,
}

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Base tracking columns to use from raw data
RAW_TRACKING_COLS = [
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "sa",
    "orientation",
    "direction",
]

# -----------------------------------------------------------------------------
# STREAM A: Player-Player Interaction
# -----------------------------------------------------------------------------
# Base features generated per timestep for Stream A
# These will be flattened across the window (t-4...t+4)
STREAM_A_BASE_FEATURES = []

# Player 1 Kinematics
p1_feats = [
    "x_position_p1",
    "y_position_p1",
    "speed_p1",
    "acceleration_p1",
    "sa_p1",
    "sin_orient_p1",
    "cos_orient_p1",
    "sin_dir_p1",
    "cos_dir_p1",
    "jerk_p1",
    "cos_orient_dir_p1",
    "centripetal_accel_p1",
]
STREAM_A_BASE_FEATURES.extend(p1_feats)

# Player 2 Kinematics
p2_feats = [
    "x_position_p2",
    "y_position_p2",
    "speed_p2",
    "acceleration_p2",
    "sa_p2",
    "sin_orient_p2",
    "cos_orient_p2",
    "sin_dir_p2",
    "cos_dir_p2",
    "jerk_p2",
    "cos_orient_dir_p2",
    "centripetal_accel_p2",
]
STREAM_A_BASE_FEATURES.extend(p2_feats)

# Interaction Dynamics (Relative)
interaction_feats = [
    "distance",
    "speed_diff",
    "accel_diff",
    "closure_rate",
    "cos_sim_dir",
    "cos_sim_orient",
    "cos_p1_facing_p2",
    "cos_p2_facing_p1",
]
STREAM_A_BASE_FEATURES.extend(interaction_feats)

# -----------------------------------------------------------------------------
# STREAM B: Player-Ground Interaction
# -----------------------------------------------------------------------------
# Base features generated per timestep for Stream B
# Strictly Ego-motion, no P2 features
STREAM_B_BASE_FEATURES = []

# Player 1 Ego-Kinematics
p1_ego_feats = [
    "x_position_p1",
    "y_position_p1",
    "speed_p1",
    "acceleration_p1",
    "sa_p1",
    "sin_orient_p1",
    "cos_orient_p1",
    "sin_dir_p1",
    "cos_dir_p1",
    "cos_orient_dir_p1",
    "centripetal_accel_p1",
]
STREAM_B_BASE_FEATURES.extend(p1_ego_feats)

# Impact Proxies (Derivatives)
impact_feats = [
    "jerk_p1",  # Derivative of acceleration
    "ang_vel_p1",  # Derivative of orientation
]
STREAM_B_BASE_FEATURES.extend(impact_feats)


# -----------------------------------------------------------------------------
# FEATURE UTILITIES
# -----------------------------------------------------------------------------
def get_flattened_feature_names(base_features, window_size=WINDOW_SIZE):
    """
    Generates the full list of feature names after temporal flattening.
    Format: {feature}_t{offset} (e.g., speed_p1_t-4, speed_p1_t0, speed_p1_t+4)
    """
    flattened_features = []
    center_idx = window_size // 2
    offsets = range(-center_idx, center_idx + 1)

    for offset in offsets:
        suffix = (
            f"_t{offset}" if offset < 0 else (f"_t+{offset}" if offset > 0 else "_t0")
        )
        for feat in base_features:
            flattened_features.append(f"{feat}{suffix}")

    return flattened_features


# Generate the full feature lists for import by other modules
STREAM_A_FEATURES = get_flattened_feature_names(STREAM_A_BASE_FEATURES)
STREAM_B_FEATURES = get_flattened_feature_names(STREAM_B_BASE_FEATURES)
