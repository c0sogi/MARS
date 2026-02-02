import os

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_41")
SUBMISSION_DIR = "./submission"

# Ensure cache and submission directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Tracking Data Paths
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

# Output Paths
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42

# Temporal Window: +/- 10 steps around the target step (Total 21 steps)
WINDOW_HALF = 10
WINDOW_SIZE = 2 * WINDOW_HALF + 1

# Gating and Sampling
GATING_THRESHOLD = 3.0  # Yards. Pairs with min_dist > 3.0 are discarded early.
ANCHOR_RATIO = 1.0  # Ratio of Random Negatives (Anchors) to Positives in Expert Set.

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
# These are the base kinematic features calculated for each timestep in the window.
# The feature engineering pipeline will flatten these over the window (e.g., r_long_t-10, ..., r_long_t+10).
BASE_KINEMATIC_FEATURES = [
    # Relative Position projected onto Relative Velocity Basis
    "r_long",  # Longitudinal distance (Time-space distance)
    "r_trans",  # Transverse distance (Miss distance)
    # Relative Acceleration projected onto Relative Velocity Basis
    "a_long",  # Longitudinal acceleration (Impact Force potential)
    "a_trans",  # Transverse acceleration (Turning Force potential)
    # Explicit Interaction Primitives
    "ttc",  # Time-To-Collision (r / v_closing)
    "jerk_p1",  # Magnitude of derivative of acceleration for P1
    "jerk_p2",  # Magnitude of derivative of acceleration for P2
    "angular_jerk",  # Rate of change of angular acceleration (proxy)
    # Standard Kinematics
    "speed_p1",
    "speed_p2",
    "accel_p1",
    "accel_p2",
    "orientation_diff",  # Difference in orientation
    "direction_diff",  # Difference in motion direction
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# LightGBM Configuration (Leaf-wise growth)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "n_estimators": 2000,  # High cap, controlled by early stopping
    "num_leaves": 256,  # Deep trees for high capacity
    "max_depth": 10,  # Limit depth to prevent overfitting with high leaves
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "is_unbalance": True,  # Handle class imbalance internally
    "random_state": SEED,
    "n_jobs": 12,
    "verbose": -1,
}

# XGBoost Configuration (Level-wise growth)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "max_depth": 10,  # Deep trees
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    # scale_pos_weight will be set dynamically during training based on ratio,
    # but a default can be provided here if needed.
    "tree_method": "hist",  # Efficient histogram-based method
    "device": "cuda",  # Use GPU if available (A100 is provided)
    "random_state": SEED,
    "n_jobs": 12,
    "verbosity": 0,
}
