import os
import numpy as np

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_7"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

SEED = 42
SCOUT_THRESHOLD = 0.01  # Threshold for Hard-Negative Mining
WINDOW_SIZE = 10  # Temporal window size (+/- 10 steps)

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Tier 1 Features: Instantaneous, low-cost features for the Scout model
# These are computed for the entire dataset (3.4M rows).
TIER_1_FEATURES = [
    "distance",
    "speed_p1",
    "speed_p2",
    "speed_diff",
    "acceleration_p1",
    "acceleration_p2",
    "acc_diff",
    "direction_diff",
    "orientation_diff",
    "is_ground",
    "step",
]

# Tier 2 Configuration: Contextual, high-cost features for the Expert model
# These are computed only for the mined subset (Positives + Hard Negatives).

# Features to apply rolling windows to
TIER_2_WINDOW_BASE_COLS = [
    "distance",
    "speed_p1",
    "speed_p2",
    "acceleration_p1",
    "acceleration_p2",
    "x_position_p1",
    "y_position_p1",
    "x_position_p2",
    "y_position_p2",
]

# Additional physics/geometry features derived in Tier 2
TIER_2_EXTRA_FEATURES = [
    "jerk_p1",
    "jerk_p2",
    "angular_jerk_p1",
    "angular_jerk_p2",
    "spatial_density",
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Scout Model (LightGBM): Lightweight, shallow trees for high-throughput filtering
LGBM_SCOUT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 800,
    "learning_rate": 0.1,
    "max_depth": 4,
    "num_leaves": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": -1,
    "seed": SEED,
}

# Expert Model (LightGBM): High capacity, deeper trees for precision
LGBM_EXPERT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "is_unbalance": True,  # Handle remaining imbalance after mining
    "verbose": -1,
    "n_jobs": -1,
    "seed": SEED,
}

# Expert Model (XGBoost): High capacity, used for ensembling with LGBM
XGB_EXPERT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,  # Conservative rebalancing
    "n_jobs": -1,
    "random_state": SEED,
    "enable_categorical": True,
    "tree_method": "hist",  # Efficient histogram-based training
}
