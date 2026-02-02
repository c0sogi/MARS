import os

# =============================================================================
# GLOBAL CONSTANTS & PATHS
# =============================================================================

# Random Seed for Reproducibility
SEED = 42

# Directory Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Tracking Data Paths
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================

# Geometric Gating: Distance threshold (yards) to filter trivial non-contacts
# Only Player-Player pairs > 2.5 yards are discarded. All Ground pairs kept.
GATING_DISTANCE = 2.5

# Invariant Kinematic Set (IKS): Radius (yards) to identify neighbors
# Used to compute set-based aggregates (min/max/mean of relative kinematics)
IKS_NEIGHBOR_RADIUS = 2.0

# Temporal Window: Number of steps before/after current step to flatten
# Window size 10 means +/- 1 second (at 10Hz)
WINDOW_SIZE = 10

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# LightGBM Configuration (Leaf-wise Growth)
# High capacity (256 leaves) with class imbalance handling
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 256,
    "max_depth": 10,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "is_unbalance": True,  # Handle class imbalance automatically
    "random_state": SEED,
    "n_jobs": 12,  # Use available vCPUs
    "verbose": -1,
}

# XGBoost Configuration (Level-wise Growth)
# Deep trees (depth 10) with GPU acceleration and positive weighting
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "booster": "gbtree",
    "max_depth": 10,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda",  # Use NVIDIA A100
    "scale_pos_weight": 10,  # Explicit weighting for imbalance (approx ratio, tuneable)
    "random_state": SEED,
    "n_jobs": 12,
    "verbosity": 0,
}
