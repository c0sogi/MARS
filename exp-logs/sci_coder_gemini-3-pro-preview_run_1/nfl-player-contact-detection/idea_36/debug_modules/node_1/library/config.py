import os
import numpy as np

# =============================================================================
# 1. GLOBAL CONSTANTS & REPRODUCIBILITY
# =============================================================================
SEED = 42
np.random.seed(SEED)


def set_seed(seed=SEED):
    """Utility to set seeds for other libraries if imported."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# =============================================================================
# 2. DIRECTORY PATHS
# =============================================================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for Idea 36 (DBRK-AME)
WORKING_DIR = "./working/idea_36"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Files (Read-Only)
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Pre-generated)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Files
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cached Data Paths
CACHED_TRAIN_FEATURES = os.path.join(CACHE_DIR, "train_features.parquet")
CACHED_VAL_FEATURES = os.path.join(CACHE_DIR, "val_features.parquet")
CACHED_TEST_FEATURES = os.path.join(CACHE_DIR, "test_features.parquet")
CACHED_HARD_NEGATIVES = os.path.join(CACHE_DIR, "hard_negative_indices.npy")

# =============================================================================
# 3. STRATEGY HYPERPARAMETERS (DBRK-AME)
# =============================================================================
# Gating & Windowing
# Relaxed Quadratic Gating Threshold (yards)
GATING_THRESHOLD = 3.0
# Timesteps to look backward and forward (Total window = 2*WINDOW_SIZE + 1)
WINDOW_SIZE = 10

# Mining & Training
# Ratio of Random Easy Negatives (Anchors) to Positives
ANCHOR_RATIO = 1.0
# Probability threshold for a negative to be considered "Hard" by Scouts
HARD_NEGATIVE_THRESHOLD = 0.05

# Feature Engineering Flags
USE_DYNAMIC_BASIS = True
USE_RELATIVE_KINEMATICS = True

# =============================================================================
# 4. MODEL CONFIGURATIONS
# =============================================================================
# Shared Training Parameters
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100

# -----------------------------------------------------------------------------
# LightGBM Expert Configuration
# -----------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 256,  # High capacity for dense numerical data
    "max_depth": 10,  # Constrain depth to prevent overfitting
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "is_unbalance": True,  # Handle class imbalance internally
    "verbosity": -1,
    "n_jobs": 12,
    "seed": SEED,
}

# -----------------------------------------------------------------------------
# XGBoost Expert Configuration
# -----------------------------------------------------------------------------
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 10,  # Consistent with LGBM
    "tree_method": "hist",  # Faster histogram-based method
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 12,
    "random_state": SEED,
    # Note: scale_pos_weight should be calculated dynamically based on batch
}

# =============================================================================
# 5. FEATURE DEFINITIONS
# =============================================================================
# Base names for dynamic basis features.
# These will be expanded by time lags: e.g., 'dist_t-10', ..., 'dist_t+10'
DYNAMIC_FEATURE_BASES = [
    "dist",  # Euclidean distance
    "v_rad",  # Radial velocity (projected on dynamic basis)
    "v_tan",  # Tangential velocity (projected on dynamic basis orthogonal)
    "a_rad",  # Radial acceleration
    "a_tan",  # Tangential acceleration
    "orientation_rel",  # Relative orientation
]

# Physics Primitives (Scalars computed over the window)
SCALAR_FEATURES = ["min_dist", "time_to_collision", "jerk_mag", "angular_jerk"]


def get_feature_names():
    """Generates the full list of flattened feature names."""
    feature_names = []
    # Time-domain lags for dynamic basis features
    for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
        suffix = f"_t{lag:+d}"  # e.g., _t-10, _t+0, _t+10
        for base in DYNAMIC_FEATURE_BASES:
            feature_names.append(f"{base}{suffix}")

    # Add scalar primitives
    feature_names.extend(SCALAR_FEATURES)
    return feature_names
