import os

# =========================================================================================
# Global Path Configurations
# =========================================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_39"

# Ensure working directory and subdirectories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "cache"), exist_ok=True)

# Input Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Input Tracking Paths
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

# Output Submission Path
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Caching Paths (Parquet/NPY for efficiency)
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
PROCESSED_TRAIN_PATH = os.path.join(CACHE_DIR, "processed_train.parquet")
PROCESSED_VAL_PATH = os.path.join(CACHE_DIR, "processed_val.parquet")
PROCESSED_TEST_PATH = os.path.join(CACHE_DIR, "processed_test.parquet")
HARD_NEGATIVE_INDICES_PATH = os.path.join(CACHE_DIR, "hard_negative_indices.npy")
BEST_THRESHOLD_PATH = os.path.join(WORKING_DIR, "models", "best_threshold.npy")

# =========================================================================================
# Global Constants & Gating
# =========================================================================================
SEED = 42
N_JOBS = 12

# Gating & Sentinel Values
GATING_THRESHOLD = 3.0  # Yards. Only process pairs with min_dist < 3.0 in the window
SENTINEL_VALUE = -1.0  # Distinct distance value for Player-Ground interactions

# =========================================================================================
# Feature Engineering Configurations
# =========================================================================================
# Time Window for Lags
# Data is 10Hz. +/- 10 steps = +/- 1.0 second context.
LAG_STEPS = 10
WINDOW_SIZE = (LAG_STEPS * 2) + 1

# Geometric Invariant Features
# These are computed per timestep and then flattened over the window.
GEOMETRIC_FEATURES = [
    "distance",
    "closing_speed",  # - (r . v_rel) / |r|
    "tangential_speed",  # sqrt(|v_rel|^2 - closing_speed^2)
    "specific_angular_momentum",  # |r x v_rel|
    "radial_acceleration",  # (r . a_rel) / |r|
    "jerk_p1",  # Derivative of acceleration magnitude
    "jerk_p2",
    "spatial_density_p1",  # Count of other players within radius
    "spatial_density_p2",
]

# =========================================================================================
# Training Configurations
# =========================================================================================
# Mining Strategy
HARD_NEGATIVE_THRESHOLD = 0.05  # Probability threshold for Scout union
ANCHOR_RATIO = 1.0  # Ratio of Random Easy Negatives to Positives (1:1)

# -----------------------------------------------------------------------------------------
# LightGBM Configuration
# -----------------------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 256,  # High capacity for complex boundaries
    "max_depth": 10,  # Prevent infinite growth
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 3000,
    "early_stopping_rounds": 100,
    "is_unbalance": True,  # Handle class imbalance internally
    "verbosity": -1,
    "n_jobs": N_JOBS,
    "random_state": SEED,
}

# Scout Config (Faster, for mining)
SCOUT_LGBM_PARAMS = LGBM_PARAMS.copy()
SCOUT_LGBM_PARAMS.update(
    {"n_estimators": 500, "num_leaves": 64, "max_depth": 8, "early_stopping_rounds": 50}
)

# -----------------------------------------------------------------------------------------
# XGBoost Configuration
# -----------------------------------------------------------------------------------------
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 3000,
    "early_stopping_rounds": 100,
    "tree_method": "hist",
    "device": "cuda",  # Leverage NVIDIA A100
    "n_jobs": N_JOBS,
    "random_state": SEED,
    # Note: scale_pos_weight will be calculated dynamically during training based on dataset
}

# Scout Config (Faster, for mining)
SCOUT_XGB_PARAMS = XGB_PARAMS.copy()
SCOUT_XGB_PARAMS.update(
    {"n_estimators": 500, "max_depth": 8, "early_stopping_rounds": 50}
)
