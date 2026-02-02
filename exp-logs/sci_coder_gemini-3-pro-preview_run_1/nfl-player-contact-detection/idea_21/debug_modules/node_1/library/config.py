import os
import numpy as np

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_21"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths (Parquet files)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")
HARD_NEGATIVE_INDICES_PATH = os.path.join(WORKING_DIR, "hard_negative_indices.npy")

# Model Paths
MODEL_DIR = os.path.join(WORKING_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
STEP_DELTA = 0.1  # Seconds between steps

# Gating Strategy
# Relaxed Quadratic Gating: Keep pairs where min(dist) < threshold
GATING_THRESHOLD = 3.0  # Yards

# Sentinel Value Strategy
# Distance for Ground interactions is set to this value to allow tree splits
GROUND_DISTANCE_SENTINEL = -1.0

# Windowing Configuration for Vector Decomposition
# 10Hz data. Window size of +/- 6 frames = +/- 0.6 seconds context.
WINDOW_PRE = 6
WINDOW_POST = 6
TOTAL_WINDOW_SIZE = WINDOW_PRE + WINDOW_POST + 1

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# 1. Physics Primitives (Scalar features calculated at t=0)
PHYSICS_FEATURES = [
    "distance",
    "speed_p1",
    "speed_p2",
    "acceleration_p1",
    "acceleration_p2",
    "closing_speed",
    "time_to_collision",
    "orientation_p1",
    "orientation_p2",
    "direction_p1",
    "direction_p2",
]

# 2. Spectral Features (Computed over the window)
# High-pass filtered RMS energy of decomposed acceleration vectors
SPECTRAL_FEATURES = ["acc_radial_spectral_energy", "acc_tangential_spectral_energy"]

# 3. Raw Vector Components (To be flattened over the window)
# These base columns will be expanded into lag/lead features
RAW_VECTOR_BASE_COLS = ["rel_vel_x", "rel_vel_y", "rel_acc_x", "rel_acc_y"]


def get_feature_names():
    """
    Generates the exhaustive list of feature column names used for training.
    Combines physics primitives, spectral features, and flattened window vectors.
    """
    features = list(PHYSICS_FEATURES) + list(SPECTRAL_FEATURES)

    # Generate windowed feature names for raw vector components
    # Naming convention: {base}_lag_{i} or {base}_lead_{i}
    for base_col in RAW_VECTOR_BASE_COLS:
        # Current timestep (t=0)
        features.append(base_col)

        # Past timesteps (Lags)
        for i in range(1, WINDOW_PRE + 1):
            features.append(f"{base_col}_lag_{i}")

        # Future timesteps (Leads)
        for i in range(1, WINDOW_POST + 1):
            features.append(f"{base_col}_lead_{i}")

    return features


# Final list of feature columns
FEATURE_COLS = get_feature_names()

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# LightGBM: Leaf-wise growth, good for dense numerical data
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 256,
    "max_depth": 10,
    "learning_rate": 0.02,
    "n_estimators": 3000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
    "is_unbalance": True,  # Handle class imbalance internally
    "verbose": -1,
}

# XGBoost: Level-wise growth, histogram-based
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 10,
    "learning_rate": 0.02,
    "n_estimators": 3000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "grow_policy": "depthwise",
    "random_state": SEED,
    "n_jobs": -1,
    "scale_pos_weight": 10,  # Approximate imbalance ratio correction
    "verbosity": 0,
}
