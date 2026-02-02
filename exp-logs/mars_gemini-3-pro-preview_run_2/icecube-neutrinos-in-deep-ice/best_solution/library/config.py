import os
import numpy as np

# =============================================================================
# GLOBAL CONFIGURATION & REPRODUCIBILITY
# =============================================================================
SEED = 42
np.random.seed(SEED)

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
# Read-only input directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Writable working directories
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_4")
SUBMISSION_DIR = "./submission"

# Ensure essential writable directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Metadata paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")

# Data paths
SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache paths for engineered features (Parquet format)
TRAIN_FEATURES_PATH = os.path.join(IDEA_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(IDEA_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(IDEA_DIR, "test_features.parquet")

# Model artifact paths (one for each vector component)
MODEL_X_PATH = os.path.join(IDEA_DIR, "lgbm_model_x.txt")
MODEL_Y_PATH = os.path.join(IDEA_DIR, "lgbm_model_y.txt")
MODEL_Z_PATH = os.path.join(IDEA_DIR, "lgbm_model_z.txt")

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# List of features to be generated for each event
FEATURE_NAMES = [
    # Signal Magnitude
    "total_charge",
    "n_pulses",
    "log_n_pulses",
    # Spatial Center (Charge Weighted Center of Gravity)
    "center_x",
    "center_y",
    "center_z",
    # Signal Spread (Weighted Standard Deviation)
    "spread_x",
    "spread_y",
    "spread_z",
    # Temporal Evolution (Percentiles relative to first pulse)
    "time_10th",
    "time_50th",
    "time_90th",
    "time_duration",
    # Shape Descriptors (Covariance Matrix Elements)
    # These capture the orientation/linearity of the track
    "cov_xx",
    "cov_yy",
    "cov_zz",
    "cov_xy",
    "cov_xz",
    "cov_yz",
]

# Targets
TARGET_COLS_VECTOR = ["target_x", "target_y", "target_z"]  # Intermediate vector targets
TARGET_COLS_ANGLES = ["azimuth", "zenith"]  # Final angle targets

# =============================================================================
# MODEL HYPERPARAMETERS (LightGBM)
# =============================================================================
# Parameters for the LightGBM Regressor
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "n_jobs": 12,
    "seed": SEED,
    "force_col_wise": True,
}

# Training Loop Configuration
N_ESTIMATORS = 3000
EARLY_STOPPING_ROUNDS = 100

# =============================================================================
# DEBUGGING / DEVELOPMENT
# =============================================================================
# Set to an integer (e.g., 50000) to limit the number of events processed
# during feature engineering and training for rapid iteration.
# Set to None to use the full dataset.
DEBUG_SAMPLE_SIZE = None
