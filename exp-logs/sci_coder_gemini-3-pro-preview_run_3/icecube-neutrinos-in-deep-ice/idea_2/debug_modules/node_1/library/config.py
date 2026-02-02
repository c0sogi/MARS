import os
from pathlib import Path

# =============================================================================
# DIRECTORIES & PATHS
# =============================================================================
# Base directories
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working/idea_2")

# Subdirectories for outputs
CACHE_DIR = WORKING_DIR / "cache"
SUBMISSION_DIR = WORKING_DIR / "submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
SENSOR_GEOMETRY_PATH = INPUT_DIR / "sensor_geometry.csv"
SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.csv"

# Metadata Paths
TRAIN_META_PATH = METADATA_DIR / "train_metadata.parquet"
VAL_META_PATH = METADATA_DIR / "val_metadata.parquet"
TEST_META_PATH = METADATA_DIR / "test_metadata.parquet"

# Output Paths
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
N_JOBS = 12  # Available vCPUs
# Set to an integer (e.g., 500000) to limit the number of events for faster debugging.
# Set to None to use the full dataset.
DEBUG_SAMPLE_SIZE = None

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
# List of features to be extracted from pulse data
FEATURE_NAMES = [
    # Pulse Quantity
    "n_pulses",
    "aux_ratio",
    # Charge Statistics
    "charge_sum",
    "charge_mean",
    "charge_std",
    "charge_max",
    # Time Statistics
    "time_duration",
    "time_std",
    "time_mean",
    # Spatial Statistics (Geometric)
    "x_mean",
    "y_mean",
    "z_mean",
    "x_std",
    "y_std",
    "z_std",
    # Spatial Statistics (Charge-Weighted / Center of Mass)
    "x_w_mean",
    "y_w_mean",
    "z_w_mean",
    "x_w_std",
    "y_w_std",
    "z_w_std",
]

# Target definitions
# We predict the unit vector components (nx, ny, nz)
TARGET_COLS = ["target_x", "target_y", "target_z"]
# Original spherical coordinates provided in metadata
ORIG_TARGET_COLS = ["azimuth", "zenith"]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# LightGBM Regressor parameters
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbosity": -1,
    "device": "cpu",
}

# Training Control
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
