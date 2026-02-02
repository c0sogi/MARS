import os
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# Directory & File Paths
# ---------------------------------------------------------
# Base directories
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working/idea_3")

# Sub-directories for artifacts
CACHE_DIR = WORKING_DIR / "cache"
SUBMISSION_DIR = WORKING_DIR / "submission"

# Ensure working directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Specific File Paths
SENSOR_GEOMETRY_PATH = INPUT_DIR / "sensor_geometry.csv"
TRAIN_META_PATH = METADATA_DIR / "train_metadata.parquet"
VAL_META_PATH = METADATA_DIR / "val_metadata.parquet"
TEST_META_PATH = METADATA_DIR / "test_metadata.parquet"
SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.csv"

# ---------------------------------------------------------
# Hardware & Reproducibility Settings
# ---------------------------------------------------------
SEED = 42
N_THREADS = 12  # Utilizing the 12 vCPUs available

# ---------------------------------------------------------
# Feature Engineering Configuration
# ---------------------------------------------------------
# Features derived from Spatiotemporal Eigen-Decomposition
FEATURE_NAMES = [
    # --- Pulse Aggregates ---
    "charge_sum",  # Total light intensity
    "charge_mean",  # Average pulse charge
    "charge_std",  # Variability in pulse charge
    "charge_count",  # Number of pulses (event size)
    # --- Temporal Features ---
    "time_range",  # Duration of the event
    "time_std",  # Temporal spread
    "aux_ratio",  # Ratio of auxiliary (noise-like) pulses
    # --- Spatial Center of Gravity (Charge-weighted) ---
    "pos_x_mean",
    "pos_y_mean",
    "pos_z_mean",
    "pos_x_std",
    "pos_y_std",
    "pos_z_std",
    # --- Eigen-Features (SVD of position covariance) ---
    "eval_1",
    "eval_2",
    "eval_3",  # Eigenvalues (sorted desc)
    "eval_ratio_12",  # Elongation (eval_1 / eval_2)
    "eval_ratio_13",  # Flatness (eval_1 / eval_3)
    # --- Spatiotemporal Covariance (Directionality) ---
    "cov_x_t",  # Covariance between x and time
    "cov_y_t",  # Covariance between y and time
    "cov_z_t",  # Covariance between z and time
    "cov_p_t",  # Covariance between position along main axis and time
]

# Intermediate regression targets (Cartesian unit vector components)
TARGET_COLS = ["target_x", "target_y", "target_z"]

# ---------------------------------------------------------
# Model Hyperparameters (LightGBM)
# ---------------------------------------------------------
# Default parameters for the regression models.
# These are tuned for stability and performance on CPU.
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": 12,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "seed": SEED,
    "n_jobs": N_THREADS,
    "force_col_wise": True,
}

# Training loop settings
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
