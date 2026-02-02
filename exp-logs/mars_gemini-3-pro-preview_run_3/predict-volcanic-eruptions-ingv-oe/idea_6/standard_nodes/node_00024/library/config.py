import os

# ==========================================
# Path Configurations
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data & Signal Processing Configurations
# ==========================================
SAMPLING_RATE = 100  # Hz (60001 samples / 600 seconds)
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]

# Savitzky-Golay Filter Parameters (for Kinematic Features)
# Window length must be odd.
SG_WINDOW_LENGTH = 11
SG_POLYORDER = 2

# Feature Extraction Parameters
# Divide the 10-minute segment into non-overlapping windows
NUM_TEMPORAL_WINDOWS = 10

# ==========================================
# Global Training Configurations
# ==========================================
SEED = 42
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 100

# Debugging / Development
# Set DEBUG to True to run on a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 200

# ==========================================
# Model Hyperparameters
# ==========================================

# LightGBM Regressor Parameters
LGBM_PARAMS = {
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": -1,
    "objective": "regression",
    "metric": "mae",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "random_state": SEED,
    "n_jobs": 12,
    "verbosity": -1,
    "force_col_wise": True,
}

# XGBoost Regressor Parameters
# Optimized for NVIDIA A100 GPU usage
XGB_PARAMS = {
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "random_state": SEED,
    "n_jobs": 12,
    "tree_method": "hist",
    "device": "cuda",
}
