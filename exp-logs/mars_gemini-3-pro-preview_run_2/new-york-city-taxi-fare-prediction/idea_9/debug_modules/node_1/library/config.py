import os

# =============================================================================
# Global Configuration & Hyperparameters
# =============================================================================

# -----------------------------------------------------------------------------
# File Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_9"

# Ensure the working directory exists for caching
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths (Parquet)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Submission Output
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# -----------------------------------------------------------------------------
# Data Cleaning & Preprocessing
# -----------------------------------------------------------------------------
# NYC Bounding Box for clamping coordinates
# Format: (min_lon, max_lon, min_lat, max_lat)
# Covers NYC boroughs and major airports (JFK, LGA, EWR)
NYC_BOUNDING_BOX = (-74.5, -72.8, 40.5, 41.8)

# Target Column
TARGET_COL = "fare_amount"

# -----------------------------------------------------------------------------
# Feature Engineering (Global-Prior Augmented)
# -----------------------------------------------------------------------------
# Spatial Discretization: Round coordinates to this many decimal places
# 3 decimal places approx 110m resolution
COORD_PRECISION = 3

# Number of folds for Vectorized Target Encoding (Background-Augmented K-Fold)
NUM_FOLDS = 5

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
# Size of the stable random subsample for model training (5 Million)
TRAIN_SUBSAMPLE_SIZE = 5_000_000

# Random Seed for reproducibility
RANDOM_SEED = 42

# Debugging / Development Mode
# If DEBUG is True, use DEBUG_SIZE for training to speed up development
DEBUG = False
DEBUG_SIZE = 100_000

# -----------------------------------------------------------------------------
# Model Hyperparameters (XGBoost)
# -----------------------------------------------------------------------------
# Optimized for NVIDIA A100 GPU (device="cuda")
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 9,
    "learning_rate": 0.03,
    "n_estimators": 10000,  # High cap, controlled by early stopping
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": 12,
    "random_state": RANDOM_SEED,
}

# Training Loop Controls
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100

# -----------------------------------------------------------------------------
# Post-Processing
# -----------------------------------------------------------------------------
# Minimum fare floor to apply to predictions
MIN_FARE = 2.50
