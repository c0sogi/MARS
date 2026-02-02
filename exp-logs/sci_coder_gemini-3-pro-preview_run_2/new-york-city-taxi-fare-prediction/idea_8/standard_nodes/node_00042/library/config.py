import os

# =============================================================================
# File Paths and Directories
# =============================================================================
# Using the metadata parquet files as the source of truth
TRAIN_DATA_PATH = "./metadata/train.parquet"
VAL_DATA_PATH = "./metadata/val.parquet"
TEST_DATA_PATH = "./metadata/test.parquet"

# Output paths
SUBMISSION_PATH = "./submission/submission.csv"
CACHE_DIR = "./working/idea_8/"
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "xgb_model.json")

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# =============================================================================
# Global Configuration
# =============================================================================
SEED = 42
NUM_WORKERS = 12  # Utilizing available vCPUs

# =============================================================================
# Data Pipeline Parameters
# =============================================================================
# Subsample Size: 5 million rows.
# This balances having enough data for the gradient booster to learn
# without introducing the noise/instability of the full 55M dataset.
SUBSAMPLE_SIZE = 5_000_000

# Valid NYC Bounding Box
# Used to clamp coordinates to valid range, preventing GPS artifacts.
# Covers NYC boroughs and major airports (JFK, LGA, EWR).
BBOX = {"min_long": -74.50, "max_long": -72.80, "min_lat": 40.50, "max_lat": 41.80}

# Global Feature Extraction Parameters
# Grid Resolution: Rounding to 3 decimal places (approx 110m).
# This creates high-resolution bins for the "Route Avg Fare" lookup.
GRID_ROUNDING = 3

# K-Folds for Target Encoding
# Used in Stage 1 to calculate global stats without leakage.
K_FOLDS_TARGET_ENCODING = 5

# Post-processing
MIN_FARE_FLOOR = 2.50

# =============================================================================
# Model Hyperparameters
# =============================================================================
# Training Control
EARLY_STOPPING_ROUNDS = 50

# XGBoost Configuration
# Optimized for Regression (RMSE) on GPU.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 9,  # Sufficient depth for spatial complexity
    "subsample": 0.8,  # Row subsampling for robustness
    "colsample_bytree": 0.8,  # Column subsampling
    "n_estimators": 5000,  # High cap, controlled by early stopping
    "n_jobs": NUM_WORKERS,
    "device": "cuda",  # Leveraging NVIDIA A100
    "tree_method": "hist",  # Efficient histogram-based training
    "random_state": SEED,
    "verbosity": 0,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}
