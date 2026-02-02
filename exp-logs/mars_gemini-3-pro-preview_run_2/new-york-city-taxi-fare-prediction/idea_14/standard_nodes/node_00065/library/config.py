import os
import numpy as np

# =============================================================================
# Global Configuration & Reproducibility
# =============================================================================
SEED = 42
np.random.seed(SEED)

# =============================================================================
# Directory Paths
# =============================================================================
# Input directories (Read-Only)
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Output/Working directories
# We use idea_14 as the specific working directory for this experiment
WORKING_DIR = "./working/idea_14"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# File Paths
# =============================================================================
# Data paths derived from metadata generation
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Submission output path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Geospatial Constants
# =============================================================================
# NYC Bounding Box (Approximate limits for valid taxi rides)
# Used for clamping and filtering
NYC_BB = {"min_lon": -74.50, "max_lon": -72.80, "min_lat": 40.50, "max_lat": 41.80}

# Geohash precision levels for Multi-Scale Interaction
# Level 5: ~5km (Macro/District)
# Level 6: ~1km (Meso/Neighborhood)
# Level 7: ~150m (Micro/Building blocks)
GEOHASH_LEVELS = [5, 6, 7]

# =============================================================================
# Dual-Hygiene Filtering Parameters
# =============================================================================

# 1. STRICT FILTERING (The "Wisdom")
# Used ONLY for generating global statistics (Route/Geohash Priors) from the full dataset.
# Goal: Remove noise and outliers to get clean expected values.
STRICT_FILTER = {
    "fare_min": 2.5,
    "fare_max": 200.0,  # Exclude extremely high fares that might be data errors
    "fare_per_km_max": 10.0,  # Exclude traffic jams/waiting time dominance
}

# 2. LOOSE FILTERING (The "Learner")
# Used for the Training Set fed into the model.
# Goal: Remove physical impossibilities but retain valid heavy-tail events.
LOOSE_FILTER = {
    "fare_min": 2.5,
    "fare_max": 1000.0,  # Retain valid high-value trips (e.g., to Newark/Hamptons)
    "min_dist_km": 0.001,  # Avoid division by zero, remove 0-distance trips
}

# =============================================================================
# Feature Engineering Configuration
# =============================================================================
# K-Fold Target Encoding
NUM_FOLDS = 5

# =============================================================================
# Training Configuration
# =============================================================================
# Subsample size for training the XGBoost model (to fit in memory/time constraints)
# The full dataset is used for stats, but we train on a robust subsample.
TRAIN_SUBSAMPLE_SIZE = 5_000_000

# XGBoost Hyperparameters
# Optimized for A100 GPU usage
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 9,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 10,
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": 12,
    "random_state": SEED,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# Training loop parameters
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50
