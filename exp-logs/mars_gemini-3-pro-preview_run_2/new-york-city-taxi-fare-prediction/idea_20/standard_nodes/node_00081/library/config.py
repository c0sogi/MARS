import os

# =============================================================================
# Directories and File Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_20")
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Global Constants
# =============================================================================
SEED = 42

# NYC Bounding Box [lon_min, lat_min, lon_max, lat_max]
# Used to clamp coordinates to valid NYC area, preventing linear extrapolation
# issues outside the dense training manifold.
NYC_BBOX = [-74.50, 40.50, -72.80, 41.80]

# =============================================================================
# Data Hygiene & Filtering Criteria
# =============================================================================
# Wisdom Criteria: Used to generate clean, robust statistical priors (Moments).
# These strict filters ensure that the "Expected Price" features are not skewing
# by extreme outliers or bad data (e.g., 0 distance but high fare).
WISDOM_CRITERIA = {"min_fare": 2.50, "max_fare": 200.00, "max_fare_per_km": 10.00}

# Learner Criteria: Used for the training dataset rows.
# These loose filters retain valid high-fare outliers (e.g., >$200) to ensure
# the model learns to predict the "Heavy Tail" required for the RMSE metric.
LEARNER_CRITERIA = {"min_fare": 2.50}

# =============================================================================
# Feature Engineering Configuration
# =============================================================================
# Hierarchical Spatial Levels for Geohashing
# Levels 5, 6, and 7 provide a "Pyramid of Priors" ranging from regional to block-level.
GEOHASH_LEVELS = [5, 6, 7]

# =============================================================================
# Model & Training Configuration
# =============================================================================
# Training Subsample Size
# Limits the training set to a manageable size for the 24h runtime limit while
# providing sufficient density for the XGBoost histogram bins.
TRAIN_SUBSAMPLE_SIZE = 5_000_000

# XGBoost Hyperparameters
# Optimized for the NVIDIA A100 GPU using 'hist' method and 'cuda' device.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",
    "n_estimators": 5000,  # High cap, controlled by early stopping
    "learning_rate": 0.02,  # Lower LR for better convergence on complex manifold
    "max_depth": 8,  # Sufficient depth to capture spatial interactions
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,  # L1 Regularization
    "reg_lambda": 1.0,  # L2 Regularization
    "n_jobs": 12,
    "random_state": SEED,
}

# Training Control
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
