import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for caching intermediate files (idea_13 specific)
WORKING_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths (using generated metadata)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA HYGIENE & PREPROCESSING
# =============================================================================
# Bounding box for NYC to filter/clamp coordinates
# Covers NYC and major airports (JFK, LGA, EWR)
NYC_BOUNDING_BOX = {
    "min_lat": 40.5,
    "max_lat": 41.0,
    "min_lon": -74.3,
    "max_lon": -73.7,
}

# Feature Engineering: Spatial Resolutions for Target Encoding
# 3 decimal places ~ 100m (Micro)
# 2 decimal places ~ 1km (Meso)
# 1 decimal place ~ 10km (Macro)
SPATIAL_RESOLUTIONS = [3, 2, 1]

# Number of folds for Vectorized Subtraction (Target Encoding)
NUM_FOLDS = 5

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Subsample size for the training set (Learner)
# The prompt specifies 5 million rows for stability
SUBSAMPLE_SIZE = 5_000_000

# Random Seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# MODEL HYPERPARAMETERS (XGBoost)
# =============================================================================
XGB_PARAMS = {
    # Objective and Metric
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    # Tree Construction
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 5000,  # High number, controlled by early stopping
    "min_child_weight": 10,
    # Regularization & Sampling
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    # Hardware Acceleration
    "tree_method": "hist",
    "device": "cuda",  # Use A100 GPU
    "n_jobs": 12,
    # Reproducibility
    "random_state": RANDOM_SEED,
}

# Training Loop Settings
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
