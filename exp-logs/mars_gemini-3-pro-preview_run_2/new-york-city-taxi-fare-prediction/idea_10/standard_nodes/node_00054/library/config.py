import os

# =============================================================================
# File Paths and Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_10_fixed"  # Dedicated cache directory for this strategy
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Global Constants & Reproducibility
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# Data Processing Parameters
# =============================================================================
# Grid resolution for global route statistics (3 decimal places approx 100m)
GRID_PRECISION = 3

# Subsample size for the training set to maintain gradient stability
# The full dataset is used for the Global Knowledge Base (priors),
# but we train on a stable subsample.
SUBSAMPLE_SIZE = 5_000_000

# NYC Bounding Box for Coordinate Clamping
# Prevents linear extrapolation errors for coordinates far outside the city
NYC_MIN_LON = -74.50
NYC_MAX_LON = -72.80
NYC_MIN_LAT = 40.50
NYC_MAX_LAT = 41.80

# =============================================================================
# Physics-Consistent Filtering Thresholds
# =============================================================================
# Maximum absolute fare to consider valid for training (removes $93k outliers)
MAX_FARE = 500.0

# Maximum Fare per Kilometer ($/km) to filter impossible price/distance ratios
MAX_FARE_PER_KM = 50.0

# Minimum fare floor for post-processing predictions
MIN_FARE = 2.50

# =============================================================================
# Model Hyperparameters (XGBoost)
# =============================================================================
# Configuration for the Residual-Learning XGBoost model
XGB_PARAMS = {
    "objective": "reg:squarederror",  # L2 Loss for residual prediction
    "eval_metric": "rmse",
    "tree_method": "gpu_hist",  # Use GPU acceleration
    "max_depth": 8,  # Depth of trees
    "learning_rate": 0.05,  # Step size shrinkage
    "n_estimators": 2000,  # Maximum number of boosting rounds
    "subsample": 0.8,  # Row subsampling for bagging
    "colsample_bytree": 0.8,  # Column subsampling
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
    "n_jobs": 12,  # Number of CPU threads
    "random_state": RANDOM_SEED,
    "early_stopping_rounds": 50,  # Stop if validation score doesn't improve
}
