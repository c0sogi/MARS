import os

# ==========================================
# Project Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for Idea 3 to store cached parquet files
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Final Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
# Sensor column names found in the raw data files
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]

# Global Random Seed for reproducibility across numpy, pandas, and models
SEED = 42

# ==========================================
# Feature Engineering Configuration
# ==========================================
# Number of non-overlapping windows to divide the signal into for temporal evolution features
NUM_WINDOWS = 10

# ==========================================
# Model Hyperparameters (LightGBM)
# ==========================================
# Configuration for LightGBM Regressor targeting MAE
LGBM_PARAMS = {
    "objective": "mae",
    "metric": "mae",
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_jobs": 12,  # optimized for 12 vCPUs
    "random_state": SEED,
    "verbose": -1,  # Silent execution
}

# Early stopping configuration to prevent overfitting
EARLY_STOPPING_ROUNDS = 100
