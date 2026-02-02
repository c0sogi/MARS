import os
import torch

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths (using metadata as requested)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Sample Submission
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Caching Paths for processed data (Parquet)
CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
N_FOLDS = 5

# Target and ID columns
ID_COL = "Id"
TARGET_COL = "Cover_Type"

# Class mapping information
# The dataset typically contains classes 1, 2, 3, 4, 6, 7.
# XGBoost requires 0-indexed classes. We will map them during processing.
NUM_CLASSES = 7

# ==========================================
# Feature Engineering Configuration
# ==========================================
PCA_N_COMPONENTS = 10

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Main model parameters
XGB_PARAMS = {
    "n_estimators": 3000,  # High cap, controlled by early stopping
    "learning_rate": 0.05,  # eta
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": NUM_CLASSES,
    "tree_method": "hist",  # GPU acceleration
    "device": DEVICE,
    "n_jobs": 12,
    "random_state": SEED,
    "eval_metric": "merror",
    "verbosity": 0,
}

# Early stopping configuration
EARLY_STOPPING_ROUNDS = 50

# ==========================================
# Debug / Runtime Control
# ==========================================
# Set DEBUG to True to run on a small subset of data for quick testing
DEBUG = False
DEBUG_SAMPLE_SIZE = 50000

# Control whether to load from cache
USE_CACHE = True
