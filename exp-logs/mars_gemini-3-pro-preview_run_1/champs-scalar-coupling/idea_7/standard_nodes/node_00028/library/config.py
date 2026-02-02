import os
import numpy as np

# ==========================================
# Global Configuration & Runtime Flags
# ==========================================
RANDOM_SEED = 42
DEBUG_MODE = False  # Set to True for fast prototyping
DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use in debug mode

# ==========================================
# Directory Structure & File Paths
# ==========================================
# Input Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working Directory (for Caching and Models)
WORKING_DIR = "./working/idea_7"
MODEL_SAVE_DIR = os.path.join(WORKING_DIR, "xgb_models")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")

# Submission Directory
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache File Paths (Parquet)
CACHE_STRUCTURES_PROCESSED = os.path.join(CACHE_DIR, "structures_processed.parquet")
CACHE_TRAIN_FEATURES = os.path.join(CACHE_DIR, "train_features.parquet")
CACHE_VAL_FEATURES = os.path.join(CACHE_DIR, "val_features.parquet")
CACHE_TEST_FEATURES = os.path.join(CACHE_DIR, "test_features.parquet")

# ==========================================
# Physical Constants & Domain Knowledge
# ==========================================
# Covalent Radii (Angstroms) for Adaptive Thresholding
# Values approximated from Alvarez (2008)
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57}

# Tolerance to add to r_i + r_j to determine connectivity
BOND_RADIUS_TOLERANCE = 0.3

# Atom and Coupling Types
ATOM_TYPES = ["H", "C", "N", "O", "F"]
COUPLING_TYPES = ["1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN"]

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
# Optimized for A100 GPU and High-Order Feature Interactions
XGB_PARAMS = {
    "objective": "reg:absoluteerror",  # Optimizing MAE directly
    "eval_metric": "mae",
    "tree_method": "hist",  # Efficient histogram-based algorithm
    "device": "cuda",  # Utilize NVIDIA A100
    "max_depth": 11,  # Deep trees for complex interactions (10-12 range)
    "learning_rate": 0.01,  # Low LR for robust convergence
    "n_estimators": 40000,  # High ceiling, controlled by early stopping
    "colsample_bytree": 0.4,  # Heavy regularization for high-dim features
    "subsample": 0.8,  # Row subsampling
    "reg_alpha": 0.1,  # L1 Regularization
    "reg_lambda": 1.0,  # L2 Regularization
    "n_jobs": 12,  # CPU threads for data loading/pre-proc
    "random_state": RANDOM_SEED,
    "verbosity": 1,
}

# Training Control
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 500  # Print metrics every N rounds
