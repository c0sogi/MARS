import os
import numpy as np

# =============================================================================
# DIRECTORIES AND FILE PATHS
# =============================================================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for Idea 12 caching
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_12")
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache File Paths
CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "X_train_augmented.parquet")
CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "X_val_augmented.parquet")
CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "X_test_augmented.parquet")

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42
N_JOBS = 12  # Utilize available vCPUs
PROB_CLIP_EPS = 1e-15  # For log loss calculation

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================
# Features provided in the CSVs
FEATURE_GROUPS = ["margin", "shape", "texture"]
NUM_FEATURES_PER_GROUP = 64

# Meta-features to be extracted from images
META_FEATURES = ["aspect_ratio", "solidity", "extent", "eccentricity"]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Linear Branch & Kernel Branch Classifier (Logistic Regression)
# Using a broad logarithmic grid as specified in the idea
# Range: 1e-2 to 1e4, with sufficient density
LR_CS_GRID = np.logspace(-3, 5, 50)

LOGISTIC_REGRESSION_PARAMS = {
    "Cs": LR_CS_GRID,
    "cv": 3,
    "penalty": "l2",
    "solver": "lbfgs",
    "scoring": "neg_log_loss",
    "max_iter": 5000,  # High iteration count to ensure convergence
    "random_state": RANDOM_SEED,
    "n_jobs": N_JOBS,
    "refit": True,  # Essential for final prediction
}

# 2. Generative Branch (LDA)
# Using Ledoit-Wolf shrinkage ('auto') and lsqr solver
LDA_PARAMS = {"solver": "lsqr", "shrinkage": "auto", "store_covariance": True}
