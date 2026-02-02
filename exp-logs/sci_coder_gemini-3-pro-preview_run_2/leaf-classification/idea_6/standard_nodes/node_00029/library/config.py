import os
import numpy as np

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific working directory for this idea
IDEA_ID = "idea_8"
CACHE_DIR = os.path.join(WORKING_DIR, IDEA_ID)

# Ensure necessary directories exist (except input which is read-only)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
VAL_SIZE = 0.2  # Fraction used for validation split (reflected in metadata)

# File paths for metadata
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# --- Logistic Regression Base Learner ---
# Solver and penalty configuration
LR_SOLVER = "lbfgs"
LR_PENALTY = "l2"
LR_MAX_ITER = 5000  # Increased to ensure convergence

# Hyperparameter Search Grid for C (Inverse Regularization Strength)
# Covering the range [0.01, 10000] with higher density
LR_C_GRID = np.logspace(-2, 4, 20)

# Cross-Validation settings for tuning
CV_FOLDS = 3

# --- Linear Discriminant Analysis (LDA) ---
# Parameters for the generative anchor model
LDA_SOLVER = "lsqr"  # Least squares solution
LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

# =============================================================================
# SUBMISSION CONFIGURATION
# =============================================================================
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
