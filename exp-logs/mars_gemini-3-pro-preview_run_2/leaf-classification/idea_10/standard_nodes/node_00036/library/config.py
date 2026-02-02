import os
import numpy as np

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Random Seed for Reproducibility
RANDOM_SEED = 42

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_10"
SUBMISSION_DIR = "./submission"

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths (Using Metadata Splits)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Paths
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Dictionary mapping View Name to Feature Column Prefix
# The dataset contains 3 sets of 64 features each.
FEATURE_VIEWS = {"Margin": "margin", "Shape": "shape", "Texture": "texture"}

# List of all available views including the Global (combined) view
ALL_VIEWS = ["Global", "Margin", "Shape", "Texture"]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Cross-Validation Settings
N_FOLDS = 3
SCORING_METRIC = "neg_log_loss"

# Logistic Regression Hyperparameters
# Grid for 'Cs' in LogisticRegressionCV (Inverse of regularization strength)
# Log-spaced values to cover high regularization to low regularization
LR_CS = np.logspace(-2, 4, 20)

# LDA Hyperparameters
# Solver and shrinkage are typically handled in the model definition,
# but we note the preference for 'lsqr' or 'eigen' with shrinkage here.
LDA_SOLVER = "lsqr"
LDA_SHRINKAGE = "auto"

# =============================================================================
# RUNTIME / DEBUGGING
# =============================================================================

# Set to an integer (e.g., 100) to limit training data size for debugging.
# Set to None for full training run.
DEBUG_SAMPLE_SIZE = None
