import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_7"
SUBMISSION_DIR = "./submission"

# Input Data Paths (Metadata)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
# Note: Directories should be created by the processing scripts, not on import.
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths
# Used to store preprocessed arrays to save time on re-runs
CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "X_train_transformed.npy")
CACHE_VAL_PATH = os.path.join(WORKING_DIR, "X_val_transformed.npy")
CACHE_TEST_PATH = os.path.join(WORKING_DIR, "X_test_transformed.npy")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
N_JOBS = 12  # Utilize available vCPUs

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"

# Columns to exclude when identifying feature columns
# 'file_path' is generated in metadata, 'full_path' might be generated during EDA
EXCLUDE_COLS = ["id", "species", "file_path", "full_path"]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Base Estimator: Linear Discriminant Analysis (LDA)
# 'lsqr' solver is required to use shrinkage.
# 'auto' shrinkage uses the Ledoit-Wolf lemma for covariance estimation.
LDA_SOLVER = "lsqr"
LDA_SHRINKAGE = "auto"

# Preprocessing Pipeline
# Yeo-Johnson is preferred over Box-Cox as it handles negative values (though data is mostly positive)
# and stabilizes variance for LDA.
USE_POWER_TRANSFORM = True
POWER_TRANSFORM_METHOD = "yeo-johnson"

# Standard Scaling is applied after Power Transform
USE_STANDARD_SCALING = True

# =============================================================================
# DEBUGGING AND VALIDATION
# =============================================================================
# If True, the pipeline can optionally limit the dataset size for rapid prototyping
DEBUG = False
DEBUG_SAMPLES = 100  # Only used if DEBUG is True
