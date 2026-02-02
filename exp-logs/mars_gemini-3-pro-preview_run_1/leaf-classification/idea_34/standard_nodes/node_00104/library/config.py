import os
import numpy as np

# -----------------------------------------------------------------------------
# Global Configuration & Constants
# -----------------------------------------------------------------------------

# Random Seed for reproducibility across all modules
SEED = 42

# Strict Data Type Enforcement
# Using float64 is critical for the Cholesky decomposition and OAS covariance
# estimation to avoid numerical instability and spectral truncation.
DTYPE = np.float64

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Cache directory specific to this experimental idea (Idea 34)
# This is used to store intermediate processed data (parquet/npy)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_34")

# Directory for final submission output
SUBMISSION_DIR = "./submission"

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------

# Metadata files containing stratified splits and file paths
TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

# Final submission file path
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data & Model Constants
# -----------------------------------------------------------------------------

# Feature prefixes used to identify feature columns in the dataset
FEATURE_PREFIXES = ["margin", "shape", "texture"]

# Epsilon for probability clipping (as defined in the metric description)
EPSILON = 1e-15

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

# Ensure critical directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
