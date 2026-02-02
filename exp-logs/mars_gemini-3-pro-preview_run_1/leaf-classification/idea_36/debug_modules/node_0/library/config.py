import os
import numpy as np

# ==========================================
# Global Configuration & Hyperparameters
# ==========================================

# Random Seed for reproducibility
SEED = 42

# Floating Point Precision
# The solution requires float64 for Cholesky decomposition stability and exact inference
FLOAT_PRECISION = np.float64

# ==========================================
# Directory & File Paths
# ==========================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for caching intermediate files (Parquet/Numpy)
WORKING_DIR = "./working/idea_36"
SUBMISSION_DIR = "./submission"

# Ensure mutable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Dataset Paths (using generated metadata)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Paths
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Directory
CACHE_DIR = WORKING_DIR

# ==========================================
# Feature Configuration
# ==========================================

# The dataset contains 3 types of features, each with 64 attributes
FEATURE_TYPES = ["margin", "shape", "texture"]
N_ATTRIBUTES_PER_TYPE = 64


def get_feature_columns(sort_alphanumeric=True):
    """
    Generates the list of 192 feature column names.

    Args:
        sort_alphanumeric (bool): If True, sorts features alphanumerically
                                  (e.g., margin_1, margin_10, margin_11...)
                                  to match high-performance memory layouts.

    Returns:
        list: List of feature column names.
    """
    features = []
    for f_type in FEATURE_TYPES:
        for i in range(1, N_ATTRIBUTES_PER_TYPE + 1):
            features.append(f"{f_type}_{i}")

    if sort_alphanumeric:
        # Standard string sort results in alphanumeric ordering
        # e.g., margin_1, margin_10, margin_11, ..., margin_2, margin_20
        features.sort()

    return features


# ==========================================
# Optimization Hyperparameters
# ==========================================

# Parameters for the Metric-Optimized Shrinkage Grid Search
# We search for the optimal alpha that minimizes Multi-class Log Loss
GRID_SIZE = 100  # Number of alpha candidates to evaluate
CV_FOLDS = 5  # Number of stratified folds for internal validation
ALPHA_BOUNDS = (
    0.001,
    0.999,
)  # Search range for shrinkage intensity (0=Empirical, 1=Identity)
