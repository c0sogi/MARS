import os
import numpy as np

# =============================================================================
# Global Configuration & Reproducibility
# =============================================================================

# Random seed for reproducibility across all operations
SEED = 42

# Floating-point precision for high-precision linear algebra
# Using float64 is critical for the stability of the covariance matrix inversion
FLOAT_PRECISION = np.float64

# =============================================================================
# Directory Paths
# =============================================================================

# Input directories (Read-Only)
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Metadata file paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching intermediate data (Write Access)
# Specific to Idea 39 to avoid conflicts with other runs
WORKING_DIR = "./working/idea_39"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory (Write Access)
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Model Hyperparameters
# =============================================================================

# Oracle Approximating Shrinkage (OAS) Estimator Parameters
# 'assume_centered=True' is used because we manually compute residuals (X - mu)
# to ensure geometric consistency in the latent space.
OAS_PARAMS = {
    "assume_centered": True,
}

# Epsilon for probability clipping to avoid log(0) in Log Loss metric
# Probabilities are clipped to [CLIP_EPSILON, 1 - CLIP_EPSILON]
CLIP_EPSILON = 1e-15

# =============================================================================
# Feature Engineering Configuration
# =============================================================================

# List of additional geometric features to extract from binary images
# These provide invariant shape descriptors orthogonal to the provided histograms
GEOMETRIC_FEATURES = [
    "aspect_ratio",
    "solidity",
    "extent",
    "eccentricity",
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
    "hu_moment_7",
]

# Preprocessing Pipeline Configuration
# 1. Yeo-Johnson Transform (stabilize variance/normality)
# 2. Standard Scaler (zero mean, unit variance)
# Note: standardize=False in PowerTransformer is crucial as we chain it with StandardScaler
PIPELINE_CONFIG = {
    "yeo_johnson": True,
    "yeo_johnson_standardize": False,
    "standard_scaler": True,
}

# Enforce alphanumeric sorting of columns to ensure deterministic memory layout
SORT_COLUMNS = True
