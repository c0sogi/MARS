import os
import numpy as np

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
WORKING_DIR = "./working"

# Cache directory for idea_67 (Global High-Precision OAS-LDA)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_67")
SUBMISSION_DIR = "./submission"

# Ensure writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
N_JOBS = 12
# Strict double precision is required for the OAS-LDA backbone to maintain
# numerical stability with small sample sizes and high dimensionality.
FLOAT_PRECISION = np.float64

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# 1. Tabular Feature Groups
# The 192 pre-extracted features are split into 3 semantic groups based on prefixes.
# This factorization increases the N/P ratio for covariance estimation.
TABULAR_FEATURE_GROUPS = {"margin": "margin", "shape": "shape", "texture": "texture"}

# 2. Geometric Feature Extraction
# The "Golden 5" robust scalar descriptors extracted from binary images.
GEOMETRIC_FEATURES = ["Area", "Eccentricity", "Solidity", "Extent", "AspectRatio"]

# =============================================================================
# PIPELINE HYPERPARAMETERS
# =============================================================================
# Preprocessing settings for the sanitized pipeline
PREPROCESSING_PARAMS = {
    "variance_threshold": 0.0,  # Remove strictly constant features
    "yeo_johnson": True,  # Apply Yeo-Johnson power transformation
    "standardize": True,  # Apply Standard Scaling (Z-score)
}

# Model settings for the Factorized OAS Ensemble
MODEL_PARAMS = {
    # We manually compute residuals (X - mean), so OAS should assume centered data.
    "oas_assume_centered": True,
    # When summing logits from N independent experts, we sum the log-priors N times.
    # We must subtract (N-1) * log(prior) to recover the correct joint posterior.
    "logit_aggregation_correction": True,
}
