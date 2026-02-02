import os
import numpy as np

# ==============================================================================
# DIRECTORY CONFIGURATION
# ==============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
WORKING_DIR = "./working"

# Specific cache directory for this idea (Idea 33)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_33")
os.makedirs(CACHE_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==============================================================================
# GLOBAL SETTINGS & REPRODUCIBILITY
# ==============================================================================
RANDOM_STATE = 42
FLOAT_PRECISION = np.float64  # Use double precision to prevent numerical noise

# ==============================================================================
# HYPERPARAMETERS FOR DUAL-GAUSSIANIZED STRATEGY
# ==============================================================================

# Group B: Regularized Non-Parametric Experts
# Constrain n_quantiles to prevent overfitting to the empirical distribution
N_QUANTILES = 50
QUANTILE_OUTPUT_DIST = "normal"

# Group A & C: Parametric Gaussian Anchors
# Standard Yeo-Johnson power transformation
POWER_METHOD = "yeo-johnson"

# LDA Configuration
# Library of shrinkage estimators for the covariance matrix
LDA_SOLVER = "lsqr"  # 'lsqr' or 'eigen' supports shrinkage
LDA_SHRINKAGE_CANDIDATES = [0.0001, 0.001, 0.01, 0.1]  # Fixed shrinkage values
LDA_AUTOMATIC_SHRINKAGE = ["auto"]  # sklearn's Ledoit-Wolf implementation via 'auto'
# Note: OAS and explicit Ledoit-Wolf will be handled as separate expert classes in the model definition

# ==============================================================================
# FEATURE CONFIGURATION
# ==============================================================================
# Prefixes for the provided feature sets
FEATURE_PREFIXES = ["margin", "shape", "texture"]
NUM_FEATURES_PER_GROUP = 64

# Macro Feature Configuration (Group C)
# These are extracted from binary images
MACRO_FEATURE_NAMES = [
    "hu_moment_0",
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
    "aspect_ratio",
    "solidity",
    "extent",
    "eccentricity",
]
