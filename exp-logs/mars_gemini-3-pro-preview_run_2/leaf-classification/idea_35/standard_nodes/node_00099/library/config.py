import os
import numpy as np

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Reproducibility
RANDOM_SEED = 42

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working and Cache Directories
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_35")

# Submission
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Create necessary directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA TYPES & PRECISION
# =============================================================================
# Strictly use float64 to minimize numerical noise at the metric floor
FLOAT_PRECISION = np.float64

# =============================================================================
# PREPROCESSING HYPERPARAMETERS
# =============================================================================

# Pipeline A & C: Parametric Gaussian Anchors
# Uses PowerTransformer
POWER_TRANSFORM_METHOD = "yeo-johnson"

# Pipeline B: Regularized Non-Parametric Experts
# Uses QuantileTransformer
# Strictly constrained to 30 quantiles to prevent overfitting (Lesson 76)
QUANTILE_TRANSFORM_N_QUANTILES = 30
QUANTILE_TRANSFORM_OUTPUT_DIST = "normal"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Linear Discriminant Analysis (LDA) Configuration
# Library of shrinkage estimators for the Global View Experts
# 'oas': Oracle Approximating Shrinkage
# float: Fixed shrinkage parameter for lsqr/eigen solver
LDA_SHRINKAGE_OPTIONS = ["oas", 0.001, 0.01, 0.1]

# Macro View Expert Configuration
# Uses Ledoit-Wolf shrinkage (standard 'auto' in sklearn LDA)
MACRO_LDA_SOLVER = "lsqr"
MACRO_LDA_SHRINKAGE = "auto"

# =============================================================================
# EVALUATION & POST-PROCESSING
# =============================================================================
# Clipping epsilon to avoid log(0) extremes in Log Loss metric
PROB_CLIP_EPS = 1e-15
