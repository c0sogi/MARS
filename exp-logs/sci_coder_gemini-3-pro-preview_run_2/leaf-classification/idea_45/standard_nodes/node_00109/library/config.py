import os
import numpy as np

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
WORKING_DIR = "./working/idea_45"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Metadata
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS & REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42
FLOAT_PRECISION = np.float64  # Strictly use float64 to minimize numerical noise
N_JOBS = 12  # Utilize available vCPUs

# Metric Clipping (Multi-class Log Loss)
# Predicted probabilities are replaced with max(min(p, 1-1e-15), 1e-15)
PROB_CLIP_MIN = 1e-15
PROB_CLIP_MAX = 1.0 - 1e-15

# =============================================================================
# CACHING CONFIGURATION
# =============================================================================
# Caching intermediate processed features to speed up iterative development
# We use .npy for fast numerical I/O of float64 arrays

# Global Features (Original 192 features)
CACHE_TRAIN_GLOBAL = os.path.join(WORKING_DIR, "X_train_global.npy")
CACHE_VAL_GLOBAL = os.path.join(WORKING_DIR, "X_val_global.npy")
CACHE_TEST_GLOBAL = os.path.join(WORKING_DIR, "X_test_global.npy")

# Physical Features (Extracted Morphometrics)
CACHE_TRAIN_PHYSICAL = os.path.join(WORKING_DIR, "X_train_physical.npy")
CACHE_VAL_PHYSICAL = os.path.join(WORKING_DIR, "X_val_physical.npy")
CACHE_TEST_PHYSICAL = os.path.join(WORKING_DIR, "X_test_physical.npy")

# Targets and Classes
CACHE_Y_TRAIN = os.path.join(WORKING_DIR, "y_train.npy")
CACHE_Y_VAL = os.path.join(WORKING_DIR, "y_val.npy")
CACHE_CLASSES = os.path.join(WORKING_DIR, "classes.npy")
CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

# Expert Predictions Library (for Ensemble Selection)
CACHE_VAL_PREDS_LIB = os.path.join(WORKING_DIR, "val_preds_library.npy")
CACHE_TEST_PREDS_LIB = os.path.join(WORKING_DIR, "test_preds_library.npy")
CACHE_EXPERT_NAMES = os.path.join(WORKING_DIR, "expert_names.npy")

# =============================================================================
# EXPERT LIBRARY HYPERPARAMETERS
# =============================================================================

# 1. Group A: Marginal Statistical Anchors
# Baseline LDA models using provided features with robust shrinkage
ANCHOR_SHRINKAGE_VALUES = [0.001, 0.01]  # Fixed shrinkage values
ANCHOR_SOLVERS = ["lsqr", "eigen"]  # Solvers that support shrinkage

# 2. Group B: Rotational Statistical Experts
# PCA used strictly for rotation (Variance Preserving), not dimensionality reduction
ROTATION_WHITEN = False  # Critical: Do not whiten to avoid noise amplification
ROTATION_COMPONENTS = None  # Keep full rank

# 3. Group C: Polynomial Physical Experts
# Feature extraction and expansion settings
PHYSICAL_POLY_DEGREE = 2
PHYSICAL_POLY_INTERACTION_ONLY = False
PHYSICAL_POLY_INCLUDE_BIAS = False

# Image Preprocessing for Physical Extraction
# Threshold to detect if background is white (1) or black (0) to invert polarity
# Leaf images are binary. If corners are white, leaf is black -> Invert.
POLARITY_CHECK_THRESHOLD = 0.5

# =============================================================================
# ENSEMBLE SELECTION HYPERPARAMETERS
# =============================================================================
# Greedy Forward Selection settings
SELECTION_MAX_ITER = 100  # Maximum size of the ensemble
SELECTION_TOLERANCE = 0.0  # Stop if improvement is <= tolerance
SELECTION_WITH_REPLACEMENT = (
    True  # Allow selecting the same expert multiple times (integer weighting)
)
