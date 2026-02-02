import os
import numpy as np

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Image Directory
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Submission Paths
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths for Deterministic Processing
CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features_mc.parquet")
CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features_mc.parquet")
CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features_mc.parquet")

# =============================================================================
# 2. GENERAL CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
N_JOBS = 12  # Available vCPUs
USE_FLOAT64 = True  # Enforce double precision for numerical stability
DEBUG_MODE = False  # Set to True to run on a small subset for testing
DEBUG_SAMPLE_SIZE = 50

# =============================================================================
# 3. DATA PROCESSING & AUGMENTATION (MONTE-CARLO)
# =============================================================================
# Number of Monte-Carlo simulations per image to estimate robust shape descriptors
N_AUGMENTATIONS = 10

# Affine transformation parameters for generating perturbations
# Used to simulate measurement noise and capture geometric stability
AUGMENTATION_PARAMS = {
    "rotation_range": (-180, 180),  # Degrees
    "scale_range": (0.9, 1.1),  # Factor
    "shear_range": (-15, 15),  # Degrees
    "translation_range": (-0.05, 0.05),  # Fraction of dimension
}

# Feature groups provided in the dataset
PROVIDED_FEATURE_GROUPS = ["margin", "shape", "texture"]

# =============================================================================
# 4. MODEL HYPERPARAMETERS (EXPERT LIBRARY)
# =============================================================================
# Group A: Linear Discriminant Analysis (The Linear Anchors)
# Includes Ledoit-Wolf ('auto'), Fixed Shrinkage, and placeholder for OAS logic
LDA_CONFIGS = [
    {"solver": "lsqr", "shrinkage": "auto"},  # Ledoit-Wolf
    {"solver": "lsqr", "shrinkage": 0.001},
    {"solver": "lsqr", "shrinkage": 0.01},
    {"solver": "lsqr", "shrinkage": 0.1},
    {"solver": "lsqr", "shrinkage": 0.5},
    # OAS is handled specially in the model factory, config marker below
    {"solver": "lsqr", "shrinkage": "OAS"},
]

# Group B: Quadratic Discriminant Analysis (The Quadratic Innovators)
# Regularized QDA to handle heteroscedasticity without singularity
QDA_CONFIGS = [
    {"reg_param": 0.0},
    {"reg_param": 0.1},
    {"reg_param": 0.5},
    {"reg_param": 0.9},
]

# Group C: Gaussian Naive Bayes (The Diagonal Stabilizers)
# High-bias fallback with varying variance smoothing
GNB_CONFIGS = [
    {"var_smoothing": 1e-9},
    {"var_smoothing": 1e-5},
    {"var_smoothing": 1e-3},
    {"var_smoothing": 1e-1},
]

# =============================================================================
# 5. ENSEMBLE SELECTION STRATEGY
# =============================================================================
# Greedy Forward Selection parameters
SELECTION_MAX_ITER = 100  # Maximum number of experts to add
SELECTION_TOLERANCE = 1e-6  # Minimum improvement required to continue adding

# Probability Clipping to avoid LogLoss extremes
# Predicted probabilities are clipped to [CLIP_MIN, 1 - CLIP_MIN]
CLIP_MIN = 1e-15
CLIP_MAX = 1.0 - 1e-15
