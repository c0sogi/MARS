import os
import numpy as np

# =============================================================================
# DIRECTORY & FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for Idea 55 (SDPGE)
WORKING_DIR = "./working/idea_55"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42
FLOAT_PRECISION = np.float64  # Strictly use float64 to minimize numerical noise
N_CLASSES = 99
VAL_SIZE = 0.2

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
# The dataset contains 192 features in total (3 groups * 64 features).
# We define slices to easily extract these views from the global feature matrix.
# Assumes X is ordered as [Margin, Shape, Texture] after dropping metadata columns.
SLICE_MARGIN = slice(0, 64)
SLICE_SHAPE = slice(64, 128)
SLICE_TEXTURE = slice(128, 192)

FEATURE_SLICES = {
    "margin": SLICE_MARGIN,
    "shape": SLICE_SHAPE,
    "texture": SLICE_TEXTURE,
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Linear Discriminant Analysis (LDA) Configuration
# Solver 'lsqr' or 'eigen' is required to use shrinkage.
LDA_SOLVER = "lsqr"

# Shrinkage Grid for the Expert Library
# Topology A (Global) uses fixed small shrinkage (e.g., 0.001, 0.01).
# Topology C (Stratified) explores a wider library.
SHRINKAGE_GRID = [0.001, 0.01, 0.1, 0.2, 0.5, "auto"]

# Preprocessing Hyperparameters
N_QUANTILES = 50  # For QuantileTransformer (Robust Topology)
POLY_DEGREE = 2  # For PolynomialFeatures (Physical & Interaction Topologies)
YEO_JOHNSON_STANDARDIZE = True

# =============================================================================
# IMAGE PROCESSING CONFIGURATION
# =============================================================================
# Threshold for determining if the image background is white (inverted).
# If the mean of corner pixels > threshold, the image is inverted.
POLARITY_THRESHOLD = 0.5

# =============================================================================
# SCORING & SUBMISSION
# =============================================================================
# Clipping to avoid log loss extremes: max(min(p, 1-10^-15), 10^-15)
CLIP_MIN = 1e-15
CLIP_MAX = 1.0 - 1e-15
