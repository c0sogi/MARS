import os
import numpy as np
import cv2

# ==========================================
# Directories and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea's cache to avoid collisions
WORKING_DIR = "./working/idea_optimized"
SUBMISSION_DIR = "./submission"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Constants & Hyperparameters
# ==========================================
SEED = 42

# ------------------------------------------
# Image Processing Parameters
# ------------------------------------------
# Fixed threshold to avoid instability of Otsu's method on lossy images
BINARY_THRESHOLD_VALUE = 127

# Invert binary threshold to ensure leaf is foreground (white) and background is black
BINARY_THRESHOLD_TYPE = cv2.THRESH_BINARY_INV

# Use lossless contour approximation to maximize boundary fidelity
CONTOUR_MODE = cv2.CHAIN_APPROX_NONE

# ------------------------------------------
# Feature Extraction Configuration
# ------------------------------------------
# The "Golden 5" robust geometric descriptors strictly required by the strategy
GEOMETRIC_FEATURES = [
    "Area",  # Absolute Scale
    "Eccentricity",  # Elongation (Ellipse fit)
    "Solidity",  # Roughness (Area / ConvexArea)
    "Extent",  # Rectangularity (Area / BoundingRectArea)
    "Aspect_Ratio",  # Orientation (BoundingWidth / BoundingHeight)
]

# ------------------------------------------
# Pipeline Sanitization & Numerical Precision
# ------------------------------------------
# Variance threshold to act as a circuit breaker for constant features (e.g., empty bins)
# Applied before scaling to prevent noise explosion.
VARIANCE_THRESHOLD = 0.0

# Mandatory precision for OAS covariance estimation and Linear Algebra
FLOAT_PRECISION = np.float64

# Clipping epsilon for log-loss metric calculation
PROB_CLIP_EPS = 1e-15

# ------------------------------------------
# Data Management
# ------------------------------------------
# Columns to exclude when identifying tabular feature columns
EXCLUDE_COLUMNS = ["id", "species", "file_path"]

# Debugging flags (can be used by other modules to limit dataset size)
DEBUG = False
DEBUG_SAMPLE_SIZE = 50  # Number of samples to use if DEBUG is True
