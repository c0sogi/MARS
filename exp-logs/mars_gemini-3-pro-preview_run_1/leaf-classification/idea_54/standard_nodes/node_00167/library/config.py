import os
import cv2
import numpy as np

# ==========================================
# Directory and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_54"
SUBMISSION_DIR = "./submission"

# Ensure the working and submission directories exist
# Note: While config usually just defines paths, creating them here ensures
# they exist for any module importing this config.
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Image Directory
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Metadata Files (Pre-generated)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache Directory for deterministic processing
CACHE_DIR = WORKING_DIR

# Final Submission File
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Constants
# ==========================================
RANDOM_SEED = 42
NUM_CLASSES = 99

# ==========================================
# Image Processing Configuration
# ==========================================
# Polarity Correction: The dataset has black leaves on white background.
# We use THRESH_BINARY_INV to invert this so the leaf becomes the foreground (white).
BINARY_THRESHOLD_TYPE = cv2.THRESH_BINARY_INV
BINARY_THRESHOLD_VALUE = 127

# Contour Extraction: Use CHAIN_APPROX_NONE to keep all boundary points (lossless)
# for maximum fidelity in shape descriptors.
CONTOUR_APPROX_METHOD = cv2.CHAIN_APPROX_NONE

# ==========================================
# Feature Engineering Configuration
# ==========================================
# The Parsimonious Geometric Basis (6 scalar descriptors)
# 1. Area (Absolute Scale)
# 2. Eccentricity (Elongation via Ellipse Fit)
# 3. Solidity (Roughness)
# 4. Extent (Rectangularity)
# 5. Aspect Ratio (Orientation)
# 6. Mean Thickness (Internal Topology via Distance Transform)
GEOMETRIC_FEATURES = [
    "area",
    "eccentricity",
    "solidity",
    "extent",
    "aspect_ratio",
    "mean_thickness",
]

# Tabular Feature Prefixes (192 total features)
TABULAR_PREFIXES = ["margin", "shape", "texture"]
NUM_TABULAR_FEATURES_PER_SET = 64

# ==========================================
# Model & Pipeline Configuration
# ==========================================
# High-Precision Inference: Use float64 to avoid spectral truncation issues
# in the OAS covariance estimation and linear scoring.
FLOAT_PRECISION = np.float64

# Preprocessing Pipeline
APPLY_POWER_TRANSFORM = True  # Yeo-Johnson
STANDARDIZE_FEATURES = True  # StandardScaler

# OAS Estimator Settings
OAS_ASSUME_CENTERED = True
