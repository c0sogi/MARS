import os
import numpy as np

# ==========================================
# Directory Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Specific working directory for this solution iteration (Idea 61)
WORKING_DIR = "./working/idea_61"
CACHE_DIR = WORKING_DIR  # Alias for clarity in data loading modules

# Submission directory
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Global Hyperparameters
# ==========================================
SEED = 42
VAL_SIZE = 0.2

# ==========================================
# Data Types & Precision
# ==========================================
# Enforce float64 for high-precision linear algebra (OAS-LDA)
FLOAT_PRECISION = np.float64

# ==========================================
# Feature Definitions
# ==========================================
# The 7 Orthogonal-Geometric Basis features derived from image processing
GEOMETRIC_FEATURES = [
    "Area",  # Absolute Scale
    "Mean_Thickness",  # Internal Topology (Euclidean Distance Transform)
    "Eccentricity",  # Elongation (Ellipse fit)
    "Solidity",  # Roughness
    "Extent",  # Rectangularity
    "Aspect_Ratio",  # Orientation
    "Roundness",  # Compactness (Non-linear ratio)
]

# Pre-extracted tabular feature groups provided in the dataset
TABULAR_FEATURE_GROUPS = ["margin", "shape", "texture"]

# Column names for identification
ID_COL = "id"
TARGET_COL = "species"
FILE_PATH_COL = "file_path"
