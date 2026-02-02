import os
import numpy as np

# ==========================================
# Global Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Specific working directory for this solution strategy
# Using 'idea_59' as the unique identifier for this run's cache and artifacts
IDEA_ID = "idea_60"
CACHE_DIR = os.path.join(WORKING_DIR, IDEA_ID)
os.makedirs(CACHE_DIR, exist_ok=True)

# Data File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Image Directory
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Submission Paths
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
OUTPUT_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Reproducibility and Precision
# ==========================================
SEED = 42
# Enforce float64 for high-precision linear algebra (OAS/SVD)
FLOAT_PRECISION = np.float64

# ==========================================
# Feature Configuration
# ==========================================
# The suite of integral/robust geometric features derived from image processing
# Includes External Geometry, Internal Topology, and Invariant Shapes
GEOMETRIC_FEATURES = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Mean_Thickness",
    "Solidity",
    "Extent",
    "Eccentricity",
    "Roundness",
]

# Tabular feature prefixes provided in the original dataset
TABULAR_FEATURE_PREFIXES = ["margin", "shape", "texture"]
NUM_TABULAR_FEATURES_PER_SET = 64

# ==========================================
# Pipeline Hyperparameters
# ==========================================
# Variance Thresholding: 0.0 means remove features with zero variance (constants)
# This is the "Sanitization Barrier"
VARIANCE_THRESHOLD = 0.0

# Debugging / Development controls
# Set to None to use full dataset, or an integer (e.g., 100) for quick debugging
DEBUG_SAMPLE_SIZE = None

# Although the Linear Discriminant is analytical, we define this for interface consistency
# or if an iterative solver fallback is ever required.
MAX_EPOCHS = 1
