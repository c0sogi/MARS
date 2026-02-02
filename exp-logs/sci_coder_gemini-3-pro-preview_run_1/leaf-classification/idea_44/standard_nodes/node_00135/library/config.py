import os
import random
import numpy as np

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR_NAME = "images"

# Working directory for intermediate artifacts
# Specific subdirectory for this solution idea to prevent cache collisions
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_44")

# Submission directory
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42


def set_seed(seed=SEED):
    """
    Sets the random seed for python's random module and numpy to ensure
    reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "file_path"

# The dataset has 99 classes
NUM_CLASSES = 99

# =============================================================================
# PRECISION & NUMERICS
# =============================================================================
# Strict requirement for Double Precision to handle ill-conditioned covariance
NUMERIC_DTYPE = np.float64

# Metric clipping to avoid log(0)
PROB_CLIP_MIN = 1e-15
PROB_CLIP_MAX = 1.0 - 1e-15

# Compute resources
N_JOBS = 12

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# 1. Tabular Features (Pre-extracted)
TABULAR_PREFIXES = ["margin", "shape", "texture"]

# 2. Geometric Features (Dual-Envelope Morphological Fusion)
# Defined explicitly to ensure deterministic column ordering and memory layout
GEOMETRIC_FEATURES = [
    # --- Absolute Scale (Size) ---
    "Area",
    "Perimeter",
    "Convex_Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Equivalent_Diameter",
    # --- Scanner Frame (Axis-Aligned Bounding Box) ---
    "AABB_Width",
    "AABB_Height",
    "AABB_Aspect_Ratio",
    "AABB_Extent",
    # --- Object Frame (Minimum Area Rectangle) ---
    "MinBox_Width",
    "MinBox_Height",
    "MinBox_Aspect_Ratio",
    "MinBox_Extent",
    # --- Internal Morphology (Distance Transform) ---
    "Inscribed_Circle_Radius",
    # --- Explicit Invariants ---
    "Solidity",
    "Convexity",
    "Roundness",
    "Compactness",
]

# =============================================================================
# PREPROCESSING PIPELINE HYPERPARAMETERS
# =============================================================================
# Apply Yeo-Johnson to stabilize variance of geometric features
USE_YEO_JOHNSON = True
# Standardize features after transformation
USE_STANDARD_SCALER = True
# Yeo-Johnson implementation parameter
YEO_JOHNSON_STANDARDIZE = False

# =============================================================================
# MODEL HYPERPARAMETERS (OAS LINEAR DISCRIMINANT)
# =============================================================================
# OAS (Oracle Approximating Shrinkage) configuration
# We manually center data, so we tell OAS it is centered to save computation/precision
OAS_ASSUME_CENTERED = True
