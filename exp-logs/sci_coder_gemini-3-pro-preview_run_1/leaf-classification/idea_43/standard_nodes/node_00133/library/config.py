import os
import numpy as np

# =============================================================================
# GLOBAL PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_43")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# REPRODUCIBILITY & PRECISION
# =============================================================================
SEED = 42
# Strict requirement for double precision to avoid metric floor issues
PRECISION_TYPE = np.float64
PRECISION_STR = "float64"

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
# Tabular features provided in the dataset
TABULAR_PREFIXES = ["margin", "shape", "texture"]
NUM_TABULAR_FEATURES_PER_GROUP = 64

# Visual Geometric Features (Dual-Envelope Strategy)
# These will be extracted from the binary images
VISUAL_FEATURES = [
    # Absolute Scale (Size)
    "area",
    "perimeter",
    "convex_perimeter",
    "major_axis_length",
    "minor_axis_length",
    "equivalent_diameter",
    # Axis-Aligned Envelope (Environment)
    "bounding_width",
    "bounding_height",
    "aspect_ratio",
    "extent",
    # Intrinsic Envelope (Object)
    "min_box_width",
    "min_box_height",
    "min_box_aspect_ratio",
    "solidity",
    # Internal Morphology
    "inscribed_circle_radius",
    # Topology
    "convexity",
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# OAS Linear Discriminant settings
OAS_ASSUME_CENTERED = True

# Preprocessing settings
USE_YEO_JOHNSON = True
STANDARDIZE = True
