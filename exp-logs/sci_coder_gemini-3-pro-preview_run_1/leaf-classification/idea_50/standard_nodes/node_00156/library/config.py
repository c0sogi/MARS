import os

# =============================================================================
# GLOBAL CONFIGURATION & CONSTANTS
# =============================================================================

# Random Seed for Reproducibility
SEED = 42

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Cache directory for the specific idea implementation
# This ensures deterministic data processing artifacts are stored separately
CACHE_DIR = os.path.join(WORKING_DIR, "idea_51")

# Submission directory
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Schema
# -----------------------------------------------------------------------------
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "file_path"

# -----------------------------------------------------------------------------
# Feature Engineering Configuration
# -----------------------------------------------------------------------------

# 1. New Geometric Features (Scalars)
# These are the robust scalar features to be extracted from the binary images
# to replace the high-dimensional shape histograms.
# We select a parsimonious set of robust descriptors (Cite 00120, 00140)
# and include Equivalent_Diameter for absolute size signal (Cite 00118).
GEOMETRIC_FEATURES = [
    "Aspect_Ratio",
    "Extent",
    "Solidity",
    "Eccentricity",
    "Roundness",
    "Equivalent_Diameter",
]

# 2. Original Features to Drop
# We explicitly identify the 64 'shape' columns to be removed from the dataset.
# These are likely noisy, high-dimensional approximations of geometry.
# Corrected naming convention to match dataset (Cite 00154).
SHAPE_COLS_TO_DROP = [f"shape{i}" for i in range(1, 65)]

# 3. Original Features to Keep
# We retain the margin and texture features which provide complementary information.
MARGIN_COLS = [f"margin_{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture_{i}" for i in range(1, 65)]

# Helper list of all numerical columns present in the raw CSVs (excluding ID)
# Used for initial loading before feature engineering.
RAW_FEATURE_COLS = MARGIN_COLS + SHAPE_COLS_TO_DROP + TEXTURE_COLS

# -----------------------------------------------------------------------------
# Image Processing
# -----------------------------------------------------------------------------
# Specific flags for OpenCV to ensure consistent contour extraction
# Note: These are constants to be used in the feature extraction module.
CV_THRESH_BINARY_INV = 1  # cv2.THRESH_BINARY_INV
CV_CHAIN_APPROX_NONE = (
    1  # cv2.CHAIN_APPROX_NONE (value depends on cv2 version, usually 1 or 2)
)
