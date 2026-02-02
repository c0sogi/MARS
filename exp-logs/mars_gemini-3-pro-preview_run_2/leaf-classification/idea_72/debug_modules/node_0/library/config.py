import os
import numpy as np

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
METADATA_DIR = "./metadata"

# Working directory for caching intermediate files (Parquet/NPY)
# Specific to 'idea_72' as per instructions
WORKING_DIR = "./working/idea_72"

# Directory for final submission
SUBMISSION_DIR = "./submission"

# Metadata File Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output File Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
RANDOM_SEED = 42
N_CLASSES = 99

# Precision Policy: Double precision is critical for maintaining numerical stability
# in high-dimensional covariance estimations (LDA/QDA with shrinkage).
FLOAT_PRECISION = np.float64

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
# Programmatically generate column names based on dataset description
# 64 attributes per feature group
MARGIN_COLS = [f"margin_{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape_{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture_{i}" for i in range(1, 65)]

# Combined feature list
ALL_FEATURE_COLS = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# Special Columns
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "image_path"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def setup_directories():
    """
    Creates necessary directories for the project if they do not exist.
    This ensures caching and submission saving do not fail.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
