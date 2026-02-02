import os
import numpy as np

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------

# Random Seed for reproducibility
SEED = 42

# Data Type
# Using float64 to guarantee high-precision arithmetic throughout the pipeline
DTYPE = np.float64

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_23"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------

TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Column Definitions
# -----------------------------------------------------------------------------

ID_COL = "id"
TARGET_COL = "species"

# Hardcoded list of 192 feature columns to enforce deterministic feature ordering.
# Structure:
#   - margin_1 to margin_64
#   - shape_1 to shape_64
#   - texture_1 to texture_64
FEATURES = [
    f"{feat}_{i}" for feat in ["margin", "shape", "texture"] for i in range(1, 65)
]
