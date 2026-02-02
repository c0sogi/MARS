import os

# ==========================================
# Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# File Paths
# ==========================================
# Using the metadata files as the source of truth for the split data
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output submission path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
CLIP_EPSILON = 1e-15

# ==========================================
# Column Definitions
# ==========================================
ID_COL = "id"
TARGET_COL = "species"

# Generate the 192 feature names
# Structure: [feature_type]_[1-64]
# Types: margin, shape, texture
_prefixes = ["margin", "shape", "texture"]
_raw_feature_list = []

for prefix in _prefixes:
    for i in range(1, 65):
        _raw_feature_list.append(f"{prefix}_{i}")

# Sort alphanumerically to ensure deterministic ordering across all modules.
# Note: Standard string sort places 'margin_10' before 'margin_2'.
FEATURE_COLS = sorted(_raw_feature_list)
