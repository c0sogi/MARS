import os

# -----------------------------------------------------------------------------
# Global Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Idea-specific cache directory
IDEA_NAME = "idea_29"
CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

# Ensure the cache directory exists immediately
os.makedirs(CACHE_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Submission Paths
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
SEED = 42
TARGET_COL = "species"
ID_COL = "id"

# -----------------------------------------------------------------------------
# Feature Columns
# -----------------------------------------------------------------------------
# We strictly define the order of features to ensure consistency between
# training and inference. The dataset provides 3 sets of 64 features.

_MARGIN_COLS = [f"margin_{i}" for i in range(1, 65)]
_SHAPE_COLS = [f"shape_{i}" for i in range(1, 65)]
_TEXTURE_COLS = [f"texture_{i}" for i in range(1, 65)]

# Combined Feature List (Length: 192)
FEATURE_COLS = _MARGIN_COLS + _SHAPE_COLS + _TEXTURE_COLS

# Sanity check to prevent silent errors if logic changes
assert len(FEATURE_COLS) == 192, f"Expected 192 features, but found {len(FEATURE_COLS)}"
