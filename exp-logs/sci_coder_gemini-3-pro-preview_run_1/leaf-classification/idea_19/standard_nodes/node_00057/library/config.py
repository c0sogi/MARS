import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_19"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Constants
# ==========================================
SEED = 42
ID_COL = "id"
TARGET_COL = "species"
NUM_CLASSES = 99

# Metric / Probability Clipping
# Probabilities are clipped to [EPSILON, 1 - EPSILON] to avoid log(0)
EPSILON = 1e-15

# ==========================================
# Feature Columns
# ==========================================
# The dataset contains 3 sets of features, each with 64 attributes.
# We generate the names programmatically and sort them to ensure
# strict alphanumeric order (e.g., margin_1, margin_10, margin_11...)
# to prevent implicit column permutation errors during loading.

_feature_types = ["margin", "shape", "texture"]
_feature_indices = range(1, 65)

FEATURE_COLS = sorted([f"{ft}{i}" for ft in _feature_types for i in _feature_indices])

# Validation to ensure correct count
assert len(FEATURE_COLS) == 192, f"Expected 192 features, got {len(FEATURE_COLS)}"
