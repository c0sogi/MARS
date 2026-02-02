import os

# =============================================================================
# Global Configuration
# =============================================================================

# Random Seed for reproducibility
SEED = 42

# =============================================================================
# Directory Paths
# =============================================================================

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Cache directory specific to this solution iteration
CACHE_DIR = os.path.join(WORKING_DIR, "idea_19")

# Ensure necessary writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# File Paths
# =============================================================================

# Metadata CSVs containing features and stratified splits
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Sample submission file for format reference
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# Dataset Schema
# =============================================================================

ID_COL = "id"
TARGET_COL = "species"
N_CLASSES = 99

# Feature Columns Definition
# We explicitly generate the 192 feature names (margin, shape, texture)
# and sort them alphanumerically. This ensures a deterministic column order
# is enforced during data loading, preventing implicit permutation errors.
_feature_types = ["margin", "shape", "texture"]
_feature_indices = range(1, 65)

# Generate all combinations (e.g., margin1, shape1, texture1, ...)
_raw_features = [f"{ft}{i}" for ft in _feature_types for i in _feature_indices]

# Sort alphanumerically (e.g., margin1, margin10, margin11, ..., margin2)
FEATURE_COLS = sorted(_raw_features)
