import os

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_66"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Specific file paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Global Configuration
# =============================================================================
RANDOM_SEED = 42
N_CLASSES = 99

# =============================================================================
# Data Column Definitions
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "image_path"

# Feature Groups (1-based indexing as per dataset description)
MARGIN_COLS = [f"margin{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture{i}" for i in range(1, 65)]

# Combined Feature List
ALL_PROVIDED_FEATURES = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS
