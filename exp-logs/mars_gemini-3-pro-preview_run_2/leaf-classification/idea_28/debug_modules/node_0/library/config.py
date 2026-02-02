import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata file paths (Pre-split data)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission output path
SUBMISSION_PATH = "./submission/submission.csv"

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
RANDOM_SEED = 42

# Column names
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "image_path"

# =============================================================================
# FEATURE COLUMNS
# =============================================================================
# The dataset provides 64 features for each of the three categories.
# We generate these names programmatically to match the CSV headers.

MARGIN_COLS = [f"margin_{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape_{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture_{i}" for i in range(1, 65)]

# Combined list of all 192 provided 'Micro' features
ALL_PROVIDED_FEATURES = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# Shrinkage grid for Linear Discriminant Analysis (LDA) solvers.
# Used to find the optimal regularization for high-dimensional covariance matrices.
SHRINKAGE_GRID = [0.0001, 0.001, 0.01, 0.1, 0.5]
