import os

# =============================================================================
# Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# File Paths
# =============================================================================
# Metadata files (Stratified splits)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw input files
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output files
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Data Configuration
# =============================================================================
ID_COLUMN = "id"
TARGET_COLUMN = "species"

# Generate the list of 192 feature column names
# 3 types * 64 attributes each
FEATURE_COLUMNS = []
for feature_type in ["margin", "shape", "texture"]:
    for i in range(1, 65):
        FEATURE_COLUMNS.append(f"{feature_type}{i}")

# =============================================================================
# Model Hyperparameters
# =============================================================================
SEED = 42

# Preprocessing
POWER_TRANSFORM_METHOD = "yeo-johnson"

# Global Expert Parameters
# 'lsqr' is required for shrinkage
GLOBAL_LDA_SOLVER = "lsqr"
# 'auto' enables Ledoit-Wolf shrinkage
GLOBAL_LDA_SHRINKAGE = "auto"

# =============================================================================
# Evaluation Metrics
# =============================================================================
# Epsilon for clipping probabilities to avoid log(0)
CLIPPING_EPSILON = 1e-15
