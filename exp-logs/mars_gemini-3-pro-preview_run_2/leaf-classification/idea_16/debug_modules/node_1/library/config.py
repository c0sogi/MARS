import os
import numpy as np

# =============================================================================
# GLOBAL SEED
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Metadata paths (Pre-split CSVs)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw data paths
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Output path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"

# Generate feature column names dynamically
# Each feature type has 64 attributes
_NUM_ATTRIBUTES = 64

MARGIN_COLS = [f"margin{i}" for i in range(1, _NUM_ATTRIBUTES + 1)]
SHAPE_COLS = [f"shape{i}" for i in range(1, _NUM_ATTRIBUTES + 1)]
TEXTURE_COLS = [f"texture{i}" for i in range(1, _NUM_ATTRIBUTES + 1)]

# Master list of all 192 features
ALL_FEATURE_COLS = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# =============================================================================
# VIEW CONFIGURATIONS
# =============================================================================
# Dictionary defining the subsets of features for different model views
VIEWS = {
    "Global": ALL_FEATURE_COLS,
    "Margin": MARGIN_COLS,
    "Shape": SHAPE_COLS,
    "Texture": TEXTURE_COLS,
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Logistic Regression (Discriminative Linear)
# Using a dense grid for C to find optimal regularization
LR_CS_GRID = np.logspace(-4, 4, 100)
LR_CV_FOLDS = 5
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 5000

# Random Forest (Discriminative Non-Linear)
RF_N_ESTIMATORS = 500

# Calibration (Isotonic Regression)
CALIBRATION_CV = 3
CALIBRATION_METHOD = "isotonic"
