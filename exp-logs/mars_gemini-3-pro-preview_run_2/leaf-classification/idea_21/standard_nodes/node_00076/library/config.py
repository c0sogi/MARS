import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_21"
SUBMISSION_DIR = "./submission"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

# Output submission path
SUBMISSION_OUTPUT = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache file paths for processed data
CACHE_TRAIN_IMG_FEATURES = os.path.join(WORKING_DIR, "train_morph_features.parquet")
CACHE_VAL_IMG_FEATURES = os.path.join(WORKING_DIR, "val_morph_features.parquet")
CACHE_TEST_IMG_FEATURES = os.path.join(WORKING_DIR, "test_morph_features.parquet")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
RANDOM_SEED = 42
N_JOBS = 12  # Number of vCPUs available

# =============================================================================
# FEATURE COLUMN DEFINITIONS
# =============================================================================
# Original provided features (1-indexed in the dataset)
MARGIN_COLS = [f"margin{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture{i}" for i in range(1, 65)]

# Combined original features
ORIGINAL_FEATURE_COLS = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# New Morphological Features to be engineered
# These names must match the dictionary keys produced by the feature engineering module
MORPHOLOGICAL_COLS = [
    "aspect_ratio",
    "eccentricity",
    "extent",
    "solidity",
    "hu_moment_0",
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
]

# ID and Target columns
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "image_path"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Expert A & B & C: Linear Discriminant Analysis
# Using Ledoit-Wolf shrinkage for high-dimensional stability
LDA_PARAMS = {"solver": "lsqr", "shrinkage": "auto"}

# Expert D: Logistic Regression (Calibrated Backup)
# L2 penalty, L-BFGS solver
LOGREG_PARAMS = {
    "penalty": "l2",
    "C": 10.0,  # Slightly looser regularization as features are clean
    "solver": "lbfgs",
    "max_iter": 2000,
    "multi_class": "multinomial",
    "n_jobs": N_JOBS,
    "random_state": RANDOM_SEED,
}

# Submission Threshold (from Task Description/Previous Best)
SUBMISSION_THRESHOLD = 4.301624233889309e-13

# Calibration for Logistic Regression
CALIBRATION_PARAMS = {"method": "isotonic", "cv": 5}

# Ensemble Selection
# Number of iterations for Greedy Forward Selection
SELECTION_ITERATIONS = 20
