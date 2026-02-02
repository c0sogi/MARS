import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_23"
SUBMISSION_DIR = "./submission"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# DATA CONSTANTS
# =============================================================================
RANDOM_SEED = 42
ID_COL = "id"
TARGET_COL = "species"
N_CLASSES = 99

# =============================================================================
# EXPERT LIBRARY HYPERPARAMETERS
# =============================================================================

# Group C: Fixed Shrinkage LDA Experts
# These values override the analytical 'auto' (Ledoit-Wolf) shrinkage.
# We span orders of magnitude to capture different degrees of regularization.
SHRINKAGE_GRID = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

# Group D: Gaussian Naive Bayes Experts
# 'var_smoothing' adds a portion of the largest variance to all features for stability.
# This effectively acts as a diagonal regularization parameter.
VAR_SMOOTHING_GRID = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5]

# =============================================================================
# ENSEMBLE SELECTION PARAMETERS
# =============================================================================
# Number of iterations for the Greedy Forward Selection algorithm
SELECTION_ITERATIONS = 100

# =============================================================================
# IMAGE PROCESSING CONSTANTS
# =============================================================================
# Threshold for converting images to strict binary if needed (0-255 scale)
BINARY_THRESHOLD = 127
