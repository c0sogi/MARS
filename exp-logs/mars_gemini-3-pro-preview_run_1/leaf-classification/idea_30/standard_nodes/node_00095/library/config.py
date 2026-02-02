import os
import numpy as np

# ==========================================
# Global Configuration & Constants
# ==========================================

# Random Seed for Reproducibility
SEED = 42

# ------------------------------------------
# Directory Paths
# ------------------------------------------
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for Idea 30 to store cached files/models
WORKING_DIR = "./working/optimized"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission File Path
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ------------------------------------------
# Data Definitions
# ------------------------------------------
ID_COL = "id"
TARGET_COL = "species"

# Feature Column Definitions
# The dataset contains 3 sets of 64 attributes each.
# We hardcode the generation to ensure deterministic ordering.
MARGIN_COLS = [f"margin{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture{i}" for i in range(1, 65)]

# Combined Feature List (192 features)
# Order: Margin -> Shape -> Texture
ALL_FEATURES = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# ------------------------------------------
# Pipeline Configuration (Idea 30)
# ------------------------------------------
# Precision settings: We use float64 for high precision matrix operations
FLOAT_PRECISION = np.float64

# Preprocessing Hyperparameters
USE_YEO_JOHNSON = True
YEO_JOHNSON_STANDARDIZE = False  # We apply standard scaler separately
USE_STANDARD_SCALER = True

# Model Hyperparameters (OAS + Cholesky)
OAS_ASSUME_CENTERED = True

# Post-processing
# Probability clipping to avoid log(0)
PROB_CLIP_MIN = 1e-15
PROB_CLIP_MAX = 1.0 - 1e-15

# ------------------------------------------
# Caching Filenames
# ------------------------------------------
# Define paths for cached processed data to speed up iterations
CACHE_TRAIN_X = os.path.join(WORKING_DIR, "X_train_processed.npy")
CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "y_train_processed.npy")
CACHE_VAL_X = os.path.join(WORKING_DIR, "X_val_processed.npy")
CACHE_VAL_Y = os.path.join(WORKING_DIR, "y_val_processed.npy")
CACHE_TEST_X = os.path.join(WORKING_DIR, "X_test_processed.npy")
CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")
CACHE_CLASSES = os.path.join(WORKING_DIR, "classes.npy")
