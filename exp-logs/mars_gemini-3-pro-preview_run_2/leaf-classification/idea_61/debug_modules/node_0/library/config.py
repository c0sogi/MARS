import os
import numpy as np

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for Idea 61 as per requirements
WORKING_DIR = "./working/idea_61"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
# Using metadata files which contain the correct splits and image paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Raw Input Paths
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
N_JOBS = 12  # Utilizing available vCPUs
FLOAT_PRECISION = np.float64  # Strict double precision requirement

# =============================================================================
# DATASET SCHEMA
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "image_path"

# Feature Groups
# The dataset provides 64 attributes for each of the three feature types
MARGIN_COLS = [f"margin_{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape_{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture_{i}" for i in range(1, 65)]

# Combined Global View
ALL_FEATURE_COLS = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# Image Processing Constants
IMG_THRESHOLD_CORNER_MEAN = 0.5  # For polarity correction

# =============================================================================
# MODEL HYPERPARAMETERS (FDME STRATEGY)
# =============================================================================
# Group A: Global Statistical Anchors
LDA_SOLVER = "lsqr"
# Shrinkage candidates for the ensemble library
LDA_SHRINKAGE_CANDIDATES = [0.001, 0.01, 0.1, "auto"]

# Group B: Physical Polynomial Experts
POLY_DEGREE = 2

# Group C: Factorized Discriminative-Interaction Experts
# Project each domain to this many components before interaction
FACTORIZED_N_COMPONENTS = 9

# Ensemble Selection
SELECTION_ITERATIONS = 100  # Max iterations for greedy forward selection
SELECTION_TOLERANCE = 1e-6  # Improvement threshold
