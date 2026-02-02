import os

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for caching intermediate files (parquet/npy)
CACHE_DIR = "./working/idea_67"
# Directory for final submission
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "image_path"

# Feature Groups
# The dataset contains 64 attributes for each feature type, indexed 1 to 64.
MARGIN_COLS = [f"margin{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture{i}" for i in range(1, 65)]

# Combined list of all pre-extracted features
ALL_FEATURE_COLS = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# LDA Configuration
# Fixed shrinkage values for Group A (Global Statistical Anchors)
LDA_SHRINKAGE_FIXED = [0.001, 0.01]

# Broader library of shrinkage values for other experts (Groups B & C)
# 'auto' corresponds to Ledoit-Wolf lemma
LDA_SHRINKAGE_LIBRARY = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, "auto"]

# Robust Preprocessing
QUANTILE_N_QUANTILES = 50
QUANTILE_OUTPUT_DIST = "normal"

# Dimensionality Reduction / Bottleneck
# Number of components for LDA Transformer (Discriminative Bottleneck)
LDA_N_COMPONENTS = 9

# Image Processing / Morphometrics
# Threshold for mean corner pixel intensity to detect inverted images (white background vs black background)
INVERT_THRESHOLD = 0.5
