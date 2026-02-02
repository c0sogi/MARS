import os
import numpy as np

# =============================================================================
# DIRECTORIES AND FILE PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific data directories
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
CACHE_DIR = os.path.join(WORKING_DIR, "idea_54")

# Ensure writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Output submission path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42
FLOAT_PRECISION = (
    np.float64
)  # Strictly use double precision to minimize numerical noise

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
# The dataset provides 192 extracted features divided into 3 groups of 64
N_FEATURES_PER_GROUP = 64

# Generate column names for the provided features
MARGIN_COLS = [f"margin_{i}" for i in range(1, N_FEATURES_PER_GROUP + 1)]
SHAPE_COLS = [f"shape_{i}" for i in range(1, N_FEATURES_PER_GROUP + 1)]
TEXTURE_COLS = [f"texture_{i}" for i in range(1, N_FEATURES_PER_GROUP + 1)]

# Aggregated list of all tabular features
ALL_PROVIDED_FEATURES = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# Interaction pairs for Group C (Cross-Domain Interaction Experts)
# Tuples of (Feature Group 1 Name, Feature Group 1 Cols, Feature Group 2 Name, Feature Group 2 Cols)
INTERACTION_PAIRS = [
    ("margin", MARGIN_COLS, "texture", TEXTURE_COLS),
    ("shape", SHAPE_COLS, "texture", TEXTURE_COLS),
    ("margin", MARGIN_COLS, "shape", SHAPE_COLS),
]

# =============================================================================
# IMAGE PROCESSING CONFIGURATION
# =============================================================================
# Parameters for Polarity-Corrected Morphometrics
INVERT_THRESHOLD = (
    0.5  # Mean pixel value of corners to trigger inversion (white background detection)
)
CORNER_MARGIN = 5  # Number of pixels to check in corners

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Global Linear LDA (Group A)
LDA_SOLVER = "lsqr"  # Solver supporting shrinkage
SHRINKAGE_VALUES = [0.001, 0.01]  # Fixed shrinkage values
SHRINKAGE_AUTO = "auto"  # Corresponds to Ledoit-Wolf/OAS

# Preprocessing Hyperparameters
QUANTILE_N_QUANTILES = 50
QUANTILE_OUTPUT_DIST = "normal"
POWER_METHOD = "yeo-johnson"

# Cross-Domain Interaction (Group C)
BOTTLENECK_N_COMPONENTS = 15  # Dimension of discriminative subspace before expansion
POLY_DEGREE = 2  # Degree for polynomial expansion

# Ensemble Selection
SELECTION_METRIC = "log_loss"
