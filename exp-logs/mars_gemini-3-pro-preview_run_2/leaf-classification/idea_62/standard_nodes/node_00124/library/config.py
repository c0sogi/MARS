import os
import numpy as np

# =============================================================================
# 1. DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
WORKING_DIR = "./working"

# Cache directory for Idea 62 (HDB-PGE)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_62")
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# 3. DATA DEFINITIONS
# =============================================================================
N_CLASSES = 99
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "image_path"

# Feature Column Groups
# Each feature type has 64 attributes
MARGIN_COLS = [f"margin{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture{i}" for i in range(1, 65)]

# Combined Tabular Features
ALL_TABULAR_COLS = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# =============================================================================
# 4. MODEL HYPERPARAMETERS (IDEA 62)
# =============================================================================

# Precision: Use float64 to minimize numerical noise at the metric floor
FLOAT_PRECISION = np.float64

# --- Preprocessing Experts ---
# Robust Transformer settings (Group A)
QUANTILE_N_QUANTILES = 50
QUANTILE_OUTPUT_DIST = "normal"

# Power Transformer settings
POWER_METHOD = "yeo-johnson"

# --- LDA Experts ---
# Solver configuration
LDA_SOLVER = "lsqr"  # Required for shrinkage

# Shrinkage candidates for Group A (Global Statistical Anchors)
# and base estimators in other groups
LDA_SHRINKAGE_CANDIDATES = [0.001, 0.01]

# --- Bottleneck Topologies ---
# Group C: Global Discriminative-Interaction Experts
# Projects 192 features -> 15 discriminative components before poly expansion
LDA_COMPONENTS_GLOBAL = 15

# Group D: Stratified Discriminative-Interaction Experts
# Projects 64 features -> 9 discriminative components per view (Margin, Shape, Texture)
LDA_COMPONENTS_STRATIFIED = 9

# --- Polynomial Expansion ---
# Applied after bottleneck projection
POLY_DEGREE = 2
POLY_INTERACTION_ONLY = False  # Include squared terms
POLY_INCLUDE_BIAS = False

# --- Image Morphometrics (Group B) ---
# Polarity correction: Invert image if corner pixel mean > threshold
POLARITY_THRESHOLD = 0.5

# =============================================================================
# 5. ENSEMBLE SELECTION & EVALUATION
# =============================================================================
# Greedy Forward Selection settings
SELECTION_ITERATIONS = 50
SELECTION_WITH_REPLACEMENT = True

# Metric Clipping (Log Loss stability)
PROB_CLIP_MIN = 1e-15
PROB_CLIP_MAX = 1.0 - 1e-15
