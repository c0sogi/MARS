import os
import numpy as np

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working/idea_9"

# Ensure working and submission directories exist
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
ID_COL = "id"
TARGET_COL = "species"
GENUS_COL = "genus"  # New target for the coarse-grained supervisor

# Feature Groups
# We will dynamically identify these in the pipeline, but we can define prefixes
MARGIN_PREFIX = "margin"
SHAPE_PREFIX = "shape"
TEXTURE_PREFIX = "texture"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Dimensionality Reduction (Quadratic Branch)
# Retain 95% variance as specified
PCA_VARIANCE = 0.95

# 2. Logistic Regression (Discriminative Linear & Quadratic Branches)
# Solver: lbfgs is standard for multiclass with L2
LR_SOLVER = "lbfgs"
LR_PENALTY = "l2"
LR_MAX_ITER = 10000  # High iteration count to ensure convergence
LR_CV_FOLDS = 5

# Regularization Grid (Cs)
# Constrained to high-signal regime: 10^-2 to 10^4
# We use 20 steps to give the CV ample granularity
LR_CS_GRID = np.logspace(-2, 4, 20).tolist()

# 3. Linear Discriminant Analysis (Generative Linear Branch)
# Solver 'lsqr' supports shrinkage
LDA_SOLVER = "lsqr"
# 'auto' enables Ledoit-Wolf shrinkage
LDA_SHRINKAGE = "auto"

# 4. Polynomial Features (Quadratic Branch)
POLY_DEGREE = 2
POLY_INTERACTION_ONLY = False
POLY_INCLUDE_BIAS = False

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Weights for the Soft-Voting Ensemble (Species Level)
# These can be tuned, but starting with equal weights is robust
WEIGHT_LINEAR = 1.0
WEIGHT_GENERATIVE = 1.0
WEIGHT_QUADRATIC = 1.0
