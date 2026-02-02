import os
import numpy as np

# =============================================================================
# GLOBAL SEED
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for caching intermediate files (Idea 26 specific)
WORKING_DIR = "./working/idea_26"
os.makedirs(WORKING_DIR, exist_ok=True)

# Directory for final submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# DATA PROCESSING HYPERPARAMETERS
# =============================================================================
# Data Type for high-precision density estimation
DTYPE = np.float64

# Monte-Carlo Augmentation Settings
N_AUGMENTATIONS = (
    10  # Number of perturbations per image to estimate probabilistic features
)

# Affine Transformation Parameters for Augmentation
AUG_ROTATION_RANGE = 15  # Degrees (+/-)
AUG_SCALE_RANGE = 0.1  # Fraction (+/- 10%)
AUG_SHEAR_RANGE = 10  # Degrees (+/-)

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# LDA Shrinkage Estimators
# 'auto' corresponds to Ledoit-Wolf lemma
# 'oas' corresponds to Oracle Approximating Shrinkage
# Floats correspond to fixed shrinkage coefficients
SHRINKAGE_LIST = ["auto", "oas", 0.001, 0.01, 0.1, 0.2, 0.5]

# QDA Regularization Parameters
# Used for the low-dimensional morphological view
REG_PARAM_LIST = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

# =============================================================================
# FEATURE GROUPS
# =============================================================================
RAW_FEATURE_PREFIXES = ["margin", "shape", "texture"]
