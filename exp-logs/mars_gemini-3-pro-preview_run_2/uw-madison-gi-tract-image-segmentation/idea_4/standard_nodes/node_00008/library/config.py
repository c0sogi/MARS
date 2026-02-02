import os

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# Data Processing Parameters
# ==========================================
# Target size for resizing images.
# Input images vary (e.g., 266x266, 360x310). Resizing ensures consistent spatial feature scale.
IMG_SIZE = (266, 266)

# Pixel intensity normalization range
NORMALIZATION_RANGE = (0, 1)

# Number of channels for 2.5D stacking.
# 3 channels correspond to slices: [i-1, i, i+1]
CHANNELS = 3

# Context depth: 1 means we use 1 slice above and 1 slice below.
CONTEXT_DEPTH = 1

# ==========================================
# Superpixel (SLIC) Parameters
# ==========================================
# Approximate number of superpixels to generate per image
N_SEGMENTS = 1000

# Balances color proximity and space proximity.
# Higher values result in more regular/square shapes.
COMPACTNESS = 10.0

# Width of Gaussian smoothing kernel applied before segmentation
SIGMA = 1.0

# ==========================================
# Class Definitions
# ==========================================
CLASSES = ["background", "large_bowel", "small_bowel", "stomach"]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}
ID_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ==========================================
# Dataset Balancing
# ==========================================
# Fraction of 'background' superpixels to retain in the training set.
# Used to mitigate extreme class imbalance (background vs organs).
BACKGROUND_SAMPLE_RATE = 0.10

# ==========================================
# LightGBM Hyperparameters
# ==========================================
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": NUM_CLASSES,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}

# ==========================================
# Training Loop Parameters
# ==========================================
# Maximum number of boosting rounds
N_ESTIMATORS = 1000

# Stop training if validation metric doesn't improve for this many rounds
EARLY_STOPPING_ROUNDS = 50

# Global random seed for reproducibility
SEED = 42

# ==========================================
# Debugging / Development
# ==========================================
# If True, runs the pipeline on a small subset of cases
DEBUG = False

# Number of cases to use when DEBUG is True
DEBUG_SAMPLE_SIZE = 50
